# SimpleCNN 训练数据在线构造

## 1. 作用、分块与符号

本文件描述 SimpleCNN 当前使用的训练、验证和独立评估数据流。完整合成序列不预先展开为大量重叠局部块；训练阶段从压缩二值观测中在线裁剪，验证和评估阶段从固定标准网格恢复相同形状的块。

一份完整序列的二值观测记为

$$
X_{t,r}\in\{0,1\},
\qquad
t=0,\ldots,F-1,
\qquad
r=0,\ldots,R-1.
$$

当前合成数据取 $F=300$、$R=300000$；每个距离 bin 对应 $1\ \mathrm m$。与 [SimpleCNN 方法说明](SimpleCNN.md) 及 [后处理方法说明](postprocess.md) 一致，局部块长度和距离跨度为

$$
F_{\mathrm{line}}=20,
\qquad
W_R=10000\ \mathrm m.
$$

以时间起点 $t_n$、距离起点 $s$ 截取的局部块记为

$$
\mathcal B(t_n,s)
=
\left\{
X_{t_n+\tau,\,s+d}:
\tau=0,\ldots,F_{\mathrm{line}}-1,\
d=0,\ldots,W_R-1
\right\}.
$$

块内时间索引为 $\tau$，中心时刻为 $\tau_0=9.5$。网络对该块输出 $(\hat q,\hat\rho,\hat\nu)$，其预测直线为

$$
\hat\ell_\tau
=
\hat\rho+\hat\nu(\tau-\tau_0).
$$

设合成器保存的潜在真实距离为 $r^{\mathrm{true}}_t$。在块 $\mathcal B(t_n,s)$ 中，逐帧目标响应掩码、响应的块内距离和响应数定义为

$$
I_\tau
=
\mathbf 1\!\left(
\text{第 }t_n+\tau\text{ 帧存在目标响应，且其距离落在 }[s,s+W_R)
\right),
$$

$$
d_\tau
=
r^{\mathrm{hit}}_{t_n+\tau}-s,
\qquad I_\tau=1,
$$

$$
H=\sum_{\tau=0}^{F_{\mathrm{line}}-1}I_\tau.
$$

其中 $r^{\mathrm{hit}}_t$ 是实际注入目标响应所在的 $1\ \mathrm m$ 整数 bin。$I_\tau=0$ 时 $d_\tau$ 没有物理定义，返回张量以 $-1$ 填充。

## 2. 完整序列字段与局部观测读取

每份序列保存在独立目录的 `data.npz` 中。训练和固定网格使用的字段如下。

| 字段 | 数据格式 | 当前用途 |
| --- | --- | --- |
| `observation_packed` | `uint8`，$[F,\lceil R/8\rceil]=[300,37500]$ | $X_{t,r}$ 沿距离轴的位打包结果；网络输入唯一来自该字段。 |
| `target_hit` | `bool`，$[F]$ | 给出逐帧是否额外注入目标响应，用于构造 $I_\tau$。 |
| `target_hit_bin` | `int32`，$[F]$ | $I_\tau=1$ 时给出 $r^{\mathrm{hit}}_t$；漏检帧的值为 $-1$。 |
| `target_true_range_m` | `float32`，$[F]$ | 给出 $r^{\mathrm{true}}_t$，仅用于判定潜在轨迹与块的空间关系。 |

`target_true_range_m` 不进入网络输入，也不参与损失。几何监督只使用 `target_hit` 与 `target_hit_bin` 导出的 $(I_\tau,d_\tau)$。

对于距离起点 $s$，仅解包覆盖该局部块的字节。令

$$
j_0=\left\lfloor\frac{s}{8}\right\rfloor,
\qquad
o=s\bmod 8,
\qquad
J=\left\lceil\frac{o+W_R}{8}\right\rceil.
$$

读取 $[t_n,t_n+F_{\mathrm{line}})\times[j_0,j_0+J)$ 的压缩片段，以与合成器一致的 `big` 位序解包，再保留偏移 $o$ 之后的 $W_R$ 个 bin。该过程直接得到

$$
U_{\tau,d}=X_{t_n+\tau,s+d}
\in\{0,1\},
\qquad
U\in\{0,1\}^{20\times10000}.
$$

当 $s$ 不是 $8$ 的倍数时，字节边界外最多额外解包 $7$ 个 bin；随机裁剪因此同时覆盖不同的距离模 $8$ 相位。

随后按局部距离索引 $d\bmod8$ 无损重排，得到与 [SimpleCNN 方法说明](SimpleCNN.md) 第 2 节相同的输入张量：

$$
Z_{c,\tau,k}=U_{\tau,8k+c},
\qquad
c=0,\ldots,7,
\quad
k=0,\ldots,1249,
$$

$$
[20,10000]\longrightarrow[8,20,1250].
$$

进入网络前，$Z$ 从二值数组转换为 `float32`；当前数据流不添加背景归一化、显著性变换或其他预处理通道。

## 3. 完整序列划分与无限训练流

训练、验证和测试以完整序列为最小划分单位。一份序列的不同时间窗或不同距离块不会被分入不同集合。初次创建实验时，从数据根目录枚举配置数量 $N_{\mathrm{src}}$ 的完整序列，按固定划分种子随机排列，并依照当前比例分为

$$
\mathcal D_{\mathrm{train}},
\qquad
\mathcal D_{\mathrm{val}},
\qquad
\mathcal D_{\mathrm{test}}.
$$

该划分连同源序列相对路径持久化到实验目录；后续恢复训练直接复用同一份划分。

分布式训练先把 $\mathcal D_{\mathrm{train}}$ 按 rank 无重叠切分，再在每个 rank 内按 worker 无重叠切分。每个 worker 缓存 $\min\!\left(K_{\mathrm{cache}},\lvert\mathcal D_{\mathrm{worker}}\rvert\right)$ 份已解压的完整序列，当前默认 $K_{\mathrm{cache}}=4$。缓存中的一份序列累计产生 $Q_{\mathrm{src}}$ 个可见正样本后触发替换，当前默认 $Q_{\mathrm{src}}=64$：当缓存未覆盖本 worker 的全部来源时，当前位置替换为一份未在缓存中的新序列；当缓存已覆盖全部来源时，仅将该序列的计数清零。

各 worker 的随机序列由全局随机种子、rank、worker 编号和数据流代次共同确定。恢复训练时数据流代次加一，因此恢复后的在线裁剪使用新的随机序列，而不从旧流开头重复取样。

训练流没有有限 epoch。训练预算由优化器更新次数控制；缓存替换时按随机排列依次取用来源，排列走完后重新随机排列。

### 3.1 每个训练 batch 的正负比例

设每张卡的 batch 大小为 $B$，可见正样本目标比例为 $\alpha_{\mathrm{pos}}$。当前 batch 内的样本数为

$$
N_{\mathrm{pos}}
=
\min\!\left(
B-1,
\max\!\left(1,\left\lfloor B\alpha_{\mathrm{pos}}+\frac12\right\rfloor\right)
\right),
\qquad
N_{\mathrm{neg}}=B-N_{\mathrm{pos}}.
$$

当前默认 $B=32$、$\alpha_{\mathrm{pos}}=0.25$，即每个本地 batch 含 $8$ 个可见正样本和 $24$ 个负样本。该取整规则适用于任意 $B\ge2$，无需令 $B$ 被 $4$ 整除。正、负样本构造完成后在 batch 内随机重排。

## 4. 训练样本与标签

### 4.1 潜在轨迹和块关系

记当前时间窗的潜在轨迹距离范围为

$$
r_{\min}=\min_{0\le\tau<20}r^{\mathrm{true}}_{t_n+\tau},
\qquad
r_{\max}=\max_{0\le\tau<20}r^{\mathrm{true}}_{t_n+\tau}.
$$

相对于块的半开距离区间 $[s,s+W_R)$，潜在轨迹具有以下三种关系：

$$
\begin{cases}
\text{完整包含}, & r_{\min}\ge s\ \text{且}\ r_{\max}<s+W_R,\\
\text{部分相交}, & \exists\tau,\ r^{\mathrm{true}}_{t_n+\tau}\in[s,s+W_R),\ \text{但不完整包含},\\
\text{完全不相交}, & \nexists\tau,\ r^{\mathrm{true}}_{t_n+\tau}\in[s,s+W_R).
\end{cases}
$$

以置信度标签 $q^*$、置信度掩码 $m_q$ 和几何掩码 $m_{\mathrm{line}}$ 表示监督关系。它们分别对应 batch 返回的 `q`、`q_valid` 和 `is_positive`。

| 样本类型 | 潜在轨迹关系与响应数 | $q^*$ | $m_q$ | $m_{\mathrm{line}}$ |
| --- | --- | ---: | ---: | ---: |
| 可见正样本 | 完整包含，且 $H>0$ | $1$ | $1$ | $1$ |
| 零证据样本 | 完整包含，且 $H=0$ | 无定义 | $0$ | $0$ |
| 部分相交负样本 | 部分相交 | $0$ | $1$ | $0$ |
| 背景负样本 | 完全不相交 | $0$ | $1$ | $0$ |

该定义与 [SimpleCNN 方法说明](SimpleCNN.md) 第 3、5 节一致。零证据样本在张量中以 $q=0$ 占位，但 $m_q=0$，不产生置信度损失；它也不产生几何损失。部分相交块中的 $(I_\tau,d_\tau)$ 仍按实际响应位置生成，但 $m_{\mathrm{line}}=0$，因此这些响应不参与几何损失。

### 4.2 可见正样本

在线训练先均匀选择时间起点

$$
t_n\in\{0,1,\ldots,F-F_{\mathrm{line}}\}.
$$

正样本空间起点要求完整潜在轨迹与块边界保留 $g_{\mathrm{pos}}$ 的距离裕量。有效整数起点区间为

$$
\left\lceil
\max\!\left(0,r_{\max}+g_{\mathrm{pos}}-W_R\right)
\right\rceil
\le s\le
\left\lfloor
\min\!\left(R-W_R,r_{\min}-g_{\mathrm{pos}}\right)
\right\rfloor.
$$

当前默认 $g_{\mathrm{pos}}=100\ \mathrm m$。在有效区间中，$30\%$ 的正样本从推理/验证标准距离起点集合中均匀选择：

$$
\mathcal S_R
=
\{0,9000,18000,\ldots,288000,290000\}.
$$

其余 $70\%$ 从全部有效整数起点均匀选择；若有效区间内没有标准起点，则直接采用随机起点。采样结果仅在 $H>0$ 时作为可见正样本进入训练，因此在线训练流不包含零证据样本。

### 4.3 负样本

每个负样本关联一个同 batch 的正样本时间窗；当 $N_{\mathrm{neg}}>N_{\mathrm{pos}}$ 时，正样本上下文按循环次序复用。设四类负样本的权重为

$$
w_{\mathrm{local}},\quad
w_{\mathrm{same}},\quad
w_{\mathrm{random}},\quad
w_{\mathrm{partial}},
$$

则每次抽取类别 $k$ 的概率为

$$
p_k=\frac{w_k}{
w_{\mathrm{local}}+w_{\mathrm{same}}+w_{\mathrm{random}}+w_{\mathrm{partial}}}.
$$

当前四个权重均为 $1$，故每类负样本的长期占比均为 $1/4$。在默认正样本比例下，每类负样本长期约占 batch 的 $18.75\%$。

对于完全不相交背景块，当前时间窗的可用距离起点集合为

$$
\mathcal A_{\mathrm{dis}}
=
\left\{
s\in\mathbb Z:
0\le s\le R-W_R,\\
s+W_R\le r_{\min}-g_{\mathrm{neg}}
\quad\text{或}\quad
s\ge r_{\max}+g_{\mathrm{neg}}
\right\},
$$

其中当前默认保护距离为 $g_{\mathrm{neg}}=100\ \mathrm m$。三类完全不相交背景负样本均使用 $q^*=0$、$m_q=1$、$m_{\mathrm{line}}=0$，并返回全零 $I_\tau$ 与填充值 $d_\tau=-1$。

1. **同时间窗局部负样本：** 使用关联正样本的 $(t_n,r_{\min},r_{\max})$。先按可用起点数选择轨迹近侧的左、右可行区间，再在离轨迹最近的 $L_{\mathrm{local}}$ 个起点内均匀选择。当前 $L_{\mathrm{local}}=3000\ \mathrm m$。

2. **同时间窗分层负样本：** 使用关联正样本的 $t_n$ 和 $\mathcal A_{\mathrm{dis}}$。以块中心 $s+W_R/2$ 划分近场 $[0,10)\ \mathrm{km}$、过渡区 $[10,50)\ \mathrm{km}$ 与远场 $[50,300)\ \mathrm{km}$；先在存在可行起点的距离层中均匀选择一层，再在该层的可用整数起点中均匀选择。

3. **随机时空负样本：** 从当前 worker 的缓存序列中随机选择来源，随机选择 $t_n$，再按相同的三层规则从 $\mathcal A_{\mathrm{dis}}$ 选择 $s$。

4. **部分相交负样本：** 使用关联正样本的 $t_n$，从潜在轨迹进入块、但被左边界或右边界截断的起点集合中选择 $s$。该块满足“部分相交”关系，以 $q^*=0$、$m_q=1$、$m_{\mathrm{line}}=0$ 监督；其中可出现实际目标响应点，但它们不参与几何损失。

当前训练流不在含有可见目标的同一输入块上附加随机错误直线标签。SimpleCNN 从一个输入块直接回归唯一 $(\hat q,\hat\rho,\hat\nu)$；随机候选直线与该块真实可见轨迹会形成矛盾监督。

## 5. batch 张量与损失掩码

每个训练 batch 或标准网格 batch 返回以下张量：

| 数学量 | 返回字段 | 数据格式 | 含义 |
| --- | --- | --- | --- |
| $Z$ | `x` | `float32`，$[B,8,20,1250]$ | 无损重排后的原始二值观测。 |
| $I$ | `I` | `bool`，$[B,20]$ | 逐帧实际目标响应掩码。 |
| $d$ | `d` | `int64`，$[B,20]$ | 有效帧的块内响应距离；无响应帧为 $-1$。 |
| $q^*$ | `q` | `float32`，$[B]$ | 二值置信度目标；零证据样本的存储值为 $0$。 |
| $m_q$ | `q_valid` | `bool`，$[B]$ | 置信度损失掩码。 |
| $m_{\mathrm{line}}$ | `is_positive` | `bool`，$[B]$ | 几何损失的块级掩码。 |

与 [SimpleCNN 方法说明](SimpleCNN.md) 第 5.2 节一致，置信度损失只对 $m_q=1$ 的样本计算；几何损失同时要求

$$
m_{\mathrm{line}}=1
\qquad\text{且}\qquad
I_\tau=1.
$$

因此 $d_\tau=-1$ 永不进入 Huber 损失。每个可见正样本先按自身响应数 $H$ 归一化，再在可见正样本块之间求平均；目标响应较少的样本不会因 $H$ 较小而被降低块级几何权重。

## 6. 固定验证、测试与独立评估网格

训练使用随机在线裁剪；验证、测试和独立评估使用固定标准网格。标准距离步进与 [postprocess.md](postprocess.md) 的全局搜索一致：

$$
S_R=9000\ \mathrm m,
\qquad
\mathcal S_R=\{0,9000,18000,\ldots,288000,290000\}.
$$

相邻块重叠

$$
O_R=W_R-S_R=1000\ \mathrm m.
$$

设固定网格的时间步进为 $S_T^{\mathrm{val}}\in\mathbb Z_{\ge1}$。对于 $F$ 帧序列，时间起点集合为

$$
\mathcal T_{\mathrm{val}}
=
\left\{
0,S_T^{\mathrm{val}},2S_T^{\mathrm{val}},\ldots,
\left\lfloor\frac{F-F_{\mathrm{line}}}{S_T^{\mathrm{val}}}\right\rfloor
S_T^{\mathrm{val}}
\right\}
\cup
\left\{F-F_{\mathrm{line}}\right\}.
$$

末尾时间窗始终存在，因此最后一帧被至少一个 $20$ 帧窗口覆盖。固定网格为

$$
\mathcal V
=
\left\{
\mathcal B(t_n,s):
t_n\in\mathcal T_{\mathrm{val}},\ s\in\mathcal S_R
\right\}.
$$

当前训练的 $S_T^{\mathrm{val}}=5$ 帧；设为 $1$ 时，序列中每个可行的连续 $20$ 帧窗口均进入验证。该步进独立于后处理 TRACK 阶段的 $S_T^{\mathrm{track}}$。

固定网格清单保存每个块的来源序列、$(t_n,s)$、$(I,d)$、$m_q$ 和 $m_{\mathrm{line}}$，并记录来源文件大小与纳秒级修改时间。缓存命中时先校验 schema、数组形状、数据类型和来源签名；训练或独立评估的 rank $0$ 还会随机复算 $10$ 个块的标签。源文件或网格参数变化后，清单重新生成。

训练启动时生成验证集的固定网格；测试集网格在独立测试时创建。独立评估入口还支持直接以数据根目录下的全部序列构成 $\mathcal V$，此时不读取训练/验证/测试划分。

多卡完整评估按完整来源序列将固定网格分给各 rank，不填充重复块。设置有限评估 batch 数时，训练期间的验证在各 rank 所属网格内有放回抽样；独立评估在各 rank 所属网格内无放回抽取有限子集，并按来源顺序读取以减少重复解压。评估 batch 大小只影响单次前向的并行规模，不改变单个块的标签或形状。

## 7. 运行时一致性检查

数据读取阶段验证以下约束：

- 压缩观测的形状为 $[F,\lceil R/8\rceil]$，且 $F\ge F_{\mathrm{line}}$；
- 三个目标标签数组均为 $[F]$，潜在轨迹距离位于 $[0,R)$；
- 可见响应的距离 bin 位于 $[0,R)$；
- 固定网格中的 $s$ 位于 $[0,R-W_R]$，有效 $d_\tau$ 位于 $[0,W_R)$；
- 固定网格中 $I_\tau=1$ 与 $d_\tau\ge0$ 完全对应；
- $m_{\mathrm{line}}=1$ 的块至少含一个实际目标响应，且同时满足 $m_q=1$；
- 训练 rank 与 worker 获得非空、互不重叠的完整序列来源。

这些检查将位序错误、标签错位、分块越界和跨集合来源泄漏在训练或评估开始前显式报告。
