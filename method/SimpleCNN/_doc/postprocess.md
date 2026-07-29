# SimpleCNN：高效流式推理与后处理

## 1. 目标与约束

SimpleCNN 对一个 $20\times10000$ 时间—距离块输出：

$$
(\hat q,\hat\rho,\hat\nu),
$$

其中 $\hat q$ 是当前块包含可见完整目标轨迹的置信度，$\hat\rho$ 是中心时刻的块内距离，$\hat\nu$ 是斜率，单位为 $\mathrm{m/frame}$。

本文设计一个面向实时运行的“捕获—跟踪—重捕获”后处理方法。第一版遵循以下约束：

1. 每批候选只使用 $\hat q$ 选择 Top-1，不维护 Top-$K$ 多假设；
2. 使用运动连续性过滤不合理跳变；
3. 暂不读取原始二值数据对候选直线重新打分；
4. 跟踪稳定后只推理目标附近的少量距离块；
5. 初版的全局扫描只在捕获和重捕获时执行。

当前 SimpleCNN 的三个输出 head 都位于完整 CNN 和共享 MLP 之后。只跳过 $\rho,\nu$ 两个标量 head 几乎不能减少计算量，因此本文不把“全局只计算 $q$ head”视为有效加速手段。主要加速来源是减少每个步进实际送入 CNN 的距离块数量。

本文默认沿用：

$$
F_{\mathrm{line}}=20,\qquad
W_R=10000\ \mathrm m,\qquad
S_R=9000\ \mathrm m,
$$

完整 $0\sim300\ \mathrm{km}$ 距离轴共划分为 $M=34$ 个标准块。

---

## 2. 统一候选的时间和距离坐标

### 2.1 CNN 输出对应的时间

当前20帧窗口记为 $\{t-19,t-18,\ldots,t\}$，其中 $t$ 是最新帧。块内时间索引为 $\tau=0,\ldots,19$，中心时刻为 $\tau_0=\frac{19}{2}=9.5$。对于起始距离为 $r_m$ 的第 $m$ 个块
- 窗口中心帧距离为：$\hat r_t^{\mathrm{center},(m)}=r_m+\hat\rho_t^{(m)}$
- 窗口最新帧距离为：$\hat z_t^{(m)}=r_m+\hat\rho_t^{(m)}+9.5\hat\nu_t^{(m)}$
- 下一帧候选位置为：$\hat z_{t+1}^{(m)}=r_m+\hat\rho_t^{(m)}+10.5\hat\nu_t^{(m)}$

后续的候选比较、距离统计和运动连续性判断统一使用最新帧位置 $\hat z_t$。只有需要画出完整20帧直线时，才直接使用中心距离 $\hat\rho$。

### 2.2 Top-1 候选

对本次实际推理的块集合 $\mathcal M_t$，预测目标存在概率最高的块索引为：

$$
m_t^*=
\underset{m\in\mathcal M_t}{\arg\max}\;
\hat q_t^{(m)}.
$$

当前步进的 Top-1 候选为：

$$
C_t
=
\left(
\hat q_t^{(m_t^*)},\hat z_t^{(m_t^*)},\nu_t^{(m_t^*)},m_t^*
\right).
$$

不同重叠块可能对同一条轨迹给出不同的块内参数。因此比较局部候选和全局候选时，应比较换算后的 $(\hat z_t,\hat\nu_t)$，不能要求二者具有相同块编号。

---

## 3. 状态与总体流程

后处理器维护三个工作状态：

```text
CAPTURE：全局捕获
    ↓ 
    ↓ 多次全局 Top-1 在运动补偿后形成稳定聚集
    ↓ 
TRACK：局部跟踪
    ↓ 
    ↓ 连续多次无有效局部候选
    ↓ 
RECAPTURE：全局重捕获
    ↓ 
    ↓ 重新形成稳定候选
    ↓ 
TRACK
```

运行时至少维护：

- 当前状态 `mode`；
- 最新估计距离 $r_t$；
- 最新估计斜率 $\nu_t$，单位为 $\mathrm{m/frame}$；
- 捕获候选缓存 $\mathcal C$；
- 当前局部搜索等级 $L$；
- 连续有效计数 $N_{\mathrm{good}}$；
- 连续无效计数 $N_{\mathrm{bad}}$；
- 当前输出是否由 CNN 测量更新，还是仅由运动模型外推。

---

## 4. 捕获模式

### 4.1 全局 Top-1 序列

捕获模式还不知道目标大致位于哪个距离，因此需要在完整 $0\sim300\ \mathrm{km}$ 距离范围内搜索。第 $i$ 次扫描时，考察当前时刻 $t_i$ 之前最近 20 帧窗口，取全部34个标准距离块组成一个 batch 送入 SimpleCNN。得到 Top-1 候选 $C_{t_i}$

由于相邻20帧窗口高度重叠，捕获阶段不一定需要每到一帧就重复一次全局扫描。定义捕获扫描步进为：

$$
t_{i+1}-t_i
=
S_T^{\mathrm{cap}},
\qquad
S_T^{\mathrm{cap}}=2\sim5\ \text{frames}.
$$
单次全局 Top-1 可能来自背景误报，因此还需要观察它能否在多次扫描中持续出现在一致的运动轨迹附近。在 $t$ 时刻，已缓存最近 $K_t$ 次扫描的 Top-1 候选
$$
\mathcal C_t
=
\left\{
C_{t_1},C_{t_2},\ldots,C_{t_{K_t}}
\right\},
\qquad
t_1<t_2<\cdots<t_{K_t}\le t.
$$
设 Top-1 后续的缓存容量上限为 $K_{\mathrm{cap}}$，使用先进先出的滑动缓存。后续捕获判断就是检查这组候选在经过运动补偿后，是否集中到同一个距离区域。

### 4.2 运动补偿到同一参考时刻

目标在捕获期间持续移动，不能直接对不同时刻的 $\hat z_{t_i}$ 做固定距离直方图。以当前最新时刻 $t$ 为参考，把此前 $t_i$ 时刻 Top-1 候选 $C_{t_i}$ 的目标距离预测 $\hat z_{t_i}$ 外推到 $t$ 时刻，即
$$
\tilde z_{t_i\rightarrow t}
=
\hat z_{t_i}
+(t-t_i)\hat\nu_{t_i}.
$$
这里的 $R_{\mathrm{cap}}$ 是判断多次外推位置是否聚集在同一目标附近的**位置一致性半径**，通常只有数百米；它不是进入跟踪后的实际 CNN 搜索宽度。捕获成功后，得到的候选位置只用于选择一个完整宽度为 $W_R=10\ \mathrm{km}$ 的跟踪块。

### 4.3 捕获确认

第一版不需要在冷启动阶段加入过多硬条件。先将缓存中全部外推位置的中位数作为候选捕获位置：

$$
r_t^{\mathrm{cap}}
=
\operatorname{median}
\left\{
\tilde z_{t_i\rightarrow t}:i=1,\ldots,K_t
\right\}.
$$

再定义支持该中位数的历史候选集合及其数量：

$$
\mathcal A_t^{\mathrm{cap}}
=
\left\{
i:
\left|
\tilde z_{t_i\rightarrow t}-r_t^{\mathrm{cap}}
\right|
\le R_{\mathrm{cap}}
\right\},
\qquad
N_t^{\mathrm{cap}}
=
\left|\mathcal A_t^{\mathrm{cap}}\right|.
$$

由于捕获只需把目标定位到一个10 km块内，而非立即精确拟合航迹，因此仅在以下条件满足时切换到 `TRACK`：

1. 缓存已满，保证已经考察足够的时间长度：$K_t=K_{\mathrm{cap}}$；
2. 支持中位捕获位置的候选数足够多：$N_t^{\mathrm{cap}}\ge M_{\mathrm{cap}}$。

阈值可以用比例形式描述：
$$
M_{\mathrm{cap}}
=
\left\lceil
\eta_{\mathrm{cap}}K_{\mathrm{cap}}
\right\rceil.
$$

建议取 $\eta_{\mathrm{cap}}>0.5$，即要求严格多数的历史候选支持同一中位位置；这样少量偶发的全局 Top-1 误报不会主导捕获结果。每次扫描内部仍由 $\hat q$ 选择 Top-1；捕获确认阶段只使用位置一致性，不再为 $\hat q$ 增加额外门限。

初始斜率只取支持中位捕获位置的候选对应 $\hat\nu_{t_i}$ 的中位数：

$$
\nu_t^{\mathrm{cap}}
=
\operatorname{median}
\left\{
\hat\nu_{t_i}:
i\in\mathcal A_t^{\mathrm{cap}}
\right\}.
$$

推理块不需要固定到标准距离网格：训练数据本身包含随机距离裁剪，因此捕获后可直接以20帧窗口中心时刻的预测距离为中心截取一个总宽度为 $W_R$ 的块。令：

$$
c_t^{\mathrm{cap}}
=
r_t^{\mathrm{cap}}
-\frac{19}{2}\nu_t^{\mathrm{cap}}.
$$

将块起点取为最接近的整数距离 bin，并限制在完整测距范围内：

$$
s_{\mathrm{cap}}
=
\operatorname{clip}_{[0,\,300000-W_R]}
\left(
\operatorname{round}
\left[
c_t^{\mathrm{cap}}-\frac{W_R}{2}
\right]
\right).
$$

初始跟踪区域为：

$$
\mathcal B_{\mathrm{cap}}
=
[s_{\mathrm{cap}},s_{\mathrm{cap}}+W_R).
$$

切换到 `TRACK` 时初始化：

$$
r_t=r_t^{\mathrm{cap}},
\qquad
\nu_t=\nu_t^{\mathrm{cap}},
\qquad
L=0.
$$

因此候选聚类区只负责找到首个10 km跟踪块；它本身不是后续持续使用的搜索区域。

---

## 5. 跟踪模式

### 5.1 匀速运动预测

设上一次有效状态位于第 $t-\Delta t$ 帧，当前推理步进相隔 $\Delta t$ 帧。使用匀速模型预测：

$$
r_t^-
=
r_{t-\Delta t}
+\Delta t\,\nu_{t-\Delta t},
$$

$$
\nu_t^-=\nu_{t-\Delta t}.
$$

短时间内目标动力学变化远小于当前 CNN 的几何测量误差，因此第一版不需要更复杂的加速度模型。

### 5.2 根据预测中心距离选择局部块

局部跟踪不需要固定到标准距离网格。当前20帧窗口中心时刻的运动预测距离为：

$$
c_t^-
=
r_t^-\;-\;\frac{19}{2}\nu_t^-.
$$

以此为中心截取总宽度为 $W_R$ 的中心块，其起点为：

$$
s_{t,0}
=
\operatorname{clip}_{[0,\,300000-W_R]}
\left(
\operatorname{round}
\left[
c_t^-\;-\;\frac{W_R}{2}
\right]
\right).
$$

正常跟踪时只推理该块：

$$
\mathcal B_{t,0}
=
[s_{t,0},s_{t,0}+W_R),
\qquad
\mathcal M_t^{(0)}
=
\{\mathcal B_{t,0}\}.
$$

搜索变差时，按“增加完整 $W_R$ 块”的粒度向左右扩展。相邻局部块的起点相隔 $S_R=9\ \mathrm{km}$，因此相邻 $W_R=10\ \mathrm{km}$ 块仍保留1 km重叠：

| 搜索等级 | 实际推理块 |
|---|---|
| $L=0$ | 中心 $W_R$ 块，共1块 |
| $L=1$ | 中心块加左右各1个 $W_R$ 块，共3块 |
| $L=2$ | 中心块加左右各2个 $W_R$ 块，共5块 |
| $L=3$ | 不再做局部推理，进入全局重捕获 |

形式化地，对 $k=-L,\ldots,L$，定义：

$$
s_{t,k}
=
\operatorname{clip}_{[0,\,300000-W_R]}
\left(
s_{t,0}+kS_R
\right),
$$

$$
\mathcal B_{t,k}
=[s_{t,k},s_{t,k}+W_R),
\qquad
\mathcal M_t^{(L)}
=
\left\{
\mathcal B_{t,k}:k=-L,\ldots,L
\right\}.
$$

测距边界附近裁剪后可能出现重复块，实际实现应先去重。同一步进选中的所有局部块组成一个 batch，只执行一次 CNN 前向，再从其中选取局部 $\hat q$-Top1。

### 5.3 连续性门控

CNN 给出局部 Top-1 后，不能立刻相信它，而要先检查它是否和上一时刻的轨迹预测一致。设上一次有效状态位于第 $t-\Delta t$ 帧，已经有状态 $(r_{t-\Delta t},\nu_{t-\Delta t})$，当前推理步进相隔 $\Delta t$ 帧。使用匀速模型预测：

$$
r_t^-
=
r_{t-\Delta t}
+\Delta t\,\nu_{t-\Delta t},
$$

CNN 对当前局部块给出 Top-1 候选，其最新帧距离为 $\hat z_t$，设两者之差为：$e_t=\hat z_t-r_t^-$。先按第 5.4 节计算候选融合后的暂定状态 $(r_t^+,\nu_t^+)$；当且仅当以下条件满足时认为本次 CNN 候选可信：
- $\hat q_t\ge q_{\mathrm{keep}}$：目标存在置信度超过阈值，CNN 自己认为该块确有目标；
- $|\nu_t^+|\le V_{\mathrm{inst}}(L)$：单帧融合状态的绝对速度不超过当前搜索等级的上限；
- 当 TRACK 历史已覆盖至少 $N_{\mathrm{avg}}$ 帧时，取不晚于 $t-N_{\mathrm{avg}}$ 的最近状态锚点 $(t_0,r_{t_0})$，并要求

$$
\left|\bar\nu_{t,N}\right|
=
\left|\frac{r_t^+-r_{t_0}}{t-t_0}\right|
\le V_{\mathrm{avg}}(L).
$$

两类速度均以米/帧计，并使用真实帧号之差，而不是推理步数；因此相同参数在不同 `time_stride` 下对应相同的物理速度语义。历史不足 $N_{\mathrm{avg}}$ 帧时，仅使用单帧速度门控。


无论远距离候选的 $\hat q$ 多高，只要没有通过运动连续性门控，就不能立即把轨迹状态跳转过去。

### 5.4 状态更新

通过门控后，使用计算量很小的 $\alpha$-$\beta$ 滤波更新位置和斜率：

$$
r_t
=
r_t^-+\alpha e_t,
$$

$$
\nu_t^{\mathrm{pos}}
=
\nu_t^-
+\frac{\beta}{\Delta t}e_t.
$$

可再以较小权重 $\gamma$ 融合 CNN 直接给出的斜率：

$$
\nu_t
=
(1-\gamma)\nu_t^{\mathrm{pos}}
+\gamma\hat\nu_t.
$$

如果候选没有通过门控，本步不更新 CNN 测量，只保留：

$$
r_t=r_t^-,
\qquad
\nu_t=\nu_t^-.
$$

这允许跟踪器跨过少量低置信度或漏检步进，同时避免错误候选形成距离跳变。

---

## 6. 搜索范围的扩大与收缩

局部搜索使用迟滞机制：

- 候选变差时快速扩大搜索范围；
- 候选恢复后连续稳定若干步再缓慢收缩；
- 不因单次好坏结果在两个等级之间反复振荡。

一次局部候选可定义为“有效”，当且仅当它同时通过 $\hat q$ 门限和位置连续性门控。

### 6.1 扩大条件

第 5.3 节的门控未通过，即 $\hat q_t<q_{\mathrm{keep}}$、$|\nu_t^+|>V_{\mathrm{inst}}(L)$，或历史充分时 $|\bar\nu_{t,N}|>V_{\mathrm{avg}}(L)$，累计一次失败并清零连续有效计数。连续失败达到 $N_{\mathrm{expand}}$ 时，若 $L<2$，则：

$$
L\leftarrow L+1,
$$

并清零连续失败计数。初版只使用 $L=0,1,2$ 三个局部搜索等级；若已经处于 $L=2$ 且再次达到扩大条件，则清空捕获缓存并进入 `RECAPTURE`。

### 6.2 收缩条件

每次候选通过第 5.3 节门控时，连续有效计数加一并清零连续失败计数。连续获得 $N_{\mathrm{shrink}}$ 次有效候选后：

$$
L\leftarrow\max(0,L-1).
$$

随后清零连续有效计数，重新累计下一次收缩所需的有效候选。一般应取：

$$
N_{\mathrm{shrink}}>N_{\mathrm{expand}},
$$

实现“快速扩大、缓慢收缩”。初版不再增加单独的强候选门限；后续若观察到搜索范围过早收缩，再引入更严格的 $q$ 或位置新息条件。

---

## 7. 初版暂不启用周期性全局看门狗

正常跟踪时只推理第 5.2 节选择的 $1/3/5$ 个局部块，不在固定周期额外执行全局34块扫描。这样初版只需验证“局部跟踪是否能在候选失效后通过重捕获恢复”，避免引入全局/局部候选匹配、看门狗周期和额外前向等尚未标定的规则。

后续若发现局部轨迹会在候选持续有效时静默跑偏，再单独加入周期性全局看门狗，并用固定验证序列标定其周期和全局—局部一致性门限。

---

## 8. 重捕获模式

当局部搜索已处于 $L=2$、且再次连续失败达到 $N_{\mathrm{expand}}$ 时，进入 `RECAPTURE`。

初版的重捕获不引入新的确认规则，而是直接复用第 4 节的全局捕获流程：

1. 清空旧的捕获缓存，并丢弃旧轨迹状态；
2. 每次扫描推理全部34个标准距离块，保留全局 $\hat q$-Top-1；
3. 按第 4.2 节进行运动补偿，并按第 4.3 节的“中位捕获位置 + 支持数”确认新轨迹；
4. 确认成功后，按第 4.3 节初始化状态并进入 `TRACK`。

因此，`RECAPTURE` 与冷启动 `CAPTURE` 的全局扫描和确认逻辑相同，只是前者在输出和评估中标记为“丢失后的重新捕获”。状态机本身的 `range_current_m` 和 `range_next_m` 均为空；运行入口可将丢失前最后可靠状态匀速外推为临时输出，并显式标记为不可靠，且绝不将其回灌给重捕获决策。

---

## 9. 每步输出

后处理器每个实时步进建议返回：

```python
{
    "mode": ...,                 # CAPTURE / TRACK / RECAPTURE
    "range_current_m": ...,      # 最新帧滤波距离
    "range_next_m": ...,         # 下一帧预测距离
    "speed_m_per_frame": ...,    # 当前估计斜率
    "candidate_q": ...,          # 本步 Top-1 的 q
    "candidate_range_m": ...,    # 本步 Top-1 的最新帧距离
    "measurement_updated": ...,  # 本步是否接受了 CNN 测量
    "search_level": ...,         # 当前局部搜索等级
    "blocks_evaluated": ...,     # 本步实际推理块数
}
```

下一帧距离为：

$$
\hat r_{t+1}=r_t+\nu_t.
$$

在 `CAPTURE` 或 `RECAPTURE` 尚未确认轨迹前，`range_current_m` 和 `range_next_m` 应返回空值，而不是强行输出当前全局 Top-1。

---

## 10. 参数初值与标定

以下数值只作为第一版搜索范围，最终应由新模型的固定验证集统计确定：

| 参数 | 初始建议 | 作用 |
|---|---:|---|
| $S_T^{\mathrm{cap}}$ | 2～5 帧 | 捕获阶段全局扫描步进 |
| $K_{\mathrm{cap}}$ | 6～10 | 冷启动候选缓存长度 |
| $\eta_{\mathrm{cap}}$ | 0.6～0.8 | 中位捕获位置的最小支持比例 |
| $R_{\mathrm{cap}}$ | 由验证集标定 | 外推位置相对中位捕获位置的一致性半径 |
| $q_{\mathrm{keep}}$ | 由验证集标定 | 接受局部 CNN 候选的最低分类分数 |
| $V_{\mathrm{inst}}(0),V_{\mathrm{inst}}(1),V_{\mathrm{inst}}(2)$ | 由验证集标定 | 三个局部搜索等级的单帧融合状态绝对速度门限（米/帧） |
| $V_{\mathrm{avg}}(0),V_{\mathrm{avg}}(1),V_{\mathrm{avg}}(2)$ | 由验证集标定 | 三个局部搜索等级的最近历史平均速度门限（米/帧） |
| $N_{\mathrm{avg}}$ | 20 帧 | 平均速度门控的最短历史跨度 |
| $N_{\mathrm{expand}}$ | 1～2 | 扩大局部范围所需连续失败数 |
| $N_{\mathrm{shrink}}$ | 3～5 | 缩小局部范围所需连续强有效数 |
| $\alpha$ | 0.6～0.9 | 位置测量更新强度 |
| $\beta$ | 0.05～0.3 | 由位置新息修正斜率的强度 |
| $\gamma$ | 0 | 初版不直接融合 CNN 斜率，仅保留位置新息的斜率修正 |

$q_{\mathrm{keep}}$ 不应直接解释为真实概率。训练时人为设置了正负样本比例，因此 $\hat q$ 首先是分类分数。阈值应使用与真实推理相同的全局34块网格和局部动态裁剪分别统计，尤其要统计：

- 可见正样本的 $\hat q$ 分布；
- 背景块的 $\hat q$ 分布；
- 每个完整时间窗内34块最大 $\hat q$ 的误报分布；
- 全局 Top-1 为真实目标块时的比例；
- 距离误差和斜率误差的 P50、P90、P95、P99。

距离门限 $R_{\mathrm{cap}}$ 应根据距离误差分位数设置；$V_{\mathrm{inst}}(L)$ 与 $V_{\mathrm{avg}}(L)$ 应结合目标速度上限和验证集状态速度分布标定。运动模型已经补偿了正常位移，速度门限主要阻止 CNN 局部噪声将状态带入不合理运动。

---

## 11. 最小实现伪代码

```python
if mode in ("CAPTURE", "RECAPTURE"):
    candidate = global_q_top1(last_20_frames)
    capture_buffer.append(candidate)
    capture = median_capture_with_support(capture_buffer)

    if capture_is_confirmed(capture):
        state = initialize_track(capture)
        mode = "TRACK"

elif mode == "TRACK":
    prediction = motion_predict(state)
    blocks = select_local_blocks(prediction, search_level)
    local_candidate = local_q_top1(last_20_frames, blocks)

    if candidate_passes_gate(local_candidate, prediction):
        state = alpha_beta_update(state, local_candidate)
        record_good_step()
        shrink_search_after_hysteresis()
    else:
        state = prediction
        record_bad_step()

        if should_expand_after_hysteresis():
            if search_level < 2:
                search_level += 1
            else:
                capture_buffer.clear()
                clear_track_state()
                mode = "RECAPTURE"
```

---

## 12. 评估指标

后处理不能只报告单块回归损失。初版至少应在完整300帧序列上报告：

1. 捕获成功率；
2. 首次捕获延迟；
3. 逐帧有效输出覆盖率；
4. 当前帧距离 MAE 和 P95 误差；
5. 下一帧预测 MAE 和 P95 误差；
6. 超过固定距离阈值的跳变次数；
7. 丢失后的重捕获成功率和平均重捕获延迟；
8. 每步平均推理块数；
9. 平均、P95 和 P99 端到端推理时间。

错误轨迹切换次数、RMSE 以及三种状态的时间占比可作为后续诊断指标，待基本闭环稳定后再加入。

第一版应同时保留“每步全局34块 Top-1”作为对照。只有在跟踪精度和覆盖率接近或优于该对照，同时显著降低平均块数和高分位延迟时，局部自适应后处理才算真正有效。
