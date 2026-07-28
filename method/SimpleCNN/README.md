# SimpleCNN 训练代码

本目录实现 `_doc/SimpleCNN.md` 与 `_doc/dataloader.md` 中定义的原始二值 CNN 基线：

- 输入为在线裁剪、无损距离重排后的 `[B, 8, 20, 1250]`；
- 输出为 `q`、块内中心距离 `rho` 与斜率 `nu`；
- 损失为质量头 BCE 加逐帧实际响应距离 bin 的掩码 Huber 损失；
- 训练使用无限在线数据流和按 optimizer step 的预算；
- 验证/测试使用固定的标准 20 帧 × 34 距离块网格；
- 支持 PyTorch DDP、AMP、断点恢复和 rank 0 W&B 日志。

## 快速检查

在 `method/SimpleCNN/.env` 中填写当前机器可访问的数据和输出目录；可参考：

```dotenv
LT_ACCELERATOR=auto
LT_DISTRIBUTED_BACKEND=auto
LT_AMP=auto
LT_PIN_MEMORY=auto
LT_DATA_ROOT=/path/to/synthetic/gen
LT_OUTPUT_ROOT=/path/to/LineTracker/method/SimpleCNN/runs
LT_DISTRIBUTED_TIMEOUT_MINUTES=360
```

`.env` 只保存设备、路径和 I/O 等本地运行时设置；模型、采样、损失、训练预算与 W&B 模式仍由 profile 与 `--set` 控制。

VS Code 请从仓库的 `.vscode/launch.json` 选择 CUDA、Ascend 或 CPU 配置。它会使用当前选择的 Python 解释器，建议在本机选择 `linetracker-py311`。


PyTorch 必须先按目标平台安装：NVIDIA 使用匹配 CUDA 的官方构建，Ascend 使用与 CANN 匹配的 PyTorch 和 `torch_npu`。之后再执行 `pip install -r requirements.txt` 安装 NumPy、W&B 和 tqdm；通用 requirements 不声明 `torch`，以免覆盖厂商版本。

首次构建大规模 validation/test cache 时，其他 rank 会在 process group 中等待 rank 0；默认超时为 360 分钟，可由 `.env` 的 `LT_DISTRIBUTED_TIMEOUT_MINUTES` 调整。rank 0 若构建失败，完整异常会广播给所有 rank，避免其余进程无提示挂住。
`SimpleCNN · 多卡正式训练 · auto` 会根据当前可用后端选择 CUDA 或 Ascend，并按 `CUDA_VISIBLE_DEVICES` 或 `ASCEND_RT_VISIBLE_DEVICES` 中的可见卡数启动对应数量的进程。修改该启动项中的索引列表即可选择要使用的卡。

命令行也可用于无调试的本地训练。先以单卡 debug profile 验证环境、压缩 bit 裁剪和训练链路：

```bash
cd method/SimpleCNN
torchrun --standalone --nproc_per_node=1 train/train_ddp.py --config debug
```

正式 profile：

```bash
torchrun --standalone --nproc_per_node=4 train/train_ddp.py --config simplecnn_v1
```

可用 `--set` 覆盖任一 profile 字段，例如：

```bash
torchrun --standalone --nproc_per_node=2 train/train_ddp.py \
  --config simplecnn_v1 \
  --set batch_size_per_gpu=16 \
  --set validation_time_stride=1 \
  --set total_optimizer_steps=10000
```

禁用 W&B：

```bash
torchrun --standalone --nproc_per_node=1 train/train_ddp.py \
  --config simplecnn_v1 \
  --set wandb_mode=disabled
```

每次训练会在 `runs/<run_name>/<experiment_slug>/<timestamp>/` 下保存解析后的配置、固定数据划分、`last.pt`、`best.pt` 与有限数量的 step checkpoint。例如当前8卡正式配置会生成 `limit50k-gbs1024-lr5e-4-pos25-vs5-models-cfg<8位哈希>`。其中 `gbs` 是包含 DDP 和梯度累积的全局训练 batch，`models` 表示使用 s 容量规格。

`cfg` 哈希覆盖模型、采样、损失、优化器、训练预算和验证协议等配置，但排除本机数据/输出路径、resume、W&B 和纯 I/O 设置。这样路径保持可读，同时任何未展示的关键算法参数变化仍会得到不同目录。恢复历史 checkpoint 时始终沿用其原 run 目录。W&B 运行名称采用相同摘要加时间戳。

验证、测试和全量独立评估网格统一缓存在 `runs/_cache/val_test_grid/`，cache key 包含数据划分/评估集合、网格参数及源文件大小/mtime。

训练启动只按需构建 validation cache；test cache 在独立评估首次使用时构建。完整评估按 source 分配给不同 rank。限量评估由 `max_eval_batch_num` 控制，每次验证都会有放回重新采样；同一次 `val` 的指标同时写入 W&B 并用于更新 `best.pt`。一个 batch 只访问一个 source，以减少完整序列反复解压。

## 独立评估

```bash
torchrun --standalone --nproc_per_node=4 eval/eval_ddp.py \
  --eval-data-root /path/to/synthetic/gen/<dataset_name> \
  --eval-batch-size-per-gpu 512 \
  --max-eval-batch-num 200 \
  --time-stride 5
```

评估集合始终为 `--eval-data-root` 下所有含 `data.npz` 的一级样本目录，不读取训练时的 train/val/test 划分。`--eval-batch-size-per-gpu` 仅控制每张卡的前向批大小。`--max-eval-batch-num` 为每个 DDP rank 的最大 batch 数，正数时从该 rank 分到的全部网格块中无放回随机抽样；不同 rank 分到的 source 本就不重叠，因此全局也不会重复计数同一个块。多卡合计理论最多为 `world_size × max_eval_batch_num` 个 batch。将其设为 `0` 才会按 source 分片完整遍历全量网格。`--time-stride=1` 可覆盖全部可行的 20 帧时间窗起点，较大的值则按固定步进抽取窗口。

评估结束后仅由 rank 0 将最终 all-reduce 指标写入 `<checkpoint 的上级运行目录>/eval/eval_<时间戳>.json`；该 JSON 同时记录 checkpoint、数据根目录、网格规模、采样参数、DDP 信息与全部指标。

## 恢复训练

```bash
torchrun --standalone --nproc_per_node=4 train/train_ddp.py \
  --resume runs/simplecnn_v1/<experiment_slug>/<timestamp>/checkpoints/last.pt
```

恢复时会加载模型、优化器、AMP scaler、step、best 验证协议、数据流代次和 W&B run id。每次恢复都会将数据流代次加一，并为各 rank/worker 派生新的可复现随机流，因此不会从在线数据流开头重新播放；它不会试图精确保存多 worker 的内部逐样本 RNG 状态。
如果验证 cache 的数据指纹或 best 评估协议发生变化，恢复流程会先用尚未更新的 checkpoint 在新验证集上评估一次，并以该结果重新建立 `best.pt` 基线。
