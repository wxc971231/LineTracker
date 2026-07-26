# SimpleCNN 训练代码

本目录实现 `_doc/SimpleCNN.md` 与 `_doc/dataloader.md` 中定义的原始二值 CNN 基线：

- 输入为在线裁剪、无损距离重排后的 `[B, 8, 20, 1250]`；
- 输出为 `q`、块内中心距离 `rho` 与斜率 `nu`；
- 损失为质量头 BCE 加逐帧实际响应距离 bin 的掩码 Huber 损失；
- 训练使用无限在线数据流和按 optimizer step 的预算；
- 验证/测试使用固定的标准 20 帧 × 34 距离块网格；
- 支持 PyTorch DDP、AMP、断点恢复和 rank 0 W&B 日志。

## 快速检查

先以单卡 debug profile 验证环境、压缩 bit 裁剪和训练链路：

```bash
cd /home/pc5090/Code/github/LineTracker/method/SimpleCNN
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
  --set wandb_enabled=false
```

每次训练会在 `runs/<run_name>/<timestamp>/` 下保存解析后的配置、固定数据划分、验证/测试网格清单、`last.pt`、`best.pt` 与有限数量的 step checkpoint。

## 独立评估

```bash
torchrun --standalone --nproc_per_node=4 eval/evaluate.py \
  --run-dir runs/simplecnn_v1/<timestamp> \
  --split test
```

## 恢复训练

```bash
torchrun --standalone --nproc_per_node=4 train/train_ddp.py \
  --resume runs/simplecnn_v1/<timestamp>/checkpoints/last.pt
```

恢复时会加载模型、优化器、AMP scaler、step 和 W&B run id。在线训练流会从新的可复现源数据循环继续，而不会试图精确重放多 worker 的内部随机裁剪状态。
