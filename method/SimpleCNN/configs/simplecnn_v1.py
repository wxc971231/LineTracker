"""SimpleCNN-v1 的正式基线 profile。"""

from __future__ import annotations

from .base import SimpleCNNConfig


def get_config() -> SimpleCNNConfig:
    """返回与文档中 20×10000 原始二值基线一致的默认配置。"""
    return SimpleCNNConfig(
        run_name="simplecnn_v1",                # 实验名；输出保存到 runs/simplecnn_v1/<时间戳>/。
        batch_size_per_gpu=32,                  # 每张 GPU 的训练 micro-batch 大小。
        positive_fraction=0.25,                 # batch 内可见正样本比例，默认约为 1:3 正负比。
        source_cache_size=4,                    # 每个 rank/worker 同时缓存的完整 .npz 序列数量。
        source_positive_quota=64,               # 一份序列提供 64 个正样本后，尝试换入新的背景序列。
        validation_time_stride=5,               # 固定验证网格的时间步进（帧），即每 0.25 s 取一个窗口。
        eval_batch_size_per_gpu=64,             # 每张 GPU 验证、测试时一次前向的标准块数量。
        total_optimizer_steps=50_000,           # 训练总 optimizer 更新次数；在线数据流不以 epoch 终止。
        learning_rate=3e-4,                     # AdamW 的峰值学习率；warmup 后从该值开始余弦衰减。
        warmup_ratio=0.05,                      # 前 5% optimizer steps 线性 warmup。
        weight_decay=1e-4,                      # AdamW 权重衰减系数。
        dropout=0.10,                           # MLP 共享特征层的 dropout 概率。
        huber_delta_bins=3.0,                   # 逐帧距离 Huber loss 的二次/线性切换阈值（bin，即 m）。
        wandb_project="LineTracker-SimpleCNN",
    )
