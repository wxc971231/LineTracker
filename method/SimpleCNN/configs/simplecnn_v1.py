"""SimpleCNN-v1 的正式基线 profile。"""

from __future__ import annotations

from .base import SimpleCNNConfig


def get_config() -> SimpleCNNConfig:
    """返回与文档中 20×10000 原始二值基线一致的默认配置。"""
    return SimpleCNNConfig(
        run_name="simplecnn_v1",
        batch_size_per_gpu=32,
        positive_fraction=0.25,
        source_cache_size=4,
        source_positive_quota=64,
        validation_time_stride=5,
        eval_batch_size_per_gpu=64,
        total_optimizer_steps=50_000,
        learning_rate=3e-4,
        warmup_ratio=0.05,
        weight_decay=1e-4,
        dropout=0.10,
        huber_delta_bins=3.0,
        wandb_enabled=True,
        wandb_project="LineTracker-SimpleCNN",
    )
