"""用于验证数据、DDP 和前向链路的小型单卡 profile。"""

from __future__ import annotations

from .base import SimpleCNNConfig


def get_config() -> SimpleCNNConfig:
    """返回可快速完成冒烟训练的保守配置。"""
    return SimpleCNNConfig(
        run_name="debug",
        batch_size_per_gpu=4,
        source_cache_size=2,
        source_positive_quota=8,
        num_workers=0,
        validation_time_stride=20,
        eval_batch_size_per_gpu=4,
        total_optimizer_steps=20,
        log_interval_steps=1,
        eval_interval_steps=10,
        checkpoint_interval_steps=10,
        wandb_enabled=False,
        wandb_mode="disabled",
        compile_model=False,
    )
