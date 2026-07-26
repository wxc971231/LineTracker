"""用于验证数据、DDP 和前向链路的小型单卡 profile。"""

from __future__ import annotations

from .base import SimpleCNNConfig


def get_config() -> SimpleCNNConfig:
    """返回可快速完成冒烟训练的保守配置。"""
    return SimpleCNNConfig(
        run_name="debug",               # 实验系列名，结果保存到 runs/debug/<参数摘要>/<时间戳>/。
        batch_size_per_gpu=4,           # 每张 GPU 的训练 micro-batch 大小；设小以便快速检查。
        source_cache_size=2,            # 每个训练 worker 同时缓存的完整 .npz 序列数。
        source_positive_quota=8,        # 单份序列累计提供 8 个正样本后，尝试替换为新序列。
        num_workers=0,                  # DataLoader 子进程数；0 表示在训练主进程中加载，便于调试。
        validation_time_stride=20,      # 验证时间窗步进（帧）；每秒取一个 20 帧窗口以缩短检查时间。
        eval_batch_size_per_gpu=4,      # 每张 GPU 验证时的 batch 大小。
        total_optimizer_steps=20,       # debug 只进行 20 次 optimizer 更新。
        log_interval_steps=1,           # 每个训练 step 都输出一次日志，方便立刻发现异常。
        eval_interval_steps=10,         # 每 10 个 step 在固定验证网格上评估一次。
        checkpoint_interval_steps=10,   # 每 10 个 step 保存一次 last 与阶段性 checkpoint。
        wandb_mode="disabled",          # 唯一 W&B 开关：不导入、不初始化，也不联网写入。
        compile_model=False,            # 关闭 torch.compile，减少首次编译等待并简化断点调试。
        wandb_project="LineTracker-SimpleCNN",
    )
