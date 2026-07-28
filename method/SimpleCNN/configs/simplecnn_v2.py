"""采用二值 q 标签、部分相交负样本和归一化几何损失的正式 v2 profile。"""

from __future__ import annotations

from .base import SimpleCNNConfig


def get_config() -> SimpleCNNConfig:
    """以 20260726 的 50k/model-n 实验为对照，只引入 v2 必要变化。"""
    return SimpleCNNConfig(
        # 实验与数据划分；data_root/output_root 继续由 .env 按机器覆盖。
        run_name="simplecnn_v2",
        seed=42,
        split_seed=42,
        source_sample_limit=50_000,
        train_fraction=0.80,
        val_fraction=0.10,
        test_fraction=0.10,

        # 输入分块保持旧实验不变。
        frames_per_window=20,
        range_bins=300_000,
        block_width_m=10_000,
        spatial_step_m=9_000,
        packed_bitorder="big",

        # 在线训练流保持旧实验不变。
        num_workers=0,
        source_cache_size=4,
        source_positive_quota=64,
        positive_fraction=0.25,
        standard_positive_fraction=0.30,
        positive_margin_m=100,
        negative_guard_m=100,
        negative_local_span_m=3_000,

        # v2 使用四类等概率负样本；绝对权重取 1，不影响其 1/4 概率。
        negative_local_weight=1.0,
        negative_same_time_weight=1.0,
        negative_random_weight=1.0,
        negative_partial_weight=1.0,

        # batch 与验证预算保持旧实验不变。
        batch_size_per_gpu=128,
        eval_batch_size_per_gpu=512,
        max_eval_batch_num=128,
        validation_time_stride=5,

        # 保持 model-n 容量，便于与旧实验做可解释对照。
        model_type="n",
        input_channels=8,
        hidden_dim=256,
        dropout=0.10,
        max_speed_per_frame_m=17.0,

        # v2 的 q/几何监督契约由代码实现；新增固定几何损失缩放。
        lambda_q=1.0,
        lambda_line=1.0,
        huber_delta_bins=3.0,
        line_loss_scale=5_000.0,

        # 优化器、训练预算和调度保持旧实验不变。
        total_optimizer_steps=50_000,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        min_learning_rate_ratio=0.10,
        warmup_ratio=0.03,
        weight_decay=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        grad_clip_norm=10.0,
        amp="auto",
        compile_model=False,

        # 日志与 checkpoint 频率保持旧实验不变。
        log_interval_steps=20,
        eval_interval_steps=2_000,
        checkpoint_interval_steps=1_000,
        keep_last_checkpoints=2,
        wandb_project="LineTracker-SimpleCNN",
        wandb_entity=None,
        wandb_mode="online",
    )
