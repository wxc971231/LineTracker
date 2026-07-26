"""SimpleCNN 的单机单卡/多卡 DDP 训练入口。"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import torch

METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from configs.base import (
    apply_overrides,
    load_config_json,
    load_profile,
    parse_entrypoint_args,
    save_config_json,
)
from data.dataloader import build_train_dataloader, build_validation_dataloader, prepare_data_artifacts
from train.trainer import Trainer
from utils.checkpoint import load_checkpoint
from utils.distributed import barrier, cleanup_distributed, rank_zero_print, setup_distributed
from utils.logging import WandbLogger
from utils.seed import seed_everything


def _resolve_run(config, resume_path: Path | None, is_main: bool) -> Path:
    """新训练创建时间戳目录；恢复训练沿用原 run 目录。"""
    if resume_path is not None:
        return resume_path.resolve().parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_root / config.run_name / timestamp
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

if __name__ == "__main__":
    """准备 DDP、固定数据清单、W&B 并启动按 step 的训练。"""

    # 在 Ampere 及更新 GPU 上允许 TF32，可加速卷积和矩阵乘法。
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # 初始化 DDP 和 logger
    context = setup_distributed()

    # 入口参数，可被命令行覆盖
    args = parse_entrypoint_args(
        config="simplecnn_v1",      # configs 下的 profile 名称
        resume=None,                # 要恢复的 last.pt checkpoint 路径
        overrides=(),               # 需要覆盖的参数字段，如 ("batch_size_per_gpu=16",)。
    )

    # 尝试加载实验参数并启动训练
    logger: WandbLogger | None = None
    try:
        # 从 profile 或 ckpt 还原参数配置
        resume_path = None if args.resume is None else args.resume.resolve()
        if resume_path is not None:
            config = load_config_json(resume_path.parent.parent / "resolved_config.json")
            config.resume = resume_path
        else:
            config = load_profile(args.config)
        config = apply_overrides(config, args.set)
        config.validate()

        # 恢复训练沿用 checkpoint 所在实验目录；新训练在覆盖参数生效后创建目录。
        run_dir = (
            resume_path.parent.parent
            if resume_path is not None
            else _resolve_run(config, None, context.is_main)
        )
        barrier(context)

        # 仅由 rank 0 落盘配置、完整序列划分和固定验证/测试网格，避免多卡并发写文件
        if context.is_main:
            run_dir.mkdir(parents=True, exist_ok=True)
            save_config_json(config, run_dir / "resolved_config.json")
            prepare_data_artifacts(config, run_dir / "data")

        # 等待 rank 0 写完后，各 rank 再读取同一份划分与网格清单
        barrier(context)
        artifacts = prepare_data_artifacts(config, run_dir / "data")

        # 各 rank 使用派生种子，避免在线随机裁剪完全重复。
        effective_seed = seed_everything(config.seed, context.rank)
        rank_zero_print(
            context,
            f"run={run_dir}  world_size={context.world_size}  seed={effective_seed} "
            f"train_sources={len(artifacts.train_sources)}  val_sources={len(artifacts.validation_sources)}",
        )

        # 构造数据加载器
        train_loader = build_train_dataloader(
            config,
            artifacts.train_sources,
            rank=context.rank,
            world_size=context.world_size,
        )
        validation_loader = build_validation_dataloader(
            config,
            artifacts.validation_manifest_path,
            rank=context.rank,
            world_size=context.world_size,
        )

        # 尝试加载 ckpt
        resume_payload = None
        resume_wandb_id = None
        if config.resume is not None:
            resume_payload = load_checkpoint(config.resume, context.device)
            resume_wandb_id = resume_payload.get("wandb_id")
        logger = WandbLogger.create(config, context, run_dir, resume_id=resume_wandb_id)
        
        # 启动训练
        trainer = Trainer(
            config,
            context,
            run_dir,
            train_loader,
            validation_loader,
            logger,
            resume_payload=resume_payload,
        )
        trainer.train()
    finally:
        if logger is not None:
            logger.finish()
        cleanup_distributed(context)
