"""SimpleCNN 的单机单卡/多卡 DDP 训练入口。"""

from __future__ import annotations

import sys
import traceback
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
from data.dataloader import (
    StandardGridDataset,
    build_train_dataloader,
    build_validation_dataloader,
    prepare_data_artifacts,
)
from runtime.settings import apply_device_defaults, apply_runtime_settings, load_runtime_settings
from train.trainer import Trainer
from utils.checkpoint import load_checkpoint
from utils.distributed import (
    barrier,
    broadcast_object,
    broadcast_path,
    cleanup_distributed,
    rank_zero_print,
    setup_distributed,
)
from utils.logging import WandbLogger
from utils.process_title import set_process_title
from utils.run_naming import build_experiment_slug
from utils.seed import seed_everything


def _resolve_run(config, resume_path: Path | None, world_size: int) -> Path:
    """新训练创建时间戳目录；恢复训练沿用原 run 目录。"""
    if resume_path is not None:
        return resume_path.resolve().parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_slug = build_experiment_slug(config, world_size)
    run_dir = config.output_root / config.run_name / experiment_slug / timestamp
    return run_dir

def _next_data_stream_generation(payload: dict | None) -> int:
    """新训练从 0 开始；每次恢复切换到一个新的可复现在线采样流。"""
    if payload is None:
        return 0
    try:
        generation = int(payload.get("data_stream_generation", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint 的 data_stream_generation 无效。") from error
    if generation < 0:
        raise ValueError("checkpoint 的 data_stream_generation 不得为负。")
    return generation + 1


'''
cd /mnt/host-model/weixc/code/LineTracker/method/SimpleCNN

LT_ACCELERATOR=npu \
ASCEND_RT_VISIBLE_DEVICES=8,9,10,11 \
HCCL_NPU_SOCKET_PORT_RANGE=auto \
HCCL_IF_BASE_PORT=63000 \
HCCL_CONNECT_TIMEOUT=1800 \
OMP_NUM_THREADS=1 \
ASCEND_GLOBAL_LOG_LEVEL=3 \
/root/miniconda3/envs/linetracker-py311/bin/python -m runtime.launch \
  train/train_ddp.py --config simplecnn_v2
'''

if __name__ == "__main__":
    """准备 DDP、固定数据清单、W&B 并启动按 step 的训练。"""

    args = parse_entrypoint_args(
        config="simplecnn_v1",      # configs 下的 profile 名称
        resume="/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/runs/simplecnn_v2/limit50k-gbs1024-lr5e-4-pos25-vs5-models-cfg11ba6304/20260728_113116/checkpoints/last.pt",
        overrides=(),               # 需要覆盖的参数字段，如 ("batch_size_per_gpu=16",)。
    )
    # 在 HCCL/NCCL 初始化前命名，使初始化失败的 rank 也能被准确识别。
    set_process_title("train", label=args.config)

    # .env 只处理机器、设备与路径；--set 仍是算法超参数的最高优先级覆盖方式。
    runtime_settings = load_runtime_settings(args.env_file)
    context = setup_distributed(runtime_settings)

    # 在 Ampere 及更新 GPU 上允许 TF32，可加速卷积和矩阵乘法。
    if context.device.type == "cuda":
        torch.set_float32_matmul_precision("high")

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
        config = apply_runtime_settings(config, runtime_settings)
        config = apply_device_defaults(config, runtime_settings, context.device.type)
        config = apply_overrides(config, args.set)
        config.validate()
        # 恢复训练时以 resolved_config 中的真实 profile 替换命令行缺省名称。
        set_process_title("train", label=config.profile, rank=context.rank)

        # 恢复训练沿用 checkpoint 所在实验目录；新训练在覆盖参数生效后创建目录。
        if resume_path is not None:
            run_dir = _resolve_run(config, resume_path, context.world_size)
        else:
            main_run_dir = (
                str(_resolve_run(config, None, context.world_size))
                if context.is_main
                else None
            )
            run_dir = Path(broadcast_path(context, main_run_dir))
        barrier(context)

        # 在构造在线 DataLoader 前读取恢复代次，确保不会重播旧数据流开头。
        resume_payload = load_checkpoint(config.resume, context.device) if config.resume is not None else None
        data_stream_generation = _next_data_stream_generation(resume_payload)
        resume_wandb_id = (
            None if resume_payload is None else resume_payload.get("wandb_id")
        )

        # 仅由 rank 0 落盘配置、完整序列划分和验证网格，再把结果广播给其他 rank。
        preparation_result = None
        rank_zero_validation_dataset: StandardGridDataset | None = None
        if context.is_main:
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                save_config_json(
                    config,
                    run_dir / "resolved_config.json",
                    world_size=context.world_size,
                )
                artifacts = prepare_data_artifacts(
                    config,
                    run_dir / "data",
                    include_test_manifest=False,
                )
                rank_zero_validation_dataset = StandardGridDataset(
                    artifacts.validation_manifest_path,
                    config,
                    verify_cached_samples=True,
                )
                preparation_result = {"artifacts": artifacts, "error": None}
            except Exception:
                preparation_result = {
                    "artifacts": None,
                    "error": traceback.format_exc(),
                }
        preparation_result = broadcast_object(context, preparation_result)
        if preparation_result["error"] is not None:
            raise RuntimeError("rank 0 数据准备失败：\n" + str(preparation_result["error"]))
        artifacts = preparation_result["artifacts"]
        if artifacts is None:
            raise RuntimeError("rank 0 未返回数据 artifacts。")

        # 各 rank 使用派生种子，避免在线随机裁剪完全重复。
        effective_seed = seed_everything(
            config.seed + data_stream_generation * 1_000_000_007,
            context.rank,
        )
        rank_zero_print(
            context,
            f"run={run_dir}  device={context.device}  backend={context.backend} "
            f"world_size={context.world_size}  seed={effective_seed}  stream={data_stream_generation} "
            f"train_sources={len(artifacts.train_sources)}  val_sources={len(artifacts.validation_sources)}",
        )

        # 构造数据加载器
        train_loader = build_train_dataloader(
            config,
            artifacts.train_sources,
            rank=context.rank,
            world_size=context.world_size,
            stream_generation=data_stream_generation,
        )
        validation_loader = build_validation_dataloader(
            config,
            artifacts.validation_manifest_path,
            rank=context.rank,
            world_size=context.world_size,
            dataset=rank_zero_validation_dataset if context.is_main else None,
        )

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
            data_stream_generation=data_stream_generation,
            validation_dataset_id=(
                f"{config.data_root}::{artifacts.validation_manifest_path.name}"
            ),
        )
        trainer.train()
    finally:
        if logger is not None:
            logger.finish()
        cleanup_distributed(context)
