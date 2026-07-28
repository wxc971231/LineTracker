"""单机单卡/多卡 DDP 评估入口，兼容 NVIDIA CUDA 与 Ascend NPU。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import types
from datetime import datetime
from pathlib import Path

import torch

METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from configs.base import SimpleCNNConfig
from data.dataloader import (
    StandardGridDataset,
    build_validation_dataloader,
    prepare_fixed_evaluation_manifest,
)
from eval.evaluator import evaluate_model
from runtime.settings import apply_device_defaults, apply_runtime_settings, load_runtime_settings
from train.model import SimpleCNN
from utils.checkpoint import load_checkpoint
from utils.distributed import broadcast_object, cleanup_distributed, rank_zero_print, setup_distributed
from utils.process_title import set_process_title
from utils.seed import seed_everything


def install_torch_npu_checkpoint_shim() -> bool:
    """让未安装 torch_npu 的 CPU/CUDA 环境可读取 NPU 保存的模型权重。

    Ascend 环境已经安装 ``torch_npu`` 时不做任何替换，仍使用原生实现。仅当
    checkpoint 中包含 NPU 张量重建符号、而本机没有 torch_npu 时，才把该存储
    视为普通 PyTorch storage；随后 ``map_location`` 会把它放到当前 CPU/CUDA 设备。
    """
    try:
        import torch_npu  # noqa: F401 - 原生 NPU 环境直接使用真实实现。
        return False
    except ModuleNotFoundError:
        pass

    torch_npu_module = types.ModuleType("torch_npu")
    utils_module = types.ModuleType("torch_npu.utils")
    storage_module = types.ModuleType("torch_npu.utils.storage")
    npu_module = types.ModuleType("torch_npu.npu")
    format_module = types.ModuleType("torch_npu.npu._format")

    class Format:
        """反序列化时仅承接 NPU 格式标记；CUDA/CPU 不需要使用该值。"""

        def __init__(self, value: int) -> None:
            self.value = value

    def rebuild_npu_tensor(storage, offset, size, stride, requires_grad, hooks, npu_format):
        """忽略 Ascend 专属 storage 格式，以标准张量方式恢复权重。"""
        del npu_format
        return torch._utils._rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hooks)

    format_module.Format = Format
    storage_module._rebuild_npu_tensor = rebuild_npu_tensor
    torch_npu_module.utils = utils_module
    torch_npu_module.npu = npu_module
    utils_module.storage = storage_module
    npu_module._format = format_module
    sys.modules.update({
        "torch_npu": torch_npu_module,
        "torch_npu.utils": utils_module,
        "torch_npu.utils.storage": storage_module,
        "torch_npu.npu": npu_module,
        "torch_npu.npu._format": format_module,
    })
    return True


def infer_model_type(checkpoint: dict[str, object]) -> tuple[str, dict[str, torch.Tensor]]:
    """从首层输出通道恢复 n/s 容量规格，并去除可选 DDP ``module.`` 前缀。"""
    raw_state_dict = checkpoint.get("model_state")
    if not isinstance(raw_state_dict, dict):
        raise KeyError("checkpoint 缺少字典类型的 model_state。")
    state_dict = raw_state_dict
    if state_dict and all(str(name).startswith("module.") for name in state_dict):
        state_dict = {str(name).removeprefix("module."): value for name, value in state_dict.items()}
    first_layer = state_dict.get("features.0.0.weight")
    if not isinstance(first_layer, torch.Tensor):
        raise KeyError("checkpoint 缺少 features.0.0.weight，无法识别模型规格。")
    model_type_by_channels = {16: "n", 24: "s"}
    try:
        model_type = model_type_by_channels[int(first_layer.shape[0])]
    except KeyError as error:
        raise RuntimeError(f"不支持的首层输出通道数：{int(first_layer.shape[0])}。") from error
    return model_type, state_dict


def write_evaluation_result(
    output_dir: Path,
    *,
    checkpoint_path: Path,
    checkpoint: dict[str, object],
    config: SimpleCNNConfig,
    context,
    model_type: str,
    shim_installed: bool,
    manifest_path: Path,
    source_count: int,
    grid_block_count: int,
    metrics: dict[str, float],
) -> Path:
    """由 rank 0 原子写入一次独立评估结果，避免覆盖已有评估记录。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone()
    filename = f"eval_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.json"
    result_path = output_dir / filename
    payload = {
        "evaluation_parameters": {
            "eval_data_root": str(config.data_root.resolve()),
            "time_stride_frames": int(config.validation_time_stride),
            "eval_batch_size_per_gpu": int(config.eval_batch_size_per_gpu),
            "max_eval_batch_num_per_rank": int(config.max_eval_batch_num),
            "without_replacement": True,
            "full_grid": config.max_eval_batch_num == 0,
            "num_workers_per_rank": int(config.num_workers),
            "world_size": int(context.world_size),
        },
        "schema_version": 1,
        "created_at": timestamp.isoformat(timespec="seconds"),
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "global_step": checkpoint.get("global_step"),
            "best_validation_loss": checkpoint.get("best_validation_loss"),
            "npu_checkpoint_shim": shim_installed,
        },
        "model": {"type": model_type},
        "evaluation_data": {
            "root": str(config.data_root.resolve()),
            "manifest": str(manifest_path.resolve()),
            "source_count": int(source_count),
            "grid_block_count": int(grid_block_count),
            "time_stride_frames": int(config.validation_time_stride),
        },
        "sampling": {
            "eval_batch_size_per_gpu": int(config.eval_batch_size_per_gpu),
            "max_eval_batch_num_per_rank": int(config.max_eval_batch_num),
            "without_replacement": True,
            "full_grid": config.max_eval_batch_num == 0,
        },
        "distributed": {
            "world_size": int(context.world_size),
            "backend": context.backend,
            "device": str(context.device),
        },
        "metrics": metrics,
    }
    temporary_path = output_dir / f".{filename}.tmp"
    with temporary_path.open("w", encoding="utf-8") as handle:
        # 保持 insertion order，使 evaluation_parameters 始终处于 JSON 顶部。
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, result_path)
    return result_path


def parse_args() -> argparse.Namespace:
    """解析指定数据路径的完整标准网格评估参数。"""
    parser = argparse.ArgumentParser(description="SimpleCNN 全量标准网格 DDP 评估")
    parser.add_argument("--checkpoint", type=Path, default=None, 
        help="要评估的模型权重路径；省略时使用本机硬编码的 best.pt。"
    )
    parser.add_argument("--eval-data-root", type=Path, default=None,
        help="评估样本根目录；其下全部含 data.npz 的一级子目录都会参与评估。",
    )
    parser.add_argument("--eval-batch-size-per-gpu", type=int, default=512,
        help="每张 CUDA/NPU 卡一次前向的标准距离块数量；仅影响显存与吞吐，不改变评估指标。",
    )
    parser.add_argument("--max_eval-batch-num", type=int, default=200,
        help="每个 DDP rank 最多评估的 batch 数；设为 0 时按 source 分片完整遍历全部评估网格。"
    )
    parser.add_argument("--time-stride", type=int, default=5,
        help="连续 20 帧标准时间窗的起点步进（帧）；1 表示使用每一个可行起点。",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="每个 rank 的 DataLoader worker 数。")
    parser.add_argument("--env-file", type=Path, default=None, help="本地运行时 .env 路径。")
    return parser.parse_args()


def main() -> None:
    """启动多卡安全的独立评估。"""
    args = parse_args()

    CHECKPOINT_PATH = Path(
        "/home/pc5090/Code/github/LineTracker/method/SimpleCNN/runs/simplecnn_v1/"
        "limit20k-gbs512-lr3e-4-pos25-vs5-modeln-cfgfcde7ff6/checkpoints/best.pt"
    )
    EVAL_DATA_ROOT = Path(
        "/home/pc5090/Code/github/LineTracker/data/synthetic/gen/"
        "F300_N200_S20260717_B-random_T-R10000-290000m-V340-A6-C10-J1-K300-Q0p35-0p95"
    ).expanduser().resolve()
    args.checkpoint = CHECKPOINT_PATH
    args.eval_data_root = EVAL_DATA_ROOT
    args.max_eval_batch_num = 0
    experiment_label = args.checkpoint.parent.parent.name
    set_process_title("eval", label=experiment_label)

    # 未显式传入时，使用本机同步回来的 NPU 集群训练 best.pt。
    if not args.eval_data_root.is_dir():
        raise FileNotFoundError(f"评估样本根目录不存在：{args.eval_data_root}")
    if args.eval_batch_size_per_gpu < 1:
        raise ValueError("eval_batch_size_per_gpu 必须为正整数。")
    if args.max_eval_batch_num < 0:
        raise ValueError("max_eval_batch_num 必须为非负整数；0 表示完整遍历。")
    if args.time_stride < 1:
        raise ValueError("time_stride 必须为正整数。")
    if args.num_workers < 0:
        raise ValueError("num_workers 必须为非负整数。")

    runtime_settings = load_runtime_settings(args.env_file)
    context = setup_distributed(runtime_settings)

    # 在 Ampere 及更新 NVIDIA GPU 上允许 TF32，加速评估卷积而不改变 NPU 路径。
    if context.device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    try:
        checkpoint_path = args.checkpoint
        shim_installed = install_torch_npu_checkpoint_shim()
        checkpoint = load_checkpoint(checkpoint_path, context.device)
        model_type, state_dict = infer_model_type(checkpoint)

        # checkpoint 未随附 resolved_config.json；按模型权重和评估命令构造必要配置。
        # 数据路径、时间步进、每卡 batch 和最大评估 batch 数均由评估参数显式指定。
        config = SimpleCNNConfig(
            data_root=args.eval_data_root,
            source_sample_limit=0,
            validation_time_stride=args.time_stride,
            eval_batch_size_per_gpu=args.eval_batch_size_per_gpu,
            max_eval_batch_num=args.max_eval_batch_num,
            num_workers=args.num_workers,
            model_type=model_type,
            wandb_mode="disabled",
        )
        config = apply_runtime_settings(config, runtime_settings)
        config.data_root = args.eval_data_root  # 命令行指定的评估路径优先于 .env 的训练路径。
        config.eval_batch_size_per_gpu = args.eval_batch_size_per_gpu
        config.validation_time_stride = args.time_stride
        config.max_eval_batch_num = args.max_eval_batch_num
        config.num_workers = args.num_workers
        config = apply_device_defaults(config, runtime_settings, context.device.type)
        config.validate()
        seed_everything(config.seed, context.rank)

        model = SimpleCNN(config).to(context.device)
        model.load_state_dict(state_dict, strict=True)
        rank_zero_print(
            context,
            f"checkpoint={checkpoint_path}  device={context.device}  "
            f"backend={context.backend}  world_size={context.world_size}  "
            f"model_type={model_type}  npu_checkpoint_shim={shim_installed}",
        )
        preparation_result = None
        rank_zero_dataset: StandardGridDataset | None = None
        if context.is_main:
            try:
                manifest_path = prepare_fixed_evaluation_manifest(config)
                rank_zero_dataset = StandardGridDataset(
                    manifest_path,
                    config,
                    verify_cached_samples=True,
                )
                preparation_result = {
                    "manifest_path": str(manifest_path),
                    "source_count": len(rank_zero_dataset.source_ids),
                    "block_count": len(rank_zero_dataset),
                    "error": None,
                }
            except Exception:
                preparation_result = {
                    "manifest_path": None,
                    "source_count": None,
                    "block_count": None,
                    "error": traceback.format_exc(),
                }
        preparation_result = broadcast_object(context, preparation_result)
        if preparation_result["error"] is not None:
            raise RuntimeError("rank 0 评估数据准备失败：\n" + str(preparation_result["error"]))
        manifest_value = preparation_result["manifest_path"]
        if manifest_value is None:
            raise RuntimeError("rank 0 未返回评估 manifest 路径。")
        manifest_path = Path(str(manifest_value))
        evaluation_scope = (
            "完整遍历全部评估网格"
            if config.max_eval_batch_num == 0
            else (
                f"每 rank 最多 {config.max_eval_batch_num} 个无放回 batch，"
                f"全部 rank 理论最多 {config.max_eval_batch_num * context.world_size} batch"
            )
        )
        rank_zero_print(
            context,
            f"eval_data_root={config.data_root}  sources={preparation_result['source_count']}  "
            f"blocks={preparation_result['block_count']}  time_stride={config.validation_time_stride}  "
            f"eval_batch_size_per_gpu={config.eval_batch_size_per_gpu}  {evaluation_scope}",
        )
        dataloader = build_validation_dataloader(
            config,
            manifest_path,
            rank=context.rank,
            world_size=context.world_size,
            max_eval_batches=config.max_eval_batch_num,
            dataset=rank_zero_dataset if context.is_main else None,
            without_replacement=True,
        )
        # 评估不反向传播；各 rank 各自前向其 sampler 分片，再由 evaluate_model all-reduce 指标即可。
        metrics = evaluate_model(model, dataloader, config, context, prefix="eval")
        if context.is_main:
            rank_zero_print(
                context,
                "[eval] " + "  ".join(f"{key}={value:.6g}" for key, value in metrics.items()),
            )
            # checkpoint 的 checkpoints/ 与 eval/ 并列；每次评估保留独立 JSON 记录。
            result_path = write_evaluation_result(
                checkpoint_path.resolve().parent.parent / "eval",
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                config=config,
                context=context,
                model_type=model_type,
                shim_installed=shim_installed,
                manifest_path=manifest_path,
                source_count=int(preparation_result["source_count"]),
                grid_block_count=int(preparation_result["block_count"]),
                metrics=metrics,
            )
            rank_zero_print(context, f"评估结果 JSON：{result_path}")
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
