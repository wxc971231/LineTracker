"""加载训练 checkpoint 后，在固定验证或测试标准网格上独立评估。"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from configs.base import load_config_json
from data.dataloader import (
    StandardGridDataset,
    build_validation_dataloader,
    prepare_evaluation_manifest,
)
from eval.evaluator import evaluate_model
from runtime.settings import apply_device_defaults, apply_runtime_settings, load_runtime_settings
from train.model import SimpleCNN
from utils.checkpoint import load_checkpoint
from utils.distributed import broadcast_object, cleanup_distributed, rank_zero_print, setup_distributed
from utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    """解析独立评估所需的 run、checkpoint 和 split 参数。"""
    parser = argparse.ArgumentParser(description="SimpleCNN 固定标准网格评估")
    parser.add_argument("--run-dir", type=Path, required=True, help="包含 resolved_config.json 和 data/ 的训练目录。")
    parser.add_argument("--checkpoint", type=Path, default=None, help="默认使用 run-dir/checkpoints/best.pt。")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--env-file", type=Path, default=None, help="本地运行时 .env 路径。")
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="忽略训练配置中的 max_eval_batch_num，完整遍历所选 split。",
    )
    return parser.parse_args()


def main() -> None:
    """启动多卡安全的独立评估。"""
    args = parse_args()
    runtime_settings = load_runtime_settings(args.env_file)
    context = setup_distributed(runtime_settings)
    try:
        config = load_config_json(args.run_dir / "resolved_config.json")
        config = apply_runtime_settings(config, runtime_settings)
        config = apply_device_defaults(config, runtime_settings, context.device.type)
        if args.full_eval:
            config.max_eval_batch_num = 0
        config.validate()
        seed_everything(config.seed, context.rank)
        checkpoint_path = args.checkpoint or args.run_dir / "checkpoints" / "best.pt"
        checkpoint = load_checkpoint(checkpoint_path, context.device)
        model = SimpleCNN(config).to(context.device)
        model.load_state_dict(checkpoint["model_state"])
        preparation_result = None
        rank_zero_dataset: StandardGridDataset | None = None
        if context.is_main:
            try:
                manifest_path = prepare_evaluation_manifest(
                    config, args.run_dir / "data", args.split
                )
                rank_zero_dataset = StandardGridDataset(
                    manifest_path,
                    config,
                    verify_cached_samples=True,
                )
                preparation_result = {
                    "manifest_path": str(manifest_path),
                    "error": None,
                }
            except Exception:
                preparation_result = {"manifest_path": None, "error": traceback.format_exc()}
        preparation_result = broadcast_object(context, preparation_result)
        if preparation_result["error"] is not None:
            raise RuntimeError("rank 0 评估数据准备失败：\n" + str(preparation_result["error"]))
        manifest_value = preparation_result["manifest_path"]
        if manifest_value is None:
            raise RuntimeError("rank 0 未返回评估 manifest 路径。")
        manifest_path = Path(str(manifest_value))
        dataloader = build_validation_dataloader(
            config,
            manifest_path,
            rank=context.rank,
            world_size=context.world_size,
            dataset=rank_zero_dataset if context.is_main else None,
        )
        metrics = evaluate_model(model, dataloader, config, context, prefix=args.split)
        if context.is_main:
            rank_zero_print(
                context,
                f"[{args.split}] " + "  ".join(f"{key}={value:.6g}" for key, value in metrics.items()),
            )
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
