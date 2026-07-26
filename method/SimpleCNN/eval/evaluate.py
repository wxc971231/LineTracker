"""加载训练 checkpoint 后，在固定验证或测试标准网格上独立评估。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from configs.base import load_config_json
from data.dataloader import build_validation_dataloader
from eval.evaluator import evaluate_model
from train.model import SimpleCNN
from utils.checkpoint import load_checkpoint
from utils.distributed import cleanup_distributed, rank_zero_print, setup_distributed
from utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    """解析独立评估所需的 run、checkpoint 和 split 参数。"""
    parser = argparse.ArgumentParser(description="SimpleCNN 固定标准网格评估")
    parser.add_argument("--run-dir", type=Path, required=True, help="包含 resolved_config.json 和 data/ 的训练目录。")
    parser.add_argument("--checkpoint", type=Path, default=None, help="默认使用 run-dir/checkpoints/best.pt。")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    return parser.parse_args()


def main() -> None:
    """启动多卡安全的独立评估。"""
    args = parse_args()
    context = setup_distributed()
    try:
        config = load_config_json(args.run_dir / "resolved_config.json")
        seed_everything(config.seed, context.rank)
        checkpoint_path = args.checkpoint or args.run_dir / "checkpoints" / "best.pt"
        checkpoint = load_checkpoint(checkpoint_path, context.device)
        model = SimpleCNN(config).to(context.device)
        model.load_state_dict(checkpoint["model_state"])
        manifest_name = "validation_grid.npz" if args.split == "val" else "test_grid.npz"
        dataloader = build_validation_dataloader(
            config,
            args.run_dir / "data" / manifest_name,
            rank=context.rank,
            world_size=context.world_size,
        )
        metrics = evaluate_model(model, dataloader, config, context, prefix=args.split)
        if context.is_main:
            rank_zero_print(context, f"[{args.split}] " + "  ".join(f"{key}={value:.6g}" for key, value in metrics.items()))
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
