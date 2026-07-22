#!/usr/bin/env python3
"""多进程批量生成含移动目标的 dataset1 风格二值时距数据。

本脚本的背景、目标、保存格式和预览图全部复用 generate_dataset1.py，唯一增加的
参数是 --workers。每个进程独立生成一个样本目录，因此不会发生数据文件写入冲突。

常用命令：

    conda run -n linetracker-py311 python data/synthetic/generate_dataset1_parallel.py \\
        --samples 100 --workers 2 --random-knobs true --seed 20260717

建议先使用 --workers 2。单个样本会同时持有 300 x 300000 的背景、完整观测矩阵和
绘图数据，盲目按 CPU 核数全开容易造成内存和磁盘写入争用。
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from generate_dataset1 import (
    _jsonable,
    build_generation_hyperparameters,
    build_parser,
    generate_sample,
    load_model,
    resolve_generation_output_dir,
    target_config_from_args,
)


_WORKER_MODEL: dict[str, np.ndarray] | None = None


def _initialize_worker(model_path: str) -> None:
    """每个子进程启动时各自读取一次拟合模型，避免在任务间重复传输大数组。"""
    global _WORKER_MODEL
    _WORKER_MODEL = load_model(Path(model_path))


def _generate_one(task: dict[str, Any]) -> dict[str, Any]:
    """子进程任务：使用共享入口生成一个样本并返回清单记录。"""
    if _WORKER_MODEL is None:
        raise RuntimeError("子进程尚未初始化拟合模型。")
    return generate_sample(
        model=_WORKER_MODEL,
        output_dir=Path(task["output_dir"]),
        index=int(task["index"]),
        frames=int(task["frames"]),
        background_seed=int(task["background_seed"]),
        target_seed=int(task["target_seed"]),
        prefix=str(task["prefix"]),
        level=float(task["level"]),
        decay=float(task["decay"]),
        near_field=float(task["near_field"]),
        gain=float(task["gain"]),
        cluster=float(task["cluster"]),
        block_frames=int(task["block_frames"]),
        validate=bool(task["validate"]),
        target_config=dict(task["target_config"]),
        preview_margin_m=float(task["preview_margin_m"]),
        preview_point_size=float(task["preview_point_size"]),
    )


def _build_tasks(
    args: argparse.Namespace,
    target_config: dict[str, float | int],
) -> list[dict[str, Any]]:
    """在主进程预先固定每个样本的旋钮和种子，保证并行顺序不影响可复现性。"""
    knob_rng = np.random.default_rng(args.seed)
    tasks: list[dict[str, Any]] = []
    for index in range(args.samples):
        if args.random_knobs:
            level, decay, near_field, gain, cluster = knob_rng.uniform(
                -1.0, 1.0, size=5
            )
        else:
            level, decay, near_field, gain, cluster = (
                args.level,
                args.decay,
                args.near_field,
                args.gain,
                args.cluster,
            )
        tasks.append(
            {
                "output_dir": str(args.output_dir),
                "index": index,
                "frames": args.frames,
                "background_seed": args.seed + index,
                "target_seed": args.seed + 1_000_000 + index,
                "prefix": args.prefix,
                "level": float(level),
                "decay": float(decay),
                "near_field": float(near_field),
                "gain": float(gain),
                "cluster": float(cluster),
                "block_frames": args.block_frames,
                "validate": args.validate,
                "target_config": target_config,
                "preview_margin_m": args.preview_margin_m,
                "preview_point_size": args.preview_point_size,
            }
        )
    return tasks


def build_parallel_parser() -> argparse.ArgumentParser:
    """复用串行脚本的全部参数，并添加多进程调度参数。"""
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="并行子进程数；建议从 2 开始，根据内存和磁盘带宽酌情增大",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    """并行调度样本生成，并在主进程统一写出批次清单。"""
    parser = build_parallel_parser()
    args = parser.parse_args(argv)

    args.samples = 200
    args.workers = 5

    if args.samples <= 0:
        parser.error("--samples 必须为正整数")
    if args.workers <= 0:
        parser.error("--workers 必须为正整数")

    output_root = Path(args.output_dir)
    args.output_dir = resolve_generation_output_dir(args.output_dir, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_config = target_config_from_args(args)
    hyperparameters = build_generation_hyperparameters(args)
    hyperparameters["parallel"] = {"workers": int(args.workers)}
    tasks = _build_tasks(args, target_config)
    records: list[dict[str, Any]] = []

    # 使用 spawn 避免继承父进程中的 NumPy/Matplotlib 状态；每个 worker 只加载一次模型。
    context = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(str(args.model.resolve()),),
    ) as executor:
        futures = {
            executor.submit(_generate_one, task): int(task["index"])
            for task in tasks
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            record = future.result()
            records.append(record)
            print(f"已完成 {completed_count}/{args.samples}：样本 {index:04d}")

    # 任务完成顺序不固定；按样本编号写清单，保持与串行脚本一致的记录顺序。
    records.sort(key=lambda record: int(record["index"]))
    manifest_path = args.manifest or args.output_dir / "batch_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": str(args.model.resolve()),
        "output_root": str(output_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "samples": args.samples,
        "frames_per_sample": args.frames,
        "random_knobs": args.random_knobs,
        "target_config": target_config,
        "workers": args.workers,
        "hyperparameters": hyperparameters,
        "records": records,
    }
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已并行生成 {len(records)} 个含目标二值样本目录：{args.output_dir}")
    print(f"批次清单：{manifest_path}")


if __name__ == "__main__":
    main()
