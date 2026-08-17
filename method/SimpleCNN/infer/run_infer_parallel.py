"""按样本把 SimpleCNN 流式推理静态分配到多个 NPU。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from time import perf_counter
from typing import Any


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from data.dataloader import PackedSource, SourceRecord, standard_distance_starts
from infer.adaptive_tracker.infer import AdaptiveInferenceConfig, run_source as run_adaptive_source
from infer.common.complexity import estimate_model_complexity
from infer.common.model_loader import load_inference_bundle
from infer.common.multi_device import (
    execute_device_tasks,
    partition_round_robin,
    resolve_npu_devices,
)
from infer.common.output import configure_logger, create_output_dir, safe_name, write_json
from infer.common.runner import ModelRunner
from infer.global_top1.infer import GlobalTop1Config, run_source as run_global_source
from infer.run_infer import (
    _METHOD_TITLES,
    _adaptive_summary,
    _checkpoint_step,
    _compact_trajectory_summary,
    _effective_sample_bounds,
    _figure_title,
    _method_config,
    _method_directory,
    _sample_compute_summary,
    _sample_directory,
    _select_records,
    _static_model_complexity,
    _timing_summary,
    _trajectory_summary,
    _workload_summary,
    _write_method_parameter_snapshot,
    _write_sample,
    build_parser as build_single_parser,
)
from utils.process_title import set_process_title


def _parallel_output_dir(base_dir: Path, devices: Sequence[str]) -> Path:
    tag = safe_name("-".join(device.replace(":", "") for device in devices))
    output_dir = base_dir / f"parallel-{tag}"
    (output_dir / "samples").mkdir(parents=True, exist_ok=True)
    return output_dir


def _run_device_worker(task: Mapping[str, Any]) -> dict[str, object]:
    """在一个固定 NPU 上依次处理已静态分配的样本。"""
    args = task["args"]
    device = str(task["device"])
    records = tuple(task["records"])
    output_dir = Path(task["output_dir"])
    if not isinstance(args, argparse.Namespace):
        raise TypeError("并行推理 worker 未收到 argparse.Namespace 参数。")
    if not all(isinstance(record, SourceRecord) for record in records):
        raise TypeError("并行推理 worker 的 records 必须全部为 SourceRecord。")

    worker_start = perf_counter()
    set_process_title("infer-parallel", label=f"{args.method}-{device}", infer_rank=False)
    bundle = load_inference_bundle(args.checkpoint, data_root=args.data_root, device=device)
    method_config = _method_config(args)
    tracker_config = (
        method_config.validate(bundle.config)
        if isinstance(method_config, AdaptiveInferenceConfig)
        else None
    )
    if isinstance(method_config, GlobalTop1Config):
        method_config.validate()
    complexity = estimate_model_complexity(
        bundle.model,
        input_channels=bundle.config.input_channels,
        input_height=bundle.config.frames_per_window,
        input_width=bundle.config.block_width_m // bundle.config.input_channels,
    )
    runner = ModelRunner(
        bundle.model,
        bundle.config,
        bundle.device,
        max_blocks_per_forward=args.max_blocks_per_forward,
    )
    if args.warmup:
        warmup_source = PackedSource(records[0], bundle.config)
        runner.warmup(warmup_source, range_starts_m=standard_distance_starts(bundle.config))

    logger = configure_logger(f"parallel-{args.method}-{device}")
    for record in records:
        source = PackedSource(record, bundle.config)
        if isinstance(method_config, GlobalTop1Config):
            result = run_global_source(source, runner, bundle.config, method_config)
        else:
            assert tracker_config is not None
            result = run_adaptive_source(source, runner, bundle.config, method_config, tracker_config)
        if not isinstance(result, Mapping):
            raise TypeError("方法 run_source 必须返回字典。")

        trajectory = _compact_trajectory_summary(
            _trajectory_summary(result, source, jump_threshold_m=args.jump_threshold_m)
        )
        workload = _workload_summary(result)
        timing = _timing_summary(args.method, result)
        compute = _sample_compute_summary(complexity, workload)
        tracker: dict[str, float | int] = {}
        if args.method == "adaptive_tracker":
            tracker = _adaptive_summary(result, frames_per_window=bundle.config.frames_per_window)
        sample_dir = _sample_directory(output_dir, record.source_id)
        method_dir = _method_directory(sample_dir, args.method, args.time_stride)
        _write_sample(
            method_dir,
            method=args.method,
            result=result,
            source=source,
            trajectory=trajectory,
            compute=compute,
            timing=timing,
            tracker=tracker,
            save_figures=args.save_figures,
            figure_dpi=args.figure_dpi,
            config=bundle.config,
            time_stride=args.time_stride,
            title=_figure_title(args.method, record.source_id, args.time_stride),
            logger=logger,
        )
        logger.info("device=%s source=%s 已完成", device, record.source_id)

    return {
        "device": device,
        "sample_count": len(records),
        "wall_time_s": perf_counter() - worker_start,
    }


def run(args: argparse.Namespace) -> Path:
    """启动每 NPU 一个 worker 的样本级并行推理。"""
    if args.device != "auto":
        raise ValueError("并行入口请使用 --devices 指定 NPU，不支持 --device。")
    devices = resolve_npu_devices(args.devices)
    method_config = _method_config(args)
    reference_bundle = load_inference_bundle(args.checkpoint, data_root=args.data_root, device="cpu")
    if isinstance(method_config, AdaptiveInferenceConfig):
        method_config.validate(reference_bundle.config)
    else:
        method_config.validate()
    records = _select_records(reference_bundle, args)
    complexity = estimate_model_complexity(
        reference_bundle.model,
        input_channels=reference_bundle.config.input_channels,
        input_height=reference_bundle.config.frames_per_window,
        input_width=reference_bundle.config.block_width_m // reference_bundle.config.input_channels,
    )
    checkpoint_step = _checkpoint_step(reference_bundle.checkpoint)
    base_dir = create_output_dir(
        data_root=reference_bundle.config.data_root,
        checkpoint_path=reference_bundle.checkpoint_path,
        checkpoint_step=checkpoint_step,
    )
    output_dir = _parallel_output_dir(base_dir, devices)
    write_json(
        output_dir / "_infer_config.json",
        {
            "schema_version": 6,
            "checkpoint_path": reference_bundle.checkpoint_path,
            "checkpoint_global_step": checkpoint_step,
            "resolved_config_path": reference_bundle.resolved_config_path,
            "data_root": reference_bundle.config.data_root,
            "effective_devices": list(devices),
            "model_complexity_per_block": _static_model_complexity(complexity),
        },
    )
    _write_method_parameter_snapshot(
        output_dir,
        args=args,
        bundle=reference_bundle,
        checkpoint_step=checkpoint_step,
        method_config=method_config,
        complexity=complexity,
        effective_device=",".join(devices),
    )
    assignments = partition_round_robin(records, devices)
    tasks = {
        device: {
            "args": args,
            "device": device,
            "records": device_records,
            "output_dir": str(output_dir),
        }
        for device, device_records in assignments.items()
    }
    logger = configure_logger(f"parallel-{args.method}-{reference_bundle.checkpoint_path.stem}")
    logger.info("开始多 NPU 推理：devices=%s samples=%d", ",".join(devices), len(records))
    wall_start = perf_counter()
    worker_results = execute_device_tasks(tasks, _run_device_worker)
    wall_time_s = perf_counter() - wall_start
    write_json(
        output_dir / "parallel_summary.json",
        {
            "schema_version": 1,
            "devices": list(devices),
            "sample_count": len(records),
            "wall_time_s": wall_time_s,
            "workers": [worker_results[device] for device in sorted(worker_results)],
        },
    )
    logger.info("多 NPU 推理完成：wall=%.3f s，结果已写入 %s", wall_time_s, output_dir)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = build_single_parser()
    parser.description = "按样本将 SimpleCNN 流式推理静态分配到多个 NPU。"
    parser.add_argument(
        "--devices",
        help="逻辑 NPU，例如 0,1,2；缺省时使用全部 ASCEND_RT_VISIBLE_DEVICES。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("多 NPU 推理已由用户中断。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
