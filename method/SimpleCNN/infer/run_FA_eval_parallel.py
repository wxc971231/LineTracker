"""按样本把纯背景虚警评估静态分配到多个 NPU。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, cast


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from data.dataloader import PackedSource, SourceRecord, standard_distance_starts
from infer.adaptive_tracker.infer import AdaptiveInferenceConfig
from infer.common.complexity import estimate_model_complexity
from infer.common.model_loader import load_inference_bundle
from infer.common.multi_device import (
    execute_device_tasks,
    partition_round_robin,
    resolve_npu_devices,
)
from infer.common.output import configure_logger, write_json
from infer.common.runner import ModelRunner
from infer.run_FA_eval import (
    FalseAlarmSummary,
    _FRAME_INTERVAL_S,
    _aggregate_false_alarm_summaries,
    _evaluate_source,
    _output_dir,
    build_parser as build_single_parser,
)
from infer.run_infer import (
    _checkpoint_step,
    _select_records,
    _static_model_complexity,
    adaptive_config_from_args,
)
from utils.process_title import set_process_title


def _run_device_worker(task: Mapping[str, Any]) -> dict[str, object]:
    """在指定 NPU 上顺序完成其分配到的纯背景样本。"""
    args = task["args"]
    device = str(task["device"])
    records = tuple(task["records"])
    output_dir = Path(task["output_dir"])
    if not isinstance(args, argparse.Namespace):
        raise TypeError("并行虚警 worker 未收到 argparse.Namespace 参数。")
    if not all(isinstance(record, SourceRecord) for record in records):
        raise TypeError("并行虚警 worker 的 records 必须全部为 SourceRecord。")

    worker_start = perf_counter()
    set_process_title("false-alarm-parallel", label=device, infer_rank=False)
    bundle = load_inference_bundle(args.checkpoint, data_root=args.data_root, device=device)
    method_config: AdaptiveInferenceConfig = adaptive_config_from_args(args)
    tracker_config = method_config.validate(bundle.config)
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

    logger = configure_logger(f"parallel-false-alarm-{device}")
    summaries: list[FalseAlarmSummary] = []
    for record in records:
        source = PackedSource(record, bundle.config)
        summary = _evaluate_source(
            source,
            runner=runner,
            method_config=method_config,
            tracker_config=tracker_config,
            complexity=complexity,
            output_dir=output_dir,
            args=args,
            logger=logger,
        )
        summaries.append(summary)
        logger.info(
            "device=%s source=%s FAR=%.3f%% (含外推 %.3f%%)",
            device,
            record.source_id,
            100.0 * float(summary["false_alarm_frame_rate"]),
            100.0 * float(summary["false_alarm_frame_rate_including_unreliable"]),
        )
    return {
        "device": device,
        "sample_count": len(records),
        "wall_time_s": perf_counter() - worker_start,
        "summaries": summaries,
    }


def run(args: argparse.Namespace) -> Path:
    """启动每 NPU 一个 worker 的纯背景虚警评估。"""
    if args.device != "auto":
        raise ValueError("并行入口请使用 --devices 指定 NPU，不支持 --device。")
    devices = resolve_npu_devices(args.devices)
    reference_bundle = load_inference_bundle(args.checkpoint, data_root=args.data_root, device="cpu")
    method_config: AdaptiveInferenceConfig = adaptive_config_from_args(args)
    method_config.validate(reference_bundle.config)
    records = _select_records(reference_bundle, args)
    complexity = estimate_model_complexity(
        reference_bundle.model,
        input_channels=reference_bundle.config.input_channels,
        input_height=reference_bundle.config.frames_per_window,
        input_width=reference_bundle.config.block_width_m // reference_bundle.config.input_channels,
    )
    checkpoint_step = _checkpoint_step(reference_bundle.checkpoint)
    output_dir = _output_dir(
        data_root=reference_bundle.config.data_root,
        checkpoint_path=reference_bundle.checkpoint_path,
        checkpoint_step=checkpoint_step,
        method_config=method_config,
    )
    write_json(
        output_dir / "config.json",
        {
            "schema_version": 2,
            "method": "adaptive_tracker",
            "checkpoint_path": reference_bundle.checkpoint_path,
            "checkpoint_global_step": checkpoint_step,
            "resolved_config_path": reference_bundle.resolved_config_path,
            "data_root": reference_bundle.config.data_root,
            "requested_device": args.device,
            "effective_devices": list(devices),
            "sample_start": args.sample_start,
            "sample_stop": args.sample_stop,
            "max_blocks_per_forward": args.max_blocks_per_forward,
            "warmup": args.warmup,
            "save_figures": args.save_figures,
            "figure_dpi": args.figure_dpi,
            "adaptive_tracker": asdict(method_config),
            "model_complexity_per_block": _static_model_complexity(complexity),
        },
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
    logger = configure_logger(f"parallel-false-alarm-{reference_bundle.checkpoint_path.stem}")
    logger.info("开始多 NPU 纯背景虚警评估：devices=%s samples=%d", ",".join(devices), len(records))
    wall_start = perf_counter()
    worker_results = execute_device_tasks(tasks, _run_device_worker)
    wall_time_s = perf_counter() - wall_start
    summaries: list[FalseAlarmSummary] = []
    for result in worker_results.values():
        summaries.extend(cast(list[FalseAlarmSummary], result["summaries"]))
    aggregate = _aggregate_false_alarm_summaries(summaries, frame_interval_s=_FRAME_INTERVAL_S)
    write_json(
        output_dir / "summary.json",
        {
            "schema_version": 3,
            "method": "adaptive_tracker",
            "metric_definition": {
                "false_alarm_frame": "可评估帧中最终 prediction_m 为有限数且不是 RECAPTURE 外推。",
                "false_alarm_frame_including_unreliable": "可评估帧中最终 prediction_m 为有限数，包含 RECAPTURE 外推。",
                "evaluable_frame_start": int(reference_bundle.config.frames_per_window),
                "frame_interval_s": _FRAME_INTERVAL_S,
            },
            "aggregate": aggregate,
            "parallel": {
                "devices": list(devices),
                "wall_time_s": wall_time_s,
                "workers": [
                    {key: value for key, value in worker_results[device].items() if key != "summaries"}
                    for device in sorted(worker_results)
                ],
            },
        },
    )
    logger.info("多 NPU 虚警评估完成：wall=%.3f s，结果已写入 %s", wall_time_s, output_dir)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = build_single_parser()
    parser.description = "按样本将纯背景虚警评估静态分配到多个 NPU。"
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
        print("多 NPU 虚警评估已由用户中断。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
