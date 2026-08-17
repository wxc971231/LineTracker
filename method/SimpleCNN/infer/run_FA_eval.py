"""评估 adaptive_tracker 在纯背景序列上的帧级虚警率。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import math
from pathlib import Path
import sys
from typing import Any, TypedDict

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = METHOD_ROOT.parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from data.dataloader import PackedSource, standard_distance_starts
from infer.adaptive_tracker.infer import run_source as run_adaptive_source
from infer.common.complexity import estimate_model_complexity
from infer.common.model_loader import load_inference_bundle
from infer.common.output import configure_logger, create_output_dir, safe_name, write_json
from infer.common.plotting import plot_false_alarm_diagnostic
from infer.common.runner import ModelRunner
from infer.run_infer import (
    _checkpoint_step,
    _nonnegative_int,
    _positive_int,
    _sample_compute_summary,
    _static_model_complexity,
    _timing_summary,
    _select_records,
    add_adaptive_tracker_arguments,
    adaptive_config_from_args,
)
from utils.process_title import set_process_title


_FRAME_INTERVAL_S = 0.05
_DEFAULT_CHECKPOINT = Path(
    "/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/runs/simplecnn_v2/"
    "limit50k-gbs1024-lr5e-4-pos25-vs5-models-cfg11ba6304/20260728_113116/"
    "checkpoints/best.pt"
)
_DEFAULT_DATA_ROOT = REPOSITORY_ROOT / (
    "data/synthetic/gen/F300_N1000_S20260816_B-random_"
    "T-R10000-290000m-V340-A6-C10-J1-K0-Q0-0"
)


class FalseAlarmSummary(TypedDict):
    """单个纯背景序列的帧级虚警汇总，也是 JSON 落盘字段契约。"""

    frames: int
    evaluable_frames: int
    false_alarm_frames: int
    false_alarm_frame_rate: float
    false_alarm_frames_including_unreliable: int
    false_alarm_frame_rate_including_unreliable: float
    reliable_false_alarm_frames: int
    unreliable_false_alarm_frames: int
    false_alarm_event_count: int
    false_alarm_event_durations_frames: list[int]
    false_alarm_event_count_including_unreliable: int
    false_alarm_event_durations_frames_including_unreliable: list[int]
    false_alarm_event_duration_total_frames: int
    false_alarm_event_duration_mean_frames: float
    false_alarm_event_duration_max_frames: int
    false_alarm_event_duration_mean_s: float
    false_alarm_event_duration_max_s: float
    false_alarm_event_duration_total_frames_including_unreliable: int
    false_alarm_event_duration_mean_frames_including_unreliable: float
    false_alarm_event_duration_max_frames_including_unreliable: int
    false_alarm_event_duration_mean_s_including_unreliable: float
    false_alarm_event_duration_max_s_including_unreliable: float


def _one_dimensional(values: object, name: str, *, frames: int, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1 or array.size != frames:
        raise ValueError(f"{name} 必须是长度为 {frames} 的一维数组，实际形状为 {array.shape}。")
    return array


def _alarm_event_durations(alarm_mask: np.ndarray) -> np.ndarray:
    """返回每个连续报警段的长度（帧）。"""
    padded = np.concatenate((np.zeros(1, dtype=bool), alarm_mask, np.zeros(1, dtype=bool)))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return (stops - starts).astype(np.int32, copy=False)


def _source_false_alarm_summary(
    result: Mapping[str, object],
    *,
    frames: int,
    frames_per_window: int,
    frame_interval_s: float,
) -> FalseAlarmSummary:
    """从一次自适应推理结果中统计纯背景样本的报警时间。"""
    if not 0 <= frames_per_window <= frames:
        raise ValueError("frames_per_window 必须位于 [0, frames] 内。")
    if not math.isfinite(frame_interval_s) or frame_interval_s <= 0.0:
        raise ValueError("frame_interval_s 必须为正有限数。")

    prediction = _one_dimensional(
        result.get("prediction_m"), "prediction_m", frames=frames, dtype=np.float64
    )
    unreliable = _one_dimensional(
        result.get("unreliable_prediction_mask", np.zeros(frames, dtype=bool)),
        "unreliable_prediction_mask",
        frames=frames,
        dtype=bool,
    )
    evaluable_mask = np.arange(frames) >= frames_per_window
    all_alarm_mask = evaluable_mask & np.isfinite(prediction)
    reliable_alarm_mask = all_alarm_mask & ~unreliable
    durations = _alarm_event_durations(reliable_alarm_mask)
    durations_including_unreliable = _alarm_event_durations(all_alarm_mask)
    false_alarm_frames = int(reliable_alarm_mask.sum())
    false_alarm_frames_including_unreliable = int(all_alarm_mask.sum())
    evaluable_frames = int(evaluable_mask.sum())
    reliable_frames = false_alarm_frames
    unreliable_frames = int(np.count_nonzero(all_alarm_mask & unreliable))
    duration_values = durations.astype(np.float64, copy=False)
    duration_values_including_unreliable = durations_including_unreliable.astype(
        np.float64, copy=False
    )

    return {
        "frames": int(frames),
        "evaluable_frames": evaluable_frames,
        "false_alarm_frames": false_alarm_frames,
        "false_alarm_frame_rate": float(false_alarm_frames / max(evaluable_frames, 1)),
        "false_alarm_frames_including_unreliable": false_alarm_frames_including_unreliable,
        "false_alarm_frame_rate_including_unreliable": float(
            false_alarm_frames_including_unreliable / max(evaluable_frames, 1)
        ),
        "reliable_false_alarm_frames": reliable_frames,
        "unreliable_false_alarm_frames": unreliable_frames,
        "false_alarm_event_count": int(durations.size),
        "false_alarm_event_durations_frames": durations.tolist(),
        "false_alarm_event_count_including_unreliable": int(
            durations_including_unreliable.size
        ),
        "false_alarm_event_durations_frames_including_unreliable": (
            durations_including_unreliable.tolist()
        ),
        "false_alarm_event_duration_total_frames": int(durations.sum()),
        "false_alarm_event_duration_mean_frames": (
            float(duration_values.mean()) if durations.size else 0.0
        ),
        "false_alarm_event_duration_max_frames": int(durations.max()) if durations.size else 0,
        "false_alarm_event_duration_mean_s": (
            float(duration_values.mean() * frame_interval_s) if durations.size else 0.0
        ),
        "false_alarm_event_duration_max_s": (
            float(durations.max() * frame_interval_s) if durations.size else 0.0
        ),
        "false_alarm_event_duration_total_frames_including_unreliable": int(
            durations_including_unreliable.sum()
        ),
        "false_alarm_event_duration_mean_frames_including_unreliable": (
            float(duration_values_including_unreliable.mean())
            if durations_including_unreliable.size
            else 0.0
        ),
        "false_alarm_event_duration_max_frames_including_unreliable": (
            int(durations_including_unreliable.max())
            if durations_including_unreliable.size
            else 0
        ),
        "false_alarm_event_duration_mean_s_including_unreliable": (
            float(duration_values_including_unreliable.mean() * frame_interval_s)
            if durations_including_unreliable.size
            else 0.0
        ),
        "false_alarm_event_duration_max_s_including_unreliable": (
            float(durations_including_unreliable.max() * frame_interval_s)
            if durations_including_unreliable.size
            else 0.0
        ),
    }


def _aggregate_false_alarm_summaries(
    summaries: Sequence[FalseAlarmSummary],
    *,
    frame_interval_s: float,
) -> dict[str, float | int]:
    """将逐样本虚警统计聚合为总体帧级与样本级比率。"""
    if not math.isfinite(frame_interval_s) or frame_interval_s <= 0.0:
        raise ValueError("frame_interval_s 必须为正有限数。")
    evaluable_frames = sum(int(summary["evaluable_frames"]) for summary in summaries)
    false_alarm_frames = sum(int(summary["false_alarm_frames"]) for summary in summaries)
    false_alarm_frames_including_unreliable = sum(
        int(summary["false_alarm_frames_including_unreliable"])
        for summary in summaries
    )
    reliable_frames = sum(int(summary["reliable_false_alarm_frames"]) for summary in summaries)
    unreliable_frames = sum(int(summary["unreliable_false_alarm_frames"]) for summary in summaries)
    durations = np.asarray(
        [
            duration
            for summary in summaries
            for duration in summary["false_alarm_event_durations_frames"]
        ],
        dtype=np.float64,
    )
    durations_including_unreliable = np.asarray(
        [
            duration
            for summary in summaries
            for duration in summary["false_alarm_event_durations_frames_including_unreliable"]
        ],
        dtype=np.float64,
    )
    samples_with_false_alarm = sum(int(summary["false_alarm_frames"]) > 0 for summary in summaries)
    samples_with_false_alarm_including_unreliable = sum(
        int(summary["false_alarm_frames_including_unreliable"]) > 0
        for summary in summaries
    )
    sample_count = len(summaries)
    return {
        "sample_count": sample_count,
        "evaluable_frames": evaluable_frames,
        "false_alarm_frames": false_alarm_frames,
        "false_alarm_frame_rate": float(false_alarm_frames / max(evaluable_frames, 1)),
        "false_alarm_frames_including_unreliable": false_alarm_frames_including_unreliable,
        "false_alarm_frame_rate_including_unreliable": float(
            false_alarm_frames_including_unreliable / max(evaluable_frames, 1)
        ),
        "reliable_false_alarm_frames": reliable_frames,
        "unreliable_false_alarm_frames": unreliable_frames,
        "samples_with_false_alarm": samples_with_false_alarm,
        "sample_false_alarm_rate": float(samples_with_false_alarm / max(sample_count, 1)),
        "samples_with_false_alarm_including_unreliable": (
            samples_with_false_alarm_including_unreliable
        ),
        "sample_false_alarm_rate_including_unreliable": float(
            samples_with_false_alarm_including_unreliable / max(sample_count, 1)
        ),
        "false_alarm_event_count": int(durations.size),
        "false_alarm_event_duration_total_frames": int(durations.sum()),
        "false_alarm_event_duration_mean_frames": float(durations.mean()) if durations.size else 0.0,
        "false_alarm_event_duration_max_frames": int(durations.max()) if durations.size else 0,
        "false_alarm_event_duration_mean_s": (
            float(durations.mean() * frame_interval_s) if durations.size else 0.0
        ),
        "false_alarm_event_duration_max_s": (
            float(durations.max() * frame_interval_s) if durations.size else 0.0
        ),
        "false_alarm_event_count_including_unreliable": int(
            durations_including_unreliable.size
        ),
        "false_alarm_event_duration_total_frames_including_unreliable": int(
            durations_including_unreliable.sum()
        ),
        "false_alarm_event_duration_mean_frames_including_unreliable": (
            float(durations_including_unreliable.mean())
            if durations_including_unreliable.size
            else 0.0
        ),
        "false_alarm_event_duration_max_frames_including_unreliable": (
            int(durations_including_unreliable.max())
            if durations_including_unreliable.size
            else 0
        ),
        "false_alarm_event_duration_mean_s_including_unreliable": (
            float(durations_including_unreliable.mean() * frame_interval_s)
            if durations_including_unreliable.size
            else 0.0
        ),
        "false_alarm_event_duration_max_s_including_unreliable": (
            float(durations_including_unreliable.max() * frame_interval_s)
            if durations_including_unreliable.size
            else 0.0
        ),
    }


def _assert_pure_background(source: PackedSource) -> None:
    """拒绝含有任意实际目标响应的样本，防止误用有目标数据集。"""
    if np.any(source.target_hit):
        raise ValueError(
            f"{source.record.source_id} 包含实际目标响应，不能用于纯背景虚警评估。"
        )


def _output_dir(*, data_root: Path, checkpoint_path: Path, checkpoint_step: object, args: argparse.Namespace) -> Path:
    base_dir = create_output_dir(
        data_root=data_root,
        checkpoint_path=checkpoint_path,
        checkpoint_step=checkpoint_step,
    )
    name = safe_name(
        f"adaptive_tracker_stride{args.time_stride}_captureq{args.capture_q_min:g}_trackq{args.q_keep:g}"
    )
    path = base_dir / "false_alarm" / name
    (path / "samples").mkdir(parents=True, exist_ok=True)
    return path


def _evaluate_source(
    source: PackedSource,
    *,
    runner: ModelRunner,
    method_config: Any,
    tracker_config: Any,
    complexity: Any,
    output_dir: Path,
    args: argparse.Namespace,
    logger: Any,
) -> FalseAlarmSummary:
    """运行并落盘一个纯背景样本，供单卡和多 NPU 入口共同调用。"""
    _assert_pure_background(source)
    result = run_adaptive_source(source, runner, runner.config, method_config, tracker_config)
    false_alarm = _source_false_alarm_summary(
        result,
        frames=source.frames,
        frames_per_window=runner.config.frames_per_window,
        frame_interval_s=_FRAME_INTERVAL_S,
    )
    workload = result["workload"]
    if not isinstance(workload, Mapping):
        raise TypeError("adaptive_tracker 推理结果的 workload 必须为字典。")
    sample_dir = output_dir / "samples" / safe_name(source.record.source_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    figure_error: str | None = None
    if args.save_figures and false_alarm["false_alarm_frames_including_unreliable"]:
        try:
            figure = plot_false_alarm_diagnostic(
                source,
                result,
                config=runner.config,
                title=f"adaptive_tracker 纯背景虚警：{source.record.source_id}",
                false_alarm=false_alarm,
                capture_q_min=method_config.capture_q_min,
                q_keep=method_config.q_keep,
                frame_interval_s=_FRAME_INTERVAL_S,
            )
            figure.savefig(sample_dir / "visualize.png", dpi=args.figure_dpi)
            import matplotlib.pyplot as plt

            plt.close(figure)
        except Exception as error:
            figure_error = f"{type(error).__name__}: {error}"
            logger.exception("样本 %s 的虚警诊断图保存失败。", source.record.source_id)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "source_id": source.record.source_id,
        "method": "adaptive_tracker",
        "false_alarm": false_alarm,
        "compute": _sample_compute_summary(complexity, workload),
        "timing": _timing_summary("adaptive_tracker", result),
    }
    if figure_error is not None:
        payload["figure_error"] = figure_error
    write_json(sample_dir / "metrics.json", payload)
    return false_alarm


def run(args: argparse.Namespace) -> Path:
    """在全部选定纯背景样本上运行 adaptive_tracker 并写出虚警统计。"""
    bundle = load_inference_bundle(args.checkpoint, data_root=args.data_root, device=args.device)
    set_process_title("false-alarm-eval", label=bundle.checkpoint_path.stem, infer_rank=False)
    method_config = adaptive_config_from_args(args)
    tracker_config = method_config.validate(bundle.config)
    selected_records = _select_records(bundle, args)
    runner = ModelRunner(
        bundle.model,
        bundle.config,
        bundle.device,
        max_blocks_per_forward=args.max_blocks_per_forward,
    )
    complexity = estimate_model_complexity(
        bundle.model,
        input_channels=bundle.config.input_channels,
        input_height=bundle.config.frames_per_window,
        input_width=bundle.config.block_width_m // bundle.config.input_channels,
    )
    checkpoint_step = _checkpoint_step(bundle.checkpoint)
    output_dir = _output_dir(
        data_root=bundle.config.data_root,
        checkpoint_path=bundle.checkpoint_path,
        checkpoint_step=checkpoint_step,
        args=args,
    )
    write_json(
        output_dir / "config.json",
        {
            "schema_version": 1,
            "method": "adaptive_tracker",
            "checkpoint_path": bundle.checkpoint_path,
            "checkpoint_global_step": checkpoint_step,
            "resolved_config_path": bundle.resolved_config_path,
            "data_root": bundle.config.data_root,
            "requested_device": args.device,
            "effective_device": str(bundle.device),
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

    first_source: PackedSource | None = None
    if args.warmup:
        first_source = PackedSource(selected_records[0], bundle.config)
        runner.warmup(first_source, range_starts_m=standard_distance_starts(bundle.config))

    logger = configure_logger(f"false-alarm-{bundle.checkpoint_path.stem}")
    logger.info("开始纯背景虚警评估：device=%s，输出=%s", bundle.device, output_dir)
    source_summaries: list[FalseAlarmSummary] = []
    for offset, record in enumerate(selected_records):
        source = first_source if offset == 0 and first_source is not None else PackedSource(record, bundle.config)
        false_alarm = _evaluate_source(
            source,
            runner=runner,
            method_config=method_config,
            tracker_config=tracker_config,
            complexity=complexity,
            output_dir=output_dir,
            args=args,
            logger=logger,
        )
        source_summaries.append(false_alarm)
        logger.info(
            "[%d/%d] source=%s FAR=%.3f%% (含外推 %.3f%%) events=%d (含外推 %d)",
            offset + 1,
            len(selected_records),
            record.source_id,
            100.0 * float(false_alarm["false_alarm_frame_rate"]),
            100.0 * float(false_alarm["false_alarm_frame_rate_including_unreliable"]),
            int(false_alarm["false_alarm_event_count"]),
            int(false_alarm["false_alarm_event_count_including_unreliable"]),
        )

    aggregate = _aggregate_false_alarm_summaries(
        source_summaries, frame_interval_s=_FRAME_INTERVAL_S
    )
    write_json(
        output_dir / "summary.json",
        {
            "schema_version": 2,
            "method": "adaptive_tracker",
            "metric_definition": {
                "false_alarm_frame": "可评估帧中最终 prediction_m 为有限数且不是 RECAPTURE 外推。",
                "false_alarm_frame_including_unreliable": "可评估帧中最终 prediction_m 为有限数，包含 RECAPTURE 外推。",
                "evaluable_frame_start": int(bundle.config.frames_per_window),
                "frame_interval_s": _FRAME_INTERVAL_S,
            },
            "aggregate": aggregate,
        },
    )
    logger.info(
        "评估完成：samples=%d，帧级 FAR=%.3f%% (含外推 %.3f%%)，样本级 FAR=%.3f%% (含外推 %.3f%%)，结果=%s",
        aggregate["sample_count"],
        100.0 * float(aggregate["false_alarm_frame_rate"]),
        100.0 * float(aggregate["false_alarm_frame_rate_including_unreliable"]),
        100.0 * float(aggregate["sample_false_alarm_rate"]),
        100.0 * float(aggregate["sample_false_alarm_rate_including_unreliable"]),
        output_dir,
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在纯背景序列上评估 SimpleCNN adaptive_tracker 的帧级虚警率。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT, help="要加载的 best.pt 或 last.pt。")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT, help="纯背景一级样本目录根路径。")
    parser.add_argument("--sample-start", type=_nonnegative_int, default=0, help="评估样本索引范围起点。")
    parser.add_argument("--sample-stop", type=_nonnegative_int, default=None, help="评估样本索引范围终点（不含）。")
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda[:index] 或 npu[:index]。")
    parser.add_argument("--time-stride", type=_positive_int, default=5, help="TRACK 状态的时间步进（帧）。")
    parser.add_argument("--max-blocks-per-forward", type=_nonnegative_int, default=0, help="单次模型前向最多拼接的距离块数；0 表示不拆分。")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True, help="是否预热模型；预热耗时不计入统计。")
    parser.add_argument("--figures", action=argparse.BooleanOptionalAction, dest="save_figures", default=True, help="是否为发生虚警的样本保存诊断图。")
    parser.add_argument("--figure-dpi", type=_positive_int, default=180, help="虚警诊断图的保存分辨率。")

    add_adaptive_tracker_arguments(parser, title="adaptive_tracker 参数")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("评估已由用户中断。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
