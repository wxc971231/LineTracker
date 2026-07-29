"""SimpleCNN 统一流式推理入口。

同一入口固定复用 checkpoint 所属训练 run 的 resolved_config.json，并从运行时
data_root 的一级序列目录取前 N 个样本。可切换全局 34 块 Top-1 基线和
postprocess.md 定义的自适应后处理。推理候选选择只依赖模型输出与观测；真值
仅在运行结束后用于指标与诊断图。
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import math
import re
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable

import numpy as np

METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from data.dataloader import PackedSource, SourceRecord, discover_sources, standard_distance_starts
from infer.adaptive_tracker.infer import AdaptiveInferenceConfig, run_source as run_adaptive_source
from infer.common.complexity import ModelComplexity, estimate_model_complexity
from infer.common.metrics import timing_metrics, timing_metrics_from_steps, trajectory_metrics
from infer.common.model_loader import InferenceBundle, load_inference_bundle
from infer.common.output import (
    configure_logger,
    create_output_dir,
    read_json,
    safe_name,
    write_json,
    write_jsonl,
)
from infer.common.plotting import plot_source_diagnostic
from infer.common.runner import ModelRunner
from infer.global_top1.infer import GlobalTop1Config, run_source as run_global_source
from utils.process_title import set_process_title


_METHOD_DIRS = {
    "global_top1": "global_top1",
    "adaptive_tracker": "adaptive_tracker",
}
_METHOD_TITLES = {
    "global_top1": "全局 34 块 Top-1 流式基线",
    "adaptive_tracker": "CAPTURE—TRACK—RECAPTURE 自适应推理",
}
_FRAME_INTERVAL_S = 0.05


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是正整数。") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须是正整数。")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是非负整数。") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("必须是非负整数。")
    return parsed


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是有限数。") from error
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("必须是有限数。")
    return parsed


def _csv_values(raw: str, converter: Callable[[str], Any], name: str) -> tuple[Any, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"{name} 不能为空。")
    try:
        return tuple(converter(part) for part in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 应为逗号分隔的合法值：{raw!r}。") from error


def _optional_number(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if math.isfinite(numeric) else float("nan")


def _checkpoint_step(checkpoint: Mapping[str, Any]) -> int | str | None:
    value = checkpoint.get("global_step")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _select_records(
    bundle: InferenceBundle,
    args: argparse.Namespace,
) -> tuple[list[SourceRecord], dict[str, int]]:
    records = discover_sources(bundle.config.data_root)
    selected = sorted(records, key=lambda record: record.source_id)[: args.num_samples]
    data_root_indices = {record.source_id: index for index, record in enumerate(selected)}
    return selected, data_root_indices


def _sample_file_stem(method: str, time_stride: int) -> str:
    return f"{safe_name(method)}_stride{time_stride}"


def _figure_title(method: str, source_id: str, time_stride: int) -> str:
    suffix = re.search(r"(\d+)$", source_id)
    data_label = f"data{suffix.group(1)}" if suffix else source_id
    return f"{method.replace('_', '-')}-{data_label}：时间步进 {time_stride}"


def _method_config(args: argparse.Namespace) -> GlobalTop1Config | AdaptiveInferenceConfig:
    if args.method == "global_top1":
        return GlobalTop1Config(time_stride=args.time_stride)

    gates = tuple(_csv_values(args.position_gates_m, float, "position_gates_m"))
    if len(gates) != 3:
        raise ValueError("position_gates_m 必须恰好提供 L=0/1/2 三个门限。")
    return AdaptiveInferenceConfig(
        time_stride=args.time_stride,
        capture_stride=args.capture_stride,
        capture_buffer_size=args.capture_buffer_size,
        capture_support_ratio=args.capture_support_ratio,
        capture_radius_m=args.capture_radius_m,
        q_keep=args.q_keep,
        position_gate_m=gates,
        expand_after_bad=args.expand_after_bad,
        shrink_after_good=args.shrink_after_good,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )


def _step_prediction_arrays(result: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    frame_count = int(result["frame_count"])
    current = np.full(frame_count, np.nan, dtype=np.float64)
    next_frame = np.full(frame_count, np.nan, dtype=np.float64)
    for step in result.get("steps", ()):
        if not isinstance(step, Mapping):
            continue
        latest = step.get("latest_frame")
        if latest is not None:
            latest_index = int(latest)
            value = _optional_number(step.get("range_current_m"))
            if 0 <= latest_index < frame_count and math.isfinite(value):
                current[latest_index] = value
        forecast_start = step.get("forecast_frame_start")
        value = _optional_number(step.get("range_next_m"))
        if forecast_start is not None and math.isfinite(value):
            next_index = int(forecast_start)
            if 0 <= next_index < frame_count:
                next_frame[next_index] = value
    return current, next_frame


def _trajectory_summary(
    result: Mapping[str, Any],
    source: PackedSource,
    *,
    jump_threshold_m: float,
) -> dict[str, float | int]:
    prediction = np.asarray(result["prediction_m"], dtype=np.float64)
    truth = source.target_true_range_m
    target_hit = source.target_hit
    summary = trajectory_metrics(
        prediction,
        truth,
        target_hit=target_hit,
        jump_threshold_m=jump_threshold_m,
    )
    current, next_frame = _step_prediction_arrays(result)
    current_summary = trajectory_metrics(
        current,
        truth,
        target_hit=target_hit,
        jump_threshold_m=jump_threshold_m,
    )
    next_summary = trajectory_metrics(
        next_frame,
        truth,
        target_hit=target_hit,
        jump_threshold_m=jump_threshold_m,
    )
    for prefix, values in (("current_", current_summary), ("next_", next_summary)):
        for key, value in values.items():
            if key in {"frame_count", "jump_threshold_m"}:
                continue
            summary[f"{prefix}{key}"] = value
    return summary


def _timing_summary(result: Mapping[str, Any]) -> dict[str, float | int]:
    steps = [step for step in result.get("steps", ()) if isinstance(step, Mapping)]
    workload = result.get("workload", {})
    if not isinstance(workload, Mapping):
        raise TypeError("推理结果 workload 必须为字典。")

    end_to_end = timing_metrics_from_steps(steps, key="end_to_end_s")
    model_per_forward = [
        _optional_number(step.get("model_s")) / int(step["forward_calls"])
        for step in steps
        if int(step.get("forward_calls", 0)) > 0
    ]
    model_timing = timing_metrics(model_per_forward)
    logical_steps = int(workload.get("logical_steps", len(steps)))
    blocks = int(workload.get("blocks_evaluated", 0))
    forwards = int(workload.get("forward_calls", 0))
    return {
        "logical_steps": logical_steps,
        "blocks_evaluated": blocks,
        "forward_calls": forwards,
        "avg_blocks_per_step": float(blocks / max(logical_steps, 1)),
        "avg_forwards_per_step": float(forwards / max(logical_steps, 1)),
        "end_to_end_total_s": _optional_number(workload.get("end_to_end_s")),
        "end_to_end_mean_ms": 1000.0 * _optional_number(end_to_end["mean_s"]),
        "end_to_end_p50_ms": 1000.0 * _optional_number(end_to_end["p50_s"]),
        "end_to_end_p95_ms": 1000.0 * _optional_number(end_to_end["p95_s"]),
        "end_to_end_p99_ms": 1000.0 * _optional_number(end_to_end["p99_s"]),
        "model_forward_mean_ms": 1000.0 * _optional_number(model_timing["mean_s"]),
        "model_forward_p95_ms": 1000.0 * _optional_number(model_timing["p95_s"]),
        "model_forward_p99_ms": 1000.0 * _optional_number(model_timing["p99_s"]),
        "preprocess_total_s": _optional_number(workload.get("preprocess_s")),
        "model_total_s": _optional_number(workload.get("model_s")),
        "model_postprocess_total_s": _optional_number(workload.get("model_postprocess_s")),
    }


def _adaptive_summary(result: Mapping[str, Any]) -> dict[str, float | int]:
    steps = [step for step in result.get("steps", ()) if isinstance(step, Mapping)]
    frames_per_window = int(result["frames_per_window"])
    initial_confirmations = [
        step
        for step in steps
        if bool(step.get("capture_confirmed")) and step.get("mode_before") == "CAPTURE"
    ]
    recapture_starts = [
        int(step["latest_frame"])
        for step in steps
        if step.get("mode_before") == "TRACK" and step.get("mode") == "RECAPTURE"
    ]
    recapture_confirmations = [
        int(step["latest_frame"])
        for step in steps
        if bool(step.get("capture_confirmed")) and step.get("mode_before") == "RECAPTURE"
    ]
    recapture_delays: list[float] = []
    unused_confirmations = list(recapture_confirmations)
    for start in recapture_starts:
        after_start = next((frame for frame in unused_confirmations if frame > start), None)
        if after_start is not None:
            recapture_delays.append(float(after_start - start))
            unused_confirmations.remove(after_start)
    mode_counts = Counter(str(step.get("mode", "UNKNOWN")) for step in steps)
    first_delay = float("nan")
    if initial_confirmations:
        first_delay = float(int(initial_confirmations[0]["latest_frame"]) - (frames_per_window - 1))
    count = max(len(steps), 1)
    workload = result.get("workload", {})
    return {
        "capture_success": int(bool(initial_confirmations)),
        "first_capture_delay_frames": first_delay,
        "first_capture_delay_s": first_delay * _FRAME_INTERVAL_S,
        "recapture_start_count": len(recapture_starts),
        "recapture_success_count": len(recapture_delays),
        "recapture_success_rate": (
            float(len(recapture_delays) / len(recapture_starts))
            if recapture_starts else float("nan")
        ),
        "recapture_delay_frames_mean": (
            float(np.mean(recapture_delays)) if recapture_delays else float("nan")
        ),
        "recapture_delay_s_mean": (
            float(np.mean(recapture_delays) * _FRAME_INTERVAL_S)
            if recapture_delays
            else float("nan")
        ),
        "capture_step_fraction": float(mode_counts["CAPTURE"] / count),
        "track_step_fraction": float(mode_counts["TRACK"] / count),
        "recapture_step_fraction": float(mode_counts["RECAPTURE"] / count),
        "capture_scan_count": int(workload.get("capture_scans", 0)),
        "local_scan_count": int(workload.get("local_scans", 0)),
    }


def _complexity_summary(complexity: ModelComplexity, timing: Mapping[str, Any]) -> dict[str, int | float]:
    blocks = int(timing["blocks_evaluated"])
    values = complexity.scale(blocks)
    values["estimated_conv_linear_macs_per_step"] = float(
        values["estimated_conv_linear_macs_total"] / max(int(timing["logical_steps"]), 1)
    )
    values["estimated_conv_linear_flops_per_step"] = float(
        values["estimated_conv_linear_flops_total"] / max(int(timing["logical_steps"]), 1)
    )
    return values


def _macro_average(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    keys = (
        "coverage",
        "mae_m",
        "rmse_m",
        "abs_error_p95_m",
        "hit_coverage",
        "hit_mae_m",
        "jump_count",
        "current_mae_m",
        "next_mae_m",
        "avg_blocks_per_step",
        "end_to_end_mean_ms",
        "end_to_end_p95_ms",
        "capture_success",
        "recapture_success_rate",
    )
    summary: dict[str, float | int] = {"source_count": len(rows)}
    for key in keys:
        values = np.asarray([_optional_number(row.get(key)) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if len(finite):
            summary[f"macro_mean_{key}"] = float(np.mean(finite))
    return summary


def _sample_directory(output_dir: Path, source_id: str) -> Path:
    path = output_dir / "samples" / safe_name(source_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _method_directory(sample_dir: Path, method: str, time_stride: int) -> Path:
    path = sample_dir / _sample_file_stem(method, time_stride)
    path.mkdir(exist_ok=True)
    return path


def _write_sample(
    method_dir: Path,
    *,
    method: str,
    result: Mapping[str, Any],
    source: PackedSource,
    summary: Mapping[str, Any],
    timing: Mapping[str, Any],
    complexity: Mapping[str, Any],
    save_figures: bool,
    figure_dpi: int,
    config: Any,
    time_stride: int,
    title: str,
    logger: Any,
) -> str | None:
    """在方法目录写入固定文件名的图、数值、摘要和逐窗日志。"""
    np.savez_compressed(
        method_dir / "prediction.npz",
        prediction_m=np.asarray(result["prediction_m"], dtype=np.float64),
        target_true_range_m=source.target_true_range_m,
        target_hit=source.target_hit,
    )
    write_jsonl(method_dir / "log.jsonl", list(result.get("steps", ())))
    figure_error: str | None = None
    if save_figures:
        try:
            figure = plot_source_diagnostic(
                source,
                result,
                config=config,
                title=title,
                metrics={**summary, **timing},
                frame_interval_s=_FRAME_INTERVAL_S,
            )
            figure.savefig(method_dir / "visualize.png", dpi=figure_dpi)
            import matplotlib.pyplot as plt

            plt.close(figure)
        except Exception as error:  # 诊断图失败不丢弃已完成的推理与数值结果。
            figure_error = f"{type(error).__name__}: {error}"
            logger.exception("样本 %s 的诊断图保存失败。", source.record.source_id)

    write_json(
        method_dir / "metrics.json",
        {
            "source_id": source.record.source_id,
            "method": method,
            "time_stride": time_stride,
            "metrics": summary,
            "timing": timing,
            "complexity": complexity,
            "result": {
                key: value for key, value in result.items()
                if key not in {"prediction_m", "steps"}
            },
            "figure_error": figure_error,
        },
    )
    return figure_error


def run(args: argparse.Namespace) -> Path:
    method_config = _method_config(args)
    if args.jump_threshold_m < 0.0:
        raise ValueError("jump_threshold_m 必须为非负数。")
    bundle = load_inference_bundle(
        args.checkpoint,
        data_root=args.data_root,
        device=args.device,
    )
    checkpoint_step = _checkpoint_step(bundle.checkpoint)
    output_dir = create_output_dir(
        data_root=bundle.config.data_root,
        checkpoint_path=bundle.checkpoint_path,
        checkpoint_step=checkpoint_step,
    )
    logger = configure_logger(f"{args.method}-{bundle.checkpoint_path.stem}")
    set_process_title(
        "infer",
        label=f"{args.method}-{bundle.checkpoint_path.stem}",
        infer_rank=False,
    )

    selected_records, data_root_indices = _select_records(bundle, args)
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
    config_path = output_dir / "infer_config.json"
    existing_config = read_json(config_path)
    if existing_config is not None:
        if (
            str(existing_config.get("checkpoint_path")) != str(bundle.checkpoint_path)
            or str(existing_config.get("data_root")) != str(bundle.config.data_root)
        ):
            raise ValueError(
                "输出目录已被其他数据集或 checkpoint 占用，拒绝混写："
                f"{config_path}"
            )
        infer_config = existing_config
    else:
        infer_config = {
            "schema_version": 2,
            "checkpoint_path": bundle.checkpoint_path,
            "checkpoint_global_step": checkpoint_step,
            "run_dir": bundle.run_dir,
            "resolved_config_path": bundle.resolved_config_path,
            "data_root": bundle.config.data_root,
            "source_selection": "data_root_prefix",
            "effective_config": asdict(bundle.config),
            "methods": {},
        }
    methods = infer_config.setdefault("methods", {})
    if not isinstance(methods, dict):
        raise ValueError(f"已有推理配置格式错误：{config_path}")
    method_file_stem = _sample_file_stem(args.method, args.time_stride)
    method_run: dict[str, Any] = {
        "title": _METHOD_TITLES[args.method],
        "device": str(bundle.device),
        "requested_num_samples": args.num_samples,
        "selected_source_ids": [record.source_id for record in selected_records],
        "arguments": vars(args),
        "method_config": asdict(method_config),
        "frame_interval_s": _FRAME_INTERVAL_S,
        "model_complexity_per_block": complexity.scale(1),
        "output_directory": method_file_stem,
        "files": {
            "visualization": "visualize.png",
            "prediction": "prediction.npz",
            "metrics": "metrics.json",
            "step_log": "log.jsonl",
        },
        "warmup": {"enabled": bool(args.warmup), "elapsed_s": None},
        "status": "running",
    }
    methods[args.method] = method_run
    write_json(config_path, infer_config)

    first_source: PackedSource | None = None
    if args.warmup:
        first_source = PackedSource(selected_records[0], bundle.config)
        warmup_start = perf_counter()
        runner.warmup(
            first_source,
            range_starts_m=standard_distance_starts(bundle.config),
        )
        method_run["warmup"]["elapsed_s"] = perf_counter() - warmup_start
        write_json(config_path, infer_config)

    logger.info(
        "开始 %s：device=%s checkpoint_step=%s data_root=%s sources=%d output=%s",
        _METHOD_TITLES[args.method],
        bundle.device,
        checkpoint_step,
        bundle.config.data_root,
        len(selected_records),
        output_dir,
    )
    rows: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []
    for selected_offset, record in enumerate(selected_records):
        source = first_source if selected_offset == 0 and first_source is not None else PackedSource(record, bundle.config)
        if args.method == "global_top1":
            result = run_global_source(source, runner, bundle.config, method_config)
        else:
            result = run_adaptive_source(source, runner, bundle.config, method_config)
        if not isinstance(result, Mapping):
            raise TypeError("方法 run_source 必须返回字典。")

        metric = _trajectory_summary(
            result,
            source,
            jump_threshold_m=args.jump_threshold_m,
        )
        timing = _timing_summary(result)
        method_metrics: dict[str, float | int] = {}
        if args.method == "adaptive_tracker":
            method_metrics = _adaptive_summary(result)
        complexity_metrics = _complexity_summary(complexity, timing)
        data_root_index = data_root_indices[record.source_id]
        sample_dir = _sample_directory(output_dir, record.source_id)
        row: dict[str, Any] = {
            "source_id": record.source_id,
            "data_root_index": data_root_index,
            "method": args.method,
            **metric,
            **timing,
            **method_metrics,
            **complexity_metrics,
        }
        method_dir = _method_directory(sample_dir, args.method, args.time_stride)
        figure_error = _write_sample(
            method_dir,
            method=args.method,
            result=result,
            source=source,
            summary={**metric, **method_metrics},
            timing=timing,
            complexity=complexity_metrics,
            save_figures=args.save_figures,
            figure_dpi=args.figure_dpi,
            config=bundle.config,
            time_stride=args.time_stride,
            title=_figure_title(args.method, record.source_id, args.time_stride),
            logger=logger,
        )
        if figure_error is not None:
            row["figure_error"] = figure_error
        rows.append(row)
        sample_summaries.append(
            {
                "source_id": record.source_id,
                "data_root_index": data_root_index,
                "directory": method_dir.relative_to(output_dir),
                "result": {
                    key: value for key, value in result.items()
                    if key not in {"prediction_m", "steps"}
                },
                **row,
            }
        )
        logger.info(
            "[%d/%d] source=%s coverage=%.1f%% mae=%s m blocks/step=%.2f p95=%.2f ms",
            selected_offset + 1,
            len(selected_records),
            record.source_id,
            100.0 * float(metric["coverage"]),
            "—" if not math.isfinite(_optional_number(metric["mae_m"])) else f"{float(metric['mae_m']):.2f}",
            float(timing["avg_blocks_per_step"]),
            _optional_number(timing["end_to_end_p95_ms"]),
        )

    method_run.update(
        {
            "status": "complete",
            "source_count": len(rows),
            "aggregate": _macro_average(rows),
            "samples": sample_summaries,
        }
    )
    write_json(config_path, infer_config)
    logger.info("推理完成：%d 个样本，汇总已写入 %s", len(rows), config_path)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("加载 checkpoint 后，从当前 data_root 的前 N 个序列执行 SimpleCNN 单进程流式推理。"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", choices=tuple(_METHOD_DIRS), default="global_top1",
        help="推理算法：全局 Top-1 基线或 postprocess.md 自适应状态机。",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/runs/simplecnn_v2/limit50k-gbs1024-lr5e-4-pos25-vs5-modeln-cfgc5fe62b8/20260727_195332/checkpoints/best.pt"),
        help="训练 run 内 checkpoints/ 下的 .pt 文件，例如 best.pt 或 last.pt。",
    )
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/host-model/weixc/code/LineTracker/data/synthetic/gen/F300_N10000_S42_B-random_T-R10000-290000m-V340-A6-C10-J1-K300-Q0p35-0p95"),
        help="当前机器的数据根目录；可覆盖默认实验数据集。",
    )
    parser.add_argument("--device", default="auto",
        help="auto、cpu、cuda[:index] 或 npu[:index]；单次推理只使用一张卡。",
    )
    parser.add_argument("--time-stride", type=_positive_int, default=5,
        help="TRACK 和全局基线每次用前20帧外推的未来帧数，也是 TRACK 滑窗步进。",
    )
    parser.add_argument("--max-blocks-per-forward", type=_nonnegative_int, default=0,
        help="单次模型前向最多拼多少距离块；0 表示同一步的所有候选一次前向。",
    )
    parser.add_argument("--jump-threshold-m", type=_finite_float, default=1000.0,
        help="离线统计预测相邻帧距离跳变的阈值（米）。",
    )
    parser.add_argument("--figure-dpi", type=_positive_int, default=180,
        help="每个样本 2×2 诊断图的保存分辨率。",
    )
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
        help="是否启用 1/3/5/34 块的模型预热；预热耗时不计入样本时延统计。",
    )
    parser.add_argument("--figures", action=argparse.BooleanOptionalAction, dest="save_figures", default=True,
        help="是否生成每个样本的中文 2×2 诊断图。",
    )

    parser.add_argument("--num-samples", type=_positive_int, default=10,
        help="从 data_root 的一级序列目录按返回顺序取前 N 个 data.npz。",
    )

    adaptive = parser.add_argument_group("adaptive_tracker 参数（仅 --method adaptive_tracker 生效）")
    adaptive.add_argument("--capture-stride", type=_positive_int, default=3,
        help="CAPTURE/RECAPTURE 阶段相邻全局扫描窗口的时间步进（帧）。",
    )
    adaptive.add_argument("--capture-buffer-size", type=_positive_int, default=8,
        help="用于稳定捕获判断的连续全局扫描候选数量。",
    )
    adaptive.add_argument("--capture-support-ratio", type=_finite_float, default=0.7,
        help="捕获成功所需的候选支持比例，取值必须在 (0, 1] 内。",
    )
    adaptive.add_argument("--capture-radius-m", type=_finite_float, default=500.0,
        help="CAPTURE/RECAPTURE 中候选聚类时的距离半径（米）。",
    )
    adaptive.add_argument("--q-keep", type=_finite_float, default=0.5,
        help="保留候选与判定跟踪成功所需的最低预测 q 值。",
    )
    adaptive.add_argument("--position-gates-m", default="1000,2000,4000",
        help="L=0/1/2 的位置连续性门限（米），以逗号分隔三个值。",
    )
    adaptive.add_argument("--expand-after-bad", type=_positive_int, default=2,
        help="连续失败达到该次数后，将局部搜索等级扩大一级。",
    )
    adaptive.add_argument("--shrink-after-good", type=_positive_int, default=4,
        help="连续成功达到该次数后，将局部搜索等级缩小一级。",
    )
    adaptive.add_argument("--alpha", type=_finite_float, default=0.8,
        help="α-β 预测器的位置更新系数 α，取值必须在 [0, 1] 内。",
    )
    adaptive.add_argument("--beta", type=_finite_float, default=0.1,
        help="α-β 预测器的速度更新系数 β，必须非负。",
    )
    adaptive.add_argument("--gamma", type=_finite_float, default=0.0,
        help="运动连续性评分中 q 与残差的融合权重 γ，取值必须在 [0, 1] 内。",
    )
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    args.method = 'adaptive_tracker'

    try:
        run(args)
    except KeyboardInterrupt:
        print("推理已由用户中断。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

