"""SimpleCNN 统一流式推理入口。

同一入口固定复用 checkpoint 所属训练 run 的 resolved_config.json，并从运行时
data_root 的一级序列目录取前 N 个样本。可切换全局 34 块 Top-1 基线和
postprocess.md 定义的自适应后处理。推理候选选择只依赖模型输出与观测；真值
仅在运行结束后用于指标与诊断图。
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import logging
import math
import re
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from configs.base import SimpleCNNConfig
from data.dataloader import PackedSource, SourceRecord, discover_sources, standard_distance_starts
from infer.adaptive_tracker.infer import AdaptiveInferenceConfig, run_source as run_adaptive_source
from infer.common.complexity import ModelComplexity, estimate_model_complexity
from infer.common.metrics import timing_metrics_from_steps, trajectory_metrics
from infer.common.model_loader import InferenceBundle, load_inference_bundle
from infer.common.output import (
    configure_logger,
    create_output_dir,
    safe_name,
    write_json,
    write_jsonl,
)
from infer.common.plotting import plot_source_diagnostic
from infer.common.runner import ModelRunner
from infer.global_top1.infer import GlobalTop1Config, run_source as run_global_source
from utils.process_title import set_process_title


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
    assert values, f"{name} 不能为空。"
    try:
        return tuple(converter(part) for part in values)
    except (TypeError, ValueError) as error:
        raise AssertionError(f"{name} 应为逗号分隔的合法值：{raw!r}。") from error


def _optional_number(value) -> float:
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


def _select_records(bundle: InferenceBundle, args: argparse.Namespace) -> list[SourceRecord]:
    """稳定选取 data_root 按 source_id 排序后的前 N 个一级样本目录。"""
    records = sorted(discover_sources(bundle.config.data_root), key=lambda record: record.source_id)
    sample_start, sample_stop = _effective_sample_bounds(args)
    return records[sample_start:sample_stop]


def _sample_file_stem(method: str, time_stride: int) -> str:
    return f"{safe_name(method)}_stride{time_stride}"


def _effective_sample_bounds(args: argparse.Namespace) -> tuple[int, int | None]:
    """兼容现有启动入口，返回本次实际用于切片的样本编号范围。"""
    sample_start = getattr(args, "sample_start", None)
    sample_stop = getattr(args, "sample_stop", None)
    # 旧启动入口仍会写入复数形式字段；仅在 CLI 参数未设置时才使用它们。
    if sample_start is None:
        sample_start = getattr(args, "samples_start", 0)
    if sample_stop is None:
        sample_stop = getattr(args, "samples_stop", None)
    return int(sample_start), None if sample_stop is None else int(sample_stop)


def _figure_title(method: str, source_id: str, time_stride: int) -> str:
    suffix = re.search(r"(\d+)$", source_id)
    data_label = f"data{suffix.group(1)}" if suffix else source_id
    return f"{method.replace('_', '-')}-{data_label}：时间步进 {time_stride}"


def _method_config(args: argparse.Namespace) -> GlobalTop1Config | AdaptiveInferenceConfig:
    if args.method == "global_top1":
        return GlobalTop1Config(time_stride=args.time_stride)
    elif args.method == "adaptive_tracker":
        instant_speed_gates = tuple(_csv_values(args.instant_speed_gates_mpf, float, "instant_speed_gates_mpf"))
        average_speed_gates = tuple(_csv_values(args.average_speed_gates_mpf, float, "average_speed_gates_mpf"))
        assert len(instant_speed_gates) == len(average_speed_gates) == 3, "两类 speed gates 都必须恰好提供 L=0/1/2 三个门限。"
        return AdaptiveInferenceConfig(
            time_stride=args.time_stride,
            capture_stride=args.capture_stride,
            capture_buffer_size=args.capture_buffer_size,
            capture_support_ratio=args.capture_support_ratio,
            capture_radius_m=args.capture_radius_m,
            q_keep=args.q_keep,
            instant_speed_gate_mpf=instant_speed_gates,
            average_speed_gate_mpf=average_speed_gates,
            speed_average_window_frames=args.speed_average_window_frames,
            expand_after_bad=args.expand_after_bad,
            shrink_after_good=args.shrink_after_good,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
        )
    else:
        raise ValueError(f"不支持的推理方法：{args.method!r}。")


def _write_method_parameter_snapshot(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    bundle: InferenceBundle,
    checkpoint_step: int | str | None,
    method_config: GlobalTop1Config | AdaptiveInferenceConfig,
    complexity: ModelComplexity,
) -> Path:
    """按方法目录名保存本次实际推理参数，供离线汇总报告复现使用。"""
    method_dir = _sample_file_stem(args.method, args.time_stride)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "method_dir": method_dir,
        "method": args.method,
        "time_stride": args.time_stride,
        "checkpoint_path": bundle.checkpoint_path,
        "checkpoint_global_step": checkpoint_step,
        "resolved_config_path": bundle.resolved_config_path,
        "data_root": bundle.config.data_root,
        "requested_device": args.device,
        "effective_device": str(bundle.device),
        "common": {
            "max_blocks_per_forward": args.max_blocks_per_forward,
            "jump_threshold_m": args.jump_threshold_m,
            "warmup": args.warmup,
            "figure_dpi": args.figure_dpi,
            "save_figures": args.save_figures,
        },
        "model_complexity_per_block": _static_model_complexity(complexity),
    }
    if isinstance(method_config, AdaptiveInferenceConfig):
        payload["adaptive_tracker"] = {
            "capture_stride": method_config.capture_stride,
            "capture_buffer_size": method_config.capture_buffer_size,
            "capture_support_ratio": method_config.capture_support_ratio,
            "capture_radius_m": method_config.capture_radius_m,
            "q_keep": method_config.q_keep,
            "instant_speed_gate_mpf": method_config.instant_speed_gate_mpf,
            "average_speed_gate_mpf": method_config.average_speed_gate_mpf,
            "speed_average_window_frames": method_config.speed_average_window_frames,
            "expand_after_bad": method_config.expand_after_bad,
            "shrink_after_good": method_config.shrink_after_good,
            "alpha": method_config.alpha,
            "beta": method_config.beta,
            "gamma": method_config.gamma,
        }
    else:
        payload["global_top1"] = {"time_stride": method_config.time_stride}
    destination = output_dir / f"{method_dir}.json"
    write_json(destination, payload)
    return destination

def _trajectory_summary(
    result: Mapping[str, Any],
    source: PackedSource,
    *,
    jump_threshold_m: float,
) -> dict[str, float | int]:
    """计算最终拼接轨迹的离线质量指标。"""
    prediction = np.asarray(result["prediction_m"], dtype=np.float64)
    unreliable = np.asarray(result.get("unreliable_prediction_mask", np.zeros(prediction.shape, dtype=bool)), dtype=bool,)
    assert unreliable.shape == prediction.shape, "unreliable_prediction_mask 必须与 prediction_m 同形状。"
    summary = trajectory_metrics(
        prediction,
        source.target_true_range_m,
        target_hit=source.target_hit,
        jump_threshold_m=jump_threshold_m,
    )
    # 覆盖率仍表示全部输出；该字段单独量化其中仅由 RECAPTURE 外推补出的部分。
    summary["unreliable_coverage"] = float(np.mean(unreliable & np.isfinite(prediction)))
    return summary


def _workload_summary(result: Mapping[str, Any]) -> dict[str, int]:
    """提取随样本和推理方法变化的实际计算量。"""
    workload = result.get("workload", {})
    assert isinstance(workload, Mapping), "推理结果 workload 必须为字典。"
    required_keys = ("logical_steps", "blocks_evaluated", "forward_calls")
    missing = [key for key in required_keys if key not in workload]
    assert not missing, f"推理结果 workload 缺少字段：{', '.join(missing)}。"
    return {key: int(workload[key]) for key in required_keys}


def _timing_summary(method: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """汇总样本时延及每个逻辑步的端到端耗时。"""
    steps = [step for step in result.get("steps", ()) if isinstance(step, Mapping)]
    workload = result.get("workload", {})
    assert isinstance(workload, Mapping), "推理结果 workload 必须为字典。"
    required_keys = ("end_to_end_s", "preprocess_s", "model_s")
    missing = [key for key in required_keys if key not in workload]
    assert not missing, f"推理结果 workload 缺少字段：{', '.join(missing)}。"

    end_to_end = timing_metrics_from_steps(steps)
    return {
        "end_to_end_total_s": _optional_number(workload["end_to_end_s"]),
        "end_to_end_mean_ms": 1000.0 * _optional_number(end_to_end["mean_s"]),
        "end_to_end_p95_ms": 1000.0 * _optional_number(end_to_end["p95_s"]),
        "preprocess_total_s": _optional_number(workload["preprocess_s"]),
        "model_total_s": _optional_number(workload["model_s"]),
        "frame_ms": {
            str(int(record["frame"])): {
                "type": str(record["type"]),
                "ms": _optional_number(record["ms"]),
            }
            for record in _frame_timing_records(method, result)
        },
    }


def _frame_timing_type(method: str, step: Mapping[str, Any]) -> str:
    """给每个逻辑预测步一个可比较的计算类型。"""
    if method == "global_top1":
        return "GLOBAL"
    mode_before = str(step.get("mode_before", "")).upper()
    if mode_before == "TRACK":
        level = step.get("scan_level")
        return "Track" if level is None else f"Track-L{int(level)}"
    if mode_before == "RECAPTURE":
        return "RECAPTURE"
    return "CAPTURE"


def _frame_timing_records(method: str, result: Mapping[str, Any]) -> list[dict[str, float | int | str]]:
    """保留每个逻辑预测步的端到端耗时，供定位不同状态下的开销。"""
    records: list[dict[str, float | int | str]] = []
    for step in result.get("steps", ()):
        if not isinstance(step, Mapping) or "frame" not in step:
            continue
        duration_s = _optional_number(step.get("end_to_end_s"))
        records.append(
            {
                "frame": int(step["frame"]),
                "type": _frame_timing_type(method, step),
                "ms": 1_000.0 * duration_s,
            }
        )
    return records


def _adaptive_summary(
    result: Mapping[str, Any],
    *,
    frames_per_window: int,
) -> dict[str, float | int]:
    """汇总自适应状态机中真正影响行为的捕获与搜索统计。"""
    steps = [step for step in result.get("steps", ()) if isinstance(step, Mapping)]
    initial_confirmations = [
        step
        for step in steps
        if bool(step.get("capture", {}).get("confirmed")) and step.get("mode_before") == "CAPTURE"
    ]
    recapture_starts = [
        int(step["frame"])
        for step in steps
        if step.get("mode_before") == "TRACK" and step.get("mode") == "RECAPTURE"
    ]
    recapture_confirmations = [
        int(step["frame"])
        for step in steps
        if bool(step.get("capture", {}).get("confirmed")) and step.get("mode_before") == "RECAPTURE"
    ]
    recapture_delays: list[float] = []
    unused_confirmations = list(recapture_confirmations)
    for start_frame in recapture_starts:
        after_start = next((frame for frame in unused_confirmations if frame > start_frame), None)
        if after_start is not None:
            recapture_delays.append(float(after_start - start_frame))
            unused_confirmations.remove(after_start)
    first_delay = float("nan")
    if initial_confirmations:
        first_delay = float(int(initial_confirmations[0]["frame"]) - (frames_per_window - 1))
    workload = result.get("workload", {})
    return {
        "capture_success": int(bool(initial_confirmations)),
        "first_capture_delay_frames": first_delay,
        "recapture_start_count": len(recapture_starts),
        "recapture_success_count": len(recapture_delays),
        "recapture_delay_frames_mean": (
            float(np.mean(recapture_delays)) if recapture_delays else float("nan")
        ),
        "capture_scan_count": int(workload.get("capture_scans", 0)),
        "local_scan_count": int(workload.get("local_scans", 0)),
    }


def _static_model_complexity(complexity: ModelComplexity) -> dict[str, int]:
    """根级配置中只保存一次的单块模型静态复杂度。"""
    return {
        "parameter_count": complexity.parameter_count,
        "estimated_conv_linear_macs_per_block": complexity.conv_linear_macs_per_block,
        "estimated_conv_linear_flops_per_block": complexity.conv_linear_flops_per_block,
    }


def _sample_compute_summary(
    complexity: ModelComplexity,
    workload: Mapping[str, int],
) -> dict[str, int]:
    """保存实际扫描量和由其换算的总 MACs/FLOPs。"""
    blocks = int(workload["blocks_evaluated"])
    return {
        **workload,
        "estimated_conv_linear_macs_total": complexity.conv_linear_macs_per_block * blocks,
        "estimated_conv_linear_flops_total": complexity.conv_linear_flops_per_block * blocks,
    }


def _compact_trajectory_summary(summary: Mapping[str, Any]) -> dict[str, float | int]:
    """保留比较方法效果所需的主轨迹指标。"""
    keys = (
        "coverage",
        "mae_m",
        "rmse_m",
        "abs_error_p95_m",
        "hit_coverage",
        "hit_mae_m",
        "jump_count",
        "unreliable_coverage",
    )
    return {key: summary[key] for key in keys}


def _compact_step_log(method: str, steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """移除仅供内存诊断图使用的预测窗边界和逐步计时。"""
    if method == "global_top1":
        keys = (
            "frame",                    # 当前时间窗的最新帧号。
            "candidate_q",              # 全局 Top-1 距离块的目标置信度。
            "candidate_block_start_m",  # Top-1 距离块在全局距离轴上的起点（米）。
            "candidate_range_m",        # Top-1 直线外推到最新帧后的全局距离（米）。
            "candidate_speed_mpf",      # Top-1 预测直线斜率，单位米/帧。
        )
        return [{key: step[key] for key in keys} for step in steps]

    internal_keys = {"forecast_frame_start", "forecast_frame_stop", "end_to_end_s"}
    return [{key: value for key, value in step.items() if key not in internal_keys} for step in steps]


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
    trajectory: Mapping[str, Any],
    compute: Mapping[str, Any],
    timing: Mapping[str, Any],
    tracker: Mapping[str, Any],
    save_figures: bool,
    figure_dpi: int,
    config: SimpleCNNConfig,
    time_stride: int,
    title: str,
    logger: logging.Logger,
) -> None:
    """写入精简的逐窗决策日志、样本摘要和可选诊断图。"""
    steps = [step for step in result.get("steps", ()) if isinstance(step, Mapping)]
    write_jsonl(method_dir / "log.jsonl", _compact_step_log(method, steps))
    figure_error: str | None = None
    if save_figures:
        try:
            figure = plot_source_diagnostic(
                source,
                result,
                config=config,
                title=title,
                metrics={**trajectory, **timing, **tracker},
                frame_interval_s=_FRAME_INTERVAL_S,
            )
            figure.savefig(method_dir / "visualize.png", dpi=figure_dpi)
            import matplotlib.pyplot as plt

            plt.close(figure)
        except Exception as error:  # 诊断图失败不丢弃已完成的推理与数值结果。
            figure_error = f"{type(error).__name__}: {error}"
            logger.exception("样本 %s 的诊断图保存失败。", source.record.source_id)

    payload: dict[str, Any] = {
        "schema_version": 5,
        "source_id": source.record.source_id,
        "method": method,
        "time_stride": time_stride,
        "trajectory": trajectory,
        "compute": compute,
        "timing": timing,
    }
    if tracker:
        payload["tracker"] = tracker
    if figure_error is not None:
        payload["figure_error"] = figure_error
    write_json(method_dir / "metrics.json", payload)


def run(args: argparse.Namespace) -> Path:
    """执行一个方法在前 N 个数据源上的单进程推理，并增量写入统一输出目录。"""
    assert args.method in _METHOD_TITLES, f"不支持的推理方法：{args.method!r}。"
    assert args.jump_threshold_m >= 0.0, "jump_threshold_m 必须为非负数。"
    method_config = _method_config(args)

    # 加载 checkpoint 所属 run 恢复结构配置与权重。
    bundle = load_inference_bundle(
        args.checkpoint,
        data_root=args.data_root,
        device=args.device,
    )
    set_process_title("infer", label=f"{args.method}-{bundle.checkpoint_path.stem}", infer_rank=False)
    tracker_config = None
    if isinstance(method_config, AdaptiveInferenceConfig):
        tracker_config = method_config.validate(bundle.config)  # 配置校验和构造只执行一次。
    else:
        method_config.validate()

    # 只计算一次模型静态复杂度，每个样本再按实际扫描块数换算总 MACs/FLOPs。
    complexity = estimate_model_complexity(
        bundle.model,
        input_channels=bundle.config.input_channels,
        input_height=bundle.config.frames_per_window,
        input_width=bundle.config.block_width_m // bundle.config.input_channels,
    )

    # 保存推理方法无关的数据/模型配置
    checkpoint_step = _checkpoint_step(bundle.checkpoint)
    output_dir = create_output_dir(
        data_root=bundle.config.data_root,
        checkpoint_path=bundle.checkpoint_path,
        checkpoint_step=checkpoint_step,
    )
    config_path = output_dir / "_infer_config.json"
    if not config_path.exists():
        write_json(
            config_path,
            {
                "schema_version": 5,
                "checkpoint_path": bundle.checkpoint_path,
                "checkpoint_global_step": checkpoint_step,
                "resolved_config_path": bundle.resolved_config_path,
                "data_root": bundle.config.data_root,
                "model_complexity_per_block": _static_model_complexity(complexity),
            },
        )
    parameter_path = _write_method_parameter_snapshot(
        output_dir,
        args=args,
        bundle=bundle,
        checkpoint_step=checkpoint_step,
        method_config=method_config,
        complexity=complexity,
    )

    # 加载评估数据和推理执行器
    selected_records = _select_records(bundle, args)
    runner = ModelRunner(
        bundle.model,
        bundle.config,
        bundle.device,
        max_blocks_per_forward=args.max_blocks_per_forward,
    )

    # 预热：让 NPU/CUDA 完成算子编译、内存分配和缓存初始化，不计推理时间
    first_source: PackedSource | None = None
    if args.warmup:
        first_source = PackedSource(selected_records[0], bundle.config)
        runner.warmup(first_source, range_starts_m=standard_distance_starts(bundle.config))

    # 启动逐样本评估
    logger = configure_logger(f"{args.method}-{bundle.checkpoint_path.stem}")
    logger.info("开始 %s：device=%s，参数快照=%s", _METHOD_TITLES[args.method], bundle.device, parameter_path)
    sample_start, _ = _effective_sample_bounds(args)
    for selected_offset, record in enumerate(selected_records):
        # 仅复用预热时的第一个 PackedSource；其余样本按需加载，控制峰值内存
        source = first_source if selected_offset == 0 and first_source is not None else PackedSource(record, bundle.config)

        # 完成样本推理。两种方法共享模型、数据和输出协议，仅在流式候选选择/状态机实现上分叉。
        if args.method == "global_top1":
            assert isinstance(method_config, GlobalTop1Config) and tracker_config is not None
            result = run_global_source(source, runner, bundle.config, method_config)
        else:
            assert isinstance(method_config, AdaptiveInferenceConfig) and tracker_config is not None
            result = run_adaptive_source(source, runner, bundle.config, method_config, tracker_config)
        assert isinstance(result, Mapping), "方法 run_source 必须返回字典。"

        # 统计推理指标
        trajectory = _compact_trajectory_summary(               # 预测轨迹质量指标
            _trajectory_summary(result, source, jump_threshold_m=args.jump_threshold_m)
        )
        workload = _workload_summary(result)                    # 实际扫描块数与前向次数
        timing = _timing_summary(args.method, result)           # 样本级时延指标
        compute = _sample_compute_summary(complexity, workload) # 计算量指标
        tracker: dict[str, float | int] = {}                    # 推理运行log
        if args.method == "adaptive_tracker":
            tracker = _adaptive_summary(result, frames_per_window=bundle.config.frames_per_window)

        # 每个样本独立落盘，包含精简日志、指标、预测数组和可选诊断图；
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
        logger.info(
            "[%d/%d] source=%s coverage=%.1f%% mae=%s m blocks=%d p95=%.2f ms",
            selected_offset + 1 + sample_start,
            len(selected_records) + sample_start,
            record.source_id,
            100.0 * float(trajectory["coverage"]),
            "—" if not math.isfinite(_optional_number(trajectory["mae_m"])) else f"{float(trajectory['mae_m']):.2f}",
            int(compute["blocks_evaluated"]),
            _optional_number(timing["end_to_end_p95_ms"]),
        )

    logger.info("推理完成：%d 个样本，结果已写入 %s", len(selected_records), output_dir)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="加载 checkpoint 后，从当前 data_root 的前 N 个序列执行 SimpleCNN 单进程流式推理。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        choices=tuple(_METHOD_TITLES),
        default="global_top1",
        help="推理方法。",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/runs/simplecnn_v2/limit50k-gbs1024-lr5e-4-pos25-vs5-modeln-cfgc5fe62b8/20260727_195332/checkpoints/best.pt"),
        help="要加载的 best.pt 或 last.pt。",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/host-model/weixc/code/LineTracker/data/synthetic/gen/F300_N10000_S42_B-random_T-R10000-290000m-V340-A6-C10-J1-K300-Q0p35-0p95"),
        help="当前机器上的一级样本目录根路径。",
    )
    # parser.add_argument(
    #     "--num-samples",
    #     type=_positive_int,
    #     default=10,
    #     help="按 source_id 排序后取 data_root 前 N 个样本。",
    # )
    parser.add_argument(
        "--sample-start", 
        type=_nonnegative_int, 
        default=None, 
        help="评估样本索引范围起点。"
    )
    parser.add_argument(
        "--sample-stop", 
        type=_nonnegative_int, 
        default=None, 
        help="评估样本索引范围终点。"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto、cpu、cuda[:index] 或 npu[:index]；单次推理只使用一张卡。",
    )
    parser.add_argument(
        "--time-stride",
        type=_positive_int,
        default=5,
        help="每个输入窗向未来外推的帧数，也是 TRACK 滑窗步进。",
    )
    parser.add_argument(
        "--max-blocks-per-forward",
        type=_nonnegative_int,
        default=0,
        help="单次模型前向最多拼多少距离块；0 表示同一步的所有候选一次前向。",
    )
    parser.add_argument(
        "--jump-threshold-m",
        type=_finite_float,
        default=1000.0,
        help="离线统计相邻预测帧距离跳变的阈值（米）。",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="预热 1/3/5/34 个距离块；预热耗时不计入样本时延。",
    )
    parser.add_argument(
        "--figure-dpi",
        type=_positive_int,
        default=180,
        help="每个样本 2×2 诊断图的保存分辨率。",
    )
    parser.add_argument(
        "--figures",
        action=argparse.BooleanOptionalAction,
        dest="save_figures",
        default=True,
        help="是否生成每个样本的中文 2×2 诊断图。",
    )

    adaptive = parser.add_argument_group("adaptive_tracker 参数（仅 --method adaptive_tracker 生效）")
    adaptive.add_argument(
        "--capture-stride",
        type=_positive_int,
        default=2,
        help="CAPTURE/RECAPTURE 相邻全局扫描窗口的时间步进（帧）。",
    )
    adaptive.add_argument(
        "--capture-buffer-size",
        type=_positive_int,
        default=8,
        help="稳定捕获判断使用的连续全局扫描候选数量。",
    )
    adaptive.add_argument(
        "--capture-support-ratio",
        type=_finite_float,
        default=0.7,
        help="捕获成功所需的候选支持比例，取值必须在 (0, 1] 内。",
    )
    adaptive.add_argument(
        "--capture-radius-m",
        type=_finite_float,
        default=500.0,
        help="CAPTURE/RECAPTURE 候选聚类半径（米）。",
    )
    adaptive.add_argument(
        "--q-keep",
        type=_finite_float,
        default=0.5,
        help="保留候选并判定跟踪成功所需的最低预测 q 值。",
    )
    adaptive.add_argument(
        "--instant-speed-gates-mpf",
        default="20,25,30",
        help="L=0/1/2 的单帧融合状态绝对速度上限（米/帧）。",
    )
    adaptive.add_argument(
        "--average-speed-gates-mpf",
        default="17,25,34",
        help="L=0/1/2 的最近 N 帧状态位移平均速度上限（米/帧）。",
    )
    adaptive.add_argument(
        "--speed-average-window-frames",
        type=_positive_int,
        default=10,
        help="平均速度门控的历史窗口长度（帧）。",
    )
    adaptive.add_argument(
        "--expand-after-bad",
        type=_positive_int,
        default=2,
        help="连续失败达到该次数后，将局部搜索等级扩大一级。",
    )
    adaptive.add_argument(
        "--shrink-after-good",
        type=_positive_int,
        default=4,
        help="连续成功达到该次数后，将局部搜索等级缩小一级。",
    )
    adaptive.add_argument(
        "--alpha",
        type=_finite_float,
        default=0.8,
        help="α-β 预测器的位置更新系数 α，取值必须在 [0, 1] 内。",
    )
    adaptive.add_argument(
        "--beta",
        type=_finite_float,
        default=0.1,
        help="α-β 预测器的速度更新系数 β，必须非负。",
    )
    adaptive.add_argument(
        "--gamma",
        type=_finite_float,
        default=0.0,
        help="候选速度与 α-β 速度融合权重 γ，取值必须在 [0, 1] 内。",
    )
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    args.checkpoint = Path('/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/runs/simplecnn_v2/limit50k-gbs1024-lr5e-4-pos25-vs5-models-cfg11ba6304/20260728_113116/checkpoints/best.pt')
    args.method = 'adaptive_tracker'
    args.time_stride = 5
    args.samples_start = 0
    args.samples_stop = 5000

    try:
        run(args)
    except KeyboardInterrupt:
        print("推理已由用户中断。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
