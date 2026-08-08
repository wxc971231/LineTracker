"""汇总 schema v5 的 SimpleCNN 推理结果

输入布局固定为：
``<run-dir>/samples/<source-id>/<method-dir>/metrics.json``
本脚本仅接受 ``schema_version == 5``，避免把旧版 v3 的计时字段与稳定版
adaptive_tracker 的逐逻辑步计时混合统计
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, cast

SCHEMA_VERSION = 5
_SOURCE_INDEX = re.compile(r"(\d+)$")
_METHOD_DIRECTORY = re.compile(r"(?P<method>.+)_stride(?P<stride>\d+)$")

_TRAJECTORY_FIELDS = (
    "coverage",
    "unreliable_coverage",
    "mae_m",
    "rmse_m",
    "hit_coverage",
    "hit_mae_m",
    "jump_count",
)
_COMPUTE_FIELDS = (
    "logical_steps",
    "blocks_evaluated",
    "forward_calls",
    "estimated_conv_linear_macs_total",
    "estimated_conv_linear_flops_total",
)
_TIMING_FIELDS = (
    "end_to_end_total_s",
    "preprocess_total_s",
    "model_total_s",
    "end_to_end_mean_ms",
    "end_to_end_p95_ms",
)
_TRACKER_FIELDS = (
    "capture_success",
    "first_capture_delay_frames",
    "recapture_start_count",
    "recapture_success_count",
    "recapture_delay_frames_mean",
    "capture_scan_count",
    "local_scan_count",
)

_TRAJECTORY_EXPLANATIONS = {
    "coverage": "输出预测距离的帧占比",
    "unreliable_coverage": "RECAPTURE 阶段不可靠外推帧占比",
    "mae_m": "coverage部分绝对距离误差的平均值（m）",
    "rmse_m": "coverage部分距离误差均方根（m）",
    "hit_coverage": "仅在实际目标响应帧上，输出有限预测距离的帧占比",
    "hit_mae_m": "仅在实际目标响应帧上，绝对距离误差的平均值（m）",
    "jump_count": "相邻输出预测距离出现超过设定阈值跳变的次数",
}
_COMPUTE_EXPLANATIONS = {
    "logical_steps": "状态机实际处理的时间窗／逻辑步数量",
    "blocks_evaluated": "送入 CNN 评估的全部时间—距离块数量",
    "forward_calls": "考虑 batch 拆分后的模型实际前向调用次数",
    "estimated_conv_linear_macs_total": "卷积层与线性层的静态估计 MAC 总数",
    "estimated_conv_linear_flops_total": "按 1 MAC = 2 FLOPs 换算的静态估计 FLOPs 总数",
}
_TIMING_EXPLANATIONS = {
    "end_to_end_total_s": "完整样本从输入裁剪到结果输出的端到端累计耗时（s）",
    "preprocess_total_s": "样本的局部解包、重排与设备传输累计耗时（s）",
    "model_total_s": "样本的 CNN 前向累计耗时（s）",
    "end_to_end_mean_ms": "样本每个逻辑步的平均端到端耗时（ms）",
    "end_to_end_p95_ms": "样本各逻辑步端到端耗时的 95% 分位（ms）",
}

_COMMON_PARAMETER_ROWS = (
    ("method_dir", "方法配置目录", "输出根目录下与本次方法对应的参数快照名称"),
    ("method", "推理方法", "候选搜索与后处理方法"),
    ("time_stride", "TRACK 时间步进（帧）", "每次输入窗向未来外推的帧数，也是 TRACK 阶段滑窗步进"),
    ("checkpoint_global_step", "Checkpoint step", "该 checkpoint 对应的优化器更新步数"),
    # ("requested_device", "请求设备", "命令行指定的设备选择"),
    ("effective_device", "实际设备", "运行时解析后实际执行模型前向的设备"),
    ("max_blocks_per_forward", "单次前向最大块数", "0 表示同一逻辑步的全部候选块一次前向；正数时会拆分 batch"),
    ("jump_threshold_m", "跳变阈值（m）", "离线统计 jump_count 时的相邻帧距离跳变阈值"),
    ("warmup", "预热", "是否在正式样本计时前执行模型预热"),
    # ("figure_dpi", "诊断图 DPI", "单样本诊断图保存分辨率"),
    # ("save_figures", "保存诊断图", "是否为每个样本生成 visualize.png"),
)
_ADAPTIVE_PARAMETER_ROWS = (
    ("capture_stride", "CAPTURE/RECAPTURE 步进（帧）", "相邻全局扫描窗口的时间步进"),
    ("capture_buffer_size", "捕获缓存长度", "用于稳定捕获判断的连续全局候选数量"),
    ("capture_support_ratio", "捕获支持比例", "确认捕获所需的一致候选比例"),
    ("capture_radius_m", "捕获聚类半径（m）", "判定全局候选在同一距离区域的半径"),
    ("q_keep", "候选 q 门限", "保留候选并判定跟踪成功所需的最低预测 q"),
    ("instant_speed_gate_mpf", "瞬时速度门限（m/frame）", "L0/L1/L2 三个搜索等级的融合状态绝对速度上限"),
    ("average_speed_gate_mpf", "平均速度门限（m/frame）", "L0/L1/L2 三个搜索等级的历史平均速度上限"),
    ("speed_average_window_frames", "平均速度历史长度（帧）", "计算平均速度门控时使用的历史状态窗口"),
    ("expand_after_bad", "扩大搜索阈值", "连续失败达到该次数后，局部搜索等级扩大一级"),
    ("shrink_after_good", "收缩搜索阈值", "连续成功达到该次数后，局部搜索等级缩小一级"),
    ("alpha", "α", "α-β 预测器的位置更新系数"),
    ("beta", "β", "α-β 预测器的速度更新系数"),
    ("gamma", "γ", "候选速度与 α-β 状态速度的融合权重"),
)


def _nonnegative_int(value: str) -> int:
    """解析命令行中的非负样本编号"""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须为非负整数") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("必须为非负整数")
    return parsed


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    """校验 JSON 节点为对象，并为后续字段读取提供明确类型"""
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} 必须为 JSON 对象")
    return value


def _finite_number(value: object, *, context: str, nullable: bool = False) -> float | None:
    """读取有限数值；仅显式允许的 nullable 字段可为 null"""
    if value is None:
        if nullable:
            return None
        raise ValueError(f"{context} 不得为 null")
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} 必须为数值，实际为 {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{context} 必须为有限数，实际为 {value!r}")
    return number


def _source_index(source_id: str) -> int:
    """从 ``dataset1_synthetic_0000`` 一类目录名读取末尾编号"""
    match = _SOURCE_INDEX.search(source_id)
    if match is None:
        raise ValueError(f"样本目录名必须以数字编号结尾，实际为 {source_id!r}")
    return int(match.group(1))


def _parse_method_directory(method_dir: str) -> tuple[str, int]:
    """将 ``adaptive_tracker_stride5`` 校验并解析为方法名和时间步进"""
    match = _METHOD_DIRECTORY.fullmatch(method_dir)
    if match is None:
        raise ValueError(
            "--method-dir 必须采用 <method>_stride<正整数> 形式，例如 adaptive_tracker_stride5"
        )
    stride = int(match.group("stride"))
    if stride < 1:
        raise ValueError("method-dir 中的 stride 必须为正整数")
    return match.group("method"), stride


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """按 NumPy 默认线性插值定义计算分位数，输入必须已排序且非空"""
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    """计算跨样本或跨逻辑步的描述统计；p95 使用线性插值定义"""
    items = sorted(float(value) for value in values)
    if not items:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(items),
        "mean": sum(items) / len(items),
        "median": _percentile(items, 0.5),
        "p95": _percentile(items, 0.95),
        "min": items[0],
        "max": items[-1],
    }


def _integer_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    """计算计数型指标的分布，并提供全部样本的整数总量"""
    items = list(values)
    return {**_stats(items), "total": int(round(sum(items)))}


def _read_json(path: Path) -> Mapping[str, Any]:
    """读取单个 JSON 文件，并在格式不正确时说明具体文件"""
    try:
        with path.open(encoding="utf-8") as handle:
            return _mapping(json.load(handle), context=str(path))
    except json.JSONDecodeError as error:
        raise ValueError(f"无法解析 JSON：{path}") from error


def _load_method_parameter_snapshot(run_dir: Path, method_dir: str) -> tuple[Path, Mapping[str, Any] | None]:
    """读取 run_infer 写入的同名方法参数快照；旧结果缺失时保留明确空值。"""
    path = run_dir / f"{method_dir}.json"
    if not path.is_file():
        return path, None
    payload = _read_json(path)
    expected_method, expected_stride = _parse_method_directory(method_dir)
    if payload.get("method_dir") != method_dir:
        raise ValueError(f"{path} 的 method_dir 与文件名不一致")
    if payload.get("method") != expected_method or payload.get("time_stride") != expected_stride:
        raise ValueError(f"{path} 的 method/time_stride 与文件名不一致")
    return path, payload


def _normalise_metrics(
    path: Path,
    *,
    source_id: str,
    expected_method: str,
    expected_stride: int,
) -> dict[str, Any]:
    """严格读取一个 schema v5 metrics.json，转成统一的扁平样本记录"""
    payload = _read_json(path)
    schema = payload.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"{path} 的 schema_version={schema!r}；本脚本仅支持 schema v{SCHEMA_VERSION}"
        )
    if payload.get("source_id") != source_id:
        raise ValueError(f"{path} 的 source_id 与父目录不一致")
    if payload.get("method") != expected_method:
        raise ValueError(
            f"{path} 的 method={payload.get('method')!r}，与目录配置 {expected_method!r} 不一致"
        )
    if payload.get("time_stride") != expected_stride:
        raise ValueError(
            f"{path} 的 time_stride={payload.get('time_stride')!r}，与目录配置 {expected_stride} 不一致"
        )

    trajectory = _mapping(payload.get("trajectory"), context=f"{path}.trajectory")
    compute = _mapping(payload.get("compute"), context=f"{path}.compute")
    timing = _mapping(payload.get("timing"), context=f"{path}.timing")
    tracker = _mapping(payload.get("tracker"), context=f"{path}.tracker")

    record: dict[str, Any] = {
        "source_id": source_id,
        "source_index": _source_index(source_id),
        "schema_version": SCHEMA_VERSION,
        "method": expected_method,
        "time_stride": expected_stride,
    }
    for field in _TRAJECTORY_FIELDS:
        record[field] = _finite_number(trajectory.get(field), context=f"{path}.trajectory.{field}")
    for field in _COMPUTE_FIELDS:
        record[field] = _finite_number(compute.get(field), context=f"{path}.compute.{field}")
    for field in _TIMING_FIELDS:
        record[field] = _finite_number(timing.get(field), context=f"{path}.timing.{field}")
    for field in _TRACKER_FIELDS:
        record[field] = _finite_number(
            tracker.get(field),
            context=f"{path}.tracker.{field}",
            nullable=field in {"first_capture_delay_frames", "recapture_delay_frames_mean"},
        )

    frame_ms = _mapping(timing.get("frame_ms"), context=f"{path}.timing.frame_ms")
    step_timings: list[dict[str, float | str]] = []
    for frame, entry in frame_ms.items():
        item = _mapping(entry, context=f"{path}.timing.frame_ms[{frame!r}]")
        step_type = item.get("type")
        if not isinstance(step_type, str) or not step_type:
            raise ValueError(f"{path}.timing.frame_ms[{frame!r}].type 必须为非空字符串")
        duration = _finite_number(item.get("ms"), context=f"{path}.timing.frame_ms[{frame!r}].ms")
        assert duration is not None
        step_timings.append({"type": step_type, "ms": duration})
    record["frame_step_count"] = len(step_timings)
    record["_step_timings"] = step_timings  # 仅用于汇总，不写入 pe r_sample.csv
    return record


def load_samples(
    run_dir: Path,
    method_dir: str,
    *,
    sample_start: int | None,
    sample_stop: int | None,
) -> tuple[list[dict[str, Any]], list[int], Path]:
    """读取指定范围内已有的 schema v5 样本，同时返回缺失 metrics 的编号"""
    if sample_start is not None and sample_stop is not None and sample_start > sample_stop:
        raise ValueError("--sample-start 不得大于 --sample-stop")
    expected_method, expected_stride = _parse_method_directory(method_dir)
    sample_root = run_dir / "samples"
    if not sample_root.is_dir():
        raise FileNotFoundError(f"未找到 samples 根目录：{sample_root}")

    source_dirs: list[tuple[int, Path]] = []
    for path in sample_root.iterdir():
        if not path.is_dir() or path.name.startswith("_"):
            continue
        index = _source_index(path.name)
        if (sample_start is None or index >= sample_start) and (sample_stop is None or index <= sample_stop):
            source_dirs.append((index, path))
    source_dirs.sort(key=lambda item: item[0])
    if not source_dirs:
        raise FileNotFoundError("指定编号范围内没有样本目录")
    if len({index for index, _ in source_dirs}) != len(source_dirs):
        raise ValueError("样本目录末尾编号重复，无法建立一一对应的统计记录")

    samples: list[dict[str, Any]] = []
    missing_indices: list[int] = []
    for index, source_dir in source_dirs:
        metrics_path = source_dir / method_dir / "metrics.json"
        if not metrics_path.is_file():
            missing_indices.append(index)
            continue
        samples.append(
            _normalise_metrics(
                metrics_path,
                source_id=source_dir.name,
                expected_method=expected_method,
                expected_stride=expected_stride,
            )
        )
    if not samples:
        raise FileNotFoundError(f"指定范围内没有 {method_dir!r} 的 metrics.json")
    return samples, missing_indices, sample_root


def _step_timing_summary(samples: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    """将所有样本的 v5 ``frame_ms`` 合并，并按状态机步类型统计"""
    by_type: defaultdict[str, list[float]] = defaultdict(list)
    for sample in samples:
        for entry in cast(list[Mapping[str, Any]], sample["_step_timings"]):
            by_type[str(entry["type"])].append(float(entry["ms"]))
    return {
        step_type: {**_stats(values), "total_ms": float(sum(values))}
        for step_type, values in sorted(by_type.items())
    }


def build_summary(
    run_dir: Path,
    method_dir: str,
    *,
    sample_start: int | None,
    sample_stop: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[int]]:
    """按轨迹、计算量、耗时和状态机四类指标构建机器可读汇总"""
    samples, missing_indices, sample_root = load_samples(
        run_dir,
        method_dir,
        sample_start=sample_start,
        sample_stop=sample_stop,
    )
    expected_method, expected_stride = _parse_method_directory(method_dir)
    indices = [int(sample["source_index"]) for sample in samples]
    logical_steps_total = sum(float(sample["logical_steps"]) for sample in samples)
    end_to_end_total_s = sum(float(sample["end_to_end_total_s"]) for sample in samples)
    capture_successes = sum(float(sample["capture_success"]) for sample in samples)
    recapture_starts = sum(float(sample["recapture_start_count"]) for sample in samples)
    recapture_successes = sum(float(sample["recapture_success_count"]) for sample in samples)

    trajectory = {field: _stats(float(sample[field]) for sample in samples) for field in _TRAJECTORY_FIELDS}
    compute = {field: _integer_stats(float(sample[field]) for sample in samples) for field in _COMPUTE_FIELDS}
    timing: dict[str, Any] = {
        field: _stats(float(sample[field]) for sample in samples)
        for field in _TIMING_FIELDS
    }
    timing["weighted_end_to_end_mean_ms"] = (
        None if logical_steps_total == 0.0 else 1_000.0 * end_to_end_total_s / logical_steps_total
    )
    timing["by_step_type"] = _step_timing_summary(samples)
    tracker = {
        "initial_capture": {
            "success_count": int(round(capture_successes)),
            "success_rate": capture_successes / len(samples),
            "delay_frames_success_only": _stats(
                float(sample["first_capture_delay_frames"])
                for sample in samples
                if float(sample["capture_success"]) > 0.0
                and sample["first_capture_delay_frames"] is not None
            ),
        },
        "recapture": {
            "start_count_total": int(round(recapture_starts)),
            "success_count_total": int(round(recapture_successes)),
            "success_rate_per_start": None if recapture_starts == 0.0 else recapture_successes / recapture_starts,
            "samples_with_recapture": int(
                sum(float(sample["recapture_start_count"]) > 0.0 for sample in samples)
            ),
            "delay_frames_per_sample_mean": _stats(
                float(sample["recapture_delay_frames_mean"])
                for sample in samples
                if sample["recapture_delay_frames_mean"] is not None
            ),
        },
        "scan_count_per_sample": {
            "capture": _integer_stats(float(sample["capture_scan_count"]) for sample in samples),
            "local": _integer_stats(float(sample["local_scan_count"]) for sample in samples),
        },
    }
    infer_config_path = run_dir / "_infer_config.json"
    infer_config = _read_json(infer_config_path) if infer_config_path.is_file() else None
    parameter_snapshot_path, parameter_snapshot = _load_method_parameter_snapshot(run_dir, method_dir)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "run_dir": str(run_dir.resolve()),
            "sample_dir": str(sample_root.resolve()),
            "method_dir": method_dir,
            "method": expected_method,
            "time_stride": expected_stride,
            "requested_source_index_range": {"start": sample_start, "stop": sample_stop},
            "sample_count": len(samples),
            "source_index_range": {"start": min(indices), "stop": max(indices)},
            "included_source_indices": indices,
            "missing_metrics_source_indices": missing_indices,
            "metrics_schema_version": SCHEMA_VERSION,
            "infer_config": infer_config,
            "parameter_snapshot_path": str(parameter_snapshot_path.resolve()),
            "parameter_snapshot": parameter_snapshot,
        },
        "trajectory": trajectory,
        "compute": compute,
        "timing": timing,
        "tracker": tracker,
    }
    return summary, samples, missing_indices


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """原子写入 UTF-8 JSON，避免中断时留下半成品汇总"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """原子写入每样本扁平统计表"""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "source_index", "source_id", "schema_version", "method", "time_stride",
        *_TRAJECTORY_FIELDS, *_COMPUTE_FIELDS, *_TIMING_FIELDS, *_TRACKER_FIELDS,
        "frame_step_count",
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # csv 的类型存根将 fieldnames 推断为 Literal 联合；运行时允许普通 str 键字典
            writer.writerow(cast(Any, row))
        temporary = Path(handle.name)
    temporary.replace(path)


def _format_number(value: object, digits: int = 4) -> str:
    """将统计字段安全格式化为 Markdown 单元格"""
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{float(cast(Any, value)):.{digits}g}"


def _format_parameter_value(value: object) -> str:
    """将 JSON 参数值渲染为紧凑、可读且适合 Markdown 表格的文本。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={_format_parameter_value(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_parameter_value(item) for item in value)
    if isinstance(value, float):
        return _format_number(value)
    return str(value)


def _project_relative_path(value: object) -> str:
    """将路径缩短为相对 LineTracker 项目根目录的形式，便于跨机器阅读报告。"""
    path = Path(str(value))
    try:
        root_index = path.parts.index("LineTracker")
    except ValueError:
        # 非本项目路径没有可靠的公共锚点，保留调用方提供的形式以免误导。
        return str(path)
    relative_parts = path.parts[root_index + 1:]
    return str(Path(*relative_parts)) if relative_parts else "."


def _format_requested_sample_range(value: object) -> str:
    """按 summary_v5 的 --sample-start/--sample-stop 配置展示编号范围。"""
    requested = _mapping(value, context="summary.input.requested_source_index_range")
    start = requested.get("start")
    stop = requested.get("stop")
    if start is None and stop is None:
        return "全部样本（未设置范围限制）"
    if start is None:
        return f"编号不大于 {stop}"
    if stop is None:
        return f"编号不小于 {start}"
    return f"{start}–{stop}（含两端）"


def _parameter_table(input_info: Mapping[str, Any]) -> list[str]:
    """从同名方法参数快照生成报告开头的推理超参数表。"""
    snapshot_path = _project_relative_path(input_info["parameter_snapshot_path"])
    snapshot_value = input_info.get("parameter_snapshot")
    if snapshot_value is None:
        return [
            "## 推理超参数",
            "",
            f"> 未找到参数快照：`{snapshot_path}`。该历史结果未记录 run_infer.py 的实际参数，无法可靠补写。",
        ]
    snapshot = _mapping(snapshot_value, context="summary.input.parameter_snapshot")
    common = _mapping(snapshot.get("common"), context="parameter_snapshot.common")
    adaptive = snapshot.get("adaptive_tracker")
    rows: list[tuple[str, object, str]] = []
    for key, label, description in _COMMON_PARAMETER_ROWS:
        if key in common:
            value = common[key]
        else:
            value = snapshot.get(key)
        rows.append((label, value, description))
    if adaptive is not None:
        adaptive_mapping = _mapping(adaptive, context="parameter_snapshot.adaptive_tracker")
        for key, label, description in _ADAPTIVE_PARAMETER_ROWS:
            rows.append((label, adaptive_mapping.get(key), description))

    lines = [
        "## 推理超参数",
        "",
        f"参数快照：`{snapshot_path}`",
        "",
        f"Checkpoint：`{_project_relative_path(snapshot.get('checkpoint_path'))}`",
        "",
        f"数据根目录：`{_project_relative_path(snapshot.get('data_root'))}`",
        "",
        "| 参数 | 取值 | 含义 |",
        "|---|---|---|",
    ]
    for label, value, description in rows:
        lines.append(f"| {label} | {_format_parameter_value(value)} | {description} |")
    return lines


def _metric_table(
    title: str,
    metrics: Mapping[str, Mapping[str, Any]],
    explanations: Mapping[str, str],
) -> list[str]:
    """生成一张通用均值/中位数/p95/min/max Markdown 表"""
    lines = [
        f"## {title}",
        "",
        "| 指标 | 指标解释 | 有效数 | 均值 | 中位数 | p95 | 最小 | 最大 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stat in metrics.items():
        lines.append(
            "| {name} | {description} | {count} | {mean} | {median} | {p95} | {min_value} | {max_value} |".format(
                name=name,
                description=explanations.get(name, "—"),
                count=_format_number(stat["count"], 0),
                mean=_format_number(stat["mean"]),
                median=_format_number(stat["median"]),
                p95=_format_number(stat["p95"]),
                min_value=_format_number(stat["min"]),
                max_value=_format_number(stat["max"]),
            )
        )
    return lines


def _build_report(summary: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]) -> str:
    """生成不依赖额外库的中文 Markdown 性能报告"""
    input_info = _mapping(summary["input"], context="summary.input")
    timing = _mapping(summary["timing"], context="summary.timing")
    tracker = _mapping(summary["tracker"], context="summary.tracker")
    initial_capture = _mapping(tracker["initial_capture"], context="summary.tracker.initial_capture")
    recapture = _mapping(tracker["recapture"], context="summary.tracker.recapture")
    lines = [
        f"# {input_info['method_dir']} 性能汇总",
        "",
        f"- 生成时间（UTC）：{summary['generated_at_utc']}",
        f"- 样本编号范围（summary_v5 配置）：{_format_requested_sample_range(input_info['requested_source_index_range'])}",
        f"- 实际纳入样本：{input_info['sample_count']} 个",
        f"- 方法：`{input_info['method']}`；时间步进：{input_info['time_stride']} 帧",
        f"- metrics schema：v{input_info['metrics_schema_version']}",
    ]
    missing = input_info["missing_metrics_source_indices"]
    if missing:
        lines.extend([f"- 缺少选定方法 metrics 的样本编号：{', '.join(map(str, missing))}"])
    lines.extend([""])
    lines.extend(_parameter_table(input_info))
    lines.extend([""])
    lines.extend(
        _metric_table(
            "轨迹质量（逐样本统计）",
            _mapping(summary["trajectory"], context="summary.trajectory"),
            _TRAJECTORY_EXPLANATIONS,
        )
    )
    lines.extend([""])
    lines.extend(
        _metric_table(
            "计算量（逐样本统计）",
            _mapping(summary["compute"], context="summary.compute"),
            _COMPUTE_EXPLANATIONS,
        )
    )
    lines.extend([""])
    timing_scalars = {field: timing[field] for field in _TIMING_FIELDS}
    lines.extend(
        _metric_table(
            "端到端耗时（逐样本统计，ms 字段按逻辑步）",
            timing_scalars,
            _TIMING_EXPLANATIONS,
        )
    )
    lines.extend([
        "",
        f"全部逻辑步加权平均端到端耗时：{_format_number(timing['weighted_end_to_end_mean_ms'])} ms",
        "",
        "## 状态机",
        "",
        f"- 初始捕获：{initial_capture['success_count']} 次成功，成功率 {_format_number(initial_capture['success_rate'])}",
        f"- 成功样本首次捕获延迟：中位数 {_format_number(_mapping(initial_capture['delay_frames_success_only'], context='capture delay')['median'])} 帧",
        f"- 重捕获：共触发 {recapture['start_count_total']} 次，成功 {recapture['success_count_total']} 次，按触发计成功率 {_format_number(recapture['success_rate_per_start'])}",
        "",
        "## 逐逻辑步端到端耗时（所有样本 pooled）",
        "",
        "| 步类型 | 步数 | 均值 ms | 中位数 ms | p95 ms | 最大 ms | 总耗时 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    by_step_type = _mapping(timing["by_step_type"], context="summary.timing.by_step_type")
    for step_type, stat_value in by_step_type.items():
        stat = _mapping(stat_value, context=f"step type {step_type}")
        lines.append(
            f"| {step_type} | {_format_number(stat['count'], 0)} | {_format_number(stat['mean'])} | "
            f"{_format_number(stat['median'])} | {_format_number(stat['p95'])} | {_format_number(stat['max'])} | "
            f"{_format_number(stat['total_ms'])} |"
        )

    worst_mae = sorted(samples, key=lambda row: float(row["mae_m"]), reverse=True)[:5]
    lines.extend(["", "## MAE 最大的 5 个样本", "", "| 样本 | MAE (m) | 覆盖率 | 端到端 p95 (ms) |", "|---|---:|---:|---:|"])
    for sample in worst_mae:
        lines.append(
            f"| {sample['source_id']} | {_format_number(sample['mae_m'])} | "
            f"{_format_number(sample['coverage'])} | {_format_number(sample['end_to_end_p95_ms'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _atomic_text(path: Path, content: str) -> None:
    """原子写入 UTF-8 文本"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    """定义 schema v5 汇总器的命令行参数"""
    parser = argparse.ArgumentParser(description="汇总 schema v5 SimpleCNN 推理输出")
    parser.add_argument("--run-dir", type=Path, default=None, help="推理 run 根目录，例如 .../F300-N10k-S42--v2-n-best-s42000")
    parser.add_argument("--method-dir", default=None, help="样本子目录名，例如 adaptive_tracker_stride5")
    parser.add_argument("--sample-start", type=_nonnegative_int, default=None, help="可选：仅纳入编号不小于该值的样本")
    parser.add_argument("--sample-stop", type=_nonnegative_int, default=None, help="可选：仅纳入编号不大于该值的样本")
    parser.add_argument("--output-dir", type=Path, default=None, help="可选：汇总输出目录；默认写入 <run-dir>/summary/")
    return parser


def main() -> None:
    """执行汇总，并将 JSON、CSV 和 Markdown 报告写入目标目录"""
    args = build_parser().parse_args()

    args.run_dir = Path('/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/infer/_output/F300-N10k-S42--v2-n-best-s42000')
    args.method_dir = 'adaptive_tracker_stride5'

    summary, samples, missing_indices = build_summary(
        args.run_dir,
        args.method_dir,
        sample_start=args.sample_start,
        sample_stop=args.sample_stop,
    )
    indices = [int(sample["source_index"]) for sample in samples]
    default_name = f"{args.method_dir}_samples{min(indices):04d}-{max(indices):04d}"
    output_dir = args.output_dir or args.run_dir / "summary" / default_name
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_csv(output_dir / "per_sample.csv", samples)
    _atomic_text(output_dir / "report.md", _build_report(summary, samples))
    print(f"已汇总 {len(samples)} 个 schema v5 样本 -> {output_dir}")
    if missing_indices:
        print(f"警告：缺少 {args.method_dir} metrics 的样本编号：{missing_indices}")


if __name__ == "__main__":
    main()