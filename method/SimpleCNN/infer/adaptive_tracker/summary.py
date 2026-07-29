"""汇总已落盘的 ``adaptive_tracker`` 性能指标，兼容 metrics schema v4/v5。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np


_METHOD = "adaptive_tracker"
_SCHEMA_VERSIONS = frozenset({4, 5})
_SOURCE_INDEX = re.compile(r"(\d+)$")
_TRAJECTORY_KEYS = (
    "coverage", "mae_m", "rmse_m", "abs_error_p95_m", "hit_coverage", "hit_mae_m", "unreliable_coverage",
)
_COMPUTE_KEYS = (
    "logical_steps", "blocks_evaluated", "forward_calls", "estimated_conv_linear_macs_total", "estimated_conv_linear_flops_total",
)
_TIMING_KEYS = (
    "end_to_end_total_s", "end_to_end_mean_ms", "end_to_end_p95_ms", "preprocess_total_s", "model_total_s",
)
_TRACKER_KEYS = (
    "capture_success", "first_capture_delay_frames", "recapture_start_count", "recapture_success_count",
    "recapture_delay_frames_mean", "capture_scan_count", "local_scan_count",
)


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    assert parsed >= 0, f"必须为非负整数，实际为 {value!r}。"
    return parsed


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    assert isinstance(value, Mapping), f"{name} 必须为 JSON 对象。"
    return value


def _finite(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _require_number(section: Mapping[str, Any], key: str, *, context: str, nullable: bool = False) -> None:
    if nullable and section.get(key) is None:
        return
    assert _finite(section.get(key)) is not None, f"{context} 缺少有限数值字段 {key!r}。"


def _stats(values: Iterable[object]) -> dict[str, float | int | None]:
    array = np.asarray([number for value in values if (number := _finite(value)) is not None], dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(array.size), "mean": float(np.mean(array)), "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)), "min": float(np.min(array)), "max": float(np.max(array)),
    }


def _integer_stats(values: Iterable[object]) -> dict[str, float | int | None]:
    finite = [number for value in values if (number := _finite(value)) is not None]
    return {**_stats(finite), "total": int(round(sum(finite)))}


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return _mapping(json.load(handle), name=str(path))


def _source_index(source_id: str) -> int:
    match = _SOURCE_INDEX.search(source_id)
    assert match is not None, f"样本目录名必须以数字编号结尾，实际为 {source_id!r}。"
    return int(match.group(1))


def _normalise_metrics(path: Path, *, time_stride: int) -> dict[str, Any]:
    """读取单样本 metrics；将 v4 compute 内耗时统一映射为 timing。"""
    payload = _read_json(path)
    source_id = path.parent.parent.name
    assert payload.get("source_id") == source_id, f"{path} 的 source_id 与目录名不一致。"
    assert payload.get("method") == _METHOD, f"{path} 不是 {_METHOD} 输出。"
    assert int(payload.get("time_stride", -1)) == time_stride, f"{path} 的 time_stride 不匹配。"
    schema = int(payload.get("schema_version", -1))
    assert schema in _SCHEMA_VERSIONS, f"{path} 的 schema v{schema} 不受支持；仅支持 v4/v5。"

    trajectory = _mapping(payload.get("trajectory"), name=f"{path}.trajectory")
    compute = _mapping(payload.get("compute"), name=f"{path}.compute")
    timing = compute if schema == 4 else _mapping(payload.get("timing"), name=f"{path}.timing")
    tracker = _mapping(payload.get("tracker"), name=f"{path}.tracker")
    for key in _TRAJECTORY_KEYS:
        _require_number(trajectory, key, context=str(path))
    _require_number(trajectory, "jump_count", context=str(path))
    for key in _COMPUTE_KEYS:
        _require_number(compute, key, context=str(path))
    for key in _TIMING_KEYS:
        _require_number(timing, key, context=str(path))
    for key in _TRACKER_KEYS:
        _require_number(tracker, key, context=str(path), nullable=key in {"first_capture_delay_frames", "recapture_delay_frames_mean"})
    assert isinstance(timing.get("frame_ms"), Mapping), f"{path} 缺少 frame_ms 对象。"
    return {
        "source_id": source_id, "source_index": _source_index(source_id), "schema_version": schema,
        "trajectory": trajectory, "compute": compute, "timing": timing, "tracker": tracker,
    }


def _step_timing_stats(samples: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    """按 CAPTURE / RECAPTURE / Track-L* 汇总逻辑步端到端耗时。"""
    by_type: defaultdict[str, list[float]] = defaultdict(list)
    for sample in samples:
        frame_ms = _mapping(sample["timing"].get("frame_ms"), name=f"{sample['source_id']}.frame_ms")
        for record in frame_ms.values():
            entry = _mapping(record, name=f"{sample['source_id']}.frame_ms 条目")
            kind, duration = entry.get("type"), _finite(entry.get("ms"))
            assert isinstance(kind, str) and duration is not None, f"{sample['source_id']} 的 frame_ms 条目无效。"
            by_type[kind].append(duration)
    return {
        kind: {**_stats(values), "total_ms": float(sum(values))}
        for kind, values in sorted(by_type.items())
    }


def _missing_indices(indices: list[int]) -> list[int]:
    present = set(indices)
    return [] if not indices else [index for index in range(indices[0], indices[-1] + 1) if index not in present]


def _root_config(output_dir: Path) -> Mapping[str, Any] | None:
    path = output_dir / "_infer_config.json"
    return _read_json(path) if path.exists() else None


def load_samples(
    output_dir: Path,
    *,
    time_stride: int,
    sample_start: int | None = None,
    sample_stop: int | None = None,
) -> list[dict[str, Any]]:
    """加载指定范围的样本，并将 v4/v5 指标统一为相同的内存结构。"""
    assert time_stride >= 1, "time_stride 必须为正整数。"
    assert sample_start is None or sample_stop is None or sample_start <= sample_stop, "sample_start 不能大于 sample_stop。"
    samples_root = output_dir / "samples"
    assert samples_root.is_dir(), f"未找到 samples 目录：{samples_root}。"

    directory_name = f"{_METHOD}_stride{time_stride}"
    samples: list[dict[str, Any]] = []
    for path in sorted(samples_root.glob(f"*/{directory_name}/metrics.json")):
        sample = _normalise_metrics(path, time_stride=time_stride)
        index = int(sample["source_index"])
        if (sample_start is None or index >= sample_start) and (sample_stop is None or index <= sample_stop):
            samples.append(sample)
    assert samples, f"未找到 {directory_name} 的有效 metrics.json。"

    indices = [int(sample["source_index"]) for sample in samples]
    assert len(indices) == len(set(indices)), "同一 source 编号出现多个 metrics.json。"
    return samples


def build_summary(
    output_dir: Path,
    *,
    time_stride: int,
    sample_start: int | None = None,
    sample_stop: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """汇总指定 stride 的已有样本，并在输出根目录原子写入 JSON。"""
    assert time_stride >= 1, "time_stride 必须为正整数。"
    assert sample_start is None or sample_stop is None or sample_start <= sample_stop, "sample_start 不能大于 sample_stop。"
    samples = load_samples(
        output_dir,
        time_stride=time_stride,
        sample_start=sample_start,
        sample_stop=sample_stop,
    )
    indices = sorted(int(sample["source_index"]) for sample in samples)
    schemas = Counter(int(sample["schema_version"]) for sample in samples)

    trajectory = {key: _stats(sample["trajectory"].get(key) for sample in samples) for key in _TRAJECTORY_KEYS}
    jumps = [sample["trajectory"].get("jump_count") for sample in samples]
    trajectory["jump_count"] = {
        **_integer_stats(jumps),
        "samples_with_jump": int(sum((_finite(value) or 0.0) > 0.0 for value in jumps)),
    }
    compute = {key: _integer_stats(sample["compute"].get(key) for sample in samples) for key in _COMPUTE_KEYS}
    timing: dict[str, Any] = {key: _stats(sample["timing"].get(key) for sample in samples) for key in _TIMING_KEYS}
    timing["by_step_type"] = _step_timing_stats(samples)

    capture_success = [sample["tracker"].get("capture_success") for sample in samples]
    recapture_starts = [sample["tracker"].get("recapture_start_count") for sample in samples]
    recapture_successes = [sample["tracker"].get("recapture_success_count") for sample in samples]
    capture_success_count = int(round(sum(_finite(value) or 0.0 for value in capture_success)))
    recapture_start_count = int(round(sum(_finite(value) or 0.0 for value in recapture_starts)))
    recapture_success_count = int(round(sum(_finite(value) or 0.0 for value in recapture_successes)))
    tracker = {
        "initial_capture": {
            "success_count": capture_success_count,
            "success_rate": capture_success_count / len(samples),
            "delay_frames_success_only": _stats(
                sample["tracker"].get("first_capture_delay_frames") for sample in samples
                if (_finite(sample["tracker"].get("capture_success")) or 0.0) > 0.0
            ),
        },
        "recapture": {
            "start_count_total": recapture_start_count,
            "success_count_total": recapture_success_count,
            "success_rate_per_start": None if recapture_start_count == 0 else recapture_success_count / recapture_start_count,
            "samples_with_recapture": int(sum((_finite(value) or 0.0) > 0.0 for value in recapture_starts)),
            "delay_frames_per_sample_mean": _stats(sample["tracker"].get("recapture_delay_frames_mean") for sample in samples),
        },
        "scan_count_per_sample": {
            "capture": _integer_stats(sample["tracker"].get("capture_scan_count") for sample in samples),
            "local": _integer_stats(sample["tracker"].get("local_scan_count") for sample in samples),
        },
    }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": _METHOD,
        "time_stride": time_stride,
        "input": {
            "output_dir": str(output_dir.resolve()),
            "sample_count": len(samples),
            "source_index_range": {"start": indices[0], "stop": indices[-1]},
            "requested_source_index_range": {"start": sample_start, "stop": sample_stop},
            "missing_source_indices": _missing_indices(indices),
            "metrics_schema_versions": {f"v{version}": count for version, count in sorted(schemas.items())},
            "infer_config": _root_config(output_dir),
        },
        "trajectory": trajectory,
        "tracker": tracker,
        "compute": compute,
        "timing": timing,
    }
    destination = output_dir / f"{_METHOD}_stride{time_stride}_samples{indices[0]:04d}-{indices[-1]:04d}.json"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(destination)
    return destination, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="汇总 adaptive_tracker 已落盘样本的推理性能。")
    parser.add_argument("--output-dir", type=Path, help="run_infer.py 生成的 _output 根目录。")
    parser.add_argument("--time-stride", type=_nonnegative_int, help="仅汇总 adaptive_tracker_stride<time_stride>；必须为正整数。")
    parser.add_argument("--sample-start", type=_nonnegative_int, default=None, help="可选：仅统计编号不小于该值的样本。")
    parser.add_argument("--sample-stop", type=_nonnegative_int, default=None, help="可选：仅统计编号不大于该值的样本。")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    args.output_dir = Path('/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/infer/_output/F300-N10k-S42--v2-n-best-s42000')
    args.time_stride = 5
    args.sample_start = 0
    args.sample_stop = 10
        
    assert args.time_stride >= 1, "time_stride 必须为正整数。"
    destination, payload = build_summary(
        args.output_dir, time_stride=args.time_stride, sample_start=args.sample_start, sample_stop=args.sample_stop,
    )
    source_range = payload["input"]["source_index_range"]
    print(
        f"已汇总 {payload['input']['sample_count']} 个样本："
        f"{source_range['start']:04d}-{source_range['stop']:04d} -> {destination}"
    )


if __name__ == "__main__":
    main()
