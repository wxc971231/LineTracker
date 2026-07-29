"""推理阶段的轨迹和耗时统计工具。

本模块只消费已经产生的预测结果和数据集真值；不参与候选块选择，
因此不会把 ``target_hit`` 等标签泄漏到推理流程中。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


_FRAME_KEYS = ("frame", "frame_index", "time_index", "current_frame", "output_frame", "latest_frame")
_PREDICTION_KEYS = (
    "range_current_m",
    "predicted_range_m",
    "prediction_m",
    "forecast_prediction_m",
    "range_m",
)
_MODE_KEYS = ("mode", "state", "tracker_mode")
_MEASUREMENT_KEYS = ("measurement_updated", "used_measurement", "cnn_measurement_updated")


def _one_dimensional(values: Any, name: str, *, dtype: Any | None = None) -> np.ndarray:
    """转换为一维数组，并在输入形状不明确时尽早报错。"""
    array = np.asarray(values if values is not None else [], dtype=dtype)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} 必须是一维数组，实际形状为 {array.shape}。")
    return array


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _optional_series(values: Any | None, length: int, name: str) -> np.ndarray | None:
    """将可选标量或一维数组对齐到候选预测数量。"""
    if values is None:
        return None
    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        return np.full(length, array.item(), dtype=object)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} 应为长度 {length} 的一维数组或标量。")
    return array



def _overlay_step_statuses(
    mode: np.ndarray,
    measurement_updated: np.ndarray,
    steps: Any,
    frame_count: int,
) -> None:
    """将运行结果中的逐步状态扩展到其实际 forecast 帧区间。"""
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        return
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        value_mode = _first_present(step, _MODE_KEYS)
        value_update = _first_present(step, _MEASUREMENT_KEYS)
        if value_mode is None and value_update is None:
            continue
        start = step.get("forecast_frame_start")
        stop = step.get("forecast_frame_stop")
        if start is not None and stop is not None:
            start = max(0, int(start))
            stop = min(frame_count, int(stop))
            indices = range(start, max(start, stop))
        else:
            frame = _first_present(step, _FRAME_KEYS)
            if frame is None:
                continue
            index = int(frame)
            indices = (index,) if 0 <= index < frame_count else ()
        for index in indices:
            if value_mode is not None:
                mode[index] = value_mode
            if value_update is not None:
                measurement_updated[index] = value_update


def prediction_series(
    prediction: np.ndarray | Sequence[float] | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    frame_count: int,
    *,
    value_key: str | None = None,
) -> dict[str, np.ndarray]:
    """把数组、字典或逐步记录归一化为按完整帧轴对齐的预测序列。

    参数
    ----
    prediction:
        可直接传入长度为 ``frame_count`` 的预测距离数组；也可传入包含
        ``range_current_m``（或 ``predicted_range_m``）与可选 ``frame``、
        ``mode``、``measurement_updated`` 的字典；还可传入这些逐步字典组成的
        列表。没有输出的帧以 ``NaN`` 表示。
    frame_count:
        完整源序列的帧数。
    value_key:
        若记录采用自定义距离字段名，可显式指定该键。

    返回
    ----
    dict
        包含 ``range_m``（float64）、``mode``（object）和
        ``measurement_updated``（object）三个长度为 ``frame_count`` 的数组。

    说明
    ----
    同一帧有多条记录时按输入顺序保留最后一条，便于直接消费 append 式日志。
    本函数只做结果整理，不读取也不使用真实标签。
    """
    if isinstance(frame_count, (bool, np.bool_)) or int(frame_count) < 1:
        raise ValueError("frame_count 必须为正整数。")
    frame_count = int(frame_count)

    values: np.ndarray
    frames: np.ndarray | None
    modes: np.ndarray | None
    updates: np.ndarray | None

    if isinstance(prediction, Mapping):
        keys = (value_key,) if value_key is not None else _PREDICTION_KEYS
        raw_values = _first_present(prediction, keys)
        if raw_values is None:
            if "steps" in prediction:
                return prediction_series(prediction["steps"], frame_count, value_key=value_key)
            raise KeyError(f"预测字典缺少距离字段，支持：{list(keys)}。")
        values = _one_dimensional(raw_values, "预测距离", dtype=np.float64)
        raw_frames = _first_present(prediction, _FRAME_KEYS)
        frames = None if raw_frames is None else _one_dimensional(raw_frames, "帧索引")
        modes = _optional_series(_first_present(prediction, _MODE_KEYS), len(values), "mode")
        updates = _optional_series(
            _first_present(prediction, _MEASUREMENT_KEYS), len(values), "measurement_updated"
        )
    elif isinstance(prediction, Sequence) and not isinstance(prediction, (str, bytes, np.ndarray)):
        if not prediction:
            values = np.empty(0, dtype=np.float64)
            frames = np.empty(0, dtype=np.int64)
            modes = None
            updates = None
        elif all(isinstance(item, Mapping) for item in prediction):
            steps = list(prediction)
            keys = (value_key,) if value_key is not None else _PREDICTION_KEYS
            values_list: list[float] = []
            frames_list: list[Any] = []
            modes_list: list[Any] = []
            updates_list: list[Any] = []
            has_mode = False
            has_update = False
            has_frame = False
            for index, step in enumerate(steps):
                raw_value = _first_present(step, keys)
                values_list.append(np.nan if raw_value is None else float(raw_value))
                raw_frame = _first_present(step, _FRAME_KEYS)
                has_frame |= raw_frame is not None
                frames_list.append(raw_frame if raw_frame is not None else index)
                raw_mode = _first_present(step, _MODE_KEYS)
                raw_update = _first_present(step, _MEASUREMENT_KEYS)
                has_mode |= raw_mode is not None
                has_update |= raw_update is not None
                modes_list.append(raw_mode)
                updates_list.append(raw_update)
            if not has_frame and len(steps) != frame_count:
                raise ValueError(
                    "逐步预测记录未包含帧索引，且记录数与完整序列帧数不一致。"
                )
            values = np.asarray(values_list, dtype=np.float64)
            frames = np.asarray(frames_list)
            modes = np.asarray(modes_list, dtype=object) if has_mode else None
            updates = np.asarray(updates_list, dtype=object) if has_update else None
        else:
            values = _one_dimensional(prediction, "预测距离", dtype=np.float64)
            frames = None
            modes = None
            updates = None
    else:
        values = _one_dimensional(prediction, "预测距离", dtype=np.float64)
        frames = None
        modes = None
        updates = None

    if frames is None:
        if len(values) != frame_count:
            raise ValueError(
                "未提供帧索引时，预测距离长度必须等于完整序列帧数："
                f"{len(values)} != {frame_count}。"
            )
        frame_indices = np.arange(frame_count, dtype=np.int64)
    else:
        if len(frames) != len(values):
            raise ValueError("帧索引长度必须与预测距离长度一致。")
        try:
            numeric_frames = np.asarray(frames, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("帧索引必须为有限整数。") from error
        if not np.all(np.isfinite(numeric_frames)) or not np.all(numeric_frames == np.floor(numeric_frames)):
            raise ValueError("帧索引必须为有限整数。")
        frame_indices = numeric_frames.astype(np.int64)
        if np.any(frame_indices < 0) or np.any(frame_indices >= frame_count):
            raise IndexError(f"帧索引必须位于 [0, {frame_count})。")

    range_m = np.full(frame_count, np.nan, dtype=np.float64)
    mode = np.full(frame_count, None, dtype=object)
    measurement_updated = np.full(frame_count, None, dtype=object)
    # Python 按顺序赋值会自然保留重复 frame 的最后一条记录。
    for offset, frame in enumerate(frame_indices):
        range_m[frame] = values[offset]
        if modes is not None:
            mode[frame] = modes[offset]
        if updates is not None:
            measurement_updated[frame] = updates[offset]
    if isinstance(prediction, Mapping) and "steps" in prediction:
        _overlay_step_statuses(mode, measurement_updated, prediction["steps"], frame_count)
    return {
        "range_m": range_m,
        "mode": mode,
        "measurement_updated": measurement_updated,
    }


def _quantile_name(value: float) -> str:
    percent = int(round(float(value) * 100.0))
    return f"p{percent}"


def _error_summary(errors_m: np.ndarray, quantiles: Sequence[float]) -> dict[str, float]:
    """生成一组只依赖有限误差值的标量统计。"""
    if not len(errors_m):
        result = {"mae_m": float("nan"), "rmse_m": float("nan"), "max_abs_error_m": float("nan")}
        result.update({f"abs_error_{_quantile_name(value)}_m": float("nan") for value in quantiles})
        return result
    absolute = np.abs(errors_m)
    result = {
        "mae_m": float(np.mean(absolute)),
        "rmse_m": float(np.sqrt(np.mean(np.square(errors_m)))),
        "max_abs_error_m": float(np.max(absolute)),
    }
    result.update(
        {
            f"abs_error_{_quantile_name(value)}_m": float(np.quantile(absolute, value))
            for value in quantiles
        }
    )
    return result


def trajectory_metrics(
    prediction_m: np.ndarray | Sequence[float],
    true_range_m: np.ndarray | Sequence[float],
    *,
    prediction_frames: np.ndarray | Sequence[int] | None = None,
    valid_mask: np.ndarray | Sequence[bool] | None = None,
    target_hit: np.ndarray | Sequence[bool] | None = None,
    jump_threshold_m: float = 1_000.0,
    quantiles: Sequence[float] = (0.50, 0.90, 0.95, 0.99),
) -> dict[str, float | int]:
    """计算流式轨迹的覆盖率、误差、命中帧和跳变指标。

    ``prediction_m`` 可以是与真值同长度的数组，未输出帧写为 ``NaN``；也可
    配合 ``prediction_frames`` 传入稀疏预测。``target_hit`` 仅用于最终离线
    评估“实际目标响应帧”子集，绝不会反馈到推理候选选择。

    ``jump_count`` 统计连续两帧都有预测时，预测位置的绝对变化超过
    ``jump_threshold_m`` 的次数；不把跨越未覆盖时间段的两点误判为跳变。
    """
    true_values = _one_dimensional(true_range_m, "真实轨迹", dtype=np.float64)
    if len(true_values) < 1:
        raise ValueError("真实轨迹不能为空。")
    if not np.isfinite(jump_threshold_m) or jump_threshold_m < 0.0:
        raise ValueError("jump_threshold_m 必须为非负有限数。")
    checked_quantiles = tuple(float(value) for value in quantiles)
    if any(not 0.0 <= value <= 1.0 for value in checked_quantiles):
        raise ValueError("quantiles 中的值必须位于 [0, 1]。")

    raw_prediction = _one_dimensional(prediction_m, "预测轨迹", dtype=np.float64)
    if prediction_frames is None:
        if len(raw_prediction) != len(true_values):
            raise ValueError("预测轨迹与真实轨迹长度不一致；稀疏预测请提供 prediction_frames。")
        aligned_prediction = raw_prediction
    else:
        frames = _one_dimensional(prediction_frames, "预测帧索引")
        if len(frames) != len(raw_prediction):
            raise ValueError("prediction_frames 长度必须与预测轨迹一致。")
        if not np.all(np.isfinite(np.asarray(frames, dtype=np.float64))):
            raise ValueError("预测帧索引必须为有限整数。")
        frames_as_float = np.asarray(frames, dtype=np.float64)
        if not np.all(frames_as_float == np.floor(frames_as_float)):
            raise ValueError("预测帧索引必须为整数。")
        indices = frames_as_float.astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= len(true_values)):
            raise IndexError("预测帧索引超出真实轨迹范围。")
        aligned_prediction = np.full(len(true_values), np.nan, dtype=np.float64)
        aligned_prediction[indices] = raw_prediction

    truth_valid = np.isfinite(true_values)
    if valid_mask is not None:
        mask = _one_dimensional(valid_mask, "有效帧掩码", dtype=bool)
        if len(mask) != len(true_values):
            raise ValueError("valid_mask 长度必须与真实轨迹一致。")
        truth_valid &= mask
    covered = truth_valid & np.isfinite(aligned_prediction)
    errors = aligned_prediction[covered] - true_values[covered]

    result: dict[str, float | int] = {
        "frame_count": int(len(true_values)),
        "valid_truth_frames": int(truth_valid.sum()),
        "covered_frames": int(covered.sum()),
        "coverage": float(covered.sum() / max(int(truth_valid.sum()), 1)),
    }
    result.update(_error_summary(errors, checked_quantiles))

    adjacent_covered = covered[:-1] & covered[1:]
    if np.any(adjacent_covered):
        displacement = np.abs(np.diff(aligned_prediction)[adjacent_covered])
        result["jump_pair_count"] = int(len(displacement))
        result["jump_count"] = int(np.sum(displacement > jump_threshold_m))
        result["jump_rate"] = float(result["jump_count"] / len(displacement))
    else:
        result["jump_pair_count"] = 0
        result["jump_count"] = 0
        result["jump_rate"] = 0.0
    result["jump_threshold_m"] = float(jump_threshold_m)

    if target_hit is not None:
        hit_mask = _one_dimensional(target_hit, "target_hit", dtype=bool)
        if len(hit_mask) != len(true_values):
            raise ValueError("target_hit 长度必须与真实轨迹一致。")
        hit_valid = truth_valid & hit_mask
        hit_covered = covered & hit_mask
        hit_errors = aligned_prediction[hit_covered] - true_values[hit_covered]
        hit_summary = _error_summary(hit_errors, checked_quantiles)
        result.update(
            {
                "hit_frames": int(hit_valid.sum()),
                "hit_covered_frames": int(hit_covered.sum()),
                "hit_coverage": float(hit_covered.sum() / max(int(hit_valid.sum()), 1)),
                "hit_mae_m": hit_summary["mae_m"],
                "hit_rmse_m": hit_summary["rmse_m"],
                "hit_max_abs_error_m": hit_summary["max_abs_error_m"],
            }
        )
        for value in checked_quantiles:
            suffix = _quantile_name(value)
            result[f"hit_abs_error_{suffix}_m"] = hit_summary[f"abs_error_{suffix}_m"]
    return result


def trajectory_metrics_from_prediction(
    prediction: np.ndarray | Sequence[float] | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    true_range_m: np.ndarray | Sequence[float],
    **kwargs: Any,
) -> dict[str, float | int]:
    """对普通预测字典或逐步日志计算 :func:`trajectory_metrics`。

    这是一层便利包装：先用 :func:`prediction_series` 对齐，再评估。其余关键字
    参数与 :func:`trajectory_metrics` 一致。
    """
    truth = _one_dimensional(true_range_m, "真实轨迹", dtype=np.float64)
    series = prediction_series(prediction, len(truth))
    return trajectory_metrics(series["range_m"], truth, **kwargs)


def timing_metrics(durations_s: np.ndarray | Sequence[float]) -> dict[str, float | int]:
    """汇总一组端到端或单阶段耗时（秒）的 mean/P50/P95/P99。

    ``NaN`` 会被忽略，负值被视为无效输入并报错。这使冷启动、模型前向和整体
    端到端耗时可以分别使用同一函数汇总。
    """
    durations = _one_dimensional(durations_s, "耗时", dtype=np.float64)
    finite = durations[np.isfinite(durations)]
    if np.any(finite < 0.0):
        raise ValueError("耗时不得为负。")
    if not len(finite):
        return {
            "count": 0,
            "total_s": 0.0,
            "mean_s": float("nan"),
            "min_s": float("nan"),
            "max_s": float("nan"),
            "p50_s": float("nan"),
            "p95_s": float("nan"),
            "p99_s": float("nan"),
        }
    return {
        "count": int(len(finite)),
        "total_s": float(np.sum(finite)),
        "mean_s": float(np.mean(finite)),
        "min_s": float(np.min(finite)),
        "max_s": float(np.max(finite)),
        "p50_s": float(np.quantile(finite, 0.50)),
        "p95_s": float(np.quantile(finite, 0.95)),
        "p99_s": float(np.quantile(finite, 0.99)),
    }


def timing_metrics_from_steps(
    steps: Sequence[Mapping[str, Any]],
    *,
    key: str = "end_to_end_s",
) -> dict[str, float | int]:
    """从逐步 JSON 风格记录中提取一个耗时字段并调用 :func:`timing_metrics`。"""
    values = [step.get(key, np.nan) for step in steps]
    return timing_metrics(values)


# 便于调用方按“计算”语义命名，保持与主 API 完全等价。
compute_trajectory_metrics = trajectory_metrics
compute_timing_metrics = timing_metrics


__all__ = [
    "compute_timing_metrics",
    "compute_trajectory_metrics",
    "prediction_series",
    "timing_metrics",
    "timing_metrics_from_steps",
    "trajectory_metrics",
    "trajectory_metrics_from_prediction",
]
