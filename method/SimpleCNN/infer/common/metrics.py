"""推理阶段的轨迹和耗时统计工具"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np


def _one_dimensional(values: Any, name: str, *, dtype: Any | None = None) -> np.ndarray:
    """转换为一维数组，并在输入形状不明确时尽早报错。"""
    array = np.asarray(values if values is not None else [], dtype=dtype)
    if array.ndim == 0:
        array = array.reshape(1)
    assert array.ndim == 1, f"{name} 必须是一维数组，实际形状为 {array.shape}。"
    return array

def _quantile_name(value: float) -> str:
    percent = int(round(float(value) * 100.0))
    return f"p{percent}"

def _error_summary(errors_m: np.ndarray) -> dict[str, float]:
    """生成一组只依赖有限误差值的标量统计。"""
    if not len(errors_m):
        result = {"mae_m": float("nan"), "rmse_m": float("nan"), "max_abs_error_m": float("nan")}
        result.update({f"abs_error_{_quantile_name(value)}_m": float("nan") for value in (0.50, 0.90, 0.95, 0.99)})
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
            for value in (0.50, 0.90, 0.95, 0.99)
        }
    )
    return result


def trajectory_metrics(
    prediction_m: np.ndarray | Sequence[float],
    true_range_m: np.ndarray | Sequence[float],
    *,
    target_hit: np.ndarray | Sequence[bool],
    jump_threshold_m: float = 1_000.0,
) -> dict[str, float | int]:
    """计算流式轨迹的覆盖率、误差、命中帧和跳变指标。

    ``prediction_m`` 必须与真值按完整帧轴对齐；未输出帧写为 ``NaN``。
    ``target_hit`` 仅用于最终离线评估 “实际目标响应帧” 子集，不会反馈到推理候选选择。

    ``jump_count`` 统计连续两帧都有预测时，预测位置的绝对变化超过
    ``jump_threshold_m`` 的次数；不把跨越未覆盖时间段的两点误判为跳变。
    """
    # 预处理真值和预测张量尺寸
    true_values = _one_dimensional(true_range_m, "真实轨迹", dtype=np.float64)          # (300,)
    aligned_prediction = _one_dimensional(prediction_m, "预测轨迹", dtype=np.float64)   # (300,)
    assert len(true_values) >= 1, "真实轨迹不能为空。"
    assert np.isfinite(jump_threshold_m) and jump_threshold_m >= 0.0, "jump_threshold_m 必须为非负有限数。"
    assert len(aligned_prediction) == len(true_values), "预测轨迹与真实轨迹长度不一致。"

    # 计算可评估帧、覆盖帧及其距离误差；预测 NaN 不参与误差统计。
    truth_valid = np.isfinite(true_values)
    covered = truth_valid & np.isfinite(aligned_prediction)     # 同时有有效真值和预测的帧。
    errors = aligned_prediction[covered] - true_values[covered] # 有符号误差；汇总函数会派生绝对误差指标。

    result: dict[str, float | int] = {
        "frame_count": int(len(true_values)),
        "valid_truth_frames": int(truth_valid.sum()),
        "covered_frames": int(covered.sum()),
        "coverage": float(covered.sum() / max(int(truth_valid.sum()), 1)),  # 分母保护仅防止空有效集除零。
    }
    result.update(_error_summary(errors))

    # 只比较时间上相邻、且两帧均有预测的位置，避免跨 NaN 空洞误报跳变。
    adjacent_covered = covered[:-1] & covered[1:]
    if np.any(adjacent_covered):
        displacement = np.abs(np.diff(aligned_prediction)[adjacent_covered])  # 单帧预测位置变化量（米）。
        result["jump_pair_count"] = int(len(displacement))
        result["jump_count"] = int(np.sum(displacement > jump_threshold_m))
        result["jump_rate"] = float(result["jump_count"] / len(displacement))
    else:
        result["jump_pair_count"] = 0
        result["jump_count"] = 0
        result["jump_rate"] = 0.0
    result["jump_threshold_m"] = float(jump_threshold_m)

    # 只在实际目标响应帧上复算覆盖率和误差；仅用于离线评估。
    hit_mask = _one_dimensional(target_hit, "target_hit", dtype=bool)
    assert len(hit_mask) == len(true_values), "target_hit 长度必须与真实轨迹一致。"
    hit_valid = truth_valid & hit_mask  # 具有真实响应且真值有效的帧。
    hit_covered = covered & hit_mask  # 上述帧中实际获得预测的子集。
    hit_errors = aligned_prediction[hit_covered] - true_values[hit_covered]
    hit_summary = _error_summary(hit_errors)
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
    for value in (0.50, 0.90, 0.95, 0.99):
        suffix = _quantile_name(value)
        result[f"hit_abs_error_{suffix}_m"] = hit_summary[f"abs_error_{suffix}_m"]
    return result


def timing_metrics_from_steps(steps: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    """汇总逐步记录中的端到端耗时（秒），忽略 ``NaN``。"""
    durations = _one_dimensional(
        [step.get("end_to_end_s", np.nan) for step in steps],
        "逐步端到端耗时",
        dtype=np.float64,
    )
    finite = durations[np.isfinite(durations)]
    assert not np.any(finite < 0.0), "耗时不得为负。"
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
