"""由跨卡累加统计量生成可记录到控制台和 W&B 的验证指标。"""

from __future__ import annotations

import math

import torch

from configs.base import SimpleCNNConfig


METRIC_NAMES = (
    "q_loss_sum",
    "q_count",
    "line_loss_sum",
    "line_count",
    "abs_error_sum",
    "squared_error_sum",
    "point_count",
    "q_abs_error_sum",
    "q_squared_error_sum",
    "q_true_positive",
    "q_false_positive",
    "q_true_negative",
    "q_false_negative",
    "block_count",
    "positive_block_count",
)


def metric_accumulator_dtype(device: torch.device) -> torch.dtype:
    """返回当前后端可用于指标累加和分布式归约的精度。"""
    # HCCL 不支持 float64 all-reduce；CUDA/CPU 继续使用 float64 统计精度。
    return torch.float32 if device.type == "npu" else torch.float64


def empty_metric_totals(device: torch.device) -> torch.Tensor:
    """创建顺序由 ``METRIC_NAMES`` 固定的数值累加器。"""
    return torch.zeros(len(METRIC_NAMES), device=device, dtype=metric_accumulator_dtype(device))


def metrics_from_totals(totals: torch.Tensor, config: SimpleCNNConfig, prefix: str) -> dict[str, float]:
    """把 all-reduce 后的累计和转换为最终标量指标。"""
    values = {name: float(totals[index].item()) for index, name in enumerate(METRIC_NAMES)}
    q_loss = values["q_loss_sum"] / max(values["q_count"], 1.0)
    line_loss = values["line_loss_sum"] / max(values["line_count"], 1.0)
    point_count = max(values["point_count"], 1.0)
    q_count = max(values["q_count"], 1.0)
    true_positive = values["q_true_positive"]
    false_positive = values["q_false_positive"]
    true_negative = values["q_true_negative"]
    false_negative = values["q_false_negative"]
    q_precision = true_positive / max(true_positive + false_positive, 1.0)
    q_recall = true_positive / max(true_positive + false_negative, 1.0)
    q_brier = values["q_squared_error_sum"] / q_count
    return {
        f"{prefix}/loss_total": config.lambda_q * q_loss + config.lambda_line * line_loss,
        f"{prefix}/loss_q": q_loss,
        f"{prefix}/loss_line": line_loss,
        f"{prefix}/point_mae_bin": values["abs_error_sum"] / point_count,
        f"{prefix}/point_rmse_bin": math.sqrt(values["squared_error_sum"] / point_count),
        f"{prefix}/q_mae": values["q_abs_error_sum"] / q_count,
        f"{prefix}/q_brier": q_brier,
        f"{prefix}/q_rmse": math.sqrt(q_brier),
        f"{prefix}/q_accuracy": (true_positive + true_negative) / q_count,
        f"{prefix}/q_precision": q_precision,
        f"{prefix}/q_recall": q_recall,
        f"{prefix}/q_f1": 2.0 * q_precision * q_recall / max(q_precision + q_recall, 1e-12),
        f"{prefix}/blocks": values["block_count"],
        f"{prefix}/visible_positive_blocks": values["positive_block_count"],
        f"{prefix}/line_supervised_blocks": values["line_count"],
        f"{prefix}/labeled_points": values["point_count"],
        f"{prefix}/q_supervised_blocks": values["q_count"],
    }
