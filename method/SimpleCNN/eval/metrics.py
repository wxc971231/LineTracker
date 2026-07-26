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
    "block_count",
    "positive_block_count",
)


def empty_metric_totals(device: torch.device) -> torch.Tensor:
    """创建顺序由 ``METRIC_NAMES`` 固定的 float64 累加器。"""
    return torch.zeros(len(METRIC_NAMES), device=device, dtype=torch.float64)


def metrics_from_totals(totals: torch.Tensor, config: SimpleCNNConfig, prefix: str) -> dict[str, float]:
    """把 all-reduce 后的累计和转换为最终标量指标。"""
    values = {name: float(totals[index].item()) for index, name in enumerate(METRIC_NAMES)}
    q_loss = values["q_loss_sum"] / max(values["q_count"], 1.0)
    line_loss = values["line_loss_sum"] / max(values["line_count"], 1.0)
    point_count = max(values["point_count"], 1.0)
    return {
        f"{prefix}/loss_total": config.lambda_q * q_loss + config.lambda_line * line_loss,
        f"{prefix}/loss_q": q_loss,
        f"{prefix}/loss_line": line_loss,
        f"{prefix}/point_mae_bin": values["abs_error_sum"] / point_count,
        f"{prefix}/point_rmse_bin": math.sqrt(values["squared_error_sum"] / point_count),
        f"{prefix}/q_mae": values["q_abs_error_sum"] / max(values["q_count"], 1.0),
        f"{prefix}/q_rmse": math.sqrt(values["q_squared_error_sum"] / max(values["q_count"], 1.0)),
        f"{prefix}/blocks": values["block_count"],
        f"{prefix}/visible_positive_blocks": values["positive_block_count"],
        f"{prefix}/line_supervised_blocks": values["line_count"],
        f"{prefix}/labeled_points": values["point_count"],
    }
