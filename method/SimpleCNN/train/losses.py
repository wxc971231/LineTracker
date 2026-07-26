"""SimpleCNN 的质量 BCE 与逐帧响应距离 bin 掩码 Huber 损失。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from configs.base import SimpleCNNConfig


@dataclass
class LossOutput:
    """反向传播所需总损失及用于跨卡加权归约的原始统计量。"""

    total: torch.Tensor
    q_loss: torch.Tensor
    line_loss: torch.Tensor
    q_loss_sum: torch.Tensor
    q_count: torch.Tensor
    line_loss_sum: torch.Tensor
    line_count: torch.Tensor
    abs_error_sum: torch.Tensor
    squared_error_sum: torch.Tensor
    point_count: torch.Tensor


def _huber_loss_per_point(
    prediction: torch.Tensor,
    target: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """返回逐元素 Huber 损失；NPU 使用等价的原生基础算子避免 CPU 回退。"""
    if prediction.device.type != "npu":
        return functional.huber_loss(prediction, target, reduction="none", delta=delta)

    residual_abs = (prediction - target).abs()
    delta_tensor = torch.as_tensor(delta, device=prediction.device, dtype=prediction.dtype)
    quadratic = 0.5 * residual_abs.square()
    linear = delta_tensor * (residual_abs - 0.5 * delta_tensor)
    return torch.where(residual_abs <= delta_tensor, quadratic, linear)


def compute_losses(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: SimpleCNNConfig,
) -> LossOutput:
    """计算文档第 5.2 节定义的联合损失。

    对第 $\tau$ 帧，预测距离 bin 为
    ``rho_m + nu_mpf * (tau - 9.5)``。仅在 $I_\tau=1$ 且该块
    ``m_line=True`` 时，才将它与实际响应 bin $d_\tau$ 比较。
    """
    # 模型前向可以使用 FP16/BF16，但损失与指标统计始终提升到 FP32。
    # 距离残差可达到几十万 bin，若沿用 FP16，平方和甚至 Huber 累加
    # 都可能在一次 batch 内溢出为 inf。
    q_logit = prediction["q_logit"].float()
    q_target = batch["q"].float()
    q_per_sample = functional.binary_cross_entropy_with_logits(
        q_logit, q_target, reduction="none"
    )
    q_loss_sum = q_per_sample.sum()
    q_count = torch.tensor(float(q_per_sample.numel()), device=q_per_sample.device)
    q_loss = q_loss_sum / q_count.clamp_min(1.0)

    frame_index = torch.arange(
        config.frames_per_window,
        device=q_per_sample.device,
        dtype=torch.float32,
    )
    tau_zero = (config.frames_per_window - 1) / 2.0
    rho_m = prediction["rho_m"].float()
    nu_mpf = prediction["nu_mpf"].float()
    predicted_bin = rho_m.unsqueeze(1) + nu_mpf.unsqueeze(1) * (
        frame_index.unsqueeze(0) - tau_zero
    )
    point_mask = batch["I"].bool() & batch["m_line"].bool().unsqueeze(1)
    target_bin = batch["d"].float()
    residual = predicted_bin - target_bin
    huber_per_point = _huber_loss_per_point(predicted_bin, target_bin, config.huber_delta_bins)

    # 先按每个块的响应数归一化，再仅在 m_line 块之间平均，
    # 避免目标响应更多的块仅因点数更多而获得更大总权重。
    valid_block = batch["m_line"].bool()
    point_count_per_block = point_mask.sum(dim=1)
    line_sum_per_block = (huber_per_point * point_mask).sum(dim=1)
    line_per_block = line_sum_per_block / point_count_per_block.clamp_min(1)
    valid_block_float = valid_block.to(dtype=predicted_bin.dtype)
    line_count = valid_block_float.sum()
    line_loss_sum = (line_per_block * valid_block_float).sum()
    line_loss = line_loss_sum / line_count.clamp_min(1.0)

    point_mask_float = point_mask.to(dtype=predicted_bin.dtype)
    abs_error_sum = (residual.abs() * point_mask_float).sum()
    squared_error_sum = (residual.square() * point_mask_float).sum()
    point_count = point_mask_float.sum()
    total = config.lambda_q * q_loss + config.lambda_line * line_loss
    return LossOutput(
        total=total,
        q_loss=q_loss,
        line_loss=line_loss,
        q_loss_sum=q_loss_sum,
        q_count=q_count,
        line_loss_sum=line_loss_sum,
        line_count=line_count,
        abs_error_sum=abs_error_sum,
        squared_error_sum=squared_error_sum,
        point_count=point_count,
    )
