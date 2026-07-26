"""固定标准网格的分布式数值评估。"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Iterator

import torch
from torch import nn

from configs.base import SimpleCNNConfig
from eval.metrics import empty_metric_totals, metrics_from_totals
from train.losses import compute_losses
from utils.distributed import DistributedContext, reduce_sum


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """将 DataLoader 返回的全部张量转移到当前 rank 设备。"""
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def autocast_context(config: SimpleCNNConfig, device: torch.device):
    """返回与训练一致的 AMP 上下文；CPU 或 off 时返回空上下文。"""
    if config.amp == "off" or device.type != "cuda":
        return nullcontext()
    if config.amp == "fp16":
        dtype = torch.float16
    elif config.amp == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    dataloader: Iterator[dict[str, torch.Tensor]],
    config: SimpleCNNConfig,
    context: DistributedContext,
    *,
    prefix: str = "val",
) -> dict[str, float]:
    """遍历固定网格，all-reduce 汇总 loss 和逐帧距离 bin 指标。"""
    was_training = model.training
    model.eval()
    totals = empty_metric_totals(context.device)
    for batch in dataloader:
        batch = move_batch_to_device(batch, context.device)
        with autocast_context(config, context.device):
            prediction = model(batch["x"])
            losses = compute_losses(prediction, batch, config)
        q_error = prediction["q"].float() - batch["q"].float()
        totals += torch.stack(
            [
                losses.q_loss_sum.detach().double(),
                losses.q_count.detach().double(),
                losses.line_loss_sum.detach().double(),
                losses.line_count.detach().double(),
                losses.abs_error_sum.detach().double(),
                losses.squared_error_sum.detach().double(),
                losses.point_count.detach().double(),
                q_error.abs().sum().detach().double(),
                q_error.square().sum().detach().double(),
                torch.tensor(float(batch["x"].shape[0]), device=context.device, dtype=torch.float64),
                batch["is_positive"].sum().detach().double(),
            ]
        )
    totals = reduce_sum(totals, context)
    if was_training:
        model.train()
    return metrics_from_totals(totals, config, prefix)
