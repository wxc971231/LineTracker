"""固定标准网格的分布式数值评估。"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Iterator

import torch
from tqdm.auto import tqdm
from torch import nn

from configs.base import SimpleCNNConfig
from eval.metrics import empty_metric_totals, metric_accumulator_dtype, metrics_from_totals
from train.losses import compute_losses
from utils.distributed import DistributedContext, reduce_sum


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """将 DataLoader 返回的全部张量转移到当前 rank 设备。"""
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def autocast_context(config: SimpleCNNConfig, device: torch.device):
    """返回与训练一致的 AMP 上下文；CPU 或 off 时返回空上下文。"""
    if config.amp == "off" or device.type == "cpu":
        return nullcontext()
    if config.amp == "fp16":
        dtype = torch.float16
    elif config.amp == "bf16":
        dtype = torch.bfloat16
    elif device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device.type == "npu":
        try:
            dtype = torch.bfloat16 if torch.npu.is_bf16_supported() else torch.float16
        except (AttributeError, RuntimeError):
            dtype = torch.float16
    else:
        dtype = torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


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
    metric_dtype = metric_accumulator_dtype(context.device)
    progress = (
        tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"{prefix} [rank 0]",
            dynamic_ncols=True,
            leave=True,
        )
        if context.is_main
        else dataloader
    )
    for batch in progress:
        batch = move_batch_to_device(batch, context.device)
        with autocast_context(config, context.device):
            prediction = model(batch["x"])
            losses = compute_losses(prediction, batch, config)
        q_error = prediction["q"].float() - batch["q"].float()
        totals += torch.stack(
            [
                losses.q_loss_sum.detach().to(dtype=metric_dtype),
                losses.q_count.detach().to(dtype=metric_dtype),
                losses.line_loss_sum.detach().to(dtype=metric_dtype),
                losses.line_count.detach().to(dtype=metric_dtype),
                losses.abs_error_sum.detach().to(dtype=metric_dtype),
                losses.squared_error_sum.detach().to(dtype=metric_dtype),
                losses.point_count.detach().to(dtype=metric_dtype),
                q_error.abs().sum().detach().to(dtype=metric_dtype),
                q_error.square().sum().detach().to(dtype=metric_dtype),
                torch.tensor(float(batch["x"].shape[0]), device=context.device, dtype=metric_dtype),
                batch["is_positive"].sum().detach().to(dtype=metric_dtype),
            ]
        )
    totals = reduce_sum(totals, context)
    if was_training:
        model.train()
    return metrics_from_totals(totals, config, prefix)
