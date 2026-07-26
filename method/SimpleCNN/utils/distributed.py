"""单机单卡/多卡 DDP 的统一初始化和数值归约工具。"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    """当前进程的设备和 DDP 身份。"""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def setup_distributed() -> DistributedContext:
    """从 torchrun 环境变量初始化 DDP；普通 python 启动时退化为单进程。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        if device.type != "cuda":
            raise RuntimeError("多进程训练需要 CUDA/NCCL；CPU 模式请使用单进程。")
        dist.init_process_group(backend="nccl")

    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def barrier(context: DistributedContext) -> None:
    """仅在 DDP 模式下同步所有 rank。"""
    if context.is_distributed:
        dist.barrier()


def cleanup_distributed(context: DistributedContext) -> None:
    """安全释放 DDP 进程组。"""
    if context.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def reduce_sum(values: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    """对数值张量执行跨 rank 求和；返回本 rank 可直接使用的结果。"""
    if context.is_distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def rank_zero_print(context: DistributedContext, *args: object, **kwargs: object) -> None:
    """只由 rank 0 输出控制台信息。"""
    if context.is_main:
        print(*args, **kwargs)
