"""单机单卡/多卡 DDP 的统一初始化和数值归约工具。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import TypeVar

import torch
import torch.distributed as dist

from runtime.settings import RuntimeSettings


T = TypeVar("T")


@dataclass(frozen=True)
class DistributedContext:
    """当前进程的设备和 DDP 身份。"""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def _npu_is_available() -> bool:
    """延迟导入 torch_npu，避免 NVIDIA/CPU 环境要求安装 Ascend 依赖。"""
    try:
        import torch_npu  # noqa: F401 - 导入会向 torch 注册 npu 后端。
    except ImportError:
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


def _resolve_device(requested: str, local_rank: int) -> tuple[torch.device, str]:
    """解析指定或自动探测的加速器，并返回设备与默认 DDP 后端。"""
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        return device, "nccl"
    if requested in {"auto", "npu"} and _npu_is_available():
        import torch_npu  # noqa: F401 - 已由此导入注册 torch.npu.config。

        torch.npu.config.allow_internal_format = True
        device = torch.device("npu", local_rank)
        torch.npu.set_device(device)
        return device, "hccl"
    if requested == "cpu" or requested == "auto":
        return torch.device("cpu"), "gloo"
    if requested == "cuda":
        raise RuntimeError("LT_ACCELERATOR=cuda，但当前 PyTorch 未检测到可用 CUDA 设备。")
    raise RuntimeError(
        "LT_ACCELERATOR=npu，但当前环境未检测到可用 Ascend NPU。"
        "请确认镜像中的 PyTorch、torch_npu 和 CANN 版本匹配。"
    )


def setup_distributed(settings: RuntimeSettings) -> DistributedContext:
    """从 torchrun 环境变量初始化 CUDA、NPU 或 CPU 的统一 DDP。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    device, default_backend = _resolve_device(settings.accelerator, local_rank)
    backend = default_backend if settings.distributed_backend == "auto" else settings.distributed_backend

    if world_size > 1 and not dist.is_initialized():
        init_kwargs: dict[str, object] = {
            "backend": backend,
            "timeout": timedelta(minutes=settings.distributed_timeout_minutes),
        }
        if device.type != "cpu":
            init_kwargs["device_id"] = device
        dist.init_process_group(**init_kwargs)

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
    )


def barrier(context: DistributedContext) -> None:
    """仅在 DDP 模式下同步所有 rank。"""
    if context.is_distributed:
        dist.barrier()


def cleanup_distributed(context: DistributedContext) -> None:
    """安全释放 DDP 进程组。"""
    if context.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def broadcast_object(context: DistributedContext, value: T | None) -> T:
    """把 rank 0 已准备好的可序列化对象广播给全部进程。"""
    values: list[T | None] = [value if context.is_main else None]
    if context.is_distributed:
        dist.broadcast_object_list(values, src=0)
    result = values[0]
    if result is None:  # pragma: no cover - 防御性检查
        raise RuntimeError("rank 0 未能广播对象。")
    return result


def broadcast_path(context: DistributedContext, path: str | None) -> str:
    """由 rank 0 生成共享路径，再安全广播给所有训练进程。"""
    return broadcast_object(context, path)


def reduce_sum(values: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    """对数值张量执行跨 rank 求和；返回本 rank 可直接使用的结果。"""
    if context.is_distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def broadcast_module_buffers(module: torch.nn.Module, context: DistributedContext) -> None:
    """评估前从 rank 0 同步 BatchNorm 等模型 buffer，允许各 rank 不等长前向。"""
    if not context.is_distributed:
        return
    for buffer in module.buffers():
        dist.broadcast(buffer, src=0)


def rank_zero_print(context: DistributedContext, *args: object, **kwargs: object) -> None:
    """只由 rank 0 输出控制台信息。"""
    if context.is_main:
        print(*args, **kwargs)
