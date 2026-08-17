"""多 NPU 推理入口共用的设备解析与静态样本分配。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import re
from collections.abc import Sequence
from multiprocessing import get_context
from typing import Callable, Mapping, TypeVar


_NPU_DEVICE = re.compile(r"(?:npu:)?(\d+)$", re.IGNORECASE)
T = TypeVar("T")
R = TypeVar("R")


def parse_npu_devices(raw: str) -> tuple[str, ...]:
    """将逗号分隔的 NPU 编号规范为 ``npu:<index>``，并拒绝歧义配置。"""
    devices: list[str] = []
    for item in raw.split(","):
        value = item.strip()
        match = _NPU_DEVICE.fullmatch(value)
        if match is None:
            raise ValueError(f"--devices 只能包含 NPU 编号或 npu:<编号>，实际为 {value!r}。")
        device = f"npu:{int(match.group(1))}"
        if device in devices:
            raise ValueError(f"--devices 包含重复 NPU：{device}。")
        devices.append(device)
    if not devices:
        raise ValueError("--devices 至少需要指定一个 NPU。")
    return tuple(devices)


def partition_round_robin(items: Sequence[T], devices: Sequence[str]) -> dict[str, tuple[T, ...]]:
    """按设备顺序轮转分配样本；每个样本恰好归属一个 NPU。"""
    if not devices:
        raise ValueError("至少需要一个设备才能分配样本。")
    buckets: dict[str, list[T]] = {device: [] for device in devices}
    for index, item in enumerate(items):
        buckets[devices[index % len(devices)]].append(item)
    return {device: tuple(items) for device, items in buckets.items() if items}


def execute_device_tasks(tasks: Mapping[str, T], worker: Callable[[T], R]) -> dict[str, R]:
    """使用 ``spawn`` 为每个指定 NPU 并发执行一个固定任务。"""
    if not tasks:
        raise ValueError("至少需要一个非空设备任务。")
    context = get_context("spawn")
    results: dict[str, R] = {}
    with ProcessPoolExecutor(
        max_workers=len(tasks),
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(worker, task): device
            for device, task in tasks.items()
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results
