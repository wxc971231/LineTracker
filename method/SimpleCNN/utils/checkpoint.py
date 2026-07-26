"""训练 snapshot 与 best checkpoint 的保存和恢复。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """先写临时文件再替换，避免训练中断留下损坏 checkpoint。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """加载 checkpoint 到当前设备。"""
    return torch.load(path, map_location=device, weights_only=False)


def rotate_checkpoints(directory: Path, keep: int) -> None:
    """保留最近的若干 step checkpoint；last.pt 和 best.pt 不参与轮转。"""
    if keep < 0:
        raise ValueError("keep 不得为负。")
    paths = sorted(directory.glob("step_*.pt"), key=lambda item: item.stat().st_mtime)
    expired = paths if keep == 0 else paths[:-keep]
    for path in expired:
        path.unlink(missing_ok=True)
