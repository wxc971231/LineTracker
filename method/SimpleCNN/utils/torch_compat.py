"""PyTorch 可选运行时后端的动态访问工具。"""

from __future__ import annotations

import importlib
from typing import Any

import torch


def import_torch_npu() -> Any:
    """导入 Ascend 扩展，使其在运行时向 ``torch`` 注册 NPU 后端。"""

    return importlib.import_module("torch_npu")


def torch_npu_backend() -> Any | None:
    """返回运行时注册的 NPU 命名空间；标准 PyTorch 环境中返回 ``None``。"""

    return getattr(torch, "npu", None)
