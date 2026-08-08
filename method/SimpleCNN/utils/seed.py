"""随机性控制：全局、rank 和 DataLoader worker 使用不同但可复现的种子。"""

from __future__ import annotations

import random

import numpy as np
import torch

from utils.torch_compat import torch_npu_backend


def seed_everything(seed: int, rank: int = 0) -> int:
    """设置当前 rank 的随机种子，并返回实际使用的种子。"""
    effective_seed = (int(seed) + int(rank) * 10_000_019) % (2**63 - 1)
    random.seed(effective_seed)
    np.random.seed(effective_seed % (2**32))
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    npu_backend = torch_npu_backend()
    if npu_backend is not None and npu_backend.is_available():
        npu_backend.manual_seed_all(effective_seed)
    return effective_seed


def worker_seed(
    base_seed: int,
    rank: int,
    worker_id: int,
    stream_generation: int = 0,
) -> int:
    """为在线数据流中的 worker 和恢复代次生成彼此分离的确定性种子。"""
    if stream_generation < 0:
        raise ValueError("stream_generation 不得为负。")
    return (
        int(base_seed)
        + int(rank) * 10_000_019
        + int(worker_id) * 1_000_003
        + int(stream_generation) * 1_000_000_007
    ) % (2**63 - 1)
