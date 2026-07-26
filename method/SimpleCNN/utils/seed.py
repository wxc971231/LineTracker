"""随机性控制：全局、rank 和 DataLoader worker 使用不同但可复现的种子。"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, rank: int = 0) -> int:
    """设置当前 rank 的随机种子，并返回实际使用的种子。"""
    effective_seed = int(seed) + int(rank) * 10_000_019
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(effective_seed)
        torch.cuda.manual_seed_all(effective_seed)
    return effective_seed


def worker_seed(base_seed: int, rank: int, worker_id: int) -> int:
    """为在线数据流中的 worker 生成彼此分离的确定性种子。"""
    return int(base_seed) + int(rank) * 10_000_019 + int(worker_id) * 1_000_003
