"""将局部预测映射到全局距离并进行候选去重的基础实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CandidateLine:
    """一个局部块回归出的全局候选线。"""

    source_index: int
    time_start: int
    q: float
    rho_global_m: float
    nu_mpf: float


def local_to_global_candidate(
    source_index: int,
    time_start: int,
    range_start: int,
    q: float,
    rho_local_m: float,
    nu_mpf: float,
) -> CandidateLine:
    """把块内中心距离转换为全局距离，保持中间帧斜率定义不变。"""
    return CandidateLine(
        source_index=source_index,
        time_start=time_start,
        q=float(q),
        rho_global_m=float(range_start + rho_local_m),
        nu_mpf=float(nu_mpf),
    )


def deduplicate_candidates(
    candidates: Iterable[CandidateLine],
    *,
    rho_threshold_m: float,
    nu_threshold_mpf: float,
) -> list[CandidateLine]:
    """按 q 降序执行简单 NMS；只在同一 source 和时间窗内去重。"""
    kept: list[CandidateLine] = []
    for candidate in sorted(candidates, key=lambda item: item.q, reverse=True):
        is_duplicate = any(
            candidate.source_index == previous.source_index
            and candidate.time_start == previous.time_start
            and abs(candidate.rho_global_m - previous.rho_global_m) < rho_threshold_m
            and abs(candidate.nu_mpf - previous.nu_mpf) < nu_threshold_mpf
            for previous in kept
        )
        if not is_duplicate:
            kept.append(candidate)
    return kept
