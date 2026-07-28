"""为 SimpleCNN 启动器、DDP rank 和 DataLoader worker 设置可辨识进程名。"""

from __future__ import annotations

import ctypes
import os
import re


PROCESS_JOB_ID_ENV = "LT_SIMPLECNN_JOB_ID"
PROCESS_LABEL_ENV = "LT_SIMPLECNN_PROCESS_LABEL"
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_component(value: object, fallback: str) -> str:
    """把进程标题字段限制为便于 shell 精确匹配的 ASCII 字符。"""
    cleaned = _SAFE_COMPONENT.sub("-", str(value)).strip("-")
    return cleaned or fallback


def ensure_process_job_id() -> str:
    """返回本次启动的稳定作业标识；启动器 PID 会由所有子进程继承。"""
    job_id = os.environ.get(PROCESS_JOB_ID_ENV)
    if not job_id:
        job_id = str(os.getpid())
        os.environ[PROCESS_JOB_ID_ENV] = job_id
    return _safe_component(job_id, str(os.getpid()))


def build_process_title(
    role: str,
    *,
    label: str | None = None,
    job_id: str | None = None,
    rank: int | str | None = None,
    worker_id: int | str | None = None,
    infer_rank: bool = True,
) -> str:
    """构造 ``SimpleCNN-角色|配置|作业|rank|worker`` 格式的进程标题。"""
    effective_label = label or os.environ.get(PROCESS_LABEL_ENV) or "unknown"
    effective_job_id = (
        job_id
        or os.environ.get(PROCESS_JOB_ID_ENV)
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or str(os.getppid())
    )
    effective_rank = rank
    if effective_rank is None and infer_rank:
        effective_rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK"))

    parts = [
        f"SimpleCNN-{_safe_component(role, 'process')}",
        _safe_component(effective_label, "unknown"),
        f"j{_safe_component(effective_job_id, str(os.getppid()))}",
    ]
    if effective_rank is not None:
        parts.append(f"r{_safe_component(effective_rank, 'unknown')}")
    if worker_id is not None:
        parts.append(f"w{_safe_component(worker_id, 'unknown')}")
    return "|".join(parts)


def _apply_process_title(title: str) -> None:
    """优先设置完整 argv 标题；依赖缺失时至少设置 Linux comm 短名称。"""
    try:
        from setproctitle import setproctitle
    except ImportError:
        try:
            # PR_SET_NAME 只保留 15 bytes，但足以让 ps 的 comm 显示 SimpleCNN 前缀。
            encoded = title.encode("ascii", errors="replace")[:15]
            libc = ctypes.CDLL(None)
            libc.prctl(15, ctypes.c_char_p(encoded), 0, 0, 0)
        except (AttributeError, OSError):
            return
    else:
        setproctitle(title)


def set_process_title(
    role: str,
    *,
    label: str | None = None,
    job_id: str | None = None,
    rank: int | str | None = None,
    worker_id: int | str | None = None,
    infer_rank: bool = True,
) -> str:
    """设置当前进程标题并返回最终文本，失败时不影响训练或评估。"""
    if label is not None:
        os.environ[PROCESS_LABEL_ENV] = _safe_component(label, "unknown")
    title = build_process_title(
        role,
        label=label,
        job_id=job_id,
        rank=rank,
        worker_id=worker_id,
        infer_rank=infer_rank,
    )
    _apply_process_title(title)
    return title
