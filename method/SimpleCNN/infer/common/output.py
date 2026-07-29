"""推理结果的可复现落盘与控制台日志。"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


_METHOD_ROOT = Path(__file__).resolve().parents[2]


def to_jsonable(value: Any) -> Any:
    """递归转换 Path、NumPy 标量和数组，供 JSON/JSONL 使用。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def safe_name(value: str) -> str:
    """将数据源或 checkpoint 名称转为安全且稳定的文件名片段。"""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return normalized.strip("._") or "unnamed"


def _compact_number(value: str) -> str:
    """将数据集命名中的大整数压缩为便于浏览的 k/m 形式。"""
    try:
        number = int(value)
    except ValueError:
        return value
    if number >= 1_000_000 and number % 1_000_000 == 0:
        return f"{number // 1_000_000}m"
    if number >= 1_000 and number % 1_000 == 0:
        return f"{number // 1_000}k"
    return str(number)


def dataset_tag(data_root: Path) -> str:
    """从合成数据目录提取 F/N/S 核心字段；未知格式时安全截断。"""
    selected: list[str] = []
    for token in data_root.name.split("_"):
        if len(token) > 1 and token[0] in {"F", "N", "S"} and token[1:].isdigit():
            selected.append(f"{token[0]}{_compact_number(token[1:])}")
    if selected:
        return "-".join(selected)
    return safe_name(data_root.name)[:64]


def checkpoint_tag(checkpoint_path: Path, checkpoint_step: object) -> str:
    """用 profile、模型类型、checkpoint 名和 step 组成短标签。"""
    profile = checkpoint_path.parents[3].name
    profile_short = profile.removeprefix("simplecnn_").replace("simplecnn", "") or profile
    run_name = checkpoint_path.parents[2].name
    model_match = re.search(r"(?:^|-)model([A-Za-z0-9]+)(?:-|$)", run_name)
    model_part = f"-{model_match.group(1)}" if model_match else ""
    step = "unknown" if checkpoint_step is None else safe_name(str(checkpoint_step))
    return safe_name(f"{profile_short}{model_part}-{checkpoint_path.stem}-s{step}")


def create_output_dir(
    *,
    data_root: Path,
    checkpoint_path: Path,
    checkpoint_step: object,
) -> Path:
    """返回稳定的 <dataset>--<checkpoint> 输出目录，可追加多个方法。"""
    output_dir = _METHOD_ROOT / "infer" / "_output" / (
        f"{dataset_tag(data_root)}--{checkpoint_tag(checkpoint_path, checkpoint_step)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)
    return output_dir


def write_json(path: Path, payload: Any) -> None:
    """原子写 JSON，避免中断时留下截断结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    """读取已有的推理配置；不存在时返回 None。"""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """逐步写入 JSONL，保留每种方法的逐窗推理日志。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_jsonable(record), ensure_ascii=False, sort_keys=False))
            handle.write("\n")


def configure_logger(label: str) -> logging.Logger:
    """创建仅输出到控制台的本次推理日志器。"""
    logger = logging.getLogger(f"simplecnn.infer.{safe_name(label)}.{id(label)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
