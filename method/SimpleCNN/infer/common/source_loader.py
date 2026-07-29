"""训练划分 manifest 与推理样本选择工具。

推理只读取训练 run 落盘的 data/split_manifest.json，绝不重新枚举或随机划分
数据集。manifest 中保存的是相对 data_root 的路径，因此同一训练 run 可以在
另一台机器上通过 .env 或 --data-root 复用。
"""

from __future__ import annotations

import json
import sys
from numbers import Integral
from pathlib import Path
from random import Random
from typing import Any, Sequence

METHOD_ROOT = Path(__file__).resolve().parents[2]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from configs.base import SimpleCNNConfig
from data.dataloader import SourceRecord

from .model_loader import InferenceBundle


_VALID_SPLITS = frozenset({"train", "val", "test"})


def split_manifest_path(run_dir: Path | str) -> Path:
    """返回训练 run 保存的 source 划分文件，并确保它存在。"""

    path = Path(run_dir).expanduser().resolve() / "data" / "split_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            "训练 run 缺少 data/split_manifest.json；推理不会为了补齐它而重新划分数据："
            f"{path}"
        )
    return path


def _safe_relative_path(raw_path: object, *, location: str, manifest_path: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"split manifest 的 {location}.relative_path 无效：{manifest_path}")
    relative = Path(raw_path)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise ValueError(
            f"split manifest 的 {location}.relative_path 必须是安全相对路径："
            f"{manifest_path}"
        )
    return relative


def _parse_split_manifest(
    manifest_path: Path,
    *,
    data_root: Path,
) -> dict[str, list[SourceRecord]]:
    try:
        content = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"split manifest 不是有效 JSON：{manifest_path}") from error
    if not isinstance(content, dict):
        raise ValueError(f"split manifest 顶层必须是对象：{manifest_path}")

    try:
        schema_version = int(content.get("schema_version", 1))
    except (TypeError, ValueError) as error:
        raise ValueError(f"split manifest 的 schema_version 无效：{manifest_path}") from error
    if schema_version not in {1, 2}:
        raise ValueError(
            f"不支持的 split manifest schema_version={schema_version}：{manifest_path}"
        )

    splits = content.get("splits")
    if not isinstance(splits, dict) or set(splits) != _VALID_SPLITS:
        raise ValueError(
            "split manifest 必须且只能包含 train、val、test 三个列表："
            f"{manifest_path}"
        )

    records_by_split: dict[str, list[SourceRecord]] = {}
    all_source_ids: set[str] = set()
    all_relative_paths: set[Path] = set()
    for split_name in ("train", "val", "test"):
        entries = splits[split_name]
        if not isinstance(entries, list):
            raise ValueError(
                f"split manifest 的 {split_name} 必须是列表：{manifest_path}"
            )
        records: list[SourceRecord] = []
        for index, item in enumerate(entries):
            location = f"{split_name}[{index}]"
            if not isinstance(item, dict):
                raise ValueError(
                    f"split manifest 的 {location} 必须是对象：{manifest_path}"
                )
            source_id = item.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(
                    f"split manifest 的 {location}.source_id 无效：{manifest_path}"
                )
            relative_path = _safe_relative_path(
                item.get("relative_path"),
                location=location,
                manifest_path=manifest_path,
            )
            if source_id in all_source_ids:
                raise ValueError(
                    f"split manifest 中 source_id={source_id!r} 重复或跨 split 泄漏："
                    f"{manifest_path}"
                )
            if relative_path in all_relative_paths:
                raise ValueError(
                    f"split manifest 中 relative_path={str(relative_path)!r} 重复或跨 split 泄漏："
                    f"{manifest_path}"
                )
            all_source_ids.add(source_id)
            all_relative_paths.add(relative_path)
            records.append(
                SourceRecord(source_id=source_id, path=data_root / relative_path)
            )
        records_by_split[split_name] = records
    return records_by_split


def load_split_manifest(
    run_dir: Path | str,
    *,
    data_root: Path | str,
) -> dict[str, list[SourceRecord]]:
    """读取完整 train/val/test 划分，并按当前机器 data_root 重建路径。"""

    root = Path(data_root).expanduser().resolve()
    return _parse_split_manifest(split_manifest_path(run_dir), data_root=root)


def load_split_sources(
    bundle_or_config: InferenceBundle | SimpleCNNConfig,
    run_dir: Path | str | None = None,
    split: str = "test",
    *,
    data_root: Path | str | None = None,
) -> list[SourceRecord]:
    """返回指定 split 的稳定 SourceRecord 列表，不创建任何新划分。

    推荐调用形式为 load_split_sources(bundle, split="test")。若只传配置，
    必须额外提供 run_dir，例如 load_split_sources(config, run_dir, "test")。
    """

    if isinstance(bundle_or_config, InferenceBundle):
        # 允许简写 load_split_sources(bundle, "val")。
        if (
            isinstance(run_dir, str)
            and run_dir in _VALID_SPLITS
            and split == "test"
        ):
            split = run_dir
            run_dir = None
        config = bundle_or_config.config
        effective_run_dir = bundle_or_config.run_dir if run_dir is None else Path(run_dir)
    elif isinstance(bundle_or_config, SimpleCNNConfig):
        if run_dir is None:
            raise ValueError("传入 SimpleCNNConfig 时必须显式提供 run_dir。")
        config = bundle_or_config
        effective_run_dir = Path(run_dir)
    else:
        raise TypeError(
            "bundle_or_config 必须是 InferenceBundle 或 SimpleCNNConfig，"
            f"实际为 {type(bundle_or_config).__name__}。"
        )

    if split not in _VALID_SPLITS:
        raise ValueError(
            f"split 仅支持 train、val、test，实际为 {split!r}。"
        )
    root = config.data_root if data_root is None else Path(data_root).expanduser().resolve()
    records = load_split_manifest(effective_run_dir, data_root=root)[split]
    if not records:
        raise ValueError(
            f"训练 run 的 {split} split 为空，不能执行推理："
            f"{split_manifest_path(effective_run_dir)}"
        )
    return records


def resolve_source_path(record: SourceRecord, data_root: Path | str) -> Path:
    """将 SourceRecord 路径安全地投影到指定 data_root 下。

    从 load_split_sources 得到的 record 已经使用该 data_root；此辅助函数主要
    用于调用方显式切换根目录时复用已有 SourceRecord。
    """

    root = Path(data_root).expanduser().resolve()
    path = record.path
    if path.is_absolute():
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"source {record.source_id!r} 的路径不位于指定 data_root 下："
                f"path={path}，data_root={root}"
            ) from error
    else:
        relative = path
    relative = _safe_relative_path(
        str(relative),
        location=f"source_id={record.source_id!r}",
        manifest_path=Path("<memory>"),
    )
    return root / relative


def _normalise_values(
    value: Sequence[Any] | Any,
    *,
    selector_name: str,
) -> list[Any]:
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    elif isinstance(value, (str, bytes)):
        values = [value]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        values = [value]
    if not values or any(item == "" for item in values):
        raise ValueError(f"{selector_name} 不能为空。")
    return values


def select_sources(
    sources: Sequence[SourceRecord],
    *,
    indices: Sequence[int] | int | str | None = None,
    source_ids: Sequence[str] | str | None = None,
    num_samples: int | None = None,
    seed: int = 42,
    all_samples: bool = False,
) -> list[SourceRecord]:
    """按 index、source_id、随机无放回或全量选择推理 source。

    四种模式必须且只能选择一种。随机模式用独立 Random(seed)，不会影响训练、
    NumPy 或 PyTorch 的随机状态。
    """

    records = list(sources)
    if not records:
        raise ValueError("候选 source 列表为空。")

    mode_count = sum(
        option is not None for option in (indices, source_ids, num_samples)
    ) + int(all_samples)
    if mode_count != 1:
        raise ValueError(
            "必须且只能指定一种样本选择方式：indices、source_ids、num_samples 或 all_samples。"
        )

    if all_samples:
        return records

    if indices is not None:
        raw_indices = _normalise_values(indices, selector_name="indices")
        selected_indices: list[int] = []
        seen_indices: set[int] = set()
        for raw_index in raw_indices:
            if isinstance(raw_index, str):
                try:
                    index = int(raw_index)
                except ValueError as error:
                    raise ValueError(f"indices 含非整数值：{raw_index!r}。") from error
            elif isinstance(raw_index, Integral) and not isinstance(raw_index, bool):
                index = int(raw_index)
            else:
                raise ValueError(f"indices 含非整数值：{raw_index!r}。")
            if not 0 <= index < len(records):
                raise IndexError(
                    f"source index={index} 越界；当前 split 共 {len(records)} 个样本。"
                )
            if index in seen_indices:
                raise ValueError(f"indices 含重复 index={index}。")
            seen_indices.add(index)
            selected_indices.append(index)
        return [records[index] for index in selected_indices]

    if source_ids is not None:
        raw_ids = _normalise_values(source_ids, selector_name="source_ids")
        by_id = {record.source_id: record for record in records}
        selected: list[SourceRecord] = []
        seen_ids: set[str] = set()
        for raw_id in raw_ids:
            if not isinstance(raw_id, str) or not raw_id:
                raise ValueError(f"source_ids 含无效 source_id：{raw_id!r}。")
            if raw_id in seen_ids:
                raise ValueError(f"source_ids 含重复 source_id={raw_id!r}。")
            try:
                record = by_id[raw_id]
            except KeyError as error:
                raise ValueError(
                    f"当前 split 中不存在 source_id={raw_id!r}。"
                ) from error
            seen_ids.add(raw_id)
            selected.append(record)
        return selected

    if not isinstance(num_samples, Integral) or isinstance(num_samples, bool):
        raise ValueError("num_samples 必须是正整数。")
    if num_samples < 1:
        raise ValueError("num_samples 必须是正整数。")
    if num_samples > len(records):
        raise ValueError(
            f"num_samples={num_samples} 超过当前 split 的样本数 {len(records)}；"
            "随机抽样不允许放回。"
        )
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError("seed 必须是整数。")
    selected_indices = Random(int(seed)).sample(range(len(records)), int(num_samples))
    return [records[index] for index in selected_indices]
