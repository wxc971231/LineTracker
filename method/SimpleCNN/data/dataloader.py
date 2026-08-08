"""SimpleCNN 的完整数据接口：在线训练流、压缩位裁剪和固定验证网格。

训练侧直接从完整 ``data.npz`` 中局部解包，不落盘重叠的 20×10000 块。
验证侧按文档中的标准时间窗和 34 个标准距离块建立固定清单。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler, get_worker_info

from configs.base import SimpleCNNConfig
from utils.process_title import set_process_title
from utils.seed import worker_seed


_GRID_CACHE_SAMPLE_CHECK_COUNT = 10
GRID_CACHE_DIR = Path(__file__).resolve().parents[1] / "runs" / "_cache" / "val_test_grid"
_GRID_CACHE_SCHEMA_VERSION = 3
_GRID_CACHE_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "source_ids",
        "source_relative_paths",
        "source_index",
        "source_sizes",
        "source_mtime_ns",
        "time_start",
        "range_start",
        "response_mask",
        "response_bin",
        "is_positive",
        "q_valid",
    }
)


def _initialize_train_worker_process(worker_id: int) -> None:
    """区分在线训练 DataLoader worker 与持有 NPU/CUDA 的 rank 进程。"""
    set_process_title("train-data", worker_id=worker_id)


def _initialize_eval_worker_process(worker_id: int) -> None:
    """区分验证/测试 DataLoader worker 与持有 NPU/CUDA 的 rank 进程。"""
    set_process_title("eval-data", worker_id=worker_id)


def _read_npz_array_header(
    archive: zipfile.ZipFile,
    field_name: str,
) -> tuple[tuple[int, ...], np.dtype[Any]]:
    """只读取 NPZ 成员的 NPY 头，不解压大型数组主体。"""
    with archive.open(f"{field_name}.npy") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version in {(2, 0), (3, 0)}:
            shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"不支持的 NPY 版本：{version}")
    return tuple(int(value) for value in shape), np.dtype(dtype)


def _grid_cache_headers_are_valid(
    archive: zipfile.ZipFile,
    config: SimpleCNNConfig,
) -> bool:
    """校验大数组 shape/dtype 契约，同时保持 cache 命中检查轻量。"""
    headers = {
        name: _read_npz_array_header(archive, name)
        for name in (
            "source_index",
            "time_start",
            "range_start",
            "response_mask",
            "response_bin",
            "is_positive",
            "q_valid",
        )
    }
    source_index_shape, source_index_dtype = headers["source_index"]
    if len(source_index_shape) != 1 or source_index_shape[0] < 1:
        return False
    sample_count = source_index_shape[0]
    expected_shapes = {
        "source_index": (sample_count,),
        "time_start": (sample_count,),
        "range_start": (sample_count,),
        "response_mask": (sample_count, config.frames_per_window),
        "response_bin": (sample_count, config.frames_per_window),
        "is_positive": (sample_count,),
        "q_valid": (sample_count,),
    }
    if any(headers[name][0] != shape for name, shape in expected_shapes.items()):
        return False
    integer_fields = ("source_index", "time_start", "range_start", "response_bin")
    if source_index_dtype.kind not in {"i", "u"}:
        return False
    if any(headers[name][1].kind not in {"i", "u"} for name in integer_fields):
        return False
    return all(
        headers[name][1] == np.dtype(np.bool_)
        for name in ("response_mask", "is_positive", "q_valid")
    )


@dataclass(frozen=True)
class SourceRecord:
    """一份完整合成序列的稳定标识和数据路径。"""

    source_id: str
    path: Path


@dataclass(frozen=True)
class DataArtifacts:
    """一次实验使用的数据划分和固定验证/测试网格清单。"""

    split_path: Path
    validation_manifest_path: Path
    test_manifest_path: Path | None
    train_sources: tuple[SourceRecord, ...]
    validation_sources: tuple[SourceRecord, ...]
    test_sources: tuple[SourceRecord, ...]


def discover_sources(data_root: Path, source_sample_limit: int = 0) -> list[SourceRecord]:
    """按生成器布局枚举一级序列目录，不查询每个 ``data.npz``。"""
    records: list[SourceRecord] = []
    with os.scandir(data_root) as entries:
        for entry in entries:
            # DirEntry 通常直接使用目录项类型，不对 data.npz 发起额外 stat。
            if not entry.is_dir():
                continue
            source_dir = Path(entry.path)
            records.append(
                SourceRecord(source_id=entry.name, path=source_dir / "data.npz")
            )
            if source_sample_limit > 0 and len(records) >= source_sample_limit:
                break
    if not records:
        raise FileNotFoundError(f"在 {data_root} 下没有找到任何序列目录。")
    return records


def standard_distance_starts(config: SimpleCNNConfig) -> tuple[int, ...]:
    """返回推理和验证共同使用的 34 个标准距离块起点。"""
    final_start = config.range_bins - config.block_width_m
    starts = list(range(0, final_start + 1, config.spatial_step_m))
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def standard_time_starts(total_frames: int, config: SimpleCNNConfig) -> tuple[int, ...]:
    """按可调验证步进生成固定长度时间窗起点，并强制覆盖最后一窗。"""
    final_start = total_frames - config.frames_per_window
    if final_start < 0:
        raise ValueError(f"序列仅有 {total_frames} 帧，不足 {config.frames_per_window} 帧窗口。")
    starts = list(range(0, final_start + 1, config.validation_time_stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


class PackedSource:
    """一份已载入内存的完整合成序列，并提供局部 bit 解包接口。"""

    REQUIRED_FIELDS = {
        "observation_packed",
        "target_hit",
        "target_hit_bin",
        "target_true_range_m",
    }

    def __init__(self, record: SourceRecord, config: SimpleCNNConfig) -> None:
        self.record = record
        with np.load(record.path, allow_pickle=False) as archive:
            missing = self.REQUIRED_FIELDS.difference(archive.files)
            if missing:
                raise KeyError(f"{record.path} 缺少字段：{sorted(missing)}")
            self.observation_packed = archive["observation_packed"].astype(np.uint8, copy=False)
            self.target_hit = archive["target_hit"].astype(np.bool_, copy=False)
            self.target_hit_bin = archive["target_hit_bin"].astype(np.int32, copy=False)
            self.target_true_range_m = archive["target_true_range_m"].astype(np.float32, copy=False)

        if self.observation_packed.ndim != 2:
            raise ValueError(
                f"{record.path} 的 observation_packed 应为二维数组，"
                f"实际形状为 {self.observation_packed.shape}。"
            )
        self.frames = int(self.observation_packed.shape[0])
        expected_packed_bins = math.ceil(config.range_bins / 8)
        if self.observation_packed.shape[1] != expected_packed_bins:
            raise ValueError(
                f"{record.path} 的 observation_packed 距离宽度为 "
                f"{self.observation_packed.shape[1]} byte，当前配置 range_bins="
                f"{config.range_bins} 需要 {expected_packed_bins} byte。"
            )
        _validate_target_arrays(
            record.path,
            self.target_hit,
            self.target_hit_bin,
            self.target_true_range_m,
            self.frames,
            config,
        )

    def extract_window(
        self,
        time_start: int,
        range_start: int,
        config: SimpleCNNConfig,
    ) -> np.ndarray:
        """局部解包出 ``[20, 10000]`` 原始二值窗口。

        只读取包含所需距离的若干字节；当 ``range_start`` 不是 8 的倍数时，
        额外解包至多 7 个 bin 后再切片，支持随机模 8 平移。
        """
        frames = config.frames_per_window
        width = config.block_width_m
        if not 0 <= time_start <= self.frames - frames:
            raise IndexError("时间窗起点越界。")
        if not 0 <= range_start <= config.range_bins - width:
            raise IndexError("距离窗起点越界。")

        first_byte = range_start // 8
        bit_offset = range_start % 8
        byte_count = math.ceil((bit_offset + width) / 8)
        byte_window = self.observation_packed[
            time_start : time_start + frames,
            first_byte : first_byte + byte_count,
        ]
        unpacked = np.unpackbits(byte_window, axis=1, bitorder=config.packed_bitorder)
        return unpacked[:, bit_offset : bit_offset + width].astype(np.uint8, copy=False)


class LabelSource:
    """仅为固定验证/测试网格读取标签，避免加载大体积观测矩阵。"""

    REQUIRED_FIELDS = {
        "target_hit",
        "target_hit_bin",
        "target_true_range_m",
    }

    def __init__(self, record: SourceRecord, config: SimpleCNNConfig) -> None:
        self.record = record
        with np.load(record.path, allow_pickle=False) as archive:
            missing = self.REQUIRED_FIELDS.difference(archive.files)
            if missing:
                raise KeyError(f"{record.path} 缺少字段：{sorted(missing)}")
            self.target_hit = archive["target_hit"].astype(np.bool_, copy=False)
            self.target_hit_bin = archive["target_hit_bin"].astype(np.int32, copy=False)
            self.target_true_range_m = archive["target_true_range_m"].astype(np.float32, copy=False)

        self.frames = int(self.target_hit.shape[0])
        _validate_target_arrays(
            record.path,
            self.target_hit,
            self.target_hit_bin,
            self.target_true_range_m,
            self.frames,
            config,
        )


def _validate_target_arrays(
    path: Path,
    target_hit: np.ndarray,
    target_hit_bin: np.ndarray,
    target_true_range_m: np.ndarray,
    frames: int,
    config: SimpleCNNConfig,
) -> None:
    """提前校验标签形状及其与当前数据配置的范围契约。"""
    expected_shape = (frames,)
    if not (
        target_hit.shape
        == target_hit_bin.shape
        == target_true_range_m.shape
        == expected_shape
    ):
        raise ValueError(f"{path} 的目标标签形状与帧数不一致。")
    if frames < config.frames_per_window:
        raise ValueError(
            f"{path} 仅有 {frames} 帧，不足当前窗口长度 "
            f"frames_per_window={config.frames_per_window}。"
        )
    if not np.all(np.isfinite(target_true_range_m)):
        raise ValueError(f"{path} 的 target_true_range_m 含非有限值。")
    if np.any(target_true_range_m < 0) or np.any(
        target_true_range_m >= config.range_bins
    ):
        raise ValueError(
            f"{path} 的 target_true_range_m 超出 [0, {config.range_bins})。"
        )
    visible_bins = target_hit_bin[target_hit]
    if np.any(visible_bins < 0) or np.any(visible_bins >= config.range_bins):
        raise ValueError(
            f"{path} 的可见 target_hit_bin 超出 [0, {config.range_bins})。"
        )


def rearrange_distance_channels(window: np.ndarray, config: SimpleCNNConfig) -> np.ndarray:
    """把 ``[20,10000]`` 原始窗口无损重排为 ``[8,20,1250]``。"""
    expected_shape = (config.frames_per_window, config.block_width_m)
    if window.shape != expected_shape:
        raise ValueError(f"期望原始窗口形状 {expected_shape}，实际为 {window.shape}。")
    grouped = window.reshape(config.frames_per_window, -1, config.input_channels)
    return grouped.transpose(2, 0, 1)


def _trajectory_relation(
    source: PackedSource | LabelSource,
    time_start: int,
    range_start: int,
    config: SimpleCNNConfig,
) -> tuple[bool, bool, float, float]:
    """判断当前块是否完整包含、部分相交或完全避开潜在轨迹。"""
    trajectory = source.target_true_range_m[time_start : time_start + config.frames_per_window]
    trajectory_min = float(np.min(trajectory))
    trajectory_max = float(np.max(trajectory))
    range_stop = range_start + config.block_width_m
    is_full = trajectory_min >= range_start and trajectory_max < range_stop
    is_intersecting = bool(np.any((trajectory >= range_start) & (trajectory < range_stop)))
    return is_full, is_intersecting, trajectory_min, trajectory_max


def build_target_labels(
    source: PackedSource | LabelSource,
    time_start: int,
    range_start: int,
    config: SimpleCNNConfig,
) -> dict[str, Any]:
    """生成二值置信度、置信度监督掩码与逐帧几何标签。"""
    frames = config.frames_per_window
    range_stop = range_start + config.block_width_m
    is_full, is_intersecting, _, _ = _trajectory_relation(source, time_start, range_start, config)

    response_mask = np.zeros(frames, dtype=np.bool_)
    response_bin = np.full(frames, -1, dtype=np.int64)
    hit = source.target_hit[time_start : time_start + frames]
    hit_bin = source.target_hit_bin[time_start : time_start + frames]
    inside = hit & (hit_bin >= range_start) & (hit_bin < range_stop)
    response_mask[inside] = True
    response_bin[inside] = hit_bin[inside].astype(np.int64) - int(range_start)

    hit_count = int(response_mask.sum())
    track_relation = 1 if is_full else (2 if is_intersecting else 0)
    is_visible_positive = bool(is_full and hit_count > 0)
    return {
        "I": response_mask,
        "d": response_bin,
        "H": hit_count,
        "q": np.float32(is_visible_positive),
        "q_valid": bool(not (is_full and hit_count == 0)),
        "is_positive": is_visible_positive,         # 是否为完整可见正样本，用于监督几何损失
        "track_relation": np.int8(track_relation),  # 0=背景，1=完整，2=部分相交
    }


def _sample_to_numpy(
    source: PackedSource,
    time_start: int,
    range_start: int,
    config: SimpleCNNConfig,
    *,
    labels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """提取原始窗口、重排并附加标签，形成一个局部块样本。"""
    if labels is None:
        labels = build_target_labels(source, time_start, range_start, config)
    raw_window = source.extract_window(time_start, range_start, config)
    return {
        "x": rearrange_distance_channels(raw_window, config),
        "I": labels["I"],
        "d": labels["d"],
        "q": labels["q"],
        "is_positive": labels["is_positive"],
        "q_valid": labels["q_valid"],
    }


def _negative_labels(config: SimpleCNNConfig) -> dict[str, Any]:
    """直接构造与潜在轨迹完全不相交的背景负样本标签。"""
    return {
        "I": np.zeros(config.frames_per_window, dtype=np.bool_),
        "d": np.full(config.frames_per_window, -1, dtype=np.int64),
        "q": np.float32(0.0),
        "is_positive": False,
        "q_valid": True,
    }


def _round_positive_count(batch_size: int, positive_fraction: float) -> int:
    """按文档规则四舍五入正样本数，并保留至少一个正负样本。"""
    rounded = math.floor(batch_size * positive_fraction + 0.5)
    return min(batch_size - 1, max(1, rounded))


@dataclass
class _CacheEntry:
    """缓存中一份序列及其已贡献的正样本次数。"""

    source: PackedSource
    positive_draws: int = 0


class _SourcePool:
    """每个 rank/worker 独立维护的随机序列缓存池。"""

    def __init__(self, records: Sequence[SourceRecord], config: SimpleCNNConfig, rng: np.random.Generator) -> None:
        if not records:
            raise ValueError("当前 rank/worker 没有可用训练序列。")
        self.records = tuple(records)
        self.config = config
        self.rng = rng
        self._order: list[int] = []
        self._cursor = 0
        self.entries: list[_CacheEntry] = []
        self._reshuffle_order()
        for _ in range(min(config.source_cache_size, len(self.records))):
            self.entries.append(
                _CacheEntry(PackedSource(self._next_record(set()), config))
            )

    def _reshuffle_order(self) -> None:
        self._order = self.rng.permutation(len(self.records)).tolist()
        self._cursor = 0

    def _next_record(self, excluded_ids: set[str]) -> SourceRecord:
        """从未使用的排列位置取下一份不在当前缓存中的序列。"""
        for _ in range(max(1, len(self.records) * 2)):
            if self._cursor >= len(self._order):
                self._reshuffle_order()
            record = self.records[self._order[self._cursor]]
            self._cursor += 1
            if record.source_id not in excluded_ids:
                return record
        # 当本 worker 只有极少来源且缓存已填满时，允许返回任意来源。
        return self.records[int(self.rng.integers(len(self.records)))]

    def choose(self) -> _CacheEntry:
        """均匀抽取当前缓存中的一份序列。"""
        return self.entries[int(self.rng.integers(len(self.entries)))]

    def note_positive(self, entry: _CacheEntry) -> None:
        """记录正样本使用次数；达到额度后用新的来源替换缓存项。"""
        entry.positive_draws += 1
        if entry.positive_draws < self.config.source_positive_quota:
            return
        if len(self.entries) >= len(self.records):
            entry.positive_draws = 0
            return
        slot = self.entries.index(entry)
        excluded = {item.source.record.source_id for item in self.entries}
        self.entries[slot] = _CacheEntry(
            PackedSource(self._next_record(excluded), self.config)
        )


@dataclass(frozen=True)
class _PositiveContext:
    """一条可见正样本的来源，用于构造匹配时间条件的负样本。"""

    source: PackedSource
    time_start: int
    range_start: int


class _BatchFactory:
    """在一个 worker 内生成严格平衡的在线训练 batch。"""

    def __init__(self, records: Sequence[SourceRecord], config: SimpleCNNConfig, seed: int) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.pool = _SourcePool(records, config, self.rng)
        weights = np.array(
            [
                config.negative_local_weight,
                config.negative_same_time_weight,
                config.negative_random_weight,
                config.negative_partial_weight,
            ],
            dtype=np.float64,
        )
        self.negative_kinds = ("local", "same_time", "random", "partial")
        self.negative_probabilities = weights / weights.sum()

    def _positive_start_bounds(self, source: PackedSource, time_start: int) -> tuple[int, int] | None:
        """计算完整轨迹带裕量落入随机正样本块时的起点整数区间。"""
        trajectory = source.target_true_range_m[time_start : time_start + self.config.frames_per_window]
        trajectory_min = float(np.min(trajectory))
        trajectory_max = float(np.max(trajectory))
        margin = self.config.positive_margin_m
        lower = math.ceil(max(0.0, trajectory_max + margin - self.config.block_width_m))
        upper = math.floor(
            min(float(self.config.range_bins - self.config.block_width_m), trajectory_min - margin)
        )
        return (lower, upper) if lower <= upper else None

    def _sample_positive_start(self, source: PackedSource, time_start: int) -> int | None:
        """混合标准块与随机有效块，为正样本选择空间起点。"""
        bounds = self._positive_start_bounds(source, time_start)
        if bounds is None:
            return None
        lower, upper = bounds
        if self.rng.random() < self.config.standard_positive_fraction:
            candidates = [
                start for start in standard_distance_starts(self.config) if lower <= start <= upper
            ]
            if candidates:
                return int(candidates[int(self.rng.integers(len(candidates)))])
        return int(self.rng.integers(lower, upper + 1))

    def sample_positive(self) -> tuple[dict[str, Any], _PositiveContext]:
        """抽取完整轨迹且至少有一次目标响应的可见正样本。"""
        for _ in range(512):
            entry = self.pool.choose()
            source = entry.source
            time_start = int(self.rng.integers(0, source.frames - self.config.frames_per_window + 1))
            range_start = self._sample_positive_start(source, time_start)
            if range_start is None:
                continue
            labels = build_target_labels(source, time_start, range_start, self.config)
            if not labels["is_positive"]:
                continue
            sample = _sample_to_numpy(source, time_start, range_start, self.config, labels=labels)
            self.pool.note_positive(entry)
            return sample, _PositiveContext(source, time_start, range_start)
        raise RuntimeError("连续 512 次无法构造可见正样本；请检查 target_hit 或裁剪参数。")

    def _disjoint_intervals(self, source: PackedSource, time_start: int) -> list[tuple[int, int]]:
        """返回与当前时间窗潜在轨迹不相交的距离块起点区间。"""
        _, _, trajectory_min, trajectory_max = _trajectory_relation(source, time_start, 0, self.config)
        guard = self.config.negative_guard_m
        maximum_start = self.config.range_bins - self.config.block_width_m
        left_upper = math.floor(trajectory_min - guard - self.config.block_width_m)
        right_lower = math.ceil(trajectory_max + guard)
        intervals: list[tuple[int, int]] = []
        if left_upper >= 0:
            intervals.append((0, min(left_upper, maximum_start)))
        if right_lower <= maximum_start:
            intervals.append((max(0, right_lower), maximum_start))
        return [(lower, upper) for lower, upper in intervals if lower <= upper]

    def _partial_start_intervals(self, source: PackedSource, time_start: int) -> list[tuple[int, int]]:
        """返回恰好由左或右边界截断潜在轨迹的块起点区间。"""
        trajectory = source.target_true_range_m[time_start : time_start + self.config.frames_per_window]
        trajectory_min = float(np.min(trajectory))
        trajectory_max = float(np.max(trajectory))
        width = self.config.block_width_m
        maximum_start = self.config.range_bins - width
        raw_intervals = (
            (math.floor(trajectory_min) + 1, math.floor(trajectory_max)),
            (math.floor(trajectory_min - width) + 1, math.floor(trajectory_max - width)),
        )
        return [
            (max(0, lower), min(maximum_start, upper))
            for lower, upper in raw_intervals
            if max(0, lower) <= min(maximum_start, upper)
        ]

    def _sample_partial_negative(self, source: PackedSource, time_start: int) -> dict[str, Any] | None:
        """抽取轨迹与块相交但被边界截断的 q*=0 困难负样本。"""
        intervals = self._partial_start_intervals(source, time_start)
        if not intervals:
            return None
        lengths = np.asarray([upper - lower + 1 for lower, upper in intervals], dtype=np.float64)
        index = int(self.rng.choice(len(intervals), p=lengths / lengths.sum()))
        lower, upper = intervals[index]
        range_start = int(self.rng.integers(lower, upper + 1))
        labels = build_target_labels(source, time_start, range_start, self.config)
        if int(labels["track_relation"]) != 2:
            raise RuntimeError("部分相交起点计算与标签关系不一致。")
        return _sample_to_numpy(source, time_start, range_start, self.config, labels=labels)

    def _choose_disjoint_start(
        self,
        source: PackedSource,
        time_start: int,
        *,
        local: bool,
    ) -> int | None:
        """抽取不相交起点：普通负样本分层均衡，局部负样本贴近航迹。"""
        intervals = self._disjoint_intervals(source, time_start)
        if not intervals:
            return None
        if not local:
            return self._sample_stratified_disjoint_start(intervals)

        lengths = np.array([upper - lower + 1 for lower, upper in intervals], dtype=np.float64)
        interval_index = int(self.rng.choice(len(intervals), p=lengths / lengths.sum()))
        lower, upper = intervals[interval_index]

        span = min(self.config.negative_local_span_m, upper - lower + 1)
        # 左侧区间的右端、右侧区间的左端最靠近轨迹。
        trajectory = source.target_true_range_m[time_start : time_start + self.config.frames_per_window]
        trajectory_center = float(np.mean(trajectory))
        interval_center = (lower + upper + self.config.block_width_m) / 2.0
        if interval_center < trajectory_center:
            return int(self.rng.integers(upper - span + 1, upper + 1))
        return int(self.rng.integers(lower, lower + span))

    def _sample_stratified_disjoint_start(self, intervals: Sequence[tuple[int, int]]) -> int:
        """在近场、过渡区、远场间均衡抽取一个不相交的距离块起点。

        三个层分别以块中心距离划分为 0--10 km、10--50 km 和
        50--300 km。先在有可行起点的层之间等概率选择，再在该层的
        可行整数起点中均匀抽取；这避免远场因可选距离范围更大而淹没
        近场和过渡区负样本。若某层没有可用位置则自动跳过。
        """
        strata = ((0, 10_000), (10_000, 50_000), (50_000, self.config.range_bins))
        candidates_by_stratum: list[list[tuple[int, int]]] = []
        half_width = self.config.block_width_m / 2.0
        for lower_center, upper_center in strata:
            # 块中心 s + W_R / 2 落在 [lower_center, upper_center) 内。
            lower_start = math.ceil(lower_center - half_width)
            upper_start = math.ceil(upper_center - half_width) - 1
            candidates = [
                (max(interval_lower, lower_start), min(interval_upper, upper_start))
                for interval_lower, interval_upper in intervals
                if max(interval_lower, lower_start) <= min(interval_upper, upper_start)
            ]
            if candidates:
                candidates_by_stratum.append(candidates)

        if not candidates_by_stratum:  # 防御性回退；理论上 intervals 非空时不会发生。
            candidates_by_stratum.append(list(intervals))
        candidates = candidates_by_stratum[int(self.rng.integers(len(candidates_by_stratum)))]
        lengths = np.array([upper - lower + 1 for lower, upper in candidates], dtype=np.float64)
        index = int(self.rng.choice(len(candidates), p=lengths / lengths.sum()))
        lower, upper = candidates[index]
        return int(self.rng.integers(lower, upper + 1))

    def _sample_normal_negative(
        self,
        source: PackedSource,
        time_start: int,
        *,
        local: bool,
    ) -> dict[str, Any] | None:
        """从 ``observation_packed`` 中抽取不含潜在轨迹的普通负样本。"""
        range_start = self._choose_disjoint_start(source, time_start, local=local)
        if range_start is None:
            return None
        return _sample_to_numpy(
            source,
            time_start,
            range_start,
            self.config,
            labels=_negative_labels(self.config),
        )

    def sample_negative(self, context: _PositiveContext) -> dict[str, Any]:
        """按配置的混合策略生成一个负样本。"""
        for _ in range(512):
            kind = self.negative_kinds[
                int(self.rng.choice(len(self.negative_kinds), p=self.negative_probabilities))
            ]
            if kind == "local":
                sample = self._sample_normal_negative(context.source, context.time_start, local=True)
            elif kind == "same_time":
                sample = self._sample_normal_negative(context.source, context.time_start, local=False)
            elif kind == "partial":
                sample = self._sample_partial_negative(context.source, context.time_start)
            else:
                source = self.pool.choose().source
                time_start = int(self.rng.integers(0, source.frames - self.config.frames_per_window + 1))
                sample = self._sample_normal_negative(source, time_start, local=False)
            if sample is not None:
                return sample
        raise RuntimeError("连续 512 次无法构造有效负样本；请检查距离范围和采样权重。")

    def make_batch(self) -> dict[str, torch.Tensor]:
        """构造一个本地 GPU batch，并在返回前随机打乱正负样本顺序。"""
        batch_size = self.config.batch_size_per_gpu
        positive_count = _round_positive_count(batch_size, self.config.positive_fraction)
        samples: list[dict[str, Any]] = []
        contexts: list[_PositiveContext] = []
        for _ in range(positive_count):
            positive, context = self.sample_positive()
            samples.append(positive)
            contexts.append(context)

        for index in range(batch_size - positive_count):
            samples.append(self.sample_negative(contexts[index % len(contexts)]))

        order = self.rng.permutation(len(samples))
        samples = [samples[int(index)] for index in order]
        return {
            "x": torch.from_numpy(np.stack([sample["x"] for sample in samples]).astype(np.float32, copy=False)),
            "I": torch.from_numpy(np.stack([sample["I"] for sample in samples]).astype(np.bool_, copy=False)),
            "d": torch.from_numpy(np.stack([sample["d"] for sample in samples]).astype(np.int64, copy=False)),
            "q": torch.from_numpy(np.asarray([sample["q"] for sample in samples], dtype=np.float32)),
            "q_valid": torch.from_numpy(np.asarray([sample["q_valid"] for sample in samples], dtype=np.bool_)),
            "is_positive": torch.from_numpy(
                np.asarray([sample["is_positive"] for sample in samples], dtype=np.bool_)
            ),
        }


class BalancedTrainBatchIterableDataset(IterableDataset[dict[str, torch.Tensor]]):
    """无限在线训练流；每次迭代直接产出一个正负比例受控的完整 batch。"""

    def __init__(
        self,
        records: Sequence[SourceRecord],
        config: SimpleCNNConfig,
        *,
        rank: int,
        stream_generation: int = 0,
    ) -> None:
        super().__init__()
        if stream_generation < 0:
            raise ValueError("stream_generation 不得为负。")
        self.records = tuple(records)
        self.config = config
        self.rank = rank
        self.stream_generation = int(stream_generation)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        worker_records = self.records[worker_id::worker_count]
        if not worker_records:
            raise RuntimeError("DataLoader worker 没有分配到任何训练序列；请减小 num_workers。")
        factory = _BatchFactory(
            worker_records,
            self.config,
            worker_seed(self.config.seed, self.rank, worker_id, self.stream_generation),
        )
        while True:
            yield factory.make_batch()


class DistributedSourceSampler(Sampler[int]):
    """按完整 source 给 DDP rank 分片，并在 source 内顺序遍历网格。"""

    def __init__(
        self,
        source_spans: Sequence[tuple[int, int]],
        rank: int,
        world_size: int,
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"无效 DDP 身份：rank={rank}, world_size={world_size}。")
        self.source_spans = tuple(source_spans[rank::world_size])

    def __iter__(self) -> Iterator[int]:
        for start, stop in self.source_spans:
            yield from range(start, stop)

    def __len__(self) -> int:
        return sum(stop - start for start, stop in self.source_spans)


class DistributedSourceReplacementSampler(Sampler[int]):
    """有放回抽取 source-local batch，保留随机性并避免反复解压序列。"""

    def __init__(
        self,
        source_spans: Sequence[tuple[int, int]],
        num_batches: int,
        batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        if num_batches < 1 or batch_size < 1:
            raise ValueError("有放回验证采样的 batch 数和 batch size 必须为正。")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"无效 DDP 身份：rank={rank}, world_size={world_size}。")
        self.source_spans = tuple(source_spans[rank::world_size])
        self.num_batches = int(num_batches) if self.source_spans else 0
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self._evaluation_index = 0

    def __iter__(self) -> Iterator[int]:
        if not self.source_spans:
            return
        evaluation_seed = (
            self.seed
            + self.rank * 10_000_019
            + self._evaluation_index * 1_000_000_007
        ) % (2**63 - 1)
        self._evaluation_index += 1
        rng = np.random.default_rng(evaluation_seed)
        lengths = np.asarray([stop - start for start, stop in self.source_spans], dtype=np.float64)
        probabilities = lengths / lengths.sum()
        for _ in range(self.num_batches):
            source_slot = int(rng.choice(len(self.source_spans), p=probabilities))
            start, stop = self.source_spans[source_slot]
            yield from rng.integers(start, stop, size=self.batch_size, dtype=np.int64).tolist()

    def __len__(self) -> int:
        return self.num_batches * self.batch_size


class DistributedSourceSubsetSampler(Sampler[int]):
    """从 rank 所属网格块中无放回抽取有限子集，并保持 source-major 访问顺序。

    每个 rank 仅处理 ``source_spans[rank::world_size]``，因此不同 rank 之间的
    source 与样本天然不重叠。rank 内先均匀无放回选择至多 ``num_batches ×
    batch_size`` 个块，再按 source-major 顺序产出，以减少 ``PackedSource`` 缓存
    的频繁切换；排序只改变访问顺序，不改变被选子集的均匀性。
    """

    def __init__(
        self,
        source_spans: Sequence[tuple[int, int]],
        num_batches: int,
        batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        if num_batches < 1 or batch_size < 1:
            raise ValueError("无放回评估采样的 batch 数和 batch size 必须为正。")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"无效 DDP 身份：rank={rank}, world_size={world_size}。")
        self.source_spans = tuple(source_spans[rank::world_size])
        self.max_sample_count = int(num_batches) * int(batch_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self._evaluation_index = 0

    def __iter__(self) -> Iterator[int]:
        if not self.source_spans:
            return
        starts = np.asarray([start for start, _ in self.source_spans], dtype=np.int64)
        lengths = np.asarray([stop - start for start, stop in self.source_spans], dtype=np.int64)
        cumulative_lengths = np.cumsum(lengths, dtype=np.int64)
        total_sample_count = int(cumulative_lengths[-1])
        selected_count = min(self.max_sample_count, total_sample_count)
        evaluation_seed = (
            self.seed
            + self.rank * 10_000_019
            + self._evaluation_index * 1_000_000_007
        ) % (2**63 - 1)
        self._evaluation_index += 1
        rng = np.random.default_rng(evaluation_seed)

        # 在 rank 的全部块上均匀无放回选子集；升序输出时同一 source 会连续访问。
        selected_offsets = np.sort(
            rng.choice(total_sample_count, size=selected_count, replace=False).astype(np.int64, copy=False)
        )
        source_slots = np.searchsorted(cumulative_lengths, selected_offsets, side="right")
        preceding_lengths = np.concatenate((np.zeros(1, dtype=np.int64), cumulative_lengths[:-1]))
        sample_indices = starts[source_slots] + selected_offsets - preceding_lengths[source_slots]
        yield from sample_indices.tolist()

    def __len__(self) -> int:
        return min(self.max_sample_count, sum(stop - start for start, stop in self.source_spans))


class StandardGridDataset(Dataset[dict[str, torch.Tensor]]):
    """从固定验证/测试清单恢复标准网格块，不做任何随机裁剪。"""

    def __init__(
        self,
        manifest_path: Path,
        config: SimpleCNNConfig,
        *,
        verify_cached_samples: bool = False,
    ) -> None:
        self.config = config
        with np.load(manifest_path, allow_pickle=False) as archive:
            uses_relative_paths = "source_relative_paths" in archive.files
            path_field = "source_relative_paths" if uses_relative_paths else "source_paths"
            raw_source_paths = archive[path_field]
            raw_source_ids = archive["source_ids"]
            raw_source_index = archive["source_index"]
            raw_time_start = archive["time_start"]
            raw_range_start = archive["range_start"]
            raw_response_mask = archive["response_mask"]
            raw_response_bin = archive["response_bin"]
            raw_is_positive = archive["is_positive"]
            raw_q_valid = archive["q_valid"]

        self._validate_manifest_arrays(
            manifest_path,
            raw_source_ids,
            raw_source_paths,
            raw_source_index,
            raw_time_start,
            raw_range_start,
            raw_response_mask,
            raw_response_bin,
            raw_is_positive,
            raw_q_valid,
        )
        if uses_relative_paths:
            self.source_paths = tuple(
                config.data_root / Path(str(path)) for path in raw_source_paths.tolist()
            )
        else:  # 兼容早期实验目录中的绝对路径清单。
            self.source_paths = tuple(Path(str(path)) for path in raw_source_paths.tolist())
        self.source_ids = tuple(str(item) for item in raw_source_ids.tolist())
        self.source_index = raw_source_index.astype(np.int32, copy=False)
        self.time_start = raw_time_start.astype(np.int32, copy=False)
        self.range_start = raw_range_start.astype(np.int32, copy=False)
        self.response_mask = raw_response_mask.astype(np.bool_, copy=False)
        self.response_bin = raw_response_bin.astype(np.int16, copy=False)
        self.is_positive = raw_is_positive.astype(np.bool_, copy=False)
        self.q_valid = raw_q_valid.astype(np.bool_, copy=False)
        self.source_spans = self._build_source_spans()
        if verify_cached_samples:
            self._verify_cached_samples(manifest_path)
        self._cache: OrderedDict[int, PackedSource] = OrderedDict()

    def _validate_manifest_arrays(
        self,
        manifest_path: Path,
        source_ids: np.ndarray,
        source_paths: np.ndarray,
        source_index: np.ndarray,
        time_start: np.ndarray,
        range_start: np.ndarray,
        response_mask: np.ndarray,
        response_bin: np.ndarray,
        is_positive: np.ndarray,
        q_valid: np.ndarray,
    ) -> None:
        """完整校验 cache 数组契约，避免损坏内容在 DataLoader worker 中延迟报错。"""
        problems: list[str] = []
        if source_ids.ndim != 1 or source_paths.ndim != 1:
            problems.append("source_ids/source_paths 必须是一维数组")
            source_count = -1
        else:
            source_count = int(source_ids.shape[0])
            if source_paths.shape[0] != source_count:
                problems.append("source_ids 与 source_paths 数量不一致")
            ids = tuple(str(item) for item in source_ids.tolist())
            paths = tuple(str(item) for item in source_paths.tolist())
            if len(set(ids)) != len(ids) or len(set(paths)) != len(paths):
                problems.append("source_ids/source_paths 不得重复")

        sample_count = int(source_index.shape[0]) if source_index.ndim == 1 else -1
        expected_shapes = {
            "source_index": (sample_count,),
            "time_start": (sample_count,),
            "range_start": (sample_count,),
            "response_mask": (sample_count, self.config.frames_per_window),
            "response_bin": (sample_count, self.config.frames_per_window),
            "is_positive": (sample_count,),
            "q_valid": (sample_count,),
        }
        arrays = {
            "source_index": source_index,
            "time_start": time_start,
            "range_start": range_start,
            "response_mask": response_mask,
            "response_bin": response_bin,
            "is_positive": is_positive,
            "q_valid": q_valid,
        }
        for name, array in arrays.items():
            if array.shape != expected_shapes[name]:
                problems.append(f"{name} 形状应为 {expected_shapes[name]}，实际为 {array.shape}")
        for name in ("source_index", "time_start", "range_start", "response_bin"):
            if not np.issubdtype(arrays[name].dtype, np.integer):
                problems.append(f"{name} 必须使用整数 dtype")
        if any(array.dtype != np.bool_ for array in (response_mask, is_positive, q_valid)):
            problems.append("response_mask/is_positive/q_valid 必须使用 bool dtype")

        shapes_valid = not any(array.shape != expected_shapes[name] for name, array in arrays.items())
        dtypes_valid = all(
            np.issubdtype(arrays[name].dtype, np.integer)
            for name in ("source_index", "time_start", "range_start", "response_bin")
        )
        if source_count < 1:
            problems.append("grid manifest 必须包含至少一个 source")
        if sample_count < 1:
            problems.append("grid manifest 必须包含至少一个样本")
        if shapes_valid and dtypes_valid and source_count > 0 and sample_count > 0:
            if np.any(source_index < 0) or np.any(source_index >= source_count):
                problems.append("source_index 存在越界值")
            if np.any(time_start < 0):
                problems.append("time_start 不得为负")
            maximum_range_start = self.config.range_bins - self.config.block_width_m
            if np.any(range_start < 0) or np.any(range_start > maximum_range_start):
                problems.append("range_start 存在越界值")
            valid_response_bin = (response_bin == -1) | (
                (response_bin >= 0) & (response_bin < self.config.block_width_m)
            )
            if not np.all(valid_response_bin):
                problems.append("response_bin 必须为 -1 或块内有效 bin")
            if not np.array_equal(response_mask, response_bin >= 0):
                problems.append("response_mask 与 response_bin 的有效位置不一致")
            if np.any(is_positive & ~response_mask.any(axis=1)):
                problems.append("is_positive 样本必须至少包含一个响应点")
            if np.any(is_positive & ~q_valid):
                problems.append("is_positive 样本必须启用 q 监督")
        if problems:
            raise ValueError(f"grid manifest 结构无效：{'；'.join(problems)}。文件：{manifest_path}")

    def _build_source_spans(self) -> tuple[tuple[int, int], ...]:
        """验证 source-major 排列并返回每份 source 的半开样本区间。"""
        if len(self.source_index) == 0:
            raise ValueError("固定评估网格不应为空。")
        starts = np.concatenate(
            (np.asarray([0]), np.flatnonzero(np.diff(self.source_index)) + 1)
        )
        stops = np.concatenate((starts[1:], np.asarray([len(self.source_index)])))
        observed = self.source_index[starts]
        expected = np.arange(len(self.source_ids), dtype=observed.dtype)
        if not np.array_equal(observed, expected):
            raise ValueError("grid manifest 必须按 source-major 连续排列且覆盖全部 source。")
        return tuple((int(start), int(stop)) for start, stop in zip(starts, stops, strict=True))

    def _verify_cached_samples(self, manifest_path: Path) -> None:
        """随机复算少量标签，尽早发现错误缓存而不读取观测矩阵。"""
        sample_count = min(_GRID_CACHE_SAMPLE_CHECK_COUNT, len(self))
        if sample_count == 0:
            return

        indices = np.random.default_rng().choice(len(self), size=sample_count, replace=False)
        label_sources: dict[int, LabelSource] = {}
        for index in indices:
            sample_index = int(index)
            source_index = int(self.source_index[sample_index])
            source = label_sources.get(source_index)
            if source is None:
                source = LabelSource(
                    SourceRecord(
                        self.source_ids[source_index],
                        self.source_paths[source_index],
                    ),
                    self.config,
                )
                label_sources[source_index] = source
            expected = build_target_labels(
                source,
                int(self.time_start[sample_index]),
                int(self.range_start[sample_index]),
                self.config,
            )
            checks = (
                ("response_mask", np.array_equal(self.response_mask[sample_index], expected["I"])),
                ("response_bin", np.array_equal(self.response_bin[sample_index], expected["d"])),
                ("q", bool(np.float32(self.is_positive[sample_index]) == expected["q"])),
                ("q_valid", bool(self.q_valid[sample_index] == expected["q_valid"])),
                ("is_positive", bool(self.is_positive[sample_index] == expected["is_positive"])),
            )
            for field_name, matched in checks:
                if not matched:
                    raise ValueError(
                        "grid 缓存标签校验失败："
                        f"manifest={manifest_path}, sample_index={sample_index}, "
                        f"source_id={self.source_ids[source_index]}, field={field_name}。"
                    )

    def __len__(self) -> int:
        return int(self.source_index.shape[0])

    def __getstate__(self) -> dict[str, Any]:
        """DataLoader worker fork/spawn 时不复制已载入的完整序列缓存。"""
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def _get_source(self, source_index: int) -> PackedSource:
        """使用小型 LRU 缓存避免验证时反复读取同一 npz。"""
        if source_index in self._cache:
            source = self._cache.pop(source_index)
            self._cache[source_index] = source
            return source
        record = SourceRecord(self.source_ids[source_index], self.source_paths[source_index])
        source = PackedSource(record, self.config)
        self._cache[source_index] = source
        while len(self._cache) > self.config.source_cache_size:
            self._cache.popitem(last=False)
        return source

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = int(self.source_index[index])
        time_start = int(self.time_start[index])
        range_start = int(self.range_start[index])
        source = self._get_source(source_index)
        raw_window = source.extract_window(time_start, range_start, self.config)
        return {
            "x": torch.from_numpy(rearrange_distance_channels(raw_window, self.config).astype(np.float32, copy=False)),
            "I": torch.from_numpy(self.response_mask[index].copy()),
            "d": torch.from_numpy(self.response_bin[index].copy()),
            "q": torch.tensor(float(self.is_positive[index]), dtype=torch.float32),
            "q_valid": torch.tensor(self.q_valid[index], dtype=torch.bool),
            "is_positive": torch.tensor(self.is_positive[index], dtype=torch.bool),
        }


def _records_to_json(records: Sequence[SourceRecord], data_root: Path) -> list[dict[str, str]]:
    """用相对 data root 的路径保存划分，便于目录整体移动后继续使用。"""
    return [
        {"source_id": record.source_id, "relative_path": str(record.path.relative_to(data_root))}
        for record in records
    ]


def _records_from_json(items: Sequence[dict[str, str]], data_root: Path) -> tuple[SourceRecord, ...]:
    """从持久化划分恢复数据路径，不逐个 resolve 触发共享文件系统查询。"""
    return tuple(
        SourceRecord(source_id=item["source_id"], path=data_root / item["relative_path"])
        for item in items
    )


def _expected_split_counts(total: int, config: SimpleCNNConfig) -> tuple[int, int, int]:
    """返回与新建 split 完全相同的 train/val/test 数量。"""
    train_count = int(total * config.train_fraction)
    val_count = int(total * config.val_fraction)
    return train_count, val_count, total - train_count - val_count


def _validate_split_manifest(content: dict[str, Any], config: SimpleCNNConfig, split_path: Path) -> None:
    """拒绝与当前划分配置不一致或结构损坏的历史 manifest。"""
    splits = content.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(f"split manifest 缺少 splits 对象：{split_path}")
    try:
        schema_version = int(content.get("schema_version", 1))
    except (TypeError, ValueError) as error:
        raise ValueError(f"split manifest 的 schema_version 无效：{split_path}") from error
    if schema_version not in {1, 2}:
        raise ValueError(f"不支持的 split manifest schema_version={schema_version}：{split_path}")
    expected_split_names = {"train", "val", "test"}
    if set(splits) != expected_split_names:
        raise ValueError(f"split manifest 必须且只能包含 train/val/test：{split_path}")
    if any(not isinstance(splits[name], list) for name in expected_split_names):
        raise ValueError(f"split manifest 的 train/val/test 必须为列表：{split_path}")
    try:
        split_lengths = tuple(len(splits[name]) for name in ("train", "val", "test"))
    except (KeyError, TypeError) as error:
        raise ValueError(f"split manifest 缺少 train/val/test 列表：{split_path}") from error
    selected_count = sum(split_lengths)
    problems: list[str] = []
    source_ids: set[str] = set()
    relative_paths: set[str] = set()
    for split_name in ("train", "val", "test"):
        for item_index, item in enumerate(splits[split_name]):
            location = f"{split_name}[{item_index}]"
            if not isinstance(item, dict):
                raise ValueError(f"split manifest 的 {location} 不是对象：{split_path}")
            source_id = item.get("source_id")
            relative_path = item.get("relative_path")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"split manifest 的 {location}.source_id 无效：{split_path}")
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(f"split manifest 的 {location}.relative_path 无效：{split_path}")
            relative = Path(relative_path)
            if relative.is_absolute() or ".." in relative.parts or relative_path == ".":
                raise ValueError(f"split manifest 的 {location}.relative_path 必须为安全相对路径：{split_path}")
            if source_id in source_ids:
                raise ValueError(f"split manifest 中 source_id={source_id!r} 重复或跨 split 泄漏：{split_path}")
            if relative_path in relative_paths:
                raise ValueError(f"split manifest 中 relative_path={relative_path!r} 重复或跨 split 泄漏：{split_path}")
            source_ids.add(source_id)
            relative_paths.add(relative_path)
    if int(content.get("selected_source_count", selected_count)) != selected_count:
        problems.append("selected_source_count 与 splits 实际数量不一致")
    if int(content.get("split_seed", config.split_seed)) != config.split_seed:
        problems.append(f"split_seed 不是当前值 {config.split_seed}")
    if int(content.get("source_sample_limit", config.source_sample_limit)) != config.source_sample_limit:
        problems.append(f"source_sample_limit 不是当前值 {config.source_sample_limit}")
    stored_fractions = content.get("fractions")
    if stored_fractions is None and schema_version >= 2:
        problems.append("schema v2 缺少 train/val/test fractions")
    elif stored_fractions is not None:
        expected_fractions = {
            "train": config.train_fraction,
            "val": config.val_fraction,
            "test": config.test_fraction,
        }
        try:
            fractions_match = all(
                float(stored_fractions[name]) == expected
                for name, expected in expected_fractions.items()
            )
        except (KeyError, TypeError, ValueError):
            fractions_match = False
        if not fractions_match:
            problems.append("train/val/test fraction 与当前配置不一致")
    expected_lengths = _expected_split_counts(selected_count, config)
    if split_lengths != expected_lengths:
        problems.append(f"划分数量为 {split_lengths}，当前配置期望 {expected_lengths}")
    if problems:
        details = "；".join(problems)
        raise ValueError(f"已有 split manifest 与当前配置不兼容：{details}。文件：{split_path}")


def _atomic_write_json(path: Path, content: dict[str, Any]) -> None:
    """同目录临时写入并原子替换 JSON，避免中断留下半份 manifest。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(content, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _build_or_load_split(config: SimpleCNNConfig, split_path: Path) -> dict[str, tuple[SourceRecord, ...]]:
    """按完整序列创建或读取可复现的 train/val/test 划分。

    新建划分时，按文件系统返回的目录顺序取前 ``source_sample_limit`` 份
    序列，再随机划分；已落盘的划分始终直接复用，避免验证集发生漂移。
    """
    if split_path.exists():
        content = json.loads(split_path.read_text(encoding="utf-8"))
        _validate_split_manifest(content, config, split_path)
        return {
            name: _records_from_json(content["splits"][name], config.data_root)
            for name in ("train", "val", "test")
        }

    records = discover_sources(config.data_root, config.source_sample_limit)
    rng = np.random.default_rng(config.split_seed)
    order = rng.permutation(len(records))
    shuffled = [records[int(index)] for index in order]
    train_count, val_count, _ = _expected_split_counts(len(records), config)
    if train_count < 1 or val_count < 1 or len(records) - train_count - val_count < 1:
        raise ValueError("完整序列数量不足以构造非空 train/val/test 划分。")
    splits = {
        "train": tuple(shuffled[:train_count]),
        "val": tuple(shuffled[train_count : train_count + val_count]),
        "test": tuple(shuffled[train_count + val_count :]),
    }
    split_content = {
        "schema_version": 2,
        "data_root": str(config.data_root),
        "split_seed": config.split_seed,
        "source_sample_limit": config.source_sample_limit,
        "selected_source_count": len(records),
        "fractions": {
            "train": config.train_fraction,
            "val": config.val_fraction,
            "test": config.test_fraction,
        },
        "splits": {name: _records_to_json(items, config.data_root) for name, items in splits.items()},
    }
    _atomic_write_json(split_path, split_content)
    return splits


def _build_grid_manifest(
    records: Sequence[SourceRecord],
    config: SimpleCNNConfig,
    output_path: Path,
) -> None:
    """批量构建固定网格标签，不读取观测矩阵或逐窗口创建 Python 对象。"""
    distance_starts = np.asarray(standard_distance_starts(config), dtype=np.int32)
    distance_count = len(distance_starts)
    frames = config.frames_per_window
    label_sources = tuple(LabelSource(record, config) for record in records)
    source_sample_counts = [
        len(standard_time_starts(source.frames, config)) * distance_count
        for source in label_sources
    ]
    total_sample_count = sum(source_sample_counts)
    fields: dict[str, np.ndarray] = {
        "source_index": np.empty(total_sample_count, dtype=np.int32),
        "time_start": np.empty(total_sample_count, dtype=np.int32),
        "range_start": np.empty(total_sample_count, dtype=np.int32),
        "response_mask": np.empty((total_sample_count, frames), dtype=np.bool_),
        "response_bin": np.empty((total_sample_count, frames), dtype=np.int16),
        "is_positive": np.empty(total_sample_count, dtype=np.bool_),
        "q_valid": np.empty(total_sample_count, dtype=np.bool_),
    }
    write_offset = 0

    for source_index, (record, source) in enumerate(
        zip(records, label_sources, strict=True)
    ):
        time_starts = np.asarray(standard_time_starts(source.frames, config), dtype=np.int32)
        time_count = len(time_starts)
        sample_count = time_count * distance_count
        if sample_count != source_sample_counts[source_index]:
            raise RuntimeError(f"构建 grid 期间 source 帧数发生变化：{record.path}")
        sample_slice = slice(write_offset, write_offset + sample_count)

        trajectory_windows = np.lib.stride_tricks.sliding_window_view(
            source.target_true_range_m, frames
        )[time_starts]
        trajectory_min = trajectory_windows.min(axis=1, keepdims=True)
        trajectory_max = trajectory_windows.max(axis=1, keepdims=True)
        range_start_grid = distance_starts.reshape(1, -1)
        range_stop_grid = range_start_grid + config.block_width_m
        is_full = (trajectory_min >= range_start_grid) & (trajectory_max < range_stop_grid)

        hit_windows = np.lib.stride_tricks.sliding_window_view(source.target_hit, frames)[time_starts]
        hit_bin_windows = np.lib.stride_tricks.sliding_window_view(source.target_hit_bin, frames)[time_starts]
        inside = (
            hit_windows[:, :, None]
            & (hit_bin_windows[:, :, None] >= range_start_grid[:, None, :])
            & (hit_bin_windows[:, :, None] < range_stop_grid[:, None, :])
        )
        response_mask = inside.transpose(0, 2, 1).reshape(sample_count, frames)
        response_bin = (
            np.where(
                inside,
                hit_bin_windows[:, :, None] - range_start_grid[:, None, :],
                -1,
            )
            .transpose(0, 2, 1)
            .reshape(sample_count, frames)
            .astype(np.int16, copy=False)
        )
        hit_count = response_mask.sum(axis=1)
        hit_count_grid = hit_count.reshape(time_count, distance_count)
        valid_line = (is_full & (hit_count_grid > 0)).reshape(-1)
        q_valid = ~(is_full & (hit_count_grid == 0)).reshape(-1)

        fields["source_index"][sample_slice] = source_index
        fields["time_start"][sample_slice] = np.repeat(time_starts, distance_count)
        fields["range_start"][sample_slice] = np.tile(distance_starts, time_count)
        fields["response_mask"][sample_slice] = response_mask
        fields["response_bin"][sample_slice] = response_bin
        fields["is_positive"][sample_slice] = valid_line
        fields["q_valid"][sample_slice] = q_valid
        write_offset += sample_count

    source_signatures = np.asarray(_record_file_signatures(records), dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(_GRID_CACHE_SCHEMA_VERSION, dtype=np.int16),
        source_ids=np.asarray([record.source_id for record in records]),
        q_valid=fields["q_valid"],
        source_relative_paths=np.asarray(_record_relative_paths(records, config.data_root)),
        source_sizes=source_signatures[:, 0],
        source_mtime_ns=source_signatures[:, 1],
        source_index=fields["source_index"],
        time_start=fields["time_start"],
        range_start=fields["range_start"],
        response_mask=fields["response_mask"],
        response_bin=fields["response_bin"],
        is_positive=fields["is_positive"],
    )


def _record_relative_paths(records: Sequence[SourceRecord], data_root: Path) -> tuple[str, ...]:
    """不访问文件系统地返回稳定相对路径。"""
    root = Path(os.path.abspath(data_root))
    relative_paths: list[str] = []
    for record in records:
        try:
            record_path = Path(os.path.abspath(record.path))
            relative_paths.append(str(record_path.relative_to(root)))
        except ValueError as error:
            raise ValueError(f"序列 {record.path} 不位于 data_root={root} 内。") from error
    return tuple(relative_paths)


def _record_file_signatures(records: Sequence[SourceRecord]) -> tuple[tuple[int, int], ...]:
    """返回会随源文件内容重写而变化的大小和纳秒 mtime。"""
    signatures: list[tuple[int, int]] = []
    for record in records:
        stat = record.path.stat()
        signatures.append((stat.st_size, stat.st_mtime_ns))
    return tuple(signatures)


def _grid_cache_path(
    split_name: str,
    records: Sequence[SourceRecord],
    config: SimpleCNNConfig,
) -> Path:
    """由全部会改变网格划分的设置和 source 身份生成缓存路径。"""
    if split_name not in {"val", "test", "eval"}:
        raise ValueError(f"grid 缓存 split 应为 val、test 或 eval，实际为 {split_name!r}。")

    relative_paths = _record_relative_paths(records, config.data_root)
    signatures = _record_file_signatures(records)
    fingerprint_payload = {
        "schema_version": _GRID_CACHE_SCHEMA_VERSION,
        "split_name": split_name,
        "source_sample_limit": config.source_sample_limit,
        "split": {
            "seed": config.split_seed,
            "train_fraction": config.train_fraction,
            "val_fraction": config.val_fraction,
            "test_fraction": config.test_fraction,
        },
        "grid": {
            "frames_per_window": config.frames_per_window,
            "range_bins": config.range_bins,
            "block_width_m": config.block_width_m,
            "spatial_step_m": config.spatial_step_m,
            "validation_time_stride": config.validation_time_stride,
        },
        "sources": [
            {
                "source_id": record.source_id,
                "relative_path": relative_path,
                "size": signature[0],
                "mtime_ns": signature[1],
            }
            for record, relative_path, signature in zip(records, relative_paths, signatures, strict=True)
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    source_limit = "all" if config.source_sample_limit == 0 else str(config.source_sample_limit)
    split_fractions = "-".join(
        f"{value:g}".replace(".", "p")
        for value in (config.train_fraction, config.val_fraction, config.test_fraction)
    )
    filename = (
        f"v{_GRID_CACHE_SCHEMA_VERSION}__{split_name}__source-total-{source_limit}"
        f"__grid-sources-{len(records)}__split-{config.split_seed}-{split_fractions}"
        f"__frames-{config.frames_per_window}__time-step-{config.validation_time_stride}"
        f"__range-{config.range_bins}__block-{config.block_width_m}"
        f"__space-step-{config.spatial_step_m}__{fingerprint}.npz"
    )
    return GRID_CACHE_DIR / filename


def _is_valid_grid_cache(
    cache_path: Path,
    records: Sequence[SourceRecord],
    config: SimpleCNNConfig,
) -> bool:
    """快速检查缓存是否完整且对应本次 source 集合，不解压大标签数组。"""
    if not cache_path.is_file():
        return False
    try:
        with zipfile.ZipFile(cache_path) as compressed_archive:
            if not _grid_cache_headers_are_valid(compressed_archive, config):
                return False
        with np.load(cache_path, allow_pickle=False) as archive:
            if not _GRID_CACHE_REQUIRED_FIELDS.issubset(archive.files):
                return False
            if int(archive["schema_version"].item()) != _GRID_CACHE_SCHEMA_VERSION:
                return False
            cached_ids = tuple(str(value) for value in archive["source_ids"].tolist())
            cached_paths = tuple(str(value) for value in archive["source_relative_paths"].tolist())
            cached_signatures = tuple(
                zip(archive["source_sizes"].tolist(), archive["source_mtime_ns"].tolist(), strict=True)
            )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False
    return (
        cached_ids == tuple(record.source_id for record in records)
        and cached_paths == _record_relative_paths(records, config.data_root)
        and cached_signatures == _record_file_signatures(records)
    )


def _build_or_load_grid_cache(
    split_name: str,
    records: Sequence[SourceRecord],
    config: SimpleCNNConfig,
) -> Path:
    """命中时复用网格缓存；未命中时原子构建，避免并发产生残缺文件。"""
    cache_path = _grid_cache_path(split_name, records, config)
    if _is_valid_grid_cache(cache_path, records, config):
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{cache_path.stem}.", suffix=".npz", dir=cache_path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        _build_grid_manifest(records, config, temporary_path)
        if not _is_valid_grid_cache(temporary_path, records, config):
            raise RuntimeError(f"生成的 grid 缓存不完整：{temporary_path}")
        os.replace(temporary_path, cache_path)
        cache_path.chmod(0o664)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return cache_path


def prepare_data_artifacts(
    config: SimpleCNNConfig,
    artifact_dir: Path,
    *,
    include_test_manifest: bool = True,
) -> DataArtifacts:
    """创建 source 划分和验证 cache；测试 cache 可按需延迟构建。"""
    # 创建实验的数据目录和数据划分
    artifact_dir.mkdir(parents=True, exist_ok=True)
    split_path = artifact_dir / "split_manifest.json"
    splits = _build_or_load_split(config, split_path)

    validation_manifest_path = _build_or_load_grid_cache("val", splits["val"], config)
    test_manifest_path = (
        _build_or_load_grid_cache("test", splits["test"], config)
        if include_test_manifest
        else None
    )

    # 返回的数据划分和val/test网格路径
    return DataArtifacts(
        split_path=split_path,
        validation_manifest_path=validation_manifest_path,
        test_manifest_path=test_manifest_path,
        train_sources=splits["train"],
        validation_sources=splits["val"],
        test_sources=splits["test"],
    )


def prepare_evaluation_manifest(
    config: SimpleCNNConfig,
    artifact_dir: Path,
    split_name: str,
) -> Path:
    """按需准备独立评估所需的单个验证或测试网格缓存。"""
    if split_name not in {"val", "test"}:
        raise ValueError(f"评估 split 应为 val 或 test，实际为 {split_name!r}。")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    splits = _build_or_load_split(config, artifact_dir / "split_manifest.json")
    return _build_or_load_grid_cache(split_name, splits[split_name], config)


def prepare_fixed_evaluation_manifest(config: SimpleCNNConfig) -> Path:
    """为 ``config.data_root`` 下全部有效序列创建或复用完整评估网格。

    与 ``prepare_evaluation_manifest`` 不同，本函数不读取、创建或依赖
    train/val/test 划分：数据根目录下每个含 ``data.npz`` 的一级子目录都会
    进入评估。调用方应将 ``max_eval_batch_num`` 设为 0，以完整遍历该网格。
    """
    records = discover_sources(config.data_root, source_sample_limit=0)
    return _build_or_load_grid_cache("eval", records, config)


def _split_records_for_rank(
    records: Sequence[SourceRecord], rank: int, world_size: int
) -> tuple[SourceRecord, ...]:
    """把训练 source 在 rank 间无重叠地划分；来源过少时显式报错。"""
    rank_records = tuple(records[rank::world_size])
    if not rank_records:
        raise RuntimeError(
            f"world_size={world_size} 大于可划分的训练序列数={len(records)}，rank {rank} 无数据。"
        )
    return rank_records


def build_train_dataloader(
    config: SimpleCNNConfig,
    records: Sequence[SourceRecord],
    *,
    rank: int,
    world_size: int,
    stream_generation: int = 0,
) -> DataLoader[dict[str, torch.Tensor]]:
    """构造每个 rank 的无限平衡 batch 数据流。"""
    rank_records = _split_records_for_rank(records, rank, world_size)
    if config.num_workers > len(rank_records):
        raise ValueError(
            f"num_workers={config.num_workers} 大于 rank {rank} 可用训练 source 数="
            f"{len(rank_records)}；请减小 num_workers 或增加训练 source。"
        )
    dataset = BalancedTrainBatchIterableDataset(
        rank_records,
        config,
        rank=rank,
        stream_generation=stream_generation,
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": None,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }
    if config.num_workers > 0:
        kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
            "worker_init_fn": _initialize_train_worker_process,
        })
    return DataLoader(**kwargs)


def build_validation_dataloader(
    config: SimpleCNNConfig,
    manifest_path: Path,
    *,
    rank: int,
    world_size: int,
    max_eval_batches: int | None = None,
    dataset: StandardGridDataset | None = None,
    num_workers: int | None = None,
    without_replacement: bool = False,
) -> DataLoader[dict[str, torch.Tensor]]:
    """构造固定标准网格验证/测试数据加载器，不填充 DDP 尾部样本。

    ``without_replacement`` 仅在 ``max_eval_batches>0`` 时生效：训练中的限量
    validation 继续沿用有放回采样；独立 eval 可显式启用无放回子集采样。
    """
    if dataset is None:
        verify_cached_samples = rank == 0 and manifest_path.parent.resolve() == GRID_CACHE_DIR.resolve()
        dataset = StandardGridDataset(
            manifest_path,
            config,
            verify_cached_samples=verify_cached_samples,
        )
    effective_max_batches = config.max_eval_batch_num if max_eval_batches is None else int(max_eval_batches)
    if effective_max_batches < 0:
        raise ValueError("max_eval_batches 不得为负。")
    effective_num_workers = config.num_workers if num_workers is None else int(num_workers)
    if effective_num_workers < 0:
        raise ValueError("num_workers 不得为负。")
    if effective_max_batches == 0:
        sampler: Sampler[int] = DistributedSourceSampler(
            dataset.source_spans,
            rank=rank,
            world_size=world_size,
        )
    elif without_replacement:
        sampler = DistributedSourceSubsetSampler(
            dataset.source_spans,
            effective_max_batches,
            config.eval_batch_size_per_gpu,
            rank=rank,
            world_size=world_size,
            seed=config.seed,
        )
    else:
        sampler = DistributedSourceReplacementSampler(
            dataset.source_spans,
            effective_max_batches,
            config.eval_batch_size_per_gpu,
            rank=rank,
            world_size=world_size,
            seed=config.seed,
        )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.eval_batch_size_per_gpu,
        "sampler": sampler,
        "num_workers": effective_num_workers,
        "pin_memory": config.pin_memory,
        "drop_last": False,
    }
    if effective_num_workers > 0:
        kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
            "worker_init_fn": _initialize_eval_worker_process,
        })
    return DataLoader(**kwargs)
