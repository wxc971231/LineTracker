"""SimpleCNN 的完整数据接口：在线训练流、压缩位裁剪和固定验证网格。

训练侧直接从完整 ``data.npz`` 中局部解包，不落盘重叠的 20×10000 块。
验证侧按文档中的标准时间窗和 34 个标准距离块建立固定清单。
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler, get_worker_info

from configs.base import SimpleCNNConfig
from utils.seed import worker_seed


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
    test_manifest_path: Path
    train_sources: tuple[SourceRecord, ...]
    validation_sources: tuple[SourceRecord, ...]
    test_sources: tuple[SourceRecord, ...]


def discover_sources(data_root: Path) -> list[SourceRecord]:
    """发现数据根目录下所有 ``*/data.npz`` 完整序列。"""
    records = [
        SourceRecord(source_id=path.parent.name, path=path.resolve())
        for path in sorted(data_root.glob("*/data.npz"))
    ]
    if not records:
        raise FileNotFoundError(f"在 {data_root} 下没有找到 */data.npz。")
    duplicate_ids = {record.source_id for record in records if sum(item.source_id == record.source_id for item in records) > 1}
    if duplicate_ids:
        raise ValueError(f"source_id 不唯一：{sorted(duplicate_ids)}")
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

    def __init__(self, record: SourceRecord) -> None:
        self.record = record
        self._background_only_packed: np.ndarray | None = None
        with np.load(record.path, allow_pickle=False) as archive:
            missing = self.REQUIRED_FIELDS.difference(archive.files)
            if missing:
                raise KeyError(f"{record.path} 缺少字段：{sorted(missing)}")
            self.observation_packed = archive["observation_packed"].astype(np.uint8, copy=False)
            self.target_hit = archive["target_hit"].astype(np.bool_, copy=False)
            self.target_hit_bin = archive["target_hit_bin"].astype(np.int32, copy=False)
            self.target_true_range_m = archive["target_true_range_m"].astype(np.float32, copy=False)

        self.frames = int(self.observation_packed.shape[0])
        self.range_bins = int(self.observation_packed.shape[1] * 8)
        if not (
            self.target_hit.shape == self.target_hit_bin.shape == self.target_true_range_m.shape == (self.frames,)
        ):
            raise ValueError(f"{record.path} 的目标标签形状与帧数不一致。")

    def _load_background_only(self) -> np.ndarray:
        """仅在需要反事实负样本时延迟读取背景矩阵。"""
        if self._background_only_packed is None:
            with np.load(self.record.path, allow_pickle=False) as archive:
                if "background_only_packed" not in archive.files:
                    raise KeyError(f"{self.record.path} 不含 background_only_packed。")
                self._background_only_packed = archive["background_only_packed"].astype(np.uint8, copy=False)
        return self._background_only_packed

    def extract_window(
        self,
        time_start: int,
        range_start: int,
        config: SimpleCNNConfig,
        *,
        background_only: bool = False,
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

        packed = self._load_background_only() if background_only else self.observation_packed
        first_byte = range_start // 8
        bit_offset = range_start % 8
        byte_count = math.ceil((bit_offset + width) / 8)
        byte_window = packed[time_start : time_start + frames, first_byte : first_byte + byte_count]
        unpacked = np.unpackbits(byte_window, axis=1, bitorder=config.packed_bitorder)
        return unpacked[:, bit_offset : bit_offset + width].astype(np.uint8, copy=False)


def rearrange_distance_channels(window: np.ndarray, config: SimpleCNNConfig) -> np.ndarray:
    """把 ``[20,10000]`` 原始窗口无损重排为 ``[8,20,1250]``。"""
    expected_shape = (config.frames_per_window, config.block_width_m)
    if window.shape != expected_shape:
        raise ValueError(f"期望原始窗口形状 {expected_shape}，实际为 {window.shape}。")
    grouped = window.reshape(config.frames_per_window, -1, config.input_channels)
    return grouped.transpose(2, 0, 1).copy()


def _trajectory_relation(
    source: PackedSource,
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
    source: PackedSource,
    time_start: int,
    range_start: int,
    config: SimpleCNNConfig,
    *,
    force_background: bool = False,
) -> dict[str, Any]:
    """生成与文档一致的 $I_\tau,d_\tau,H,q^*,m_{line}$ 标签。"""
    frames = config.frames_per_window
    range_stop = range_start + config.block_width_m
    is_full, is_intersecting, _, _ = _trajectory_relation(source, time_start, range_start, config)

    response_mask = np.zeros(frames, dtype=np.bool_)
    response_bin = np.full(frames, -1, dtype=np.int64)
    if not force_background:
        hit = source.target_hit[time_start : time_start + frames]
        hit_bin = source.target_hit_bin[time_start : time_start + frames]
        inside = hit & (hit_bin >= range_start) & (hit_bin < range_stop)
        response_mask[inside] = True
        response_bin[inside] = hit_bin[inside].astype(np.int64) - int(range_start)

    hit_count = int(response_mask.sum())
    track_relation = 1 if is_full else (2 if is_intersecting else 0)
    return {
        "I": response_mask,
        "d": response_bin,
        "H": hit_count,
        "q": np.float32(hit_count / frames),
        "m_line": bool(is_full and hit_count > 0 and not force_background),
        "is_positive": bool(is_full and hit_count > 0 and not force_background),
        "track_relation": np.int8(track_relation),  # 0=背景，1=完整，2=部分相交
    }


def _sample_to_numpy(
    source: PackedSource,
    time_start: int,
    range_start: int,
    config: SimpleCNNConfig,
    *,
    background_only: bool = False,
) -> dict[str, Any]:
    """提取原始窗口、重排并附加标签，形成一个局部块样本。"""
    raw_window = source.extract_window(
        time_start,
        range_start,
        config,
        background_only=background_only,
    )
    labels = build_target_labels(
        source,
        time_start,
        range_start,
        config,
        force_background=background_only,
    )
    labels.update(
        {
            "x": rearrange_distance_channels(raw_window, config),
            "time_start": np.int32(time_start),
            "range_start": np.int32(range_start),
        }
    )
    return labels


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
    """每个 rank/worker 独立维护的随机序列缓存池和源数据循环。"""

    def __init__(self, records: Sequence[SourceRecord], config: SimpleCNNConfig, rng: np.random.Generator) -> None:
        if not records:
            raise ValueError("当前 rank/worker 没有可用训练序列。")
        self.records = tuple(records)
        self.config = config
        self.rng = rng
        self.source_cycle = 0
        self._order: list[int] = []
        self._cursor = 0
        self.entries: list[_CacheEntry] = []
        self._reshuffle_order()
        for _ in range(min(config.source_cache_size, len(self.records))):
            self.entries.append(_CacheEntry(PackedSource(self._next_record(set()))))

    def _reshuffle_order(self) -> None:
        self._order = self.rng.permutation(len(self.records)).tolist()
        self._cursor = 0
        self.source_cycle += 1

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
        self.entries[slot] = _CacheEntry(PackedSource(self._next_record(excluded)))


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
                config.counterfactual_negative_weight,
            ],
            dtype=np.float64,
        )
        self.negative_kinds = ("local", "same_time", "random", "counterfactual")
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
            sample = _sample_to_numpy(source, time_start, range_start, self.config)
            if not sample["is_positive"]:
                continue
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
        sample = _sample_to_numpy(source, time_start, range_start, self.config)
        # 不相交块不应含目标；此处强制标签为零以保证训练定义明确。
        sample["I"] = np.zeros(self.config.frames_per_window, dtype=np.bool_)
        sample["d"] = np.full(self.config.frames_per_window, -1, dtype=np.int64)
        sample["H"] = 0
        sample["q"] = np.float32(0.0)
        sample["m_line"] = False
        sample["is_positive"] = False
        return sample

    def sample_negative(self, context: _PositiveContext) -> dict[str, Any]:
        """按配置的混合策略生成一个负样本。"""
        for _ in range(512):
            kind = self.negative_kinds[
                int(self.rng.choice(len(self.negative_kinds), p=self.negative_probabilities))
            ]
            if kind == "counterfactual":
                return _sample_to_numpy(
                    context.source,
                    context.time_start,
                    context.range_start,
                    self.config,
                    background_only=True,
                )
            if kind == "local":
                sample = self._sample_normal_negative(context.source, context.time_start, local=True)
            elif kind == "same_time":
                sample = self._sample_normal_negative(context.source, context.time_start, local=False)
            else:
                source = self.pool.choose().source
                time_start = int(self.rng.integers(0, source.frames - self.config.frames_per_window + 1))
                sample = self._sample_normal_negative(source, time_start, local=False)
            if sample is not None:
                return sample
        raise RuntimeError("连续 512 次无法构造不相交负样本；请检查距离范围和保护带。")

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
            "m_line": torch.from_numpy(np.asarray([sample["m_line"] for sample in samples], dtype=np.bool_)),
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
    ) -> None:
        super().__init__()
        self.records = tuple(records)
        self.config = config
        self.rank = rank

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
            worker_seed(self.config.seed, self.rank, worker_id),
        )
        while True:
            yield factory.make_batch()


class DistributedNoPaddingSampler(Sampler[int]):
    """验证专用 DDP sampler：按 rank 步进分片，不填充、不重复样本。"""

    def __init__(self, length: int, rank: int, world_size: int) -> None:
        self.length = int(length)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self) -> int:
        return max(0, math.ceil((self.length - self.rank) / self.world_size))


class StandardGridDataset(Dataset[dict[str, torch.Tensor]]):
    """从固定验证/测试清单恢复标准网格块，不做任何随机裁剪。"""

    def __init__(self, manifest_path: Path, config: SimpleCNNConfig) -> None:
        self.config = config
        with np.load(manifest_path, allow_pickle=False) as archive:
            self.source_paths = tuple(Path(path) for path in archive["source_paths"].tolist())
            self.source_ids = tuple(str(item) for item in archive["source_ids"].tolist())
            self.source_index = archive["source_index"].astype(np.int32, copy=False)
            self.time_start = archive["time_start"].astype(np.int32, copy=False)
            self.range_start = archive["range_start"].astype(np.int32, copy=False)
            self.response_mask = archive["response_mask"].astype(np.bool_, copy=False)
            self.response_bin = archive["response_bin"].astype(np.int64, copy=False)
            self.q = archive["q"].astype(np.float32, copy=False)
            self.m_line = archive["m_line"].astype(np.bool_, copy=False)
            self.is_positive = archive["is_positive"].astype(np.bool_, copy=False)
            self.track_relation = archive["track_relation"].astype(np.int8, copy=False)
        self._cache: OrderedDict[int, PackedSource] = OrderedDict()

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
        source = PackedSource(record)
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
            "q": torch.tensor(self.q[index], dtype=torch.float32),
            "m_line": torch.tensor(self.m_line[index], dtype=torch.bool),
            "is_positive": torch.tensor(self.is_positive[index], dtype=torch.bool),
            "track_relation": torch.tensor(self.track_relation[index], dtype=torch.int8),
            "source_index": torch.tensor(source_index, dtype=torch.int32),
            "time_start": torch.tensor(time_start, dtype=torch.int32),
            "range_start": torch.tensor(range_start, dtype=torch.int32),
        }


def _records_to_json(records: Sequence[SourceRecord], data_root: Path) -> list[dict[str, str]]:
    """用相对 data root 的路径保存划分，便于目录整体移动后继续使用。"""
    return [
        {"source_id": record.source_id, "relative_path": str(record.path.relative_to(data_root))}
        for record in records
    ]


def _records_from_json(items: Sequence[dict[str, str]], data_root: Path) -> tuple[SourceRecord, ...]:
    """从持久化划分恢复绝对数据路径。"""
    return tuple(
        SourceRecord(source_id=item["source_id"], path=(data_root / item["relative_path"]).resolve())
        for item in items
    )


def _build_or_load_split(config: SimpleCNNConfig, split_path: Path) -> dict[str, tuple[SourceRecord, ...]]:
    """按完整序列创建或读取可复现的 train/val/test 划分。

    新建划分时，先按路径稳定排序，并按 ``source_sample_limit`` 截取前 N
    份序列，再随机划分；已落盘的划分始终直接复用，避免验证集发生漂移。
    """
    if split_path.exists():
        content = json.loads(split_path.read_text(encoding="utf-8"))
        return {
            name: _records_from_json(content["splits"][name], config.data_root)
            for name in ("train", "val", "test")
        }

    records = discover_sources(config.data_root)
    if config.source_sample_limit > 0:
        records = records[: config.source_sample_limit]
    rng = np.random.default_rng(config.split_seed)
    order = rng.permutation(len(records))
    shuffled = [records[int(index)] for index in order]
    train_count = int(len(records) * config.train_fraction)
    val_count = int(len(records) * config.val_fraction)
    if train_count < 1 or val_count < 1 or len(records) - train_count - val_count < 1:
        raise ValueError("完整序列数量不足以构造非空 train/val/test 划分。")
    splits = {
        "train": tuple(shuffled[:train_count]),
        "val": tuple(shuffled[train_count : train_count + val_count]),
        "test": tuple(shuffled[train_count + val_count :]),
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(
            {
                "data_root": str(config.data_root),
                "split_seed": config.split_seed,
                "source_sample_limit": config.source_sample_limit,
                "selected_source_count": len(records),
                "splits": {name: _records_to_json(items, config.data_root) for name, items in splits.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return splits


def _build_grid_manifest(
    records: Sequence[SourceRecord],
    config: SimpleCNNConfig,
    output_path: Path,
) -> None:
    """针对 val 和 test 数据集，把固定的标准时间—距离网格的标签保存为轻量元数据清单。"""
    source_index_rows: list[int] = []       # 每个网格块所属完整序列在 records 中的索引。
    time_rows: list[int] = []               # 每个网格块的 20 帧时间窗起点（帧）。
    range_rows: list[int] = []              # 每个网格块的 10 km 距离窗起点（m/bin）。
    mask_rows: list[np.ndarray] = []        # 每块的 I_τ，形状 [20]，逐帧标记实际目标响应。
    bin_rows: list[np.ndarray] = []         # 每块的 d_τ，形状 [20]；逐帧目标响应位置，无响应帧填 -1。
    q_rows: list[np.float32] = []           # 每块的质量标签 q*=H/20。
    m_line_rows: list[bool] = []            # 每块是否参与逐帧几何 Huber 损失。
    positive_rows: list[bool] = []          # 每块是否完整包含潜在轨迹且至少有一次实际响应。
    relation_rows: list[np.int8] = []       # 潜在轨迹关系：0=不相交，1=完整包含，2=部分相交。
    distance_starts = standard_distance_starts(config)  # 推理一致的 34 个标准距离块起点。

    for source_index, record in enumerate(records):
        source = PackedSource(record)
        for time_start in standard_time_starts(source.frames, config):
            for range_start in distance_starts:
                labels = build_target_labels(source, time_start, range_start, config)
                source_index_rows.append(source_index)
                time_rows.append(time_start)
                range_rows.append(range_start)
                mask_rows.append(labels["I"])
                bin_rows.append(labels["d"])
                q_rows.append(labels["q"])
                m_line_rows.append(labels["m_line"])
                positive_rows.append(labels["is_positive"])
                relation_rows.append(labels["track_relation"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        source_ids=np.asarray([record.source_id for record in records]),
        source_paths=np.asarray([str(record.path) for record in records]),
        source_index=np.asarray(source_index_rows, dtype=np.int32),
        time_start=np.asarray(time_rows, dtype=np.int32),
        range_start=np.asarray(range_rows, dtype=np.int32),
        response_mask=np.stack(mask_rows).astype(np.bool_, copy=False),
        response_bin=np.stack(bin_rows).astype(np.int64, copy=False),
        q=np.asarray(q_rows, dtype=np.float32),
        m_line=np.asarray(m_line_rows, dtype=np.bool_),
        is_positive=np.asarray(positive_rows, dtype=np.bool_),
        track_relation=np.asarray(relation_rows, dtype=np.int8),
    )


def prepare_data_artifacts(config: SimpleCNNConfig, artifact_dir: Path) -> DataArtifacts:
    """创建一次实验需要的 source 划分和固定验证/测试网格清单。"""
    # 创建实验的数据目录和数据划分
    artifact_dir.mkdir(parents=True, exist_ok=True)
    split_path = artifact_dir / "split_manifest.json"
    splits = _build_or_load_split(config, split_path)

    # 首次运行时生成 val 与 test 数据网格，包括标准分块的位置和标签信息
    validation_manifest_path = artifact_dir / "validation_grid.npz"
    test_manifest_path = artifact_dir / "test_grid.npz"
    if not validation_manifest_path.exists():
        _build_grid_manifest(splits["val"], config, validation_manifest_path)
    if not test_manifest_path.exists():
        _build_grid_manifest(splits["test"], config, test_manifest_path)

    # 返回的数据划分和val/test网格路径
    return DataArtifacts(
        split_path=split_path,
        validation_manifest_path=validation_manifest_path,
        test_manifest_path=test_manifest_path,
        train_sources=splits["train"],
        validation_sources=splits["val"],
        test_sources=splits["test"],
    )


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
) -> DataLoader[dict[str, torch.Tensor]]:
    """构造每个 rank 的无限平衡 batch 数据流。"""
    dataset = BalancedTrainBatchIterableDataset(
        _split_records_for_rank(records, rank, world_size),
        config,
        rank=rank,
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": None,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }
    if config.num_workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(**kwargs)


def build_validation_dataloader(
    config: SimpleCNNConfig,
    manifest_path: Path,
    *,
    rank: int,
    world_size: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    """构造固定标准网格验证/测试数据加载器，不填充 DDP 尾部样本。"""
    dataset = StandardGridDataset(manifest_path, config)
    sampler = DistributedNoPaddingSampler(len(dataset), rank=rank, world_size=world_size)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.eval_batch_size_per_gpu,
        "sampler": sampler,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "drop_last": False,
    }
    if config.num_workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(**kwargs)
