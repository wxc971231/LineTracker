"""固定评估网格 cache 与 source-aware sampler 的回归测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from configs.base import SimpleCNNConfig
from data import dataloader
from data.dataloader import (
    DistributedSourceReplacementSampler,
    DistributedSourceSampler,
    SourceRecord,
    StandardGridDataset,
    _build_grid_manifest,
    build_train_dataloader,
)


class SourceAwareSamplerTests(unittest.TestCase):
    """验证按 source 分卡以及有放回 batch 的数据局部性。"""

    def test_full_sampler_shards_whole_sources_without_overlap(self) -> None:
        spans = ((0, 3), (3, 5), (5, 9), (9, 10))
        rank_zero = list(DistributedSourceSampler(spans, rank=0, world_size=2))
        rank_one = list(DistributedSourceSampler(spans, rank=1, world_size=2))

        self.assertEqual(rank_zero, [0, 1, 2, 5, 6, 7, 8])
        self.assertEqual(rank_one, [3, 4, 9])
        self.assertEqual(set(rank_zero) | set(rank_one), set(range(10)))
        self.assertFalse(set(rank_zero) & set(rank_one))

    def test_replacement_sampler_keeps_each_batch_inside_one_source(self) -> None:
        spans = ((0, 5), (5, 12), (12, 20))
        batch_size = 4
        sampler = DistributedSourceReplacementSampler(
            spans,
            num_batches=6,
            batch_size=batch_size,
            rank=0,
            world_size=1,
            seed=42,
        )
        indices = list(sampler)

        self.assertEqual(len(indices), 6 * batch_size)
        second_evaluation_indices = list(sampler)
        self.assertNotEqual(indices, second_evaluation_indices)
        for offset in range(0, len(indices), batch_size):
            batch = indices[offset : offset + batch_size]
            self.assertTrue(any(all(start <= index < stop for index in batch) for start, stop in spans))
        for offset in range(0, len(second_evaluation_indices), batch_size):
            batch = second_evaluation_indices[offset : offset + batch_size]
            self.assertTrue(
                any(all(start <= index < stop for index in batch) for start, stop in spans)
            )


class GridCacheSchemaTests(unittest.TestCase):
    """验证 schema v2 保留训练所需标签且不保存冗余数组。"""

    def test_grid_manifest_uses_compact_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            source_dir = data_root / "source-0"
            source_dir.mkdir()
            frame_count = 25
            target_hit = np.zeros(frame_count, dtype=np.bool_)
            target_hit[[3, 8, 13, 18, 23]] = True
            target_hit_bin = np.full(frame_count, 1_000, dtype=np.int32)
            target_true_range_m = np.full(frame_count, 1_000.0, dtype=np.float32)
            np.savez_compressed(
                source_dir / "data.npz",
                target_hit=target_hit,
                target_hit_bin=target_hit_bin,
                target_true_range_m=target_true_range_m,
            )
            config = SimpleCNNConfig(
                data_root=data_root,
                range_bins=10_000,
                validation_time_stride=5,
            )
            manifest_path = data_root / "grid.npz"
            records = (SourceRecord("source-0", source_dir / "data.npz"),)

            with mock.patch.object(
                dataloader, "LabelSource", wraps=dataloader.LabelSource
            ) as label_source:
                _build_grid_manifest(records, config, manifest_path)
            self.assertEqual(label_source.call_count, 1)

            with np.load(manifest_path, allow_pickle=False) as archive:
                self.assertEqual(archive["response_bin"].dtype, np.dtype(np.int16))
                self.assertNotIn("q", archive.files)
                self.assertNotIn("m_line", archive.files)
                self.assertNotIn("track_relation", archive.files)
            # verify_cached_samples=True 会逐条调用标量 build_target_labels
            # 复算小型 manifest 的全部样本，防止向量化缓存逻辑漂移。
            dataset = StandardGridDataset(
                manifest_path, config, verify_cached_samples=True
            )
            self.assertEqual(dataset.source_spans, ((0, len(dataset)),))
            self.assertEqual(len(dataset), 2)
            self.assertTrue(
                dataloader._is_valid_grid_cache(manifest_path, records, config)
            )

            with np.load(manifest_path, allow_pickle=False) as archive:
                corrupted = {name: archive[name] for name in archive.files}
            corrupted["response_bin"] = corrupted["response_bin"][:, :-1]
            np.savez_compressed(manifest_path, **corrupted)
            self.assertFalse(
                dataloader._is_valid_grid_cache(manifest_path, records, config)
            )

    def test_grid_manifest_rejects_mismatched_array_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            manifest_path = data_root / "malformed.npz"
            np.savez_compressed(
                manifest_path,
                source_ids=np.asarray(["source-0"]),
                source_relative_paths=np.asarray(["source-0/data.npz"]),
                source_index=np.asarray([0], dtype=np.int32),
                time_start=np.asarray([0], dtype=np.int32),
                range_start=np.asarray([0], dtype=np.int32),
                response_mask=np.zeros((1, 20), dtype=np.bool_),
                response_bin=np.zeros((1, 19), dtype=np.int16),
                is_positive=np.zeros(1, dtype=np.bool_),
            )

            with self.assertRaisesRegex(ValueError, "结构无效"):
                StandardGridDataset(
                    manifest_path,
                    SimpleCNNConfig(data_root=data_root),
                )

    def test_train_loader_rejects_more_workers_than_rank_sources(self) -> None:
        config = SimpleCNNConfig(num_workers=2)
        records = (SourceRecord("source-0", Path("/unused/source-0/data.npz")),)

        with self.assertRaisesRegex(ValueError, "num_workers"):
            build_train_dataloader(
                config,
                records,
                rank=0,
                world_size=1,
            )


if __name__ == "__main__":
    unittest.main()
