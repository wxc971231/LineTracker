"""配置、split 复用和 cache 失效规则的回归测试。"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from configs.base import SimpleCNNConfig, load_config_json
from data import dataloader
from data.dataloader import SourceRecord


class ConfigValidationTests(unittest.TestCase):

    def test_default_config_is_valid(self) -> None:
        config = SimpleCNNConfig()
        config.validate()
        self.assertEqual(config.eval_batch_size_per_gpu, 64)

    def test_old_best_eval_field_is_ignored_when_loading_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolved_config.json"
            values = SimpleCNNConfig().as_serializable_dict()
            values["best_eval_batch_num"] = 32
            path.write_text(json.dumps(values), encoding="utf-8")

            loaded = load_config_json(path)

        self.assertFalse(hasattr(loaded, "best_eval_batch_num"))

    """确保已知会在运行中崩溃的配置能在启动前被拒绝。"""

    def test_invalid_values_are_rejected(self) -> None:
        invalid_cases = {
            "log_interval_steps": {"log_interval_steps": 0},
            "eval_interval_steps": {"eval_interval_steps": 0},
            "checkpoint_interval_steps": {"checkpoint_interval_steps": 0},
            "eval_batch_size_per_gpu": {"eval_batch_size_per_gpu": 0},
            "num_workers": {"num_workers": -1},
            "negative_sampling_weight": {"negative_local_weight": -0.1},
            "input_channels": {"input_channels": 4},
            "range_bins": {"range_bins": 5_000},
            "dropout": {"dropout": 1.0},
            "learning_rate": {"learning_rate": 0.0},
            "learning_rate_nan": {"learning_rate": float("nan")},
            "huber_delta_inf": {"huber_delta_bins": float("inf")},
            "negative_weight_nan": {"negative_random_weight": float("nan")},
        }
        for name, overrides in invalid_cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                SimpleCNNConfig(**overrides).validate()


class SplitManifestTests(unittest.TestCase):
    """确保历史 split 只在划分配置一致时复用。"""

    def test_changed_split_settings_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            data_root.mkdir()
            for index in range(10):
                (data_root / f"source-{index}").mkdir()
            split_path = root / "artifacts" / "split_manifest.json"
            config = SimpleCNNConfig(data_root=data_root, source_sample_limit=10)

            original = dataloader._build_or_load_split(config, split_path)
            reloaded = dataloader._build_or_load_split(config, split_path)
            self.assertEqual(
                tuple(record.source_id for record in original["val"]),
                tuple(record.source_id for record in reloaded["val"]),
            )
            incompatible = (
                SimpleCNNConfig(data_root=data_root, source_sample_limit=10, split_seed=43),
                SimpleCNNConfig(
                    data_root=data_root,
                    source_sample_limit=10,
                    train_fraction=0.7,
                    val_fraction=0.2,
                    test_fraction=0.1,
                ),
                # 取整后仍是 (8, 1, 1)，也必须因原始比例变化而拒绝。
                SimpleCNNConfig(
                    data_root=data_root,
                    source_sample_limit=10,
                    train_fraction=0.801,
                    val_fraction=0.101,
                    test_fraction=0.098,
                ),
                SimpleCNNConfig(data_root=data_root, source_sample_limit=0),
            )
            for changed in incompatible:
                with self.assertRaisesRegex(ValueError, "不兼容"):
                    dataloader._build_or_load_split(changed, split_path)


    def test_duplicate_source_across_splits_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            data_root.mkdir()
            for index in range(10):
                (data_root / f"source-{index}").mkdir()
            split_path = root / "artifacts" / "split_manifest.json"
            config = SimpleCNNConfig(data_root=data_root, source_sample_limit=10)
            dataloader._build_or_load_split(config, split_path)
            content = json.loads(split_path.read_text(encoding="utf-8"))
            content["splits"]["val"][0] = dict(content["splits"]["train"][0])
            split_path.write_text(json.dumps(content), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "重复|泄漏"):
                dataloader._build_or_load_split(config, split_path)


class CacheInvalidationTests(unittest.TestCase):
    """确保源文件重写和 lazy test 设置会正确影响 cache。"""

    def test_source_mtime_changes_cache_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            source_dir = data_root / "source-0"
            source_dir.mkdir()
            source_path = source_dir / "data.npz"
            np.savez_compressed(source_path, target_hit=np.zeros(20, dtype=np.bool_))
            record = SourceRecord("source-0", source_path)
            config = SimpleCNNConfig(data_root=data_root, source_sample_limit=1)
            first_path = dataloader._grid_cache_path("val", (record,), config)
            stat = source_path.stat()
            os.utime(source_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
            second_path = dataloader._grid_cache_path("val", (record,), config)

            self.assertNotEqual(first_path, second_path)
            changed_grid = SimpleCNNConfig(data_root=data_root, validation_time_stride=1)
            third_path = dataloader._grid_cache_path("val", (record,), changed_grid)
            self.assertNotEqual(second_path, third_path)

    def test_training_can_skip_test_manifest(self) -> None:
        config = SimpleCNNConfig()
        splits = {"train": (), "val": ("val-source",), "test": ("test-source",)}
        validation_path = Path("/tmp/validation-grid.npz")
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(dataloader, "_build_or_load_split", return_value=splits),
                mock.patch.object(
                    dataloader,
                    "_build_or_load_grid_cache",
                    return_value=validation_path,
                ) as build_cache,
            ):
                artifacts = dataloader.prepare_data_artifacts(
                    config,
                    Path(directory),
                    include_test_manifest=False,
                )

        self.assertEqual(artifacts.validation_manifest_path, validation_path)
        self.assertIsNone(artifacts.test_manifest_path)
        build_cache.assert_called_once_with("val", splits["val"], config)


if __name__ == "__main__":
    unittest.main()
