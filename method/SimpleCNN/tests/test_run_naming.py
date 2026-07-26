"""实验目录摘要和配置哈希的回归测试。"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from configs.base import SimpleCNNConfig
from train.train_ddp import _resolve_run
from utils.run_naming import (
    algorithm_config_digest,
    build_experiment_slug,
    build_wandb_run_name,
)


class RunNamingTests(unittest.TestCase):
    """确保目录名可读、稳定，并能区分真实算法配置。"""

    def setUp(self) -> None:
        self.config = SimpleCNNConfig(
            run_name="simplecnn_v1",
            source_sample_limit=10_000,
            batch_size_per_gpu=64,
            gradient_accumulation_steps=1,
            learning_rate=3e-4,
            positive_fraction=0.25,
            validation_time_stride=5,
            eval_batch_size_per_gpu=256,
            max_eval_batch_num=128,
        )

    def test_current_eight_device_slug_contains_effective_values(self) -> None:
        slug = build_experiment_slug(self.config, world_size=8)

        self.assertRegex(
            slug,
            r"^limit10k-gbs512-lr3e-4-pos25-vs5-eval262144-cfg[0-9a-f]{8}$",
        )

    def test_world_size_and_full_eval_change_readable_summary(self) -> None:
        four_device = build_experiment_slug(self.config, world_size=4)
        full_eval = build_experiment_slug(
            replace(self.config, source_sample_limit=0, max_eval_batch_num=0),
            world_size=8,
        )

        self.assertIn("-gbs256-", four_device)
        self.assertIn("-eval131072-", four_device)
        self.assertIn("limitall-gbs512", full_eval)
        self.assertIn("-evalfull-", full_eval)

    def test_digest_ignores_machine_and_logging_only_fields(self) -> None:
        changed_runtime = replace(
            self.config,
            profile="another_profile",
            run_name="another_run",
            data_root=Path("/another/data"),
            output_root=Path("/another/output"),
            resume=Path("/another/last.pt"),
            pin_memory=False,
            log_interval_steps=999,
            checkpoint_interval_steps=777,
            keep_last_checkpoints=9,
            wandb_project="another-project",
            wandb_entity="another-entity",
            wandb_mode="disabled",
        )

        self.assertEqual(
            algorithm_config_digest(self.config),
            algorithm_config_digest(changed_runtime),
        )

    def test_digest_changes_with_algorithm_setting(self) -> None:
        changed_model = replace(self.config, hidden_dim=self.config.hidden_dim * 2)

        self.assertNotEqual(
            algorithm_config_digest(self.config),
            algorithm_config_digest(changed_model),
        )

    def test_new_run_directory_and_legacy_wandb_name(self) -> None:
        with mock.patch("train.train_ddp.datetime") as mocked_datetime:
            mocked_datetime.now.return_value.strftime.return_value = "20260726_153000"
            run_dir = _resolve_run(self.config, None, world_size=8)

        self.assertEqual(run_dir.name, "20260726_153000")
        self.assertEqual(run_dir.parent.name, build_experiment_slug(self.config, 8))
        self.assertEqual(run_dir.parent.parent.name, "simplecnn_v1")
        self.assertEqual(
            build_wandb_run_name(self.config, run_dir),
            f"simplecnn_v1-{run_dir.parent.name}-20260726_153000",
        )

        legacy_dir = Path("/runs/simplecnn_v1/20260726_120000")
        self.assertEqual(
            build_wandb_run_name(self.config, legacy_dir),
            "simplecnn_v1-20260726_120000",
        )
        checkpoint = legacy_dir / "checkpoints" / "last.pt"
        self.assertEqual(_resolve_run(self.config, checkpoint, 8), legacy_dir)


if __name__ == "__main__":
    unittest.main()
