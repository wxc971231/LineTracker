"""SimpleCNN 关键正确性行为的轻量 CPU 回归测试。"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from datetime import timedelta
from unittest import mock



import torch
METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from configs.base import SimpleCNNConfig
from data import dataloader


from runtime.settings import RuntimeSettings
from train.train_ddp import _next_data_stream_generation
from utils.checkpoint import rotate_checkpoints
from utils.distributed import setup_distributed
from utils.seed import worker_seed
class EvaluationArtifactTests(unittest.TestCase):
    """验证独立评估通过统一 cache API 获取 manifest。"""

    def test_prepare_evaluation_manifest_builds_only_requested_split(self) -> None:
        config = SimpleCNNConfig()
        splits = {"train": (), "val": ("val-source",), "test": ("test-source",)}
        expected = Path("/tmp/test-grid.npz")
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            with (
                mock.patch.object(dataloader, "_build_or_load_split", return_value=splits),
                mock.patch.object(
                    dataloader,
                    "_build_or_load_grid_cache",
                    return_value=expected,
                ) as build_cache,
            ):
                result = dataloader.prepare_evaluation_manifest(config, artifact_dir, "test")

        self.assertEqual(result, expected)
        build_cache.assert_called_once_with("test", splits["test"], config)

    def test_prepare_evaluation_manifest_rejects_unknown_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "val 或 test"):
            dataloader.prepare_evaluation_manifest(
                SimpleCNNConfig(),
                Path("/tmp/unused"),
                "validation",
            )


class ResumeAndRuntimeTests(unittest.TestCase):
    """验证恢复数据流、checkpoint 轮转和分布式超时。"""

    def test_resume_advances_data_stream_generation_and_worker_seed(self) -> None:
        self.assertEqual(_next_data_stream_generation(None), 0)
        self.assertEqual(_next_data_stream_generation({}), 1)
        self.assertEqual(_next_data_stream_generation({"data_stream_generation": 3}), 4)
        self.assertNotEqual(worker_seed(42, 0, 0, 0), worker_seed(42, 0, 0, 1))


    def test_zero_checkpoint_retention_removes_all_step_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index in range(3):
                (directory / f"step_{index:07d}.pt").write_text("step", encoding="utf-8")
            (directory / "last.pt").write_text("last", encoding="utf-8")
            (directory / "best.pt").write_text("best", encoding="utf-8")
            rotate_checkpoints(directory, keep=0)
            self.assertEqual(list(directory.glob("step_*.pt")), [])
            self.assertTrue((directory / "last.pt").exists())
            self.assertTrue((directory / "best.pt").exists())


    def test_process_group_uses_configured_timeout(self) -> None:
        settings = RuntimeSettings(accelerator="cpu", distributed_timeout_minutes=7)
        distributed_env = {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}
        with (
            mock.patch.dict(os.environ, distributed_env, clear=False),
            mock.patch(
                "utils.distributed._resolve_device",
                return_value=(torch.device("cpu"), "gloo"),
            ),
            mock.patch("utils.distributed.dist.is_initialized", return_value=False),
            mock.patch("utils.distributed.dist.init_process_group") as initialize,
        ):
            context = setup_distributed(settings)

        self.assertEqual(context.world_size, 2)
        self.assertEqual(initialize.call_args.kwargs["timeout"], timedelta(minutes=7))


if __name__ == "__main__":
    unittest.main()
