"""进程标题格式的轻量回归测试。"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from utils.process_title import (
    PROCESS_JOB_ID_ENV,
    PROCESS_LABEL_ENV,
    build_process_title,
    ensure_process_job_id,
    set_process_title,
)


class ProcessTitleTests(unittest.TestCase):
    def test_title_contains_role_label_job_rank_and_worker(self) -> None:
        title = build_process_title(
            "train-data",
            label="simplecnn v2",
            job_id="123",
            rank=5,
            worker_id=2,
        )
        self.assertEqual(title, "SimpleCNN-train-data|simplecnn-v2|j123|r5|w2")

    def test_worker_inherits_label_job_and_rank_from_environment(self) -> None:
        environment = {
            PROCESS_LABEL_ENV: "simplecnn_v2",
            PROCESS_JOB_ID_ENV: "456",
            "RANK": "3",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            title = build_process_title("eval-data", worker_id=1)
        self.assertEqual(title, "SimpleCNN-eval-data|simplecnn_v2|j456|r3|w1")

    def test_launcher_job_id_is_stable_and_exported(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            first = ensure_process_job_id()
            second = ensure_process_job_id()
            self.assertEqual(first, second)
            self.assertEqual(os.environ[PROCESS_JOB_ID_ENV], first)

    def test_setting_title_exports_sanitized_label(self) -> None:
        with (
            mock.patch.dict(os.environ, {PROCESS_JOB_ID_ENV: "789"}, clear=True),
            mock.patch("utils.process_title._apply_process_title") as apply_title,
        ):
            title = set_process_title("train", label="simplecnn v2", rank=0)
            self.assertEqual(os.environ[PROCESS_LABEL_ENV], "simplecnn-v2")
        apply_title.assert_called_once_with(title)


if __name__ == "__main__":
    unittest.main()
