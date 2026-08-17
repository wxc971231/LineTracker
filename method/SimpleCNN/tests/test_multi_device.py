"""多 NPU 推理调度工具的回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from infer.common.multi_device import (  # noqa: E402
    execute_device_tasks,
    parse_npu_devices,
    partition_round_robin,
    resolve_npu_devices,
)
from infer.run_FA_eval_parallel import build_parser as build_fa_parallel_parser  # noqa: E402
from infer.run_infer_parallel import build_parser as build_infer_parallel_parser  # noqa: E402


def _worker_identity(task: tuple[str, int]) -> tuple[str, int]:
    return task


class MultiDeviceTests(unittest.TestCase):
    def test_parse_npu_devices_normalizes_indices_and_rejects_duplicates(self) -> None:
        self.assertEqual(parse_npu_devices("0,npu:2, 5"), ("npu:0", "npu:2", "npu:5"))
        with self.assertRaisesRegex(ValueError, "重复"):
            parse_npu_devices("npu:0,0")
        with self.assertRaisesRegex(ValueError, "NPU"):
            parse_npu_devices("cuda:0")

    def test_partition_round_robin_keeps_each_item_on_exactly_one_device(self) -> None:
        assignments = partition_round_robin(
            ("sample_0", "sample_1", "sample_2", "sample_3", "sample_4"),
            ("npu:0", "npu:1", "npu:2"),
        )

        self.assertEqual(assignments, {
            "npu:0": ("sample_0", "sample_3"),
            "npu:1": ("sample_1", "sample_4"),
            "npu:2": ("sample_2",),
        })

    @patch.dict("os.environ", {"ASCEND_RT_VISIBLE_DEVICES": "6,7,8"}, clear=True)
    def test_resolve_npu_devices_uses_all_visible_devices(self) -> None:
        self.assertEqual(
            resolve_npu_devices(None),
            ("npu:0", "npu:1", "npu:2"),
        )

    def test_execute_device_tasks_returns_results_by_device(self) -> None:
        results = execute_device_tasks(
            {
                "npu:0": ("npu:0", 3),
                "npu:1": ("npu:1", 5),
            },
            _worker_identity,
        )
        self.assertEqual(results, {"npu:0": ("npu:0", 3), "npu:1": ("npu:1", 5)})

    def test_parallel_entry_parsers_accept_optional_npu_devices(self) -> None:
        for parser in (build_infer_parallel_parser(), build_fa_parallel_parser()):
            self.assertIsNone(parser.parse_args([]).devices)
            args = parser.parse_args(["--devices", "0,npu:2"])
            self.assertEqual(args.devices, "0,npu:2")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
