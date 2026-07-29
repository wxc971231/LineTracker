"""统一推理入口参数与 JSON 落盘约定的回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from infer.common.output import checkpoint_tag, dataset_tag, to_jsonable
from infer.run_infer import _adaptive_summary, _select_records, build_parser


class InferEntryTests(unittest.TestCase):
    def test_boolean_flags_keep_internal_field_names(self) -> None:
        parser = build_parser()
        disabled = parser.parse_args(
            [
                "--method",
                "global_top1",
                "--checkpoint",
                "/tmp/example/checkpoints/best.pt",
                "--no-warmup",
                "--no-figures",
            ]
        )
        self.assertFalse(disabled.warmup)
        self.assertFalse(disabled.save_figures)

        enabled = parser.parse_args(
            [
                "--method",
                "adaptive_tracker",
                "--checkpoint",
                "/tmp/example/checkpoints/best.pt",
            ]
        )
        self.assertTrue(enabled.warmup)
        self.assertTrue(enabled.save_figures)

    def test_select_records_uses_sorted_data_root_prefix(self) -> None:
        with TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            for source_id in ("dataset1_synthetic_0010", "dataset1_synthetic_0002", "dataset1_synthetic_0000"):
                (data_root / source_id).mkdir()
            bundle = SimpleNamespace(config=SimpleNamespace(data_root=data_root))
            args = SimpleNamespace(num_samples=2)
            records, indices = _select_records(bundle, args)
        self.assertEqual([record.source_id for record in records], ["dataset1_synthetic_0000", "dataset1_synthetic_0002"])
        self.assertEqual(indices, {"dataset1_synthetic_0000": 0, "dataset1_synthetic_0002": 1})

    def test_output_tags_are_short_and_include_model_and_step(self) -> None:
        data_tag = dataset_tag(Path("F300_N10000_S42_B-random_T-R10000-290000m"))
        checkpoint = Path(
            "/tmp/runs/simplecnn_v2/limit50k-gbs1024-lr5e-4-pos25-vs5-modelxn-cfg123/"
            "20260728_032220/checkpoints/best.pt"
        )
        self.assertEqual(data_tag, "F300-N10k-S42")
        self.assertEqual(checkpoint_tag(checkpoint, 41980), "v2-xn-best-s41980")

    def test_json_conversion_turns_nonfinite_scalars_and_arrays_into_null(self) -> None:
        value = to_jsonable(
            {
                "scalar": float("nan"),
                "array": np.asarray([1.0, np.inf, -np.inf]),
            }
        )
        self.assertIsNone(value["scalar"])
        self.assertEqual(value["array"], [1.0, None, None])


    def test_no_recapture_is_not_reported_as_zero_success_rate(self) -> None:
        summary = _adaptive_summary(
            {
                "frames_per_window": 20,
                "steps": [],
                "workload": {"capture_scans": 0, "local_scans": 0},
            },
        )
if __name__ == "__main__":
    unittest.main()

