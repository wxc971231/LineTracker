"""统一推理入口参数与 JSON 落盘约定的回归测试。"""

from __future__ import annotations

import json
import logging
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import numpy as np

METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from infer.common.complexity import ModelComplexity
from infer.common.output import checkpoint_tag, dataset_tag, to_jsonable
from infer.common.plotting import _first_output_frame

from infer.run_infer import (
    _adaptive_summary,
    _compact_step_log,
    _frame_timing_records,
    _method_output_directory_name,
    _sample_compute_summary,
    _write_sample,
    _trajectory_summary,
    _select_records,
    _static_model_complexity,
    build_parser,
)
from infer.adaptive_tracker.infer import AdaptiveInferenceConfig
from infer.global_top1.infer import GlobalTop1Config


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
        self.assertEqual(enabled.capture_q_min, 0.5)
        self.assertEqual(enabled.instant_speed_gates_mpf, "20,25,30")
        self.assertEqual(enabled.average_speed_gates_mpf, "17,25,34")
        self.assertEqual(enabled.speed_average_window_frames, 10)

    def test_select_records_uses_sorted_data_root_prefix(self) -> None:
        with TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            for source_id in ("dataset1_synthetic_0010", "dataset1_synthetic_0002", "dataset1_synthetic_0000"):
                (data_root / source_id).mkdir()
            bundle: Any = SimpleNamespace(config=SimpleNamespace(data_root=data_root))  # 最小推理 bundle 替身。
            args: Any = SimpleNamespace(samples_start=0, samples_stop=2)  # 仅覆盖该函数读取的参数。
            records = _select_records(bundle, args)
        self.assertEqual([record.source_id for record in records], ["dataset1_synthetic_0000", "dataset1_synthetic_0002"])

    def test_select_records_rejects_empty_or_reversed_ranges(self) -> None:
        with TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            (data_root / "dataset1_synthetic_0000").mkdir()
            bundle: Any = SimpleNamespace(config=SimpleNamespace(data_root=data_root))

            with self.assertRaisesRegex(ValueError, "不能小于"):
                _select_records(bundle, SimpleNamespace(sample_start=1, sample_stop=0))
            with self.assertRaisesRegex(ValueError, "范围为空"):
                _select_records(bundle, SimpleNamespace(sample_start=1, sample_stop=None))

    def test_first_output_frame_follows_actual_prediction_not_window_size(self) -> None:
        self.assertEqual(_first_output_frame(np.asarray([np.nan, np.nan, 1.0])), 2)
        self.assertEqual(_first_output_frame(np.asarray([np.nan, np.nan])), 2)

    def test_output_tags_are_short_and_include_model_and_step(self) -> None:
        data_tag = dataset_tag(Path("F300_N10000_S42_B-random_T-R10000-290000m"))
        checkpoint = Path(
            "/tmp/runs/simplecnn_v2/limit50k-gbs1024-lr5e-4-pos25-vs5-modelxn-cfg123/"
            "20260728_032220/checkpoints/best.pt"
        )
        self.assertEqual(data_tag, "F300-N10k-S42")
        self.assertEqual(checkpoint_tag(checkpoint, 41980), "v2-xn-best-s41980")

    def test_method_output_directory_name_is_executor_independent(self) -> None:
        self.assertEqual(
            _method_output_directory_name(
                AdaptiveInferenceConfig(time_stride=5, capture_q_min=0.0, q_keep=0.5)
            ),
            "adaptive_tracker_stride5_captureq0_trackq0.5",
        )
        self.assertEqual(
            _method_output_directory_name(GlobalTop1Config(time_stride=5)),
            "global_top1_stride5",
        )

    def test_sample_metrics_record_the_full_parameter_method_directory(self) -> None:
        with TemporaryDirectory() as temporary:
            method_dir = Path(temporary) / "adaptive_tracker_stride5_captureq0.5_trackq0.5"
            method_dir.mkdir()
            source: Any = SimpleNamespace(record=SimpleNamespace(source_id="dataset1_synthetic_0000"))
            result = {"steps": []}
            _write_sample(
                method_dir,
                method="adaptive_tracker",
                result=result,
                source=source,
                trajectory={},
                compute={},
                timing={},
                tracker={},
                save_figures=False,
                figure_dpi=100,
                config=SimpleNamespace(),
                time_stride=5,
                title="test",
                logger=logging.getLogger(__name__),
            )
            payload = json.loads((method_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["method_dir"], method_dir.name)

    def test_json_conversion_turns_nonfinite_scalars_and_arrays_into_null(self) -> None:
        value = to_jsonable(
            {
                "scalar": float("nan"),
                "array": np.asarray([1.0, np.inf, -np.inf]),
            }
        )
        self.assertIsNone(value["scalar"])
        self.assertEqual(value["array"], [1.0, None, None])

    def test_compact_global_log_keeps_only_candidate_decision(self) -> None:
        compact = _compact_step_log(
            "global_top1",
            [
                {
                    "frame": 19,
                    "candidate_q": 0.9,
                    "candidate_block_start_m": 117_000,
                    "candidate_range_m": 123_456.0,
                    "candidate_speed_mpf": -4.0,
                    "end_to_end_s": 0.01,
                }
            ],
        )
        self.assertEqual(
            compact,
            [
                {
                    "frame": 19,
                    "candidate_q": 0.9,
                    "candidate_block_start_m": 117_000,
                    "candidate_range_m": 123_456.0,
                    "candidate_speed_mpf": -4.0,
                }
            ],
        )

    def test_compact_adaptive_log_removes_internal_plotting_fields(self) -> None:
        compact = _compact_step_log(
            "adaptive_tracker",
            [
                {
                    "frame": 25,
                    "forecast_frame_start": 26,
                    "forecast_frame_stop": 31,
                    "mode_before": "CAPTURE",
                    "mode": "TRACK",
                    "candidate_accepted": True,
                    "measurement_updated": True,
                    "next_search_level": 0,
                    "rejected_by": ["q_keep"],
                    "candidate_q": 0.95,
                    "candidate_range_m": 123_000.0,
                    "candidate_block_start_m": 117_000,
                    "candidate_speed_mpf": -5.0,
                    "capture": {"buffer_size": 8, "support_count": 6, "confirmed": True},
                    "end_to_end_s": 0.1,
                }
            ],
        )
        self.assertNotIn("forecast_frame_start", compact[0])
        self.assertNotIn("forecast_frame_stop", compact[0])
        self.assertNotIn("end_to_end_s", compact[0])
        self.assertEqual(compact[0]["capture"], {"buffer_size": 8, "support_count": 6, "confirmed": True})
        self.assertEqual(compact[0]["rejected_by"], ["q_keep"])

    def test_frame_timing_records_identify_state_type(self) -> None:
        records = _frame_timing_records(
            "adaptive_tracker",
            {
                "steps": [
                    {"frame": 19, "mode_before": "CAPTURE", "end_to_end_s": 0.001},
                    {"frame": 24, "mode_before": "TRACK", "scan_level": 1, "end_to_end_s": 0.002},
                    {"frame": 29, "mode_before": "RECAPTURE", "end_to_end_s": 0.003},
                ]
            },
        )
        self.assertEqual(
            records,
            [
                {"frame": 19, "type": "CAPTURE", "ms": 1.0},
                {"frame": 24, "type": "Track-L1", "ms": 2.0},
                {"frame": 29, "type": "RECAPTURE", "ms": 3.0},
            ],
        )

    def test_trajectory_summary_marks_unreliable_coverage(self) -> None:
        source: Any = SimpleNamespace(  # 指标函数只读取潜在轨迹与目标命中掩码。
            target_true_range_m=np.asarray([0.0, 1.0, 2.0]),
            target_hit=np.asarray([True, True, True]),
        )
        summary = _trajectory_summary(
            {
                "prediction_m": np.asarray([0.0, 1.0, np.nan]),
                "unreliable_prediction_mask": np.asarray([False, True, False]),
            },
            source,
            jump_threshold_m=1_000.0,
        )
        self.assertAlmostEqual(summary["unreliable_coverage"], 1.0 / 3.0)

    def test_adaptive_summary_uses_compact_direct_step_records(self) -> None:
        summary = _adaptive_summary(
            {
                "steps": [
                    {"frame": 25, "mode_before": "CAPTURE", "capture": {"confirmed": True}},
                    {"frame": 30, "mode_before": "TRACK", "mode": "RECAPTURE"},
                    {"frame": 40, "mode_before": "RECAPTURE", "capture": {"confirmed": True}},
                ],
                "workload": {"capture_scans": 4, "local_scans": 5},
            },
            frames_per_window=20,
        )
        self.assertEqual(summary["capture_success"], 1)
        self.assertEqual(summary["first_capture_delay_frames"], 6.0)
        self.assertEqual(summary["recapture_success_count"], 1)
        self.assertEqual(summary["recapture_delay_frames_mean"], 10.0)
        self.assertEqual(summary["capture_scan_count"], 4)
        self.assertEqual(summary["local_scan_count"], 5)

    def test_sample_compute_keeps_actual_total_complexity(self) -> None:
        complexity = ModelComplexity(parameter_count=10, conv_linear_macs_per_block=100)
        compute = _sample_compute_summary(
            complexity,
            {
                "logical_steps": 3,
                "blocks_evaluated": 17,
                "forward_calls": 4,
            },
        )
        self.assertEqual(compute["estimated_conv_linear_macs_total"], 1_700)
        self.assertEqual(compute["estimated_conv_linear_flops_total"], 3_400)
        self.assertNotIn("end_to_end_total_s", compute)
        self.assertEqual(_static_model_complexity(complexity)["parameter_count"], 10)


if __name__ == "__main__":
    unittest.main()
