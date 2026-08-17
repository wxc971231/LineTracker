"""纯背景虚警评估入口的回归测试。"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = METHOD_ROOT.parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from infer.run_FA_eval import (
    _aggregate_false_alarm_summaries,
    _assert_pure_background,
    _source_false_alarm_summary,
    build_parser,
)
from infer.run_infer import adaptive_config_from_args


class FalseAlarmEvaluationTests(unittest.TestCase):
    def test_source_summary_excludes_warmup_and_groups_contiguous_alarm_frames(self) -> None:
        summary = _source_false_alarm_summary(
            {
                "prediction_m": np.asarray([10.0, 20.0, np.nan, 40.0, 50.0, np.nan]),
                "unreliable_prediction_mask": np.asarray([False, False, False, False, True, False]),
            },
            frames=6,
            frames_per_window=2,
            frame_interval_s=0.05,
        )

        self.assertEqual(summary["evaluable_frames"], 4)
        self.assertEqual(summary["false_alarm_frames"], 1)
        self.assertAlmostEqual(summary["false_alarm_frame_rate"], 0.25)
        self.assertEqual(summary["reliable_false_alarm_frames"], 1)
        self.assertEqual(summary["unreliable_false_alarm_frames"], 1)
        self.assertEqual(summary["false_alarm_event_count"], 1)
        self.assertEqual(summary["false_alarm_event_durations_frames"], [1])
        self.assertAlmostEqual(summary["false_alarm_event_duration_mean_s"], 0.05)
        self.assertEqual(summary["false_alarm_frames_including_unreliable"], 2)
        self.assertAlmostEqual(summary["false_alarm_frame_rate_including_unreliable"], 0.5)
        self.assertEqual(summary["false_alarm_event_durations_frames_including_unreliable"], [2])
        self.assertAlmostEqual(
            summary["false_alarm_event_duration_mean_s_including_unreliable"], 0.1
        )

    def test_source_summary_reports_zero_alarms(self) -> None:
        summary = _source_false_alarm_summary(
            {"prediction_m": np.full(5, np.nan)},
            frames=5,
            frames_per_window=2,
            frame_interval_s=0.05,
        )

        self.assertEqual(summary["evaluable_frames"], 3)
        self.assertEqual(summary["false_alarm_frames"], 0)
        self.assertEqual(summary["false_alarm_event_count"], 0)
        self.assertEqual(summary["false_alarm_event_durations_frames"], [])
        self.assertAlmostEqual(summary["false_alarm_frame_rate"], 0.0)

    def test_source_summary_counts_separated_reliable_alarm_events(self) -> None:
        summary = _source_false_alarm_summary(
            {
                "prediction_m": np.asarray(
                    [np.nan, np.nan, 10.0, 11.0, np.nan, 20.0, 21.0, np.nan]
                ),
                "unreliable_prediction_mask": np.zeros(8, dtype=bool),
            },
            frames=8,
            frames_per_window=2,
            frame_interval_s=0.05,
        )

        self.assertEqual(summary["false_alarm_frames"], 4)
        self.assertEqual(summary["false_alarm_event_count"], 2)
        self.assertEqual(summary["false_alarm_event_durations_frames"], [2, 2])
        self.assertEqual(summary["false_alarm_event_count_including_unreliable"], 2)

    def test_pure_background_validation_rejects_injected_target_hits(self) -> None:
        class Source:
            target_hit = np.asarray([False, True, False])
            record = type("Record", (), {"source_id": "sample_0001"})()

        with self.assertRaisesRegex(ValueError, "sample_0001"):
            _assert_pure_background(Source())

    def test_aggregate_summary_uses_frame_weighted_rate_and_sample_rate(self) -> None:
        aggregate = _aggregate_false_alarm_summaries(
            [
                {
                    "evaluable_frames": 4,
                    "false_alarm_frames": 1,
                    "reliable_false_alarm_frames": 1,
                    "unreliable_false_alarm_frames": 1,
                    "false_alarm_event_durations_frames": [1],
                    "false_alarm_frames_including_unreliable": 2,
                    "false_alarm_event_durations_frames_including_unreliable": [2],
                },
                {
                    "evaluable_frames": 6,
                    "false_alarm_frames": 0,
                    "reliable_false_alarm_frames": 0,
                    "unreliable_false_alarm_frames": 0,
                    "false_alarm_event_durations_frames": [],
                    "false_alarm_frames_including_unreliable": 0,
                    "false_alarm_event_durations_frames_including_unreliable": [],
                },
            ],
            frame_interval_s=0.05,
        )

        self.assertEqual(aggregate["sample_count"], 2)
        self.assertEqual(aggregate["evaluable_frames"], 10)
        self.assertEqual(aggregate["false_alarm_frames"], 1)
        self.assertAlmostEqual(aggregate["false_alarm_frame_rate"], 0.1)
        self.assertEqual(aggregate["samples_with_false_alarm"], 1)
        self.assertAlmostEqual(aggregate["sample_false_alarm_rate"], 0.5)
        self.assertEqual(aggregate["false_alarm_event_count"], 1)
        self.assertAlmostEqual(aggregate["false_alarm_event_duration_mean_s"], 0.05)
        self.assertEqual(aggregate["false_alarm_frames_including_unreliable"], 2)
        self.assertAlmostEqual(aggregate["false_alarm_frame_rate_including_unreliable"], 0.2)
        self.assertEqual(aggregate["samples_with_false_alarm_including_unreliable"], 1)
        self.assertAlmostEqual(
            aggregate["sample_false_alarm_rate_including_unreliable"], 0.5
        )
        self.assertAlmostEqual(
            aggregate["false_alarm_event_duration_mean_s_including_unreliable"], 0.1
        )

    def test_parser_defaults_to_pure_background_dataset_and_adaptive_threshold(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(
            args.data_root.name,
            "F300_N1000_S20260816_B-random_T-R10000-290000m-V340-A6-C10-J1-K0-Q0-0",
        )
        self.assertEqual(args.capture_q_min, 0.5)
        self.assertEqual(args.q_keep, 0.5)
        self.assertTrue(args.warmup)
        self.assertTrue(args.save_figures)

        method_config = adaptive_config_from_args(args)
        self.assertEqual(method_config.capture_q_min, 0.5)
        self.assertEqual(method_config.q_keep, 0.5)
        self.assertEqual(method_config.instant_speed_gate_mpf, (20.0, 25.0, 30.0))

        disabled = build_parser().parse_args(["--no-figures"])
        self.assertFalse(disabled.save_figures)

    def test_launch_configuration_includes_false_alarm_evaluation(self) -> None:
        launch_path = REPOSITORY_ROOT / ".vscode" / "launch.json"
        text = launch_path.read_text(encoding="utf-8")
        text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
        payload = json.loads(re.sub(r",\s*([}\]])", r"\1", text))
        configuration = next(
            item
            for item in payload["configurations"]
            if item["name"] == "SimpleCNN · 单卡纯背景虚警评估 · CUDA/NPU auto"
        )

        self.assertTrue(
            configuration["program"].endswith(
                "method/SimpleCNN/infer/run_FA_eval.py"
            )
        )
        self.assertEqual(configuration["args"], ["--device", "auto", "--no-warmup"])


if __name__ == "__main__":
    unittest.main()
