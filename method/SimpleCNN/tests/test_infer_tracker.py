"""纯 NumPy 自适应后处理状态机的回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from infer.adaptive_tracker.tracker import (  # noqa: E402
    AdaptiveTracker,
    Candidate,
    TrackerConfig,
    TrackerMode,
)


class AdaptiveTrackerTests(unittest.TestCase):
    """覆盖 postprocess.md 定义的捕获、跟踪、扩缩和重捕获路径。"""

    @staticmethod
    def _config(**overrides: object) -> TrackerConfig:
        values: dict[str, object] = {
            "capture_buffer_size": 3,
            "capture_support_ratio": 0.67,
            "capture_radius_m": 0.5,
            "capture_q_min": 0.5,
            "q_keep": 0.5,
            "instant_speed_gate_mpf": (10.0, 20.0, 30.0),
            "average_speed_gate_mpf": (10.0, 20.0, 30.0),
            "speed_average_window_frames": 20,
            "expand_after_bad": 1,
            "shrink_after_good": 2,
            "alpha": 0.5,
            "beta": 0.1,
            "gamma": 0.0,
            "block_width_m": 100.0,
            "block_step_m": 90.0,
            "range_min_m": 0.0,
            "range_max_m": 300.0,
            "frames_per_window": 20,
        }
        values.update(overrides)
        return TrackerConfig(**values)  # type: ignore[arg-type]

    @staticmethod
    def _candidate(q: float, latest_range_m: float, speed_m_per_frame: float = 2.0) -> Candidate:
        return Candidate(q, latest_range_m, speed_m_per_frame, block_start_m=0.0)

    def _capture(self, tracker: AdaptiveTracker) -> None:
        """在 t=4 捕获匀速 2 m/frame 的轨迹，当前距离应为 108 m。"""
        first = tracker.step(0, self._candidate(0.9, 100.0))
        self.assertEqual(first.mode, TrackerMode.CAPTURE)
        second = tracker.step(2, self._candidate(0.9, 104.0))
        self.assertEqual(second.mode, TrackerMode.CAPTURE)
        final = tracker.step(4, self._candidate(0.9, 108.0))
        self.assertEqual(final.mode, TrackerMode.TRACK)
        self.assertTrue(final.capture_confirmed)
        self.assertTrue(final.measurement_updated)
        self.assertEqual(final.capture_support_count, 3)
        self.assertAlmostEqual(final.range_current_m or 0.0, 108.0)
        self.assertAlmostEqual(final.speed_m_per_frame or 0.0, 2.0)

    def test_candidate_coordinate_conversion_uses_latest_window_frame(self) -> None:
        candidate = Candidate.from_block_prediction(
            q=0.8,
            rho_m=100.0,
            speed_m_per_frame=2.0,
            block_start_m=1_000.0,
            frames_per_window=20,
        )
        self.assertAlmostEqual(candidate.latest_range_m, 1_119.0)
        self.assertTrue(candidate.is_valid)

    def test_capture_uses_motion_compensated_median_and_support(self) -> None:
        tracker = AdaptiveTracker(
            self._config(capture_buffer_size=5, capture_support_ratio=0.6, capture_radius_m=1.0)
        )
        # 前三次候选外推到 t=4 都是 104；两次远距离误报不能破坏中位捕获。
        records = (
            (0, self._candidate(0.9, 100.0, 1.0)),
            (1, self._candidate(0.9, 101.0, 1.0)),
            (2, self._candidate(0.9, 102.0, 1.0)),
            (3, self._candidate(0.99, 5_000.0, 0.0)),
            (4, self._candidate(0.99, 8_000.0, 0.0)),
        )
        diagnostics = None
        for frame, candidate in records:
            diagnostics = tracker.step(frame, candidate)

        assert diagnostics is not None
        self.assertEqual(diagnostics.mode, TrackerMode.TRACK)
        self.assertEqual(diagnostics.capture_support_count, 3)
        self.assertAlmostEqual(diagnostics.range_current_m or 0.0, 104.0)
        self.assertAlmostEqual(diagnostics.speed_m_per_frame or 0.0, 1.0)

    def test_capture_requires_capture_q_min_before_caching_candidates(self) -> None:
        tracker = AdaptiveTracker(
            self._config(capture_buffer_size=2, capture_support_ratio=1.0)
        )

        rejected = tracker.step(0, self._candidate(0.49, 100.0))
        self.assertEqual(rejected.mode, TrackerMode.CAPTURE)
        self.assertFalse(rejected.candidate_accepted)
        self.assertEqual(rejected.rejected_by, ("capture_q_min",))
        self.assertEqual(rejected.capture_buffer_size, 0)

        tracker.step(2, self._candidate(0.5, 104.0))
        confirmed = tracker.step(4, self._candidate(0.5, 108.0))
        self.assertEqual(confirmed.mode, TrackerMode.TRACK)
        self.assertTrue(confirmed.capture_confirmed)

    def test_capture_keeps_empty_steps_without_emitting_track(self) -> None:
        tracker = AdaptiveTracker(self._config())
        first = tracker.step(0, None)
        second = tracker.step(1, None)
        self.assertEqual(first.mode, TrackerMode.CAPTURE)
        self.assertEqual(second.mode, TrackerMode.CAPTURE)
        self.assertEqual(second.capture_buffer_size, 0)
        self.assertIsNone(second.range_current_m)

    def test_track_alpha_beta_update_and_soft_speed_gate(self) -> None:
        tracker = AdaptiveTracker(self._config())
        self._capture(tracker)

        # 预测为 (110, 2)，位置新息为 10。斜率差只诊断、不做硬拒绝。
        diagnostic = tracker.step(5, self._candidate(0.9, 120.0, 20.0))
        self.assertEqual(diagnostic.mode, TrackerMode.TRACK)
        self.assertTrue(diagnostic.candidate_accepted)
        self.assertTrue(diagnostic.measurement_updated)
        self.assertAlmostEqual(diagnostic.position_residual_m or 0.0, 10.0)
        self.assertAlmostEqual(diagnostic.speed_residual_m_per_frame or 0.0, 18.0)
        self.assertAlmostEqual(diagnostic.range_current_m or 0.0, 115.0)
        self.assertAlmostEqual(diagnostic.speed_m_per_frame or 0.0, 3.0)

    def test_failures_expand_then_enter_recapture_with_unreliable_extrapolation(self) -> None:
        tracker = AdaptiveTracker(self._config())
        self._capture(tracker)

        first = tracker.step(5, self._candidate(0.1, 110.0))
        self.assertEqual(first.mode, TrackerMode.TRACK)
        self.assertEqual(first.search_level, 1)
        self.assertFalse(first.measurement_updated)
        self.assertEqual(first.rejected_by, ("q_keep",))
        self.assertAlmostEqual(first.range_current_m or 0.0, 110.0)

        second = tracker.step(6, self._candidate(0.1, 112.0))
        self.assertEqual(second.mode, TrackerMode.TRACK)
        self.assertEqual(second.search_level, 2)

        lost = tracker.step(7, self._candidate(0.1, 114.0))
        self.assertEqual(lost.mode_before, TrackerMode.TRACK)
        self.assertEqual(lost.mode, TrackerMode.RECAPTURE)
        self.assertIsNone(lost.range_current_m)
        self.assertEqual(lost.search_level, 0)
        self.assertEqual(lost.rejected_by, ("q_keep",))
        self.assertAlmostEqual(lost.extrapolation_range_m or 0.0, 114.0)
        self.assertAlmostEqual(lost.extrapolation_speed_mpf or 0.0, 2.0)
        self.assertEqual(lost.extrapolation_reference_frame, 7)

    def test_speed_gates_limit_instant_and_recent_average_state_speed(self) -> None:
        instant_tracker = AdaptiveTracker(
            self._config(
                instant_speed_gate_mpf=(17.0, 25.0, 34.0),
                average_speed_gate_mpf=(100.0, 100.0, 100.0),
            )
        )
        self._capture(instant_tracker)
        # 当前预测速度为 2；大位置残差会令暂定融合速度超过 17 m/frame。
        rejected_instant = instant_tracker.step(5, self._candidate(0.99, 300.0))
        self.assertFalse(rejected_instant.candidate_accepted)
        self.assertEqual(rejected_instant.rejected_by, ("instant_speed",))

        average_tracker = AdaptiveTracker(
            self._config(
                alpha=1.0,
                beta=0.0,
                instant_speed_gate_mpf=(100.0, 100.0, 100.0),
                average_speed_gate_mpf=(17.0, 25.0, 34.0),
            )
        )
        self._capture(average_tracker)
        for frame in (5, 10, 15, 20):
            average_tracker.step(frame, self._candidate(0.99, 108.0 + 2.0 * (frame - 4)))
        # 单帧速度仍为 2，但候选位置会使最近约 20 帧平均速度超过 17。
        rejected_average = average_tracker.step(25, self._candidate(0.99, 500.0))
        self.assertFalse(rejected_average.candidate_accepted)
        self.assertEqual(rejected_average.rejected_by, ("average_speed",))


    def test_high_q_but_excessive_speed_candidate_is_rejected(self) -> None:
        tracker = AdaptiveTracker(self._config(expand_after_bad=2))
        self._capture(tracker)
        # 预测位置为 110，但 q 再高也不能接受 190 m 的突跳。
        diagnostic = tracker.step(5, self._candidate(0.99, 300.0))
        self.assertFalse(diagnostic.candidate_accepted)
        self.assertEqual(diagnostic.rejected_by, ("instant_speed",))
        self.assertFalse(diagnostic.measurement_updated)
        self.assertAlmostEqual(diagnostic.position_residual_m or 0.0, 190.0)
        self.assertAlmostEqual(diagnostic.range_current_m or 0.0, 110.0)

    def test_good_steps_shrink_after_hysteresis(self) -> None:
        tracker = AdaptiveTracker(self._config())
        self._capture(tracker)
        # 一次失败先把搜索范围扩大到 L=1。
        tracker.step(5, self._candidate(0.1, 110.0))
        self.assertEqual(tracker.search_level, 1)

        one = tracker.step(6, self._candidate(0.9, 112.0))
        self.assertEqual(one.search_level, 1)
        two = tracker.step(7, self._candidate(0.9, 114.0))
        self.assertEqual(two.search_level, 0)

    def test_nan_candidate_and_non_increasing_frame_are_safe(self) -> None:
        tracker = AdaptiveTracker(self._config(expand_after_bad=2))
        self._capture(tracker)
        diagnostic = tracker.step(5, self._candidate(float("nan"), 110.0))
        self.assertFalse(diagnostic.candidate_accepted)
        self.assertFalse(diagnostic.measurement_updated)
        self.assertAlmostEqual(diagnostic.range_current_m or 0.0, 110.0)
        self.assertIsNone(diagnostic.position_residual_m)
        with self.assertRaises(AssertionError):
            tracker.step(5, None)
        with self.assertRaises(AssertionError):
            tracker.step(4, None)

    def test_local_blocks_are_boundary_clamped_deduplicated_and_global_grid_covers_end(self) -> None:
        tracker = AdaptiveTracker(self._config())
        self.assertEqual(tracker.global_block_starts_m(), (0, 90, 180, 200))

        # 在左边界附近捕获，L=2 的左侧块裁剪后应去重。
        for frame in (0, 2, 4):
            tracker.step(frame, self._candidate(0.9, 10.0, 0.0))
        tracker.step(5, self._candidate(0.1, 10.0, 0.0))
        tracker.step(6, self._candidate(0.1, 10.0, 0.0))
        self.assertEqual(tracker.mode, TrackerMode.TRACK)
        self.assertEqual(tracker.search_level, 2)
        self.assertEqual(tracker.local_block_starts_m(7), (0, 90, 180))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
