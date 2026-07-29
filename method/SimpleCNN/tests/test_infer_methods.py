"""全局与自适应流式推理编排的轻量回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from configs.base import SimpleCNNConfig
from data.dataloader import SourceRecord, standard_distance_starts
from infer.adaptive_tracker.infer import AdaptiveInferenceConfig, run_source as run_adaptive_source
from infer.common.runner import BatchPrediction, BatchTiming
from infer.global_top1.infer import GlobalTop1Config, run_source as run_global_source


class _FakeSource:
    def __init__(self, frames: int) -> None:
        self.frames = frames
        self.record = SourceRecord("fake_source", Path("/tmp/fake_source/data.npz"))


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[int, tuple[int, ...]]] = []

    def predict_blocks(
        self,
        source: _FakeSource,
        time_start: int,
        range_starts_m: tuple[int, ...],
    ) -> BatchPrediction:
        del source
        starts = np.asarray(range_starts_m, dtype=np.int32)
        self.calls.append((time_start, tuple(int(item) for item in starts)))
        q = np.linspace(0.1, 0.9, len(starts), dtype=np.float32)
        return BatchPrediction(
            range_starts_m=starts,
            q=q,
            rho_m=np.full(len(starts), 100.0, dtype=np.float32),
            nu_mpf=np.full(len(starts), 1.0, dtype=np.float32),
            timing=BatchTiming(
                preprocess_s=0.0,
                model_s=0.0,
                postprocess_s=0.0,
                total_s=0.0,
                forward_calls=1,
                blocks_evaluated=len(starts),
            ),
        )


class InferMethodTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimpleCNNConfig()

    def test_global_top1_uses_all_standard_blocks_and_forecasts_future_stride(self) -> None:
        source = _FakeSource(frames=27)
        runner = _FakeRunner()
        result = run_global_source(
            source,  # type: ignore[arg-type]
            runner,  # type: ignore[arg-type]
            self.config,
            GlobalTop1Config(time_stride=5),
        )

        standard_starts = standard_distance_starts(self.config)
        self.assertEqual([call[0] for call in runner.calls], [0, 5])
        self.assertTrue(all(call[1] == standard_starts for call in runner.calls))
        self.assertEqual(result["workload"]["blocks_evaluated"], 2 * len(standard_starts))
        prediction = np.asarray(result["prediction_m"])
        self.assertTrue(np.isnan(prediction[:20]).all())
        self.assertTrue(np.isfinite(prediction[20:]).all())
        self.assertEqual(len(result["steps"]), 2)

    def test_adaptive_capture_stride_is_independent_from_track_stride(self) -> None:
        source = _FakeSource(frames=40)
        runner = _FakeRunner()
        result = run_adaptive_source(
            source,  # type: ignore[arg-type]
            runner,  # type: ignore[arg-type]
            self.config,
            AdaptiveInferenceConfig(
                time_stride=5,
                capture_stride=2,
                capture_buffer_size=2,
                capture_support_ratio=1.0,
                capture_radius_m=500.0,
                q_keep=0.0,
                position_gate_m=(1_000.0, 2_000.0, 4_000.0),
                expand_after_bad=2,
                shrink_after_good=4,
            ),
        )

        # 前两次是 2 帧步进捕获；第二次确认后才切换为 5 帧 TRACK 步进。
        self.assertGreaterEqual(len(runner.calls), 3)
        self.assertEqual([call[0] for call in runner.calls[:3]], [0, 2, 7])
        first_two = [call[1] for call in runner.calls[:2]]
        self.assertTrue(all(len(starts) == 34 for starts in first_two))
        self.assertEqual(result["steps"][1]["capture_confirmed"], True)
        self.assertEqual(result["steps"][2]["mode_before"], "TRACK")
        prediction = np.asarray(result["prediction_m"])
        self.assertTrue(np.isfinite(prediction[22:27]).all())


if __name__ == "__main__":
    unittest.main()

