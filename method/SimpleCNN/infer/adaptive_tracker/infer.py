"""按 postprocess.md 实现的 CAPTURE / TRACK / RECAPTURE 流式推理方法。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from configs.base import SimpleCNNConfig
from data.dataloader import PackedSource
from infer.adaptive_tracker.tracker import AdaptiveTracker, Candidate, TrackerConfig, TrackerMode
from infer.common.runner import BatchPrediction, BatchTiming, ModelRunner


@dataclass(frozen=True)
class AdaptiveInferenceConfig:
    """状态机推理的时间步进和文档第10节后处理参数。"""

    time_stride: int
    capture_stride: int = 3
    capture_buffer_size: int = 8
    capture_support_ratio: float = 0.7
    capture_radius_m: float = 500.0
    q_keep: float = 0.5
    position_gate_m: tuple[float, ...] = (1_000.0, 2_000.0, 4_000.0)
    expand_after_bad: int = 2
    shrink_after_good: int = 4
    alpha: float = 0.8
    beta: float = 0.1
    gamma: float = 0.0

    def tracker_config(self, config: SimpleCNNConfig) -> TrackerConfig:
        return TrackerConfig(
            capture_stride=self.capture_stride,
            capture_buffer_size=self.capture_buffer_size,
            capture_support_ratio=self.capture_support_ratio,
            capture_radius_m=self.capture_radius_m,
            q_keep=self.q_keep,
            position_gate_m=self.position_gate_m,
            expand_after_bad=self.expand_after_bad,
            shrink_after_good=self.shrink_after_good,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            block_width_m=float(config.block_width_m),
            block_step_m=float(config.spatial_step_m),
            range_min_m=0.0,
            range_max_m=float(config.range_bins),
            frames_per_window=config.frames_per_window,
        )

    def validate(self, config: SimpleCNNConfig) -> None:
        if self.time_stride < 1:
            raise ValueError("time_stride 必须为正整数。")
        self.tracker_config(config)


def _top1_candidate(
    batch: BatchPrediction,
    *,
    frames_per_window: int,
) -> tuple[Candidate, int, float, float]:
    """由当前实际推理块集合的 q-Top1 生成统一最新帧坐标候选。"""
    if not (
        np.all(np.isfinite(batch.q))
        and np.all(np.isfinite(batch.rho_m))
        and np.all(np.isfinite(batch.nu_mpf))
    ):
        raise RuntimeError("模型输出含 NaN/Inf，不能进入后处理状态机。")
    slot = int(np.argmax(batch.q))
    rho_m = float(batch.rho_m[slot])
    nu_mpf = float(batch.nu_mpf[slot])
    block_start_m = int(batch.range_starts_m[slot])
    return (
        Candidate.from_block_prediction(
            q=float(batch.q[slot]),
            rho_m=rho_m,
            speed_m_per_frame=nu_mpf,
            block_start_m=block_start_m,
            frames_per_window=frames_per_window,
        ),
        block_start_m,
        rho_m,
        nu_mpf,
    )


def _empty_timing() -> BatchTiming:
    return BatchTiming(
        preprocess_s=0.0,
        model_s=0.0,
        postprocess_s=0.0,
        total_s=0.0,
        forward_calls=0,
        blocks_evaluated=0,
    )


def run_source(
    source: PackedSource,
    runner: ModelRunner,
    config: SimpleCNNConfig,
    method_config: AdaptiveInferenceConfig,
) -> dict[str, object]:
    """仅依赖二值观测执行文档定义的自适应捕获、跟踪和重捕获。"""
    method_config.validate(config)
    tracker = AdaptiveTracker(method_config.tracker_config(config))
    frame_count = int(source.frames)
    frames_per_window = int(config.frames_per_window)
    prediction_m = np.full(frame_count, np.nan, dtype=np.float64)
    step_records: list[dict[str, object]] = []
    total_blocks = 0
    total_forwards = 0
    total_preprocess_s = 0.0
    total_model_s = 0.0
    total_model_postprocess_s = 0.0
    total_end_to_end_s = 0.0
    capture_scans = 0
    local_scans = 0

    # CAPTURE / RECAPTURE 没有可输出的稳定轨迹，因此按文档的
    # capture_stride 独立扫描；确认后才按业务预测步进 time_stride 前推。
    # 这样两个步进不会在 time_stride > capture_stride 时被悄悄混为一个。
    step_index = 0
    time_start = 0
    while time_start < frame_count - frames_per_window:
        whole_start = perf_counter()
        latest_frame = int(time_start + frames_per_window - 1)
        candidate: Candidate | None = None
        selected_block_start: int | None = None
        selected_rho_m: float | None = None
        selected_nu_mpf: float | None = None
        block_starts: tuple[int, ...]
        timing = _empty_timing()

        if tracker.mode in {TrackerMode.CAPTURE, TrackerMode.RECAPTURE}:
            if tracker.capture_scan_due(latest_frame):
                block_starts = tracker.global_block_starts_m()
                batch = runner.predict_blocks(source, time_start, block_starts)
                candidate, selected_block_start, selected_rho_m, selected_nu_mpf = _top1_candidate(
                    batch,
                    frames_per_window=frames_per_window,
                )
                timing = batch.timing
                capture_scans += 1
            else:
                block_starts = ()
        else:
            block_starts = tracker.local_block_starts_m(latest_frame)
            if not block_starts:
                raise RuntimeError("TRACK 状态未生成任何局部候选块。")
            batch = runner.predict_blocks(source, time_start, block_starts)
            candidate, selected_block_start, selected_rho_m, selected_nu_mpf = _top1_candidate(
                batch,
                frames_per_window=frames_per_window,
            )
            timing = batch.timing
            local_scans += 1

        diagnostics = tracker.step(
            latest_frame,
            candidate,
            blocks_evaluated=timing.blocks_evaluated,
            block_starts_m=block_starts,
        )
        forecast_start = int(time_start + frames_per_window)
        forecast_stop = min(forecast_start + method_config.time_stride, frame_count)
        forecast_frames = np.arange(forecast_start, forecast_stop, dtype=np.int32)
        if diagnostics.range_current_m is not None and diagnostics.speed_m_per_frame is not None:
            forecast_values = (
                float(diagnostics.range_current_m)
                + (forecast_frames.astype(np.float64) - float(latest_frame))
                * float(diagnostics.speed_m_per_frame)
            )
            prediction_m[forecast_frames] = forecast_values
            range_next_m = float(forecast_values[0])
        else:
            range_next_m = None

        end_to_end_s = perf_counter() - whole_start
        total_blocks += timing.blocks_evaluated
        total_forwards += timing.forward_calls
        total_preprocess_s += timing.preprocess_s
        total_model_s += timing.model_s
        total_model_postprocess_s += timing.postprocess_s
        total_end_to_end_s += end_to_end_s

        record = diagnostics.to_dict()
        record.update(
            {
                "step_index": step_index,
                "input_time_start": int(time_start),
                "input_time_stop": int(time_start + frames_per_window),
                "latest_frame": latest_frame,
                "forecast_frame_start": forecast_start,
                "forecast_frame_stop": forecast_stop,
                "range_next_m": range_next_m,
                "candidate_block_start_m": selected_block_start,
                "candidate_rho_m": selected_rho_m,
                "candidate_nu_mpf": selected_nu_mpf,
                "forward_calls": timing.forward_calls,
                "preprocess_s": timing.preprocess_s,
                "model_s": timing.model_s,
                "model_postprocess_s": timing.postprocess_s,
                "end_to_end_s": end_to_end_s,
            }
        )
        step_index += 1
        time_start += (
            method_config.time_stride
            if tracker.mode is TrackerMode.TRACK
            else method_config.capture_stride
        )
        step_records.append(record)

    return {
        "method": "adaptive_tracker",
        "source_id": source.record.source_id,
        "source_path": str(source.record.path),
        "frame_count": frame_count,
        "frames_per_window": frames_per_window,
        "time_stride": method_config.time_stride,
        "prediction_m": prediction_m,
        "steps": step_records,
        "tracker_config": {
            key: value
            for key, value in method_config.tracker_config(config).__dict__.items()
        },
        "workload": {
            "logical_steps": len(step_records),
            "capture_scans": capture_scans,
            "local_scans": local_scans,
            "blocks_evaluated": total_blocks,
            "forward_calls": total_forwards,
            "preprocess_s": total_preprocess_s,
            "model_s": total_model_s,
            "model_postprocess_s": total_model_postprocess_s,
            "end_to_end_s": total_end_to_end_s,
        },
    }
