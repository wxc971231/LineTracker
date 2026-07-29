"""每个时间窗扫描全部标准距离块的流式 Top-1 基线。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from configs.base import SimpleCNNConfig
from data.dataloader import PackedSource, standard_distance_starts
from infer.common.runner import ModelRunner


@dataclass(frozen=True)
class GlobalTop1Config:
    """全局扫描方法唯一的算法参数。"""

    time_stride: int

    def validate(self) -> None:
        assert self.time_stride >= 1, "time_stride 必须为正整数。"


def stream_time_starts(
    frame_count: int,
    *,
    frames_per_window: int,
    time_stride: int,
) -> tuple[int, ...]:
    """只返回仍有至少一帧未来可预测的流式输入窗起点。"""
    assert frames_per_window >= 1 and time_stride >= 1, "frames_per_window 与 time_stride 必须为正整数。"
    assert frame_count > frames_per_window, "序列帧数必须大于输入时间窗长度，才能进行未来外推。"
    return tuple(range(0, frame_count - frames_per_window, time_stride))


def _forecast(
    *,
    time_start: int,
    range_start_m: int,
    rho_m: float,
    nu_mpf: float,
    forecast_frames: np.ndarray,
    frames_per_window: int,
) -> np.ndarray:
    """将块内中心距离和斜率外推到给定全局帧编号。"""
    centre_local_frame = (frames_per_window - 1) / 2.0
    local_frames = forecast_frames.astype(np.float64) - float(time_start)
    return (
        float(range_start_m)
        + float(rho_m)
        + float(nu_mpf) * (local_frames - centre_local_frame)
    )


def run_source(
    source: PackedSource,
    runner: ModelRunner,
    config: SimpleCNNConfig,
    method_config: GlobalTop1Config,
) -> dict[str, object]:
    """仅根据观测构造全局 Top-1 流式预测，不读取任何真值标签参与决策。"""
    # 方法配置已由 run_infer.run() 在样本循环外校验。
    frame_count = int(source.frames)
    frames_per_window = int(config.frames_per_window)
    range_starts = standard_distance_starts(config)
    time_starts = stream_time_starts(
        frame_count,
        frames_per_window=frames_per_window,
        time_stride=method_config.time_stride,
    )
    prediction_m = np.full(frame_count, np.nan, dtype=np.float64)
    step_records: list[dict[str, object]] = []
    total_blocks = 0
    total_forwards = 0
    total_model_s = 0.0
    total_preprocess_s = 0.0
    total_end_to_end_s = 0.0

    for time_start in time_starts:
        step_start = perf_counter()
        batch = runner.predict_blocks(source, time_start, range_starts)
        assert (
            np.all(np.isfinite(batch.q))
            and np.all(np.isfinite(batch.rho_m))
            and np.all(np.isfinite(batch.nu_mpf))
        ), f"数据源 {source.record.source_id} 在 time_start={time_start} 产生非有限模型输出。"
        candidate_slot = int(np.argmax(batch.q))
        forecast_start = time_start + frames_per_window
        forecast_stop = min(forecast_start + method_config.time_stride, frame_count)
        forecast_frames = np.arange(forecast_start, forecast_stop, dtype=np.int32)
        candidate_range_start = int(batch.range_starts_m[candidate_slot])
        candidate_rho = float(batch.rho_m[candidate_slot])
        candidate_nu = float(batch.nu_mpf[candidate_slot])
        forecast_values = _forecast(
            time_start=time_start,
            range_start_m=candidate_range_start,
            rho_m=candidate_rho,
            nu_mpf=candidate_nu,
            forecast_frames=forecast_frames,
            frames_per_window=frames_per_window,
        )
        prediction_m[forecast_frames] = forecast_values
        latest_frame = time_start + frames_per_window - 1
        latest_range_m = float(
            _forecast(
                time_start=time_start,
                range_start_m=candidate_range_start,
                rho_m=candidate_rho,
                nu_mpf=candidate_nu,
                forecast_frames=np.asarray([latest_frame], dtype=np.int32),
                frames_per_window=frames_per_window,
            )[0]
        )
        step_total_s = perf_counter() - step_start
        timing = batch.timing
        total_blocks += timing.blocks_evaluated
        total_forwards += timing.forward_calls
        total_preprocess_s += timing.preprocess_s
        total_model_s += timing.model_s
        total_end_to_end_s += step_total_s
        # ``prediction_m`` 已保存完整未来轨迹；逐步记录只保留日志复盘和延迟分位数所需字段。
        step_records.append(
            {
                "frame": int(latest_frame),
                "candidate_q": float(batch.q[candidate_slot]),
                "candidate_block_start_m": candidate_range_start,
                "candidate_range_m": latest_range_m,
                "candidate_speed_mpf": candidate_nu,
                "end_to_end_s": step_total_s,
            }
        )

    return {
        "prediction_m": prediction_m,
        "steps": step_records,
        "workload": {
            "logical_steps": len(step_records),
            "blocks_evaluated": total_blocks,
            "forward_calls": total_forwards,
            "preprocess_s": total_preprocess_s,
            "model_s": total_model_s,
            "end_to_end_s": total_end_to_end_s,
        },
    }
