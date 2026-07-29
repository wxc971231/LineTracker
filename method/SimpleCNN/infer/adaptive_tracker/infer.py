"""按 postprocess.md 实现的 CAPTURE / TRACK / RECAPTURE 流式推理方法。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from configs.base import SimpleCNNConfig
from data.dataloader import PackedSource
from infer.adaptive_tracker.tracker import AdaptiveTracker, Candidate, TrackerConfig, TrackerMode
from infer.common.runner import BatchPrediction, ModelRunner


@dataclass(frozen=True)
class AdaptiveInferenceConfig:
    """状态机推理的时间步进和文档第10节后处理参数。"""

    time_stride: int
    capture_stride: int = 3
    capture_buffer_size: int = 8
    capture_support_ratio: float = 0.7
    capture_radius_m: float = 500.0
    q_keep: float = 0.5
    instant_speed_gate_mpf: tuple[float, ...] = (17.0, 25.0, 34.0)
    average_speed_gate_mpf: tuple[float, ...] = (17.0, 25.0, 34.0)
    speed_average_window_frames: int = 20
    expand_after_bad: int = 2
    shrink_after_good: int = 4
    alpha: float = 0.8
    beta: float = 0.1
    gamma: float = 0.0

    def tracker_config(self, config: SimpleCNNConfig) -> TrackerConfig:
        return TrackerConfig(
            capture_buffer_size=self.capture_buffer_size,
            capture_support_ratio=self.capture_support_ratio,
            capture_radius_m=self.capture_radius_m,
            q_keep=self.q_keep,
            instant_speed_gate_mpf=self.instant_speed_gate_mpf,
            average_speed_gate_mpf=self.average_speed_gate_mpf,
            speed_average_window_frames=self.speed_average_window_frames,
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

    def validate(self, config: SimpleCNNConfig) -> TrackerConfig:
        """校验方法配置，并返回可跨样本复用的不可变状态机配置。"""
        assert self.time_stride >= 1, "time_stride 必须为正整数。"
        return self.tracker_config(config)


def _top1_candidate(
    batch: BatchPrediction,
    *,
    frames_per_window: int,
) -> Candidate:
    """将当前 batch 的 q-Top1 直接转换为状态机唯一需要的候选对象。"""
    assert (
        np.all(np.isfinite(batch.q))
        and np.all(np.isfinite(batch.rho_m))
        and np.all(np.isfinite(batch.nu_mpf))
    ), "模型输出含 NaN/Inf，不能进入后处理状态机。"
    slot = int(np.argmax(batch.q))
    return Candidate.from_block_prediction(
        q=float(batch.q[slot]),
        rho_m=float(batch.rho_m[slot]),
        speed_m_per_frame=float(batch.nu_mpf[slot]),
        block_start_m=int(batch.range_starts_m[slot]),
        frames_per_window=frames_per_window,
    )


def run_source(
    source: PackedSource,
    runner: ModelRunner,
    config: SimpleCNNConfig,
    method_config: AdaptiveInferenceConfig,
    tracker_config: TrackerConfig,
) -> dict[str, object]:
    """仅依赖二值观测执行文档定义的自适应捕获、跟踪和重捕获。"""
    tracker = AdaptiveTracker(tracker_config)           # 创建样本推理状态机
    frame_count = int(source.frames)                    # 300 当前源序列的总帧数  
    frames_per_window = int(config.frames_per_window)   # 20 CNN 输入的时间窗长度
    prediction_m = np.full(frame_count, np.nan, dtype=np.float64)   # (300, ) 按全局帧号保存预测距离，未覆盖帧保持 NaN
    unreliable_prediction_mask = np.zeros(frame_count, dtype=bool)  # (300, ) RECAPTURE 临时外推为 True，不影响状态机
    step_records: list[dict[str, object]] = []          # 后续写入 log.jsonl 的精简逐步诊断记录。
    total_blocks = 0                # 样本推理的距离块总数
    total_forwards = 0              # 考虑 max_blocks_per_forward 拆分后的实际前向次数
    total_preprocess_s = 0.0        # 输入裁剪、通道重排和设备传输等预处理累计耗时（秒）
    total_model_s = 0.0             # 模型前向累计耗时（秒）
    total_end_to_end_s = 0.0        # 每步完整处理累计耗时（秒）
    capture_scans = 0               # CAPTURE 与 RECAPTURE 的全局扫描次数
    local_scans = 0                 # TRACK 状态下的局部扫描次数

    # CAPTURE / RECAPTURE 模式下按 capture_stride 时间步进扫描，无可靠输出，仅依赖外推
    # TRACK 模式下按 time_stride 时间步进持续跟踪
    time_start = 0
    while time_start < frame_count - frames_per_window:
        whole_start = perf_counter()
        latest_frame = int(time_start + frames_per_window - 1)                      # 本窗口最后一帧，也是状态更新时间

        # ---- 根据当前状态选择全局捕获或局部跟踪候选块 ----
        if tracker.mode in {TrackerMode.CAPTURE, TrackerMode.RECAPTURE}:
            block_starts = tracker.global_block_starts_m()                          # 全局扫描时所有块的起始距离
            batch = runner.predict_blocks(source, time_start, block_starts)         # 所有时间-距离块拼成 batch 并行推理
            candidate = _top1_candidate(batch, frames_per_window=frames_per_window) # 全局 q-Top1 块的推理结果
            timing = batch.timing
            capture_scans += 1
        else:
            block_starts = tracker.local_block_starts_m(latest_frame)               # L0/L1/L2 对应 1/3/5 个块。
            assert block_starts, "TRACK 状态未生成任何局部候选块。"         
            batch = runner.predict_blocks(source, time_start, block_starts)         # 局部时间-距离块拼成 batch 并行推理
            candidate = _top1_candidate(batch, frames_per_window=frames_per_window) # 局部 q-Top1 块的推理结果
            timing = batch.timing
            local_scans += 1

        diagnostics = tracker.step(latest_frame, candidate)                         # 基于 q-Top1 选块检测结果进行状态转移
        
        # ---- 将当前状态外推到下一次推理前的帧；RECAPTURE 输出会显式标为不可靠 ----
        # 准备外推时间窗
        forecast_start = int(time_start + frames_per_window)
        forecast_stop = min(forecast_start + method_config.time_stride, frame_count)
        forecast_frames = np.arange(forecast_start, forecast_stop, dtype=np.int32)  # (frames_per_window, )

        # 在 TRACK 模式，且存在有效的当前位置和速度时进行可信外推
        if (
            diagnostics.mode is TrackerMode.TRACK
            and diagnostics.range_current_m is not None
            and diagnostics.speed_m_per_frame is not None
        ):
            forecast_values = (
                float(diagnostics.range_current_m)
                + (forecast_frames.astype(np.float64) - float(latest_frame))
                * float(diagnostics.speed_m_per_frame)
            )
            prediction_m[forecast_frames] = forecast_values
            unreliable_prediction_mask[forecast_frames] = False # 由于 capture_stride 可能不等于 time_stride，重捕获确认后可能更新部分不可靠外推
        # 在 RECAPTURE 模式，且已经有一份 “最后可靠 TRACK 状态” 时，依赖最后可靠状态进行不可靠外推
        elif (
            diagnostics.mode is TrackerMode.RECAPTURE
            and diagnostics.extrapolation_range_m is not None
            and diagnostics.extrapolation_speed_mpf is not None
            and diagnostics.extrapolation_reference_frame is not None
        ):
            forecast_values = (
                float(diagnostics.extrapolation_range_m)
                + (forecast_frames.astype(np.float64) - float(diagnostics.extrapolation_reference_frame))
                * float(diagnostics.extrapolation_speed_mpf)
            )
            prediction_m[forecast_frames] = forecast_values
            unreliable_prediction_mask[forecast_frames] = True

        # ---- 汇总本逻辑步的实际工作量与端到端时延 ----
        end_to_end_s = perf_counter() - whole_start 
        total_end_to_end_s += end_to_end_s          # 累计本轮步进推理总时间
        total_blocks += timing.blocks_evaluated     # 累计本轮步进推理总快数
        total_forwards += timing.forward_calls      # 累计前向调用模型次数
        total_preprocess_s += timing.preprocess_s   # 累计预处理时间
        total_model_s += timing.model_s             # 累计模型前向时间
        
        # ---- 生成精简的逐步日志；只保留 JSONL、汇总和诊断图实际消费的字段 ----
        record: dict[str, object] = {
            "frame": latest_frame,
            "forecast_frame_start": forecast_start,
            "forecast_frame_stop": forecast_stop,
            "mode_before": diagnostics.mode_before.value,
            "mode": diagnostics.mode.value,
            "candidate_accepted": diagnostics.candidate_accepted,
            "measurement_updated": diagnostics.measurement_updated,
            "next_search_level": diagnostics.search_level,
            "end_to_end_s": end_to_end_s,
        }
        if diagnostics.rejected_by:
            record["rejected_by"] = list(diagnostics.rejected_by)
        if diagnostics.mode_before in {TrackerMode.CAPTURE, TrackerMode.RECAPTURE}:
            record["capture"] = {
                "buffer_size": diagnostics.capture_buffer_size,
                "support_count": diagnostics.capture_support_count,
                "confirmed": diagnostics.capture_confirmed,
            }

        # 可能不存在（None）的诊断字段
        optional_values = {
            "state_range_m": diagnostics.range_current_m,                  # 本步结束后的可靠滤波位置；CAPTURE 未确认、RECAPTURE 时不存在
            "state_speed_mpf": diagnostics.speed_m_per_frame,              # 本步结束后的可靠滤波速度
            "scan_level": diagnostics.evaluated_search_level,              # 本步实际使用的局部搜索等级；全局扫描时不存在
            "position_residual_m": diagnostics.position_residual_m,        # CNN 候选位置相对匀速外推的位置残差（偏置误差）
            "speed_residual_mpf": diagnostics.speed_residual_m_per_frame,  # CNN 候选速度相对当前预测速度的残差（斜率误差）
            "candidate_q": candidate.q,                                    # 当前 batch 的 q-Top1 分数
            "candidate_range_m": candidate.latest_range_m,                 # q-Top1 换算到最新帧的全局距离
            "candidate_block_start_m": candidate.block_start_m,            # q-Top1 所在距离块起点
            "candidate_speed_mpf": candidate.speed_m_per_frame,            # q-Top1 的模型速度输出
        }
        record.update({key: value for key, value in optional_values.items() if value is not None})
        
        # TRACK 使用业务步进；尚未确认稳定轨迹时保持更密的捕获扫描步进。
        time_start += (method_config.time_stride if tracker.mode is TrackerMode.TRACK else method_config.capture_stride)
        step_records.append(record)

    # ---- 返回统一输出协议：轨迹、可靠性掩码、逐步日志与样本级工作量 ----
    return {
        "prediction_m": prediction_m,
        "unreliable_prediction_mask": unreliable_prediction_mask,
        "steps": step_records,
        "workload": {
            "logical_steps": len(step_records),
            "capture_scans": capture_scans,
            "local_scans": local_scans,
            "blocks_evaluated": total_blocks,
            "forward_calls": total_forwards,
            "preprocess_s": total_preprocess_s,
            "model_s": total_model_s,
            "end_to_end_s": total_end_to_end_s,
        },
    }
