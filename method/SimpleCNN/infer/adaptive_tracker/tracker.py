"""严格按 ``_doc/postprocess.md`` 实现的纯 NumPy 捕获—跟踪状态机。

本模块不读取原始二值时距图、也不调用 PyTorch 模型。调用方负责把一次
CNN batch 的 q-Top1 转换为 :class:`Candidate` 后传给 :meth:`AdaptiveTracker.step`。
这样状态机可独立测试，也可同时服务于 CPU、CUDA 与 Ascend NPU 推理入口。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np


class TrackerMode(str, Enum):
    """后处理的三个工作状态。"""
    CAPTURE = "CAPTURE"
    TRACK = "TRACK"
    RECAPTURE = "RECAPTURE"

def _is_finite_number(value: object) -> bool:
    """仅接受可转换为有限 float 的标量，防止 NaN/Inf 污染状态。"""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def _as_finite_or_none(value: float) -> float | None:
    return float(value) if _is_finite_number(value) else None

def _require_positive_int(name: str, value: object) -> int:
    assert not isinstance(value, bool) and isinstance(value, (int, np.integer)) and int(value) > 0, f"{name} 必须是正整数，实际为 {value!r}。"
    return int(value)

def _require_nonnegative_int(name: str, value: object) -> int:
    assert not isinstance(value, bool) and isinstance(value, (int, np.integer)) and int(value) >= 0, f"{name} 必须是非负整数，实际为 {value!r}。"
    return int(value)

def _round_bin(value: float) -> int:
    """距离 bin 的确定性四舍五入；距离轴通常非负，避免 Python 的银行家舍入。"""
    return int(math.floor(float(value) + 0.5))


@dataclass(frozen=True)
class Candidate:
    """一次实际 CNN batch 中 q 最大块换算后的候选。

    ``latest_range_m`` 必须是文档第 2.1 节定义的最新帧位置 ``z_t``，而不是模型原始输出的块内中心距离 ``rho_m``。
    使用 `from_block_prediction` 完成 CNN 原始输出坐标转换。
    """

    q: float                            # CNN 预测块内目标存在概率
    latest_range_m: float               # CNN 预测块内最新帧距离
    speed_m_per_frame: float            # CNN 预测块内目标平均速度
    block_start_m: float | None = None  # 分块起始距离

    @classmethod
    def from_block_prediction(
        cls,
        *,
        q: float,
        rho_m: float,
        speed_m_per_frame: float,
        block_start_m: float,
        frames_per_window: int = 20,
    ) -> "Candidate":
        """从 SimpleCNN 的 ``(q, rho_m, nu_mpf)`` 创建统一时间坐标候选
        对 20 帧窗口，模型 ``rho_m`` 对应中心时刻 9.5 帧；因此最新帧位置为 ``z_t = block_start + rho_m + 9.5 * speed``
        """
        frames = _require_positive_int("frames_per_window", frames_per_window)
        center_offset = (frames - 1) / 2.0
        latest_range_m = float(block_start_m) + float(rho_m) + center_offset * float(speed_m_per_frame)
        return cls(
            q=float(q),
            latest_range_m=latest_range_m,
            speed_m_per_frame=float(speed_m_per_frame),
            block_start_m=float(block_start_m),
        )

    @property
    def is_valid(self) -> bool:
        """候选是否可安全进入聚类、门控和滤波"""
        if not all(_is_finite_number(value) for value in (self.q, self.latest_range_m, self.speed_m_per_frame)):
            return False
        if not 0.0 <= float(self.q) <= 1.0:
            return False
        if self.block_start_m is not None and not _is_finite_number(self.block_start_m):
            return False
        return True


@dataclass(frozen=True)
class TrackerConfig:
    """文档第 10 节所需的后处理参数。

    ``instant_speed_gate_mpf`` 和 ``average_speed_gate_mpf`` 的第 ``L`` 项分别限制单帧融合状态速度与最近窗口平均速度，单位均为米/帧
    默认仅实现L=0/1/2 三个局部搜索等级，L=2 再失败会进入 ``RECAPTURE``
    """

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
    block_width_m: float = 10_000.0
    block_step_m: float = 9_000.0
    range_min_m: float = 0.0
    range_max_m: float = 300_000.0
    frames_per_window: int = 20
    max_search_level: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capture_buffer_size",
            _require_positive_int("capture_buffer_size", self.capture_buffer_size),
        )
        object.__setattr__(self, "expand_after_bad", _require_positive_int("expand_after_bad", self.expand_after_bad))
        object.__setattr__(
            self,
            "shrink_after_good",
            _require_positive_int("shrink_after_good", self.shrink_after_good),
        )
        object.__setattr__(self, "frames_per_window", _require_positive_int("frames_per_window", self.frames_per_window))
        object.__setattr__(
            self,
            "speed_average_window_frames",
            _require_positive_int("speed_average_window_frames", self.speed_average_window_frames),
        )
        object.__setattr__(self, "max_search_level", _require_nonnegative_int("max_search_level", self.max_search_level))

        finite_fields = {
            "capture_support_ratio": self.capture_support_ratio,
            "capture_radius_m": self.capture_radius_m,
            "q_keep": self.q_keep,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "block_width_m": self.block_width_m,
            "block_step_m": self.block_step_m,
            "range_min_m": self.range_min_m,
            "range_max_m": self.range_max_m,
        }
        for name, value in finite_fields.items():
            assert _is_finite_number(value), f"{name} 必须是有限数，实际为 {value!r}。"
            object.__setattr__(self, name, float(value))

        assert 0.0 < self.capture_support_ratio <= 1.0, "capture_support_ratio 必须在 (0, 1] 内。"
        assert self.capture_radius_m >= 0.0, "capture_radius_m 不能为负数。"
        assert 0.0 <= self.q_keep <= 1.0, "q_keep 必须在 [0, 1] 内。"
        assert 0.0 <= self.alpha <= 1.0, "alpha 必须在 [0, 1] 内。"
        assert self.beta >= 0.0, "beta 不能为负数。"
        assert 0.0 <= self.gamma <= 1.0, "gamma 必须在 [0, 1] 内。"
        assert self.block_width_m > 0.0 and self.block_step_m > 0.0, "block_width_m 和 block_step_m 必须为正数。"
        assert self.range_max_m > self.range_min_m, "range_max_m 必须大于 range_min_m。"
        assert self.range_max_m - self.range_min_m >= self.block_width_m, "测距范围必须至少容纳一个完整 block_width_m 块。"

        for name in ("instant_speed_gate_mpf", "average_speed_gate_mpf"):
            try:
                gates = tuple(float(value) for value in getattr(self, name))
            except TypeError as exc:
                raise AssertionError(f"{name} 必须是按搜索等级排列的数值序列。") from exc
            assert len(gates) == self.max_search_level + 1, (
                f"{name} 长度必须等于 max_search_level + 1，实际为 {len(gates)}。"
            )
            assert all(math.isfinite(value) and value >= 0.0 for value in gates), f"{name} 中所有门限必须为非负有限数。"
            object.__setattr__(self, name, gates)

    @property
    def center_frame_offset(self) -> float:
        """CNN ``rho_m`` 对应窗口中心相对最新帧的偏移，20 帧时为 9.5。"""
        return (self.frames_per_window - 1) / 2.0

    @property
    def capture_min_support(self) -> int:
        """第 4.3 节的 ``ceil(eta_cap * K_cap)``。"""
        return int(math.ceil(self.capture_support_ratio * self.capture_buffer_size))

    @property
    def max_block_start_m(self) -> float:
        return self.range_max_m - self.block_width_m

    def global_block_starts_m(self) -> tuple[int, ...]:
        """返回覆盖完整测距范围的标准全局块起点，末块必覆盖右边界。"""
        starts: list[int] = []
        current = self.range_min_m
        max_start = self.max_block_start_m
        epsilon = max(abs(self.block_step_m), 1.0) * 1e-9
        while current <= max_start + epsilon:
            start = self.clamp_block_start(current)
            if not starts or start != starts[-1]:
                starts.append(start)
            current += self.block_step_m
        final_start = self.clamp_block_start(max_start)
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
        return tuple(starts)

    def clamp_block_start(self, start_m: float) -> int:
        """按文档将局部块起点 round 后裁剪到完整测距范围。"""
        bounded = min(max(float(start_m), self.range_min_m), self.max_block_start_m)
        return _round_bin(bounded)


@dataclass(frozen=True)
class StepDiagnostics:
    """单步状态机结果；仅包含状态转移和诊断真正需要的数据。"""

    frame_index: int
    mode: TrackerMode
    mode_before: TrackerMode
    range_current_m: float | None
    speed_m_per_frame: float | None
    candidate_accepted: bool
    measurement_updated: bool
    search_level: int
    evaluated_search_level: int | None
    position_residual_m: float | None
    speed_residual_m_per_frame: float | None
    capture_buffer_size: int
    capture_support_count: int
    capture_confirmed: bool
    rejected_by: tuple[str, ...]
    extrapolation_range_m: float | None
    extrapolation_speed_mpf: float | None
    extrapolation_reference_frame: int | None



@dataclass(frozen=True)
class _TimedCandidate:
    frame_index: int
    candidate: Candidate


@dataclass(frozen=True)
class _StatePoint:
    """用于按真实帧跨度计算平均状态速度的历史锚点。"""

    frame_index: int
    range_current_m: float


@dataclass(frozen=True)
class _CaptureEstimate:
    range_current_m: float
    speed_m_per_frame: float
    support_count: int


@dataclass(frozen=True)
class _StepOutcome:
    """本步状态转移产物，供 :meth:`AdaptiveTracker._diagnostics` 统一封装。"""

    candidate_accepted: bool = False
    measurement_updated: bool = False
    evaluated_search_level: int | None = None
    position_residual_m: float | None = None
    speed_residual_m_per_frame: float | None = None
    capture_support_count: int = 0
    capture_confirmed: bool = False
    rejected_by: tuple[str, ...] = ()


class AdaptiveTracker:
    """文档第 3～9 节的 CAPTURE / TRACK / RECAPTURE 状态机。

    调用方根据 :attr:`mode` 决定本步应做全局还是局部 CNN 推理；从 batch
    取得 q-Top1 后调用 :meth:`step`。
    该方法只接受已经换算到最新帧坐标的 :class:`Candidate`，不会接触模型。
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = TrackerConfig() if config is None else config
        assert isinstance(self.config, TrackerConfig), "config 必须是 TrackerConfig。"
        self.reset()

    def reset(self, mode: TrackerMode | str = TrackerMode.CAPTURE) -> None:
        """清空缓存和轨迹状态；用于开始一条新样本序列。"""
        self._mode = self._coerce_mode(mode)
        self._capture_buffer: deque[_TimedCandidate] = deque(maxlen=self.config.capture_buffer_size)
        self._range_current_m: float | None = None
        self._speed_m_per_frame: float | None = None
        self._last_state_frame: int | None = None
        self._last_step_frame: int | None = None
        # RECAPTURE 时状态机不会使用旧轨迹继续决策；下面三项只用于输出一条
        # 明确标为不可靠的可视化外推线，直至新的 CAPTURE 被确认。
        self._extrapolation_range_m: float | None = None
        self._extrapolation_speed_mpf: float | None = None
        self._extrapolation_reference_frame: int | None = None
        self._state_history: deque[_StatePoint] = deque(
            maxlen=self.config.speed_average_window_frames + 2
        )
        self._search_level = 0
        self._good_count = 0
        self._bad_count = 0

    @property
    def mode(self) -> TrackerMode:
        return self._mode

    @property
    def range_current_m(self) -> float | None:
        return self._range_current_m

    @property
    def speed_m_per_frame(self) -> float | None:
        return self._speed_m_per_frame

    @property
    def search_level(self) -> int:
        return self._search_level

    @property
    def capture_buffer_size(self) -> int:
        return len(self._capture_buffer)

    def global_block_starts_m(self) -> tuple[int, ...]:
        """暴露标准全局网格，供 CAPTURE/RECAPTURE 调用方一次组成 batch。"""
        return self.config.global_block_starts_m()

    def predict_state(self, frame_index: int) -> tuple[float, float]:
        """从最新滤波状态匀速外推到 ``frame_index``，不修改内部状态。"""
        frame = _require_nonnegative_int("frame_index", frame_index)
        assert self._mode is TrackerMode.TRACK and self._last_state_frame is not None, "当前不在 TRACK 状态，不能进行局部运动预测。"
        assert frame >= self._last_state_frame, "frame_index 不能早于当前轨迹状态帧。"
        assert self._range_current_m is not None
        assert self._speed_m_per_frame is not None
        dt = frame - self._last_state_frame
        predicted_range = self._range_current_m + dt * self._speed_m_per_frame
        assert _is_finite_number(predicted_range), "运动预测产生非有限距离；请重置或进入 RECAPTURE。"
        return float(predicted_range), float(self._speed_m_per_frame)

    def local_block_starts_m(self, frame_index: int) -> tuple[int, ...]:
        """按文档第 5.2 节生成当前搜索等级的去重局部块起点。"""
        if self._mode is not TrackerMode.TRACK:
            return ()
        predicted_range, predicted_speed = self.predict_state(frame_index)
        center_range = predicted_range - self.config.center_frame_offset * predicted_speed
        center_start = self.config.clamp_block_start(center_range - self.config.block_width_m / 2.0)

        starts: list[int] = []
        for offset in range(-self._search_level, self._search_level + 1):
            start = self.config.clamp_block_start(center_start + offset * self.config.block_step_m)
            if start not in starts:
                starts.append(start)
        return tuple(starts)

    def step(self, frame_index: int, candidate: Candidate | None) -> StepDiagnostics:
        """消费本步 q-Top1，并返回当前状态和门控诊断。

        ``frame_index`` 必须严格递增。CAPTURE/RECAPTURE 传入全局 Top-1，
        TRACK 传入局部 q-Top1。候选为 ``None`` 或包含 NaN/Inf 时不会污染内部状态，
        TRACK 中按一次失败处理。
        """
        frame = _require_nonnegative_int("frame_index", frame_index)
        assert self._last_step_frame is None or frame > self._last_step_frame, (
            f"frame_index 必须严格递增（上一帧为 {self._last_step_frame}，当前为 {frame}）。"
        )
        assert candidate is None or isinstance(candidate, Candidate), "candidate 必须是 Candidate 或 None。"

        mode_before = self._mode
        outcome = (
            self._step_capture(frame, candidate)
            if self._mode in {TrackerMode.CAPTURE, TrackerMode.RECAPTURE}
            else self._step_track(frame, candidate)
        )
        self._last_step_frame = frame
        return self._diagnostics(frame, mode_before, outcome)

    def _step_capture(self, frame_index: int, candidate: Candidate | None) -> _StepOutcome:
        candidate_valid = candidate is not None and candidate.is_valid
        support_count = 0
        capture_confirmed = False
        measurement_updated = False

        if candidate_valid:
            assert candidate is not None
            self._capture_buffer.append(_TimedCandidate(frame_index, candidate))
            estimate = self._capture_estimate(frame_index)
            support_count = 0 if estimate is None else estimate.support_count
            if (
                estimate is not None
                and len(self._capture_buffer) == self.config.capture_buffer_size
                and estimate.support_count >= self.config.capture_min_support
            ):
                self._initialise_track(frame_index, estimate)
                capture_confirmed = True
                measurement_updated = True

        return _StepOutcome(
            candidate_accepted=candidate_valid,
            measurement_updated=measurement_updated,
            capture_support_count=support_count,
            capture_confirmed=capture_confirmed,
        )

    def _step_track(self, frame_index: int, candidate: Candidate | None) -> _StepOutcome:
        if self._last_state_frame is None or self._range_current_m is None or self._speed_m_per_frame is None:
            # 防御性分支：不可能的半初始化状态不可继续输出旧轨迹。
            self._enter_recapture(preserve_extrapolation=False)
            return _StepOutcome()

        dt = frame_index - self._last_state_frame
        assert dt > 0, f"TRACK 状态的 dt 必须大于 0（last={self._last_state_frame}，current={frame_index}）。"
        evaluated_level = self._search_level
        predicted_range = self._range_current_m + dt * self._speed_m_per_frame
        predicted_speed = self._speed_m_per_frame
        if not _is_finite_number(predicted_range) or not _is_finite_number(predicted_speed):
            self._enter_recapture(preserve_extrapolation=False)
            return _StepOutcome(evaluated_search_level=evaluated_level)

        position_residual: float | None = None
        speed_residual: float | None = None
        accepted = False
        rejected_by: list[str] = []
        if candidate is None:
            rejected_by.append("missing_candidate")
        elif not candidate.is_valid:
            rejected_by.append("invalid_candidate")
        else:
            position_residual = float(candidate.latest_range_m) - predicted_range
            speed_residual = float(candidate.speed_m_per_frame) - predicted_speed
            if not _is_finite_number(position_residual) or not _is_finite_number(speed_residual):
                rejected_by.append("nonfinite_residual")
            elif float(candidate.q) < self.config.q_keep:
                rejected_by.append("q_keep")
            else:
                updated_range = predicted_range + self.config.alpha * position_residual
                position_speed = predicted_speed + self.config.beta * position_residual / dt
                updated_speed = (
                    (1.0 - self.config.gamma) * position_speed
                    + self.config.gamma * float(candidate.speed_m_per_frame)
                )
                average_speed = self._recent_average_speed_mpf(frame_index, updated_range)
                if not _is_finite_number(updated_range) or not _is_finite_number(updated_speed):
                    rejected_by.append("nonfinite_update")
                if abs(updated_speed) > self.config.instant_speed_gate_mpf[evaluated_level]:
                    rejected_by.append("instant_speed")
                if (
                    average_speed is not None
                    and abs(average_speed) > self.config.average_speed_gate_mpf[evaluated_level]
                ):
                    rejected_by.append("average_speed")
                if not rejected_by:
                    self._range_current_m = float(updated_range)
                    self._speed_m_per_frame = float(updated_speed)
                    self._last_state_frame = frame_index
                    accepted = True

        if accepted:
            self._good_count += 1
            self._bad_count = 0
            if self._good_count >= self.config.shrink_after_good:
                self._search_level = max(0, self._search_level - 1)
                self._good_count = 0
        else:
            # 门控未通过时保留匀速外推，而不是跳向高 q 的远距离假候选。
            self._range_current_m = float(predicted_range)
            self._speed_m_per_frame = float(predicted_speed)
            self._last_state_frame = frame_index
            self._bad_count += 1
            self._good_count = 0
            if self._bad_count >= self.config.expand_after_bad:
                if self._search_level < self.config.max_search_level:
                    self._search_level += 1
                    self._bad_count = 0
                else:
                    self._enter_recapture()

        if self._mode is TrackerMode.TRACK:
            assert self._range_current_m is not None
            self._append_state_history(frame_index, self._range_current_m)

        return _StepOutcome(
            candidate_accepted=accepted,
            measurement_updated=accepted,
            evaluated_search_level=evaluated_level,
            position_residual_m=position_residual,
            speed_residual_m_per_frame=speed_residual,
            rejected_by=tuple(rejected_by),
        )

    def _capture_estimate(self, reference_frame: int) -> _CaptureEstimate | None:
        if not self._capture_buffer:
            return None
        extrapolated_positions = np.asarray(
            [
                item.candidate.latest_range_m
                + (reference_frame - item.frame_index) * item.candidate.speed_m_per_frame
                for item in self._capture_buffer
            ],
            dtype=np.float64,
        )
        speeds = np.asarray(
            [item.candidate.speed_m_per_frame for item in self._capture_buffer],
            dtype=np.float64,
        )
        if not np.isfinite(extrapolated_positions).all() or not np.isfinite(speeds).all():
            return None
        capture_range = float(np.median(extrapolated_positions))
        support = np.abs(extrapolated_positions - capture_range) <= self.config.capture_radius_m
        support_count = int(np.count_nonzero(support))
        if support_count == 0:
            return None
        capture_speed = float(np.median(speeds[support]))
        if not _is_finite_number(capture_range) or not _is_finite_number(capture_speed):
            return None
        return _CaptureEstimate(capture_range, capture_speed, support_count)

    def _initialise_track(self, frame_index: int, estimate: _CaptureEstimate) -> None:
        self._mode = TrackerMode.TRACK
        self._range_current_m = estimate.range_current_m
        self._speed_m_per_frame = estimate.speed_m_per_frame
        self._last_state_frame = frame_index
        self._search_level = 0
        self._good_count = 0
        self._bad_count = 0
        # 进入 TRACK 后旧捕获候选不再参与后续重捕获，避免混入失效轨迹。
        self._capture_buffer.clear()
        self._extrapolation_range_m = None
        self._extrapolation_speed_mpf = None
        self._extrapolation_reference_frame = None
        self._state_history.clear()
        self._append_state_history(frame_index, estimate.range_current_m)

    def _enter_recapture(self, *, preserve_extrapolation: bool = True) -> None:
        """按文档第 8 节丢弃旧轨迹和旧缓存。

        保留的外推状态只服务于离线输出，绝不参与重捕获候选选择或确认。
        """
        can_extrapolate = preserve_extrapolation and all(
            value is not None and _is_finite_number(value)
            for value in (self._range_current_m, self._speed_m_per_frame, self._last_state_frame)
        )
        if can_extrapolate:
            self._extrapolation_range_m = float(self._range_current_m)
            self._extrapolation_speed_mpf = float(self._speed_m_per_frame)
            self._extrapolation_reference_frame = int(self._last_state_frame)
        else:
            self._extrapolation_range_m = None
            self._extrapolation_speed_mpf = None
            self._extrapolation_reference_frame = None
        self._mode = TrackerMode.RECAPTURE
        self._capture_buffer.clear()
        self._range_current_m = None
        self._speed_m_per_frame = None
        self._last_state_frame = None
        self._state_history.clear()
        self._search_level = 0
        self._good_count = 0
        self._bad_count = 0

    def _append_state_history(self, frame_index: int, range_current_m: float) -> None:
        self._state_history.append(_StatePoint(int(frame_index), float(range_current_m)))

    def _recent_average_speed_mpf(
        self,
        frame_index: int,
        proposed_range_m: float,
    ) -> float | None:
        """以至少 N 帧前的状态为锚点，计算候选更新后的平均速度。"""
        cutoff = frame_index - self.config.speed_average_window_frames
        anchor = next(
            (point for point in reversed(self._state_history) if point.frame_index <= cutoff),
            None,
        )
        if anchor is None:
            return None
        elapsed_frames = frame_index - anchor.frame_index
        if elapsed_frames <= 0:
            return None
        average_speed = (float(proposed_range_m) - anchor.range_current_m) / elapsed_frames
        return _as_finite_or_none(average_speed)

    def _diagnostics(
        self,
        frame_index: int,
        mode_before: TrackerMode,
        outcome: _StepOutcome,
    ) -> StepDiagnostics:
        if self._mode is TrackerMode.TRACK:
            assert self._range_current_m is not None
            assert self._speed_m_per_frame is not None
            range_current = float(self._range_current_m)
            speed = float(self._speed_m_per_frame)
            if not _is_finite_number(range_current) or not _is_finite_number(speed):
                range_current = None
                speed = None
        else:
            range_current = None
            speed = None

        extrapolation_range = None
        extrapolation_speed = None
        extrapolation_frame = None
        if self._mode is TrackerMode.RECAPTURE:
            extrapolation_range = self._extrapolation_range_m
            extrapolation_speed = self._extrapolation_speed_mpf
            extrapolation_frame = self._extrapolation_reference_frame

        return StepDiagnostics(
            frame_index=frame_index,
            mode=self._mode,
            mode_before=mode_before,
            range_current_m=range_current,
            speed_m_per_frame=speed,
            candidate_accepted=outcome.candidate_accepted,
            measurement_updated=outcome.measurement_updated,
            search_level=self._search_level,
            evaluated_search_level=outcome.evaluated_search_level,
            position_residual_m=outcome.position_residual_m,
            speed_residual_m_per_frame=outcome.speed_residual_m_per_frame,
            capture_buffer_size=len(self._capture_buffer),
            capture_support_count=outcome.capture_support_count,
            capture_confirmed=outcome.capture_confirmed,
            rejected_by=outcome.rejected_by,
            extrapolation_range_m=extrapolation_range,
            extrapolation_speed_mpf=extrapolation_speed,
            extrapolation_reference_frame=extrapolation_frame,
        )

    @staticmethod
    def _coerce_mode(mode: TrackerMode | str) -> TrackerMode:
        try:
            return TrackerMode(mode)
        except ValueError as exc:
            raise AssertionError(f"未知 TrackerMode: {mode!r}。") from exc
