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

def _is_finite_number(value) -> bool:
    """仅接受可转换为有限 float 的标量，防止 NaN/Inf 污染状态。"""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def _as_finite_or_none(value: float) -> float | None:
    """将有限数统一转换为 ``float``；非法值以 ``None`` 表示。"""
    return float(value) if _is_finite_number(value) else None

def _require_positive_int(name: str, value: object) -> int:
    """校验配置中的正整数，并返回 Python ``int``。"""
    assert not isinstance(value, bool) and isinstance(value, (int, np.integer)) and int(value) > 0, f"{name} 必须是正整数，实际为 {value!r}。"
    return int(value)

def _require_nonnegative_int(name: str, value: object) -> int:
    """校验配置中的非负整数，并返回 Python ``int``。"""
    assert not isinstance(value, bool) and isinstance(value, (int, np.integer)) and int(value) >= 0, f"{name} 必须是非负整数，实际为 {value!r}。"
    return int(value)

def _round_bin(value: float) -> int:
    """距离 bin 的确定性四舍五入；距离轴通常非负，避免 Python 的银行家舍入。"""
    return int(math.floor(float(value) + 0.5))


@dataclass(frozen=True)
class Candidate:
    """一次实际 CNN batch 中 q 最大块换算后的候选。

    ``latest_range_m`` 必须是文档第 2 节定义的最新帧位置 ``z_t``，而不是模型原始输出的块内中心距离 ``rho_m``。
    使用 `from_block_prediction` 完成 CNN 原始输出坐标转换。
    """

    q: float                            # CNN 预测的目标存在分类分数（已过 sigmoid，范围应为 [0, 1]）
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
        """将模型块内输出换算为全局距离轴、最新帧坐标下的候选。

        对 20 帧窗口，``rho_m`` 对应中心时刻而非最新帧。因此先叠加距离块起点，再沿 ``speed_m_per_frame`` 前进 9.5 帧，
        得到后续捕获、门控和状态更新统一使用的 ``latest_range_m``。
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
        """判断候选能否安全进入聚类、门控与状态更新。"""
        if not all(_is_finite_number(value) for value in (self.q, self.latest_range_m, self.speed_m_per_frame)):
            return False
        if not 0.0 <= float(self.q) <= 1.0:
            return False
        if self.block_start_m is not None and not _is_finite_number(self.block_start_m):
            return False
        return True


@dataclass(frozen=True)
class TrackerConfig:
    """后处理参数。

    ``instant_speed_gate_mpf`` 和 ``average_speed_gate_mpf`` 的第 ``L`` 项分别限制单帧融合状态速度与最近窗口平均速度，单位均为米/帧
    默认仅实现L=0/1/2 三个局部搜索等级，L=2 再失败会进入 ``RECAPTURE``
    """

    capture_buffer_size: int = 8
    capture_support_ratio: float = 0.7
    capture_radius_m: float = 500.0
    capture_q_min: float = 0.5
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
        """在状态机启动前一次性校验全部约束，不修改冻结配置。"""
        # 先校验会参与索引、计数和 deque 容量的离散参数。
        _require_positive_int("capture_buffer_size", self.capture_buffer_size)
        _require_positive_int("expand_after_bad", self.expand_after_bad)
        _require_positive_int("shrink_after_good", self.shrink_after_good)
        _require_positive_int("frames_per_window", self.frames_per_window)
        _require_positive_int("speed_average_window_frames", self.speed_average_window_frames)
        _require_nonnegative_int("max_search_level", self.max_search_level)

        # 连续参数只接受有限数值；此处仅校验，保留调用方传入的原始数值类型。
        finite_fields = {
            "capture_support_ratio": self.capture_support_ratio,
            "capture_radius_m": self.capture_radius_m,
            "capture_q_min": self.capture_q_min,
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
            assert not isinstance(value, (bool, np.bool_)) and _is_finite_number(value), (
                f"{name} 必须是有限数值，实际为 {value!r}。"
            )

        assert 0.0 < self.capture_support_ratio <= 1.0, "capture_support_ratio 必须在 (0, 1] 内。"
        assert self.capture_radius_m >= 0.0, "capture_radius_m 不能为负数。"
        assert 0.0 <= self.capture_q_min <= 1.0, "capture_q_min 必须在 [0, 1] 内。"
        assert 0.0 <= self.q_keep <= 1.0, "q_keep 必须在 [0, 1] 内。"
        assert 0.0 <= self.alpha <= 1.0, "alpha 必须在 [0, 1] 内。"
        assert self.beta >= 0.0, "beta 不能为负数。"
        assert 0.0 <= self.gamma <= 1.0, "gamma 必须在 [0, 1] 内。"
        assert self.block_width_m > 0.0 and self.block_step_m > 0.0, "block_width_m 和 block_step_m 必须为正数。"
        assert self.range_max_m > self.range_min_m, "range_max_m 必须大于 range_min_m。"
        assert self.range_max_m - self.range_min_m >= self.block_width_m, "测距范围必须至少容纳一个完整 block_width_m 块。"

        # 两组速度门限必须与 L=0,...,max_search_level 一一对应。
        for name in ("instant_speed_gate_mpf", "average_speed_gate_mpf"):
            try:
                gates = tuple(getattr(self, name))
            except TypeError as exc:
                raise AssertionError(f"{name} 必须是按搜索等级排列的数值序列。") from exc
            assert len(gates) == self.max_search_level + 1, (
                f"{name} 长度必须等于 max_search_level + 1，实际为 {len(gates)}。"
            )
            assert all(
                not isinstance(value, (bool, np.bool_))
                and _is_finite_number(value)
                and float(value) >= 0.0
                for value in gates
            ), f"{name} 中所有门限必须为非负有限数。"

    @property
    def center_frame_offset(self) -> float:
        """CNN ``rho_m`` 对应窗口中心相对最新帧的偏移，20 帧时为 9.5。"""
        return (self.frames_per_window - 1) / 2.0

    @property
    def capture_min_support(self) -> int:
        """用于捕获确认的 ``ceil(eta_cap * K_cap)``。"""
        return int(math.ceil(self.capture_support_ratio * self.capture_buffer_size))

    @property
    def max_block_start_m(self) -> float:
        """返回完整宽度距离块允许的最大起点，保证右端不越过测距范围。"""
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
    """单步状态机结果；仅包含状态转移和诊断真正需要的数据。

    ``range_current_m`` 与 ``speed_m_per_frame`` 仅在步结束后仍为
    ``TRACK`` 时有效；进入 ``RECAPTURE`` 后的临时外推参数位于
    ``extrapolation_*`` 三个字段，不能反馈给状态机决策。
    """

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
    """带产生帧号的全局 Top-1，用于捕获阶段的运动补偿。"""

    frame_index: int
    candidate: Candidate


@dataclass(frozen=True)
class _StatePoint:
    """用于按真实帧跨度计算平均状态速度的历史锚点。"""

    frame_index: int
    range_current_m: float


@dataclass(frozen=True)
class _CaptureEstimate:
    """捕获缓存聚类后的最新帧位置、支持候选斜率和支持数。"""

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
    """ CAPTURE / TRACK / RECAPTURE 状态机。

    调用方根据 :attr:`mode` 决定本步应做全局还是局部 CNN 推理；从 batch
    取得 q-Top1 后调用 :meth:`step`。
    该方法只接受已经换算到最新帧坐标的 :class:`Candidate`，不会接触模型。
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        """创建状态机；未传配置时使用文档给出的默认参数。"""
        self.config = TrackerConfig() if config is None else config
        assert isinstance(self.config, TrackerConfig), "config 必须是 TrackerConfig。"
        self.reset()

    def reset(self, mode: TrackerMode | str = TrackerMode.CAPTURE) -> None:
        """清空缓存、滤波状态与计数器；用于开始一条新样本序列。

        ``mode`` 主要服务测试：正式推理通常从 ``CAPTURE`` 开始。即使
        传入 ``TRACK``，也不会凭空构造距离和速度状态，调用方仍需负责
        后续的合法初始化。
        """
        self._mode = self._coerce_mode(mode)
        # CAPTURE 与 RECAPTURE 共享这一 FIFO 缓存，但发生模式切换时会清空旧候选。
        self._capture_buffer: deque[_TimedCandidate] = deque(maxlen=self.config.capture_buffer_size)
        # 最近一次 TRACK 内部状态对应的“最新帧”位置与速度。
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
        """返回当前工作模式：CAPTURE、TRACK 或 RECAPTURE。"""
        return self._mode

    @property
    def range_current_m(self) -> float | None:
        """返回 TRACK 状态最新帧的内部距离估计；非 TRACK 时为 ``None``。"""
        return self._range_current_m

    @property
    def speed_m_per_frame(self) -> float | None:
        """返回 TRACK 状态的内部速度估计，单位为米/帧；非 TRACK 时为 ``None``。"""
        return self._speed_m_per_frame

    @property
    def search_level(self) -> int:
        """返回下一次局部搜索使用的等级 ``L``，对应 1/3/5 个候选块。"""
        return self._search_level

    @property
    def capture_buffer_size(self) -> int:
        """返回当前捕获缓存中实际保留的全局 Top-1 候选数。"""
        return len(self._capture_buffer)

    def global_block_starts_m(self) -> tuple[int, ...]:
        """暴露标准全局网格，供 CAPTURE/RECAPTURE 调用方一次组成 batch。"""
        return self.config.global_block_starts_m()

    def predict_state(self, frame_index: int) -> tuple[float, float]:
        """将最近 TRACK 状态匀速外推到指定最新帧，不修改内部状态。

        返回 ``(预测最新帧距离, 预测速度)``。该方法只负责运动预测；窗口
        中心时刻的距离换算由 :meth:`local_block_starts_m` 完成。
        """
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
        """为指定最新帧生成当前搜索等级的局部距离块起点。

        先将状态外推到窗口最新帧，再回退到 20 帧窗口中心，使块中心与
        CNN ``rho_m`` 的时间定义一致。最后按等级 ``L`` 以 9 km 步长
        向两侧扩展，并处理测距边界造成的重复块。
        """
        if self._mode is not TrackerMode.TRACK:
            return ()

        # 将捕获/上一次 TRACK 状态从其所在时间窗最新帧匀速外推到当前待推理时间窗的最新帧
        predicted_range, predicted_speed = self.predict_state(frame_index)

        # 回退 center_frame_offset 帧，得到当前时间窗中心时刻的预测距离
        center_range = predicted_range - self.config.center_frame_offset * predicted_speed

        # 将以 center_range 为中心截取距离块，取得距离起点
        center_start = self.config.clamp_block_start(center_range - self.config.block_width_m / 2.0)

        # L=0/1/2 分别尝试中心 1 块、中心加两侧 3 块、中心加两侧 5 块。
        starts: list[int] = []
        for offset in range(-self._search_level, self._search_level + 1):
            start = self.config.clamp_block_start(center_start + offset * self.config.block_step_m)
            # 两端裁剪可能把不同 offset 映射到相同起点，避免重复前向。
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

        # 一个逻辑步只消费一个 q-Top1：全局扫描的候选用于捕获，局部扫描的候选用于跟踪。
        mode_before = self._mode
        outcome = (
            self._step_capture(frame, candidate)
            if self._mode in {TrackerMode.CAPTURE, TrackerMode.RECAPTURE}
            else self._step_track(frame, candidate)
        )
        self._last_step_frame = frame
        return self._diagnostics(frame, mode_before, outcome)

    def _step_capture(self, frame_index: int, candidate: Candidate | None) -> _StepOutcome:
        """处理一次 CAPTURE 或 RECAPTURE 的全局 Top-1，并尝试确认轨迹。"""
        candidate_valid = candidate is not None and candidate.is_valid
        candidate_accepted = False
        rejected_by: list[str] = []
        support_count = 0
        capture_confirmed = False
        measurement_updated = False

        if not candidate_valid:
            if candidate is not None:
                rejected_by.append("invalid_candidate")
        else:
            assert candidate is not None
            if float(candidate.q) < self.config.capture_q_min:
                rejected_by.append("capture_q_min")
            else:
                # 仅缓存通过捕获 q 门限的候选；缓存满后最早候选由 deque 自动弹出。
                candidate_accepted = True
                self._capture_buffer.append(_TimedCandidate(frame_index, candidate))
                estimate = self._capture_estimate(frame_index)
                support_count = 0 if estimate is None else estimate.support_count
                # 必须同时满足“缓存已满”和“中位位置获得足够支持”才切换 TRACK。
                if (
                    estimate is not None
                    and len(self._capture_buffer) == self.config.capture_buffer_size
                    and estimate.support_count >= self.config.capture_min_support
                ):
                    self._initialise_track(frame_index, estimate)   # 转入 TRACK 模式，初始化状态机
                    capture_confirmed = True
                    measurement_updated = True

        return _StepOutcome(
            candidate_accepted=candidate_accepted,
            measurement_updated=measurement_updated,
            capture_support_count=support_count,
            capture_confirmed=capture_confirmed,
            rejected_by=tuple(rejected_by),
        )

    def _step_track(self, frame_index: int, candidate: Candidate | None) -> _StepOutcome:
        """处理一次局部 q-Top1：门控、滤波更新以及搜索等级状态转移。"""
        if self._last_state_frame is None or self._range_current_m is None or self._speed_m_per_frame is None:
            # 防御性分支：不可能的半初始化状态不可继续输出旧轨迹。
            self._enter_recapture(preserve_extrapolation=False)
            return _StepOutcome()

        # 先把内部状态外推到当前窗口最新帧；后续所有残差均在该时刻比较。
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
        # 依次做候选存在性、数值有效性、q 门限和速度连续性检查。
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
                # 先基于位置残差形成暂定 alpha-beta 更新，门控通过才真正写回状态。
                updated_range = predicted_range + self.config.alpha * position_residual
                position_speed = predicted_speed + self.config.beta * position_residual / dt
                updated_speed = (
                    (1.0 - self.config.gamma) * position_speed
                    + self.config.gamma * float(candidate.speed_m_per_frame)
                )
                # 平均速度锚定到至少 N 帧前的历史状态，短历史阶段则不启用该门控。
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
            # 有效测量：重置失败计数；连续足够多次后才逐级收缩搜索范围。
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
            # 连续失败后快速扩大；已达最大等级仍失败时丢弃 TRACK 状态并重捕获。
            if self._bad_count >= self.config.expand_after_bad:
                if self._search_level < self.config.max_search_level:
                    self._search_level += 1
                    self._bad_count = 0
                else:
                    self._enter_recapture()

        # 被拒绝的外推状态同样进入历史：平均速度门控约束的是实际内部轨迹。
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
        """将缓存候选补偿到参考帧，并估计可用于初始化 TRACK 的状态。

        所有候选先按自身速度外推到 ``reference_frame``。外推位置的中位数
        是捕获距离；半径内的候选构成支持集，其速度中位数作为初始速度。
        """
        if not self._capture_buffer:
            return None
        # 历史缓存的 Top-1 预测须先补偿到当前参考帧，才能做空间聚类。
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
        
        # 用中位数抑制少量远距离误报，再以固定半径选出一致候选。
        capture_range = float(np.median(extrapolated_positions))
        support = np.abs(extrapolated_positions - capture_range) <= self.config.capture_radius_m
        support_count = int(np.count_nonzero(support))
        if support_count == 0:
            return None

        # 取支持候选的速度中位数
        capture_speed = float(np.median(speeds[support]))
        if not _is_finite_number(capture_range) or not _is_finite_number(capture_speed):
            return None
        return _CaptureEstimate(capture_range, capture_speed, support_count)

    def _initialise_track(self, frame_index: int, estimate: _CaptureEstimate) -> None:
        """用确认后的捕获估计初始化 TRACK 状态与历史。"""
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
        # 新轨迹不继承旧 TRACK 或旧 CAPTURE 的历史，避免平均速度门控混入旧目标。
        self._state_history.clear()
        self._append_state_history(frame_index, estimate.range_current_m)

    def _enter_recapture(self, *, preserve_extrapolation: bool = True) -> None:
        """重捕获前丢弃旧轨迹和旧缓存。保留的外推状态只服务于离线输出，不参与重捕获候选选择或确认"""
        # 旧状态不参与重捕获，但可复制一份仅供输出端绘制不可靠外推。
        can_extrapolate = preserve_extrapolation and all(
            value is not None and _is_finite_number(value)
            for value in (self._range_current_m, self._speed_m_per_frame, self._last_state_frame)
        )
        if can_extrapolate:
            assert self._last_state_frame is not None
            assert self._range_current_m is not None
            assert self._speed_m_per_frame is not None
            self._extrapolation_range_m = float(self._range_current_m)
            self._extrapolation_speed_mpf = float(self._speed_m_per_frame)
            self._extrapolation_reference_frame = int(self._last_state_frame)
        else:
            self._extrapolation_range_m = None
            self._extrapolation_speed_mpf = None
            self._extrapolation_reference_frame = None
        # 进入 RECAPTURE 后完全清空决策状态；下一步必须重新做全局扫描。
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
        """向固定长度历史队列写入当前内部位置，用于后续平均速度门控。"""
        self._state_history.append(_StatePoint(int(frame_index), float(range_current_m)))

    def _recent_average_speed_mpf(
        self,
        frame_index: int,
        proposed_range_m: float,
    ) -> float | None:
        """以至少 N 帧前的状态为锚点，计算候选更新后的平均速度。

        返回 ``None`` 说明历史跨度还不足，此时调用方只做瞬时速度门控。
        """
        cutoff = frame_index - self.config.speed_average_window_frames
        # 从最近的历史点向前找，选取“不晚于 cutoff”的最近锚点。
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
        """将当前内部状态和本步结果封装为对调用方稳定的诊断记录。"""
        # 只有 TRACK 可对外声明可靠状态；RECAPTURE 的旧状态只作为临时外推字段返回。
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
        """将枚举或字符串规范化为 ``TrackerMode``，并将非法值转为明确断言错误。"""
        try:
            return TrackerMode(mode)
        except ValueError as exc:
            raise AssertionError(f"未知 TrackerMode: {mode!r}。") from exc
