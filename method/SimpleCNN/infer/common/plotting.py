"""不参与推理决策的公共诊断绘图工具。

图中的真值、目标命中标签和纯背景只用于离线复核；调用方必须在调用本模块前
完成推理，不能把本模块返回的任何信息回灌给候选块选择或状态机。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np

from configs.base import SimpleCNNConfig
from data.dataloader import PackedSource
from infer.common.metrics import trajectory_metrics


_PLOT_FIELDS = (
    "background_only_packed",
    "observation_packed",
    "background_probability_1m",
    "target_true_range_m",
    "target_hit",
    "target_hit_bin",
)
_FALSE_ALARM_PLOT_FIELDS = (
    "background_only_packed",
    "background_probability_1m",
)


def _load_plot_data(
    source: PackedSource,
    *,
    fields: Sequence[str] = _PLOT_FIELDS,
) -> dict[str, np.ndarray]:
    """从当前推理样本的 NPZ 读取指定诊断图字段。"""
    with np.load(source.record.path, allow_pickle=False) as archive:
        missing = set(fields).difference(archive.files)
        if missing:
            raise KeyError(f"{source.record.path} 缺少诊断图字段：{sorted(missing)}")
        return {name: archive[name] for name in fields}


def _unpack_local_range(
    packed: np.ndarray,
    range_start: int,
    range_stop: int,
    *,
    bitorder: str,
) -> np.ndarray:
    """仅解包绘图所需的局部距离区间，避免展开完整 300 km 二值矩阵。"""
    if bitorder not in {"big", "little"}:
        raise ValueError(f"不支持的位序：{bitorder!r}。")
    validated_bitorder = cast(Literal["big", "little"], bitorder)
    array = np.asarray(packed, dtype=np.uint8)
    assert array.ndim == 2, f"packed 观测应为二维数组，实际形状为 {array.shape}。"
    assert 0 <= range_start < range_stop <= array.shape[1] * 8, "局部距离范围超出 packed 观测可表示的范围。"
    first_byte = range_start // 8
    bit_offset = range_start % 8
    width = range_stop - range_start
    byte_count = math.ceil((bit_offset + width) / 8)
    unpacked = np.unpackbits(
        array[:, first_byte : first_byte + byte_count], axis=1, bitorder=validated_bitorder
    )
    return unpacked[:, bit_offset : bit_offset + width].astype(bool, copy=False)


def _background_probability(data: Mapping[str, np.ndarray], *, range_bins: int) -> np.ndarray:
    probability = np.asarray(data["background_probability_1m"], dtype=np.float64).reshape(-1)
    assert len(probability) == range_bins, "background_probability_1m 与距离轴长度不一致。"
    return probability


def _coarsen_probability(probability: np.ndarray, width_m: int) -> tuple[np.ndarray, np.ndarray]:
    assert width_m >= 1, "probability_bin_m 必须为正整数。"
    starts = np.arange(0, len(probability), width_m, dtype=np.int64)
    values = np.empty(len(starts), dtype=np.float64)
    centres = np.empty(len(starts), dtype=np.float64)
    for index, start in enumerate(starts):
        stop = min(int(start + width_m), len(probability))
        values[index] = float(np.nanmean(probability[start:stop]))
        centres[index] = (start + stop) * 0.5 / 1_000.0
    return centres, values


def _prediction_category(mode: Any, measurement_updated: Any) -> str:
    mode_text = "" if mode is None else str(mode).upper()
    if "RECAPTURE" in mode_text:
        return "重捕获"
    if "CAPTURE" in mode_text:
        return "捕获"
    if measurement_updated is True:
        return "测量更新"
    if measurement_updated is False:
        return "运动外推"
    return "模型预测"


def _first_output_frame(prediction_m: np.ndarray) -> int:
    """返回首个有效预测帧；全程无输出时返回完整序列长度。"""
    finite_frames = np.flatnonzero(np.isfinite(prediction_m))
    return int(finite_frames[0]) if len(finite_frames) else len(prediction_m)


def _plot_prediction_segments(
    axis: Any,
    *,
    time_seconds: np.ndarray,
    prediction_m: np.ndarray,
    mode: np.ndarray,
    measurement_updated: np.ndarray,
) -> None:
    """按连续状态段绘制预测，避免跨未覆盖帧或状态切换错误连线。"""
    colors = {
        "模型预测": "#E64B35",
        "测量更新": "#E64B35",
        "运动外推": "#377EB8",
        "捕获": "#F0A202",
        "重捕获": "#8E5EA2",
    }
    labels = {
        "模型预测": "模型预测",
        "测量更新": "CNN 测量更新",
        "运动外推": "运动模型外推",
        "捕获": "捕获阶段输出",
        "重捕获": "重捕获阶段不可靠外推",
    }
    finite_indices = np.flatnonzero(np.isfinite(prediction_m))
    if not len(finite_indices):
        return
    categories = [
        _prediction_category(mode[index], measurement_updated[index]) for index in finite_indices
    ]
    shown: set[str] = set()
    start = 0
    while start < len(finite_indices):
        stop = start + 1
        while (
            stop < len(finite_indices)
            and finite_indices[stop] == finite_indices[stop - 1] + 1
            and categories[stop] == categories[start]
        ):
            stop += 1
        indices = finite_indices[start:stop]
        category = categories[start]
        axis.plot(
            time_seconds[indices],
            prediction_m[indices] / 1_000.0,
            color=colors[category],
            linewidth=1.8,
            marker="o" if len(indices) == 1 else None,
            markersize=3.0,
            alpha=0.92,
            linestyle="--" if category == "重捕获" else "-",
            zorder=4,
            label=labels[category] if category not in shown else None,
        )
        shown.add(category)
        start = stop


def _style_axis(axis: Any) -> None:
    axis.grid(alpha=0.22, linewidth=0.65)
    axis.tick_params(axis="both", labelsize=9)
    for spine in axis.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.75)


def _apply_chinese_font(figure: Any) -> None:
    """为本张图锁定已安装的中文字体，避免离开 rc_context 后回退。"""
    from matplotlib import font_manager
    from matplotlib.text import Text

    font_path: str | None = None
    for family in ("Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei"):
        try:
            font_path = font_manager.findfont(family, fallback_to_default=False)
            break
        except ValueError:
            continue
    if font_path is None:
        return
    for text in figure.findobj(match=Text):
        properties = text.get_fontproperties().copy()
        properties.set_file(font_path)
        text.set_fontproperties(properties)


def _format_metric(value: Any, format_spec: str, fallback: str = "—") -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return format(numeric, format_spec) if np.isfinite(numeric) else fallback


def _status_series(
    steps: Sequence[Mapping[str, Any]],
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把自适应结果的逐窗状态覆盖到各自的未来预测帧。"""
    mode = np.full(frame_count, None, dtype=object)
    measurement_updated = np.full(frame_count, None, dtype=object)
    for step in steps:
        if "forecast_frame_start" not in step:
            continue  # 全局 Top-1 没有状态机标记，绘图使用默认“模型预测”。
        start = int(step["forecast_frame_start"])
        stop = int(step["forecast_frame_stop"])
        assert 0 <= start <= stop <= frame_count, "逐窗预测范围超出完整帧轴。"
        mode[start:stop] = step["mode"]
        measurement_updated[start:stop] = step["measurement_updated"]
    return mode, measurement_updated


def plot_source_diagnostic(
    source: PackedSource,
    result: Mapping[str, Any],
    *,
    config: SimpleCNNConfig,
    title: str,
    metrics: Mapping[str, Any],
    frame_interval_s: float = 0.05,
    margin_m: float = 5_000.0,
    point_size: float = 1.0,
    probability_bin_m: int = 1_000,
) -> Any:
    """为当前 ``PackedSource`` 与统一推理结果生成中文 2×2 诊断图。"""
    assert np.isfinite(frame_interval_s) and frame_interval_s > 0.0, "frame_interval_s 必须为正有限数。"
    assert np.isfinite(margin_m) and margin_m >= 0.0, "margin_m 必须为非负有限数。"
    assert point_size > 0.0, "point_size 必须为正。"

    import matplotlib.pyplot as plt  # 延迟导入，避免纯推理路径依赖 matplotlib。

    data = _load_plot_data(source)
    bitorder = config.packed_bitorder
    range_bins = int(config.range_bins)
    true_range_m = np.asarray(data["target_true_range_m"], dtype=np.float64).reshape(-1)
    target_hit = np.asarray(data["target_hit"], dtype=bool).reshape(-1)
    target_hit_bin = np.asarray(data["target_hit_bin"], dtype=np.int64).reshape(-1)
    frame_count = len(true_range_m)
    assert frame_count >= 1 and target_hit.shape == true_range_m.shape == target_hit_bin.shape, "真实轨迹、target_hit 和 target_hit_bin 必须是同长度的一维数组。"
    assert np.all(np.isfinite(true_range_m)), "target_true_range_m 含非有限值。"

    prediction_m = np.asarray(result["prediction_m"], dtype=np.float64).reshape(-1)
    assert len(prediction_m) == frame_count, "prediction_m 必须与完整真值帧轴等长。"
    steps = result["steps"]
    assert isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)), "推理结果 steps 必须为逐窗记录序列。"
    assert all(isinstance(step, Mapping) for step in steps), "推理结果 steps 必须全部为字典。"
    mode, measurement_updated = _status_series(steps, frame_count)
    base_metrics = trajectory_metrics(prediction_m, true_range_m, target_hit=target_hit)
    base_metrics.update(metrics)

    # 左上以外的两个局部观测图只围绕真实目标显示，避免异常预测拉大它们的尺度。
    target_margin_m = 6_000.0
    target_range_start = max(0, int(math.floor(float(np.min(true_range_m)) - target_margin_m)))
    target_range_stop = min(
        range_bins,
        int(math.ceil(float(np.max(true_range_m)) + target_margin_m + 1.0)),
    )
    if target_range_stop <= target_range_start:
        target_range_stop = min(range_bins, target_range_start + 1)

    finite_display = np.concatenate(
        [true_range_m[np.isfinite(true_range_m)], prediction_m[np.isfinite(prediction_m)]]
    )
    assert len(finite_display), "没有可用于绘图的轨迹位置。"
    range_start = max(0, int(math.floor(float(finite_display.min()) - margin_m)))
    range_stop = min(
        range_bins,
        int(math.ceil(float(finite_display.max()) + margin_m + 1.0)),
    )
    if range_stop <= range_start:
        range_stop = min(range_bins, range_start + 1)

    target_background = _unpack_local_range(
        data["background_only_packed"], target_range_start, target_range_stop, bitorder=bitorder
    )
    target_observation = _unpack_local_range(
        data["observation_packed"], target_range_start, target_range_stop, bitorder=bitorder
    )
    local_background = _unpack_local_range(
        data["background_only_packed"], range_start, range_stop, bitorder=bitorder
    )
    assert (
        target_background.shape[0] == frame_count
        and target_observation.shape == target_background.shape
        and local_background.shape[0] == frame_count
    ), "背景、观测和真实轨迹的帧数或局部形状不一致。"
    probability = _background_probability(data, range_bins=range_bins)
    probability_centres_km, probability_values = _coarsen_probability(probability, int(probability_bin_m))

    time_seconds = np.arange(frame_count, dtype=np.float64) * float(frame_interval_s)
    background_frames, background_bins = np.nonzero(local_background)
    target_background_frames, target_background_bins = np.nonzero(target_background)
    observation_frames, observation_bins = np.nonzero(target_observation)
    hit_mask = (
        target_hit
        & (target_hit_bin >= target_range_start)
        & (target_hit_bin < target_range_stop)
    )
    target_y_limits = (target_range_start / 1_000.0, target_range_stop / 1_000.0)
    local_y_limits = (range_start / 1_000.0, range_stop / 1_000.0)
    # 全局方法通常在输入窗后首次输出；自适应方法则要等 CAPTURE 确认。
    warmup_frames = _first_output_frame(prediction_m)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(16.0, 10.2), constrained_layout=True)
        figure.suptitle(title, fontsize=15, fontweight="normal")

        probability_axis = axes[0, 0]
        positive_probability = np.maximum(probability_values, np.finfo(np.float64).tiny)
        probability_axis.semilogy(
            probability_centres_km, positive_probability, color="#4C78A8", linewidth=1.3
        )
        probability_axis.axvspan(
            float(np.min(true_range_m)) / 1_000.0,
            float(np.max(true_range_m)) / 1_000.0,
            color="#E64B35",
            alpha=0.16,
            label="目标经过距离",
        )
        probability_axis.set(
            title="拟合背景占据概率",
            xlabel="距离（千米）",
            ylabel="每米距离单元的占据概率",
            xlim=(0.0, range_bins / 1_000.0),
        )
        probability_axis.legend(fontsize=8, loc="upper right")
        _style_axis(probability_axis)

        observation_axis = axes[0, 1]
        observation_axis.scatter(
            time_seconds[observation_frames],
            (target_range_start + observation_bins) / 1_000.0,
            s=point_size,
            marker="s",
            linewidths=0,
            alpha=0.76,
            color="#4C9AD4",
            rasterized=True,
            label="二值响应（未标注）",
        )
        observation_axis.set(
            title="含目标的局部二值时距图（未标注）",
            xlabel="时间（秒）",
            ylabel="距离（千米）",
            ylim=target_y_limits,
        )
        observation_axis.legend(fontsize=8, loc="upper right")
        _style_axis(observation_axis)

        truth_axis = axes[1, 0]
        truth_axis.scatter(
            time_seconds[target_background_frames],
            (target_range_start + target_background_bins) / 1_000.0,
            s=point_size,
            marker="s",
            linewidths=0,
            alpha=0.64,
            color="#4C9AD4",
            rasterized=True,
            label="背景响应",
        )
        truth_axis.plot(
            time_seconds,
            true_range_m / 1_000.0,
            color="#252525",
            linewidth=1.25,
            label="真实潜在轨迹",
            zorder=3,
        )
        if np.any(hit_mask):
            truth_axis.scatter(
                time_seconds[hit_mask],
                target_hit_bin[hit_mask] / 1_000.0,
                s=10,
                color="#E64B35",
                zorder=4,
                label="实际目标响应",
            )
        truth_axis.set(
            title="纯背景、真实轨迹与实际目标响应",
            xlabel="时间（秒）",
            ylabel="距离（千米）",
            ylim=target_y_limits,
        )
        truth_axis.legend(fontsize=8, loc="upper right")
        _style_axis(truth_axis)

        prediction_axis = axes[1, 1]
        prediction_axis.scatter(
            time_seconds[background_frames],
            (range_start + background_bins) / 1_000.0,
            s=point_size,
            marker="s",
            linewidths=0,
            alpha=0.48,
            color="#4C9AD4",
            rasterized=True,
            label="背景响应",
        )
        prediction_axis.plot(
            time_seconds,
            true_range_m / 1_000.0,
            color="#555555",
            linewidth=1.15,
            linestyle="--",
            alpha=0.92,
            label="真实轨迹（对照）",
            zorder=3,
        )
        _plot_prediction_segments(
            prediction_axis,
            time_seconds=time_seconds,
            prediction_m=prediction_m,
            mode=mode,
            measurement_updated=measurement_updated,
        )
        if warmup_frames > 0:
            prediction_axis.axvspan(
                0.0,
                min(frame_count, int(warmup_frames)) * frame_interval_s,
                color="#777777",
                alpha=0.08,
                label="首次输出前阶段",
                zorder=0,
            )
        prediction_axis.set(
            title=(
                "流式预测："
                f"MAE {_format_metric(base_metrics.get('mae_m'), '.1f')} 米，"
                f"覆盖率 {_format_metric(base_metrics.get('coverage'), '.1%')}，"
                f"不可靠覆盖 {_format_metric(base_metrics.get('unreliable_coverage'), '.1%')}，"
                f"总推理时间 {_format_metric(base_metrics.get('end_to_end_total_s'), '.3f')} 秒"
            ),
            xlabel="时间（秒）",
            ylabel="距离（千米）",
            ylim=local_y_limits,
        )
        prediction_axis.legend(fontsize=8, loc="upper right")
        _style_axis(prediction_axis)
        _apply_chinese_font(figure)

    return figure


def plot_false_alarm_diagnostic(
    source: PackedSource,
    result: Mapping[str, Any],
    *,
    config: SimpleCNNConfig,
    title: str,
    false_alarm: Mapping[str, Any],
    capture_q_min: float,
    q_keep: float,
    frame_interval_s: float = 0.05,
    margin_m: float = 5_000.0,
    point_size: float = 1.0,
    probability_bin_m: int = 1_000,
) -> Any:
    """绘制纯背景样本的虚警诊断图，不使用潜在目标轨迹或目标标签。"""
    if not np.isfinite(frame_interval_s) or frame_interval_s <= 0.0:
        raise ValueError("frame_interval_s 必须为正有限数。")
    if not np.isfinite(margin_m) or margin_m < 0.0:
        raise ValueError("margin_m 必须为非负有限数。")
    if point_size <= 0.0:
        raise ValueError("point_size 必须为正。")
    if not np.isfinite(capture_q_min) or not 0.0 <= capture_q_min <= 1.0:
        raise ValueError("capture_q_min 必须在 [0, 1] 内。")
    if not np.isfinite(q_keep) or not 0.0 <= q_keep <= 1.0:
        raise ValueError("q_keep 必须在 [0, 1] 内。")

    import matplotlib.pyplot as plt

    data = _load_plot_data(source, fields=_FALSE_ALARM_PLOT_FIELDS)
    range_bins = int(config.range_bins)
    prediction_m = np.asarray(result["prediction_m"], dtype=np.float64).reshape(-1)
    if prediction_m.shape != (source.frames,):
        raise ValueError("prediction_m 必须与完整帧轴等长。")
    finite_prediction = np.isfinite(prediction_m)
    if not np.any(finite_prediction):
        raise ValueError("样本没有可绘制的虚警预测。")

    steps = result.get("steps", ())
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise TypeError("推理结果 steps 必须为逐窗记录序列。")
    if not all(isinstance(step, Mapping) for step in steps):
        raise TypeError("推理结果 steps 必须全部为字典。")
    mode, measurement_updated = _status_series(steps, source.frames)

    range_start = max(0, int(math.floor(float(np.min(prediction_m[finite_prediction])) - margin_m)))
    range_stop = min(
        range_bins,
        int(math.ceil(float(np.max(prediction_m[finite_prediction])) + margin_m + 1.0)),
    )
    if range_stop <= range_start:
        range_stop = min(range_bins, range_start + 1)
    local_background = _unpack_local_range(
        data["background_only_packed"], range_start, range_stop, bitorder=config.packed_bitorder
    )
    probability = _background_probability(data, range_bins=range_bins)
    probability_centres_km, probability_values = _coarsen_probability(
        probability, int(probability_bin_m)
    )
    time_seconds = np.arange(source.frames, dtype=np.float64) * frame_interval_s
    background_frames, background_bins = np.nonzero(local_background)
    y_limits = (range_start / 1_000.0, range_stop / 1_000.0)
    first_output = _first_output_frame(prediction_m)

    step_frames = np.asarray(
        [int(step["frame"]) for step in steps if "frame" in step and "candidate_q" in step],
        dtype=np.int32,
    )
    step_q = np.asarray(
        [float(step["candidate_q"]) for step in steps if "frame" in step and "candidate_q" in step],
        dtype=np.float64,
    )

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(16.0, 10.2), constrained_layout=True)
        figure.suptitle(title, fontsize=15, fontweight="normal")

        probability_axis = axes[0, 0]
        probability_axis.semilogy(
            probability_centres_km,
            np.maximum(probability_values, np.finfo(np.float64).tiny),
            color="#4C78A8",
            linewidth=1.3,
        )
        probability_axis.axvspan(
            range_start / 1_000.0,
            range_stop / 1_000.0,
            color="#E64B35",
            alpha=0.16,
            label="虚警轨迹覆盖距离",
        )
        probability_axis.set(
            title="拟合背景占据概率",
            xlabel="距离（千米）",
            ylabel="每米距离单元的占据概率",
            xlim=(0.0, range_bins / 1_000.0),
        )
        probability_axis.legend(fontsize=8, loc="upper right")
        _style_axis(probability_axis)

        noise_axis = axes[0, 1]
        noise_axis.scatter(
            time_seconds[background_frames],
            (range_start + background_bins) / 1_000.0,
            s=point_size,
            marker="s",
            linewidths=0,
            alpha=0.6,
            color="#4C9AD4",
            rasterized=True,
            label="纯背景二值响应",
        )
        noise_axis.set(
            title="虚警距离附近的纯噪声时距图",
            xlabel="时间（秒）",
            ylabel="距离（千米）",
            ylim=y_limits,
        )
        noise_axis.legend(fontsize=8, loc="upper right")
        _style_axis(noise_axis)

        confidence_axis = axes[1, 0]
        if step_frames.size:
            confidence_axis.plot(
                time_seconds[step_frames], step_q, color="#4C78A8", linewidth=1.25, marker="o", markersize=3.2,
                label="当前窗口 q-Top1",
            )
        confidence_axis.axhline(
            capture_q_min,
            color="#F0A202",
            linewidth=1.1,
            linestyle="--",
            label="capture_q_min（捕获）",
        )
        confidence_axis.axhline(
            q_keep,
            color="#E64B35",
            linewidth=1.1,
            linestyle=":",
            label="q_keep（TRACK）",
        )
        confidence_axis.set(
            title="候选置信度与跟踪状态",
            xlabel="时间（秒）",
            ylabel="q",
            ylim=(0.0, 1.0),
        )
        mode_colors = {"CAPTURE": "#F0A202", "TRACK": "#E64B35", "RECAPTURE": "#8E5EA2"}
        shown_modes: set[str] = set()
        for step in steps:
            if "forecast_frame_start" not in step or "forecast_frame_stop" not in step:
                continue
            mode_name = str(step.get("mode", "")).upper()
            color = mode_colors.get(mode_name, "#777777")
            start_s = int(step["forecast_frame_start"]) * frame_interval_s
            stop_s = int(step["forecast_frame_stop"]) * frame_interval_s
            confidence_axis.axvspan(
                start_s,
                stop_s,
                color=color,
                alpha=0.10,
                label=mode_name if mode_name and mode_name not in shown_modes else None,
                zorder=0,
            )
            shown_modes.add(mode_name)
        confidence_axis.legend(fontsize=8, loc="upper right", ncols=2)
        _style_axis(confidence_axis)

        prediction_axis = axes[1, 1]
        prediction_axis.scatter(
            time_seconds[background_frames],
            (range_start + background_bins) / 1_000.0,
            s=point_size,
            marker="s",
            linewidths=0,
            alpha=0.44,
            color="#4C9AD4",
            rasterized=True,
            label="背景响应",
        )
        _plot_prediction_segments(
            prediction_axis,
            time_seconds=time_seconds,
            prediction_m=prediction_m,
            mode=mode,
            measurement_updated=measurement_updated,
        )
        if first_output > 0:
            prediction_axis.axvspan(
                0.0,
                min(source.frames, first_output) * frame_interval_s,
                color="#777777",
                alpha=0.08,
                label="首次报警前阶段",
                zorder=0,
            )
        prediction_axis.set(
            title=(
                "虚警轨迹："
                f"FAR {_format_metric(false_alarm.get('false_alarm_frame_rate'), '.1%')}，"
                f"含外推 {_format_metric(false_alarm.get('false_alarm_frame_rate_including_unreliable'), '.1%')}，"
                f"事件 {false_alarm.get('false_alarm_event_count', '—')}，"
                f"可靠帧 {false_alarm.get('reliable_false_alarm_frames', '—')}，"
                f"重捕获外推帧 {false_alarm.get('unreliable_false_alarm_frames', '—')}"
            ),
            xlabel="时间（秒）",
            ylabel="距离（千米）",
            ylim=y_limits,
        )
        prediction_axis.legend(fontsize=8, loc="upper right")
        _style_axis(prediction_axis)
        _apply_chinese_font(figure)

    return figure
