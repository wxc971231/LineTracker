"""不参与推理决策的公共诊断绘图工具。

图中的真值、目标命中标签和纯背景只用于离线复核；调用方必须在调用本模块前
完成推理，不能把本模块返回的任何信息回灌给候选块选择或状态机。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:  # 支持 ``python -m infer...`` 与直接从 infer 目录调试两种方式。
    from .metrics import prediction_series, trajectory_metrics
except ImportError:  # pragma: no cover - 仅用于直接运行单文件的便利路径。
    from metrics import prediction_series, trajectory_metrics


_SOURCE_FIELDS = {
    "background_only_packed",
    "observation_packed",
    "background_probability_1m",
    "target_true_range_m",
    "target_hit",
    "target_hit_bin",
    "metadata_json",
    # 允许测试或外部调用直接传未压缩矩阵。
    "background",
    "background_only",
    "observation",
}


def _config_value(config: Any | None, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _source_mapping(source: Any) -> tuple[dict[str, Any], str | None]:
    """读取绘图所需字段，兼容路径、普通字典和带 ``record.path`` 的对象。"""
    source_id: str | None = None
    source_path: Path | None = None
    if isinstance(source, (str, Path)):
        source_path = Path(source)
    elif isinstance(source, Mapping):
        data = dict(source)
        raw_id = data.get("source_id")
        source_id = None if raw_id is None else str(raw_id)
        return data, source_id
    elif hasattr(source, "record") and hasattr(source.record, "path"):
        source_path = Path(source.record.path)
        raw_id = getattr(source.record, "source_id", None)
        source_id = None if raw_id is None else str(raw_id)
    elif hasattr(source, "path"):
        source_path = Path(source.path)
        raw_id = getattr(source, "source_id", None)
        source_id = None if raw_id is None else str(raw_id)
    elif hasattr(source, "files") and hasattr(source, "__getitem__"):
        data = {name: source[name] for name in source.files if name in _SOURCE_FIELDS}
        return data, source_id
    else:
        raise TypeError("source 应为 data.npz 路径、字段字典或 PackedSource 类对象。")

    if not source_path.is_file():
        raise FileNotFoundError(f"找不到完整 source npz：{source_path}")
    with np.load(source_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files if name in _SOURCE_FIELDS}
    if source_id is None:
        source_id = source_path.parent.name
    return data, source_id


def _field(data: Mapping[str, Any], names: Sequence[str], *, required: bool = True) -> Any | None:
    for name in names:
        if name in data:
            return data[name]
    if required:
        raise KeyError(f"诊断图缺少字段，至少需要以下之一：{list(names)}。")
    return None


def _metadata(data: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = data.get("metadata_json")
    if raw is None:
        return {}
    try:
        value = raw.item() if isinstance(raw, np.ndarray) and raw.ndim == 0 else raw
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(str(value))
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def _infer_range_bins(data: Mapping[str, Any], config: Any | None) -> int:
    configured = _config_value(config, "range_bins", None)
    if configured is not None:
        return int(configured)
    probability = data.get("background_probability_1m")
    if probability is not None:
        return int(np.asarray(probability).size)
    for name in ("background", "background_only", "observation"):
        if name in data:
            return int(np.asarray(data[name]).shape[1])
    metadata_range = _metadata(data).get("range_bins")
    if metadata_range is not None:
        return int(metadata_range)
    packed = _field(data, ("observation_packed", "background_only_packed"), required=False)
    if packed is not None:
        return int(np.asarray(packed).shape[1] * 8)
    raise KeyError("无法推断距离轴长度；请传入 config.range_bins 或 background_probability_1m。")


def _unpack_local_range(
    packed: np.ndarray,
    range_start: int,
    range_stop: int,
    *,
    bitorder: str,
) -> np.ndarray:
    """仅解包绘图所需的局部距离区间，避免展开完整 300 km 二值矩阵。"""
    array = np.asarray(packed, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError(f"packed 观测应为二维数组，实际形状为 {array.shape}。")
    if not 0 <= range_start < range_stop <= array.shape[1] * 8:
        raise IndexError("局部距离范围超出 packed 观测可表示的范围。")
    first_byte = range_start // 8
    bit_offset = range_start % 8
    width = range_stop - range_start
    byte_count = math.ceil((bit_offset + width) / 8)
    unpacked = np.unpackbits(
        array[:, first_byte : first_byte + byte_count], axis=1, bitorder=bitorder
    )
    return unpacked[:, bit_offset : bit_offset + width].astype(bool, copy=False)


def _local_binary(
    data: Mapping[str, Any],
    *,
    raw_names: Sequence[str],
    packed_names: Sequence[str],
    range_start: int,
    range_stop: int,
    bitorder: str,
) -> np.ndarray:
    raw = _field(data, raw_names, required=False)
    if raw is not None:
        array = np.asarray(raw, dtype=bool)
        if array.ndim != 2:
            raise ValueError(f"二值观测应为二维数组，实际形状为 {array.shape}。")
        return array[:, range_start:range_stop]
    packed = _field(data, packed_names, required=False)
    if packed is None:
        raise KeyError(
            f"缺少 {list(raw_names)} 或 {list(packed_names)}；完整诊断图需要背景和观测。"
        )
    return _unpack_local_range(
        np.asarray(packed), range_start, range_stop, bitorder=bitorder
    )


def _background_probability(
    data: Mapping[str, Any],
    *,
    range_bins: int,
    bitorder: str,
) -> np.ndarray:
    """获取生成器保存的背景概率；缺失时才从背景二值观测重算。"""
    stored = data.get("background_probability_1m")
    if stored is not None:
        probability = np.asarray(stored, dtype=np.float64).reshape(-1)
        if len(probability) != range_bins:
            raise ValueError("background_probability_1m 与距离轴长度不一致。")
        return probability

    raw_background = _field(data, ("background", "background_only"), required=False)
    if raw_background is not None:
        array = np.asarray(raw_background, dtype=bool)
        if array.ndim != 2 or array.shape[1] != range_bins:
            raise ValueError("未压缩背景矩阵形状与距离轴长度不一致。")
        return array.mean(axis=0, dtype=np.float64)

    packed_background = _field(data, ("background_only_packed",), required=False)
    if packed_background is None:
        raise KeyError("缺少背景概率和 background_only_packed，无法绘制背景占据概率。")
    # 这是备用路径；合成数据正常都会保存 background_probability_1m。
    full_background = np.unpackbits(
        np.asarray(packed_background, dtype=np.uint8), axis=1, count=range_bins, bitorder=bitorder
    )
    return full_background.mean(axis=0, dtype=np.float64)


def _coarsen_probability(probability: np.ndarray, width_m: int) -> tuple[np.ndarray, np.ndarray]:
    if width_m < 1:
        raise ValueError("probability_bin_m 必须为正整数。")
    starts = np.arange(0, len(probability), width_m, dtype=np.int64)
    values = np.empty(len(starts), dtype=np.float64)
    centres = np.empty(len(starts), dtype=np.float64)
    for index, start in enumerate(starts):
        stop = min(int(start + width_m), len(probability))
        values[index] = float(np.nanmean(probability[start:stop]))
        centres[index] = (start + stop) * 0.5 / 1_000.0
    return centres, values


def _optional_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
        return None
    return bool(value)


def _prediction_category(mode: Any, measurement_updated: Any) -> str:
    mode_text = "" if mode is None else str(mode).upper()
    if "RECAPTURE" in mode_text:
        return "重捕获"
    if "CAPTURE" in mode_text:
        return "捕获"
    updated = _optional_bool(measurement_updated)
    if updated is True:
        return "测量更新"
    if updated is False:
        return "运动外推"
    return "模型预测"


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
        "重捕获": "重捕获阶段输出",
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


def plot_source_diagnostic(
    source: str | Path | Mapping[str, Any] | Any,
    prediction: np.ndarray | Sequence[float] | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    source_id: str | None = None,
    config: Any | None = None,
    title: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    frame_interval_s: float = 0.05,
    margin_m: float = 5_000.0,
    point_size: float = 1.0,
    probability_bin_m: int = 1_000,
    warmup_frames: int | None = None,
) -> Any:
    """生成一张用于保存的中文 2×2 流式推理诊断图。

    ``source`` 可以是完整 ``data.npz``、其字段字典，或 ``PackedSource`` 风格的
    对象。完整 NPZ 应包括 ``background_only_packed``、``observation_packed``、
    ``background_probability_1m``、``target_true_range_m``、``target_hit`` 和
    ``target_hit_bin``。为方便小型测试，也允许传入未压缩 ``background`` 与
    ``observation``。

    ``prediction`` 可传完整预测数组、数组字典，或每步 JSON 风格记录。逐步记录
    推荐使用后处理文档第 9 节的 ``frame_index``、``range_current_m``、``mode``、
    ``measurement_updated`` 字段。函数返回 matplotlib ``Figure``，调用方可直接
    ``fig.savefig(path, dpi=180)``。

    本函数只在推理结束后读取真实轨迹和命中标签来绘制第 3、4 图及计算评估值；
    它不会也不能将标签用于预测选择。
    """
    if not np.isfinite(frame_interval_s) or frame_interval_s <= 0.0:
        raise ValueError("frame_interval_s 必须为正有限数。")
    if not np.isfinite(margin_m) or margin_m < 0.0:
        raise ValueError("margin_m 必须为非负有限数。")
    if point_size <= 0.0:
        raise ValueError("point_size 必须为正。")

    # 延迟导入，使仅使用推理/指标代码的路径不要求 matplotlib。
    import matplotlib.pyplot as plt

    data, inferred_source_id = _source_mapping(source)
    display_source_id = source_id or inferred_source_id or "未命名样本"
    bitorder = str(_config_value(config, "packed_bitorder", "big"))
    if bitorder not in {"big", "little"}:
        raise ValueError("packed_bitorder 仅支持 big 或 little。")
    range_bins = _infer_range_bins(data, config)

    true_range_m = np.asarray(_field(data, ("target_true_range_m",)), dtype=np.float64).reshape(-1)
    target_hit = np.asarray(_field(data, ("target_hit",)), dtype=bool).reshape(-1)
    target_hit_bin = np.asarray(_field(data, ("target_hit_bin",)), dtype=np.int64).reshape(-1)
    frame_count = len(true_range_m)
    if frame_count < 1 or target_hit.shape != true_range_m.shape or target_hit_bin.shape != true_range_m.shape:
        raise ValueError("真实轨迹、target_hit 和 target_hit_bin 必须是同长度的一维数组。")
    if np.any(~np.isfinite(true_range_m)):
        raise ValueError("target_true_range_m 含非有限值。")

    series = prediction_series(prediction, frame_count)
    prediction_m = series["range_m"]
    base_metrics = trajectory_metrics(
        prediction_m,
        true_range_m,
        target_hit=target_hit,
    )
    if metrics is not None:
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
    if not len(finite_display):  # true_range 已在上面校验，这里仅作防御。
        raise ValueError("没有可用于绘图的轨迹位置。")
    range_start = max(0, int(math.floor(float(finite_display.min()) - margin_m)))
    range_stop = min(
        range_bins,
        int(math.ceil(float(finite_display.max()) + margin_m + 1.0)),
    )
    if range_stop <= range_start:
        range_stop = min(range_bins, range_start + 1)

    target_background = _local_binary(
        data,
        raw_names=("background", "background_only"),
        packed_names=("background_only_packed",),
        range_start=target_range_start,
        range_stop=target_range_stop,
        bitorder=bitorder,
    )
    target_observation = _local_binary(
        data,
        raw_names=("observation",),
        packed_names=("observation_packed",),
        range_start=target_range_start,
        range_stop=target_range_stop,
        bitorder=bitorder,
    )
    local_background = _local_binary(
        data,
        raw_names=("background", "background_only"),
        packed_names=("background_only_packed",),
        range_start=range_start,
        range_stop=range_stop,
        bitorder=bitorder,
    )
    if (
        target_background.shape[0] != frame_count
        or target_observation.shape != target_background.shape
        or local_background.shape[0] != frame_count
    ):
        raise ValueError("背景、观测和真实轨迹的帧数或局部形状不一致。")
    probability = _background_probability(data, range_bins=range_bins, bitorder=bitorder)
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
    if warmup_frames is None:
        warmup_frames = int(_config_value(config, "frames_per_window", 20))

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(16.0, 10.2), constrained_layout=True)
        if title is None:
            title = f"流式推理诊断：{display_source_id}"
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
            mode=series["mode"],
            measurement_updated=series["measurement_updated"],
        )
        if warmup_frames > 0:
            prediction_axis.axvspan(
                0.0,
                min(frame_count, int(warmup_frames)) * frame_interval_s,
                color="#777777",
                alpha=0.08,
                label="首次输入预热窗",
                zorder=0,
            )
        prediction_axis.set(
            title=(
                "流式预测："
                f"MAE {_format_metric(base_metrics.get('mae_m'), '.1f')} 米，"
                f"覆盖率 {_format_metric(base_metrics.get('coverage'), '.1%')}，"
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


# 更贴近运行入口命名的别名，保持主接口只有一种实现。
plot_inference_diagnostic = plot_source_diagnostic


__all__ = ["plot_inference_diagnostic", "plot_source_diagnostic"]
