#!/usr/bin/env python3
"""读取已拟合的 dataset1 模型，批量生成含移动目标的二值时距数据。

常用命令：

    # 用中等固定背景场景生成 10 个样本；每个样本只因随机种子和目标不同而不同。
    conda run -n linetracker-py311 python data/synthetic/generate_dataset1.py \
        --samples 10 --random-knobs false

    # 随机抽取五个连续旋钮，构造多样的 B-random 场景。
    conda run -n linetracker-py311 python data/synthetic/generate_dataset1.py \
        --samples 100 --random-knobs true --seed 20260717

本脚本只读取拟合结果，不读取原始 dataset1 MAT 文件。每个样本先生成二值背景，
再生成带漏检的随机目标并做逻辑 OR。所有输出始终为 0/1 布尔矩阵；概率坐标变量
不表示返回幅值或累计光子数。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


RANGE_BINS = 300_000
PROFILE_WIDTH_M = 1_000
FRAME_INTERVAL_S = 0.05
BACKGROUND_START_M = 10_000
CHUNK_RANGE_BINS = 5_000
REGIONS = (
    (1, 10_000, "0–10 km"),
    (10_000, 50_000, "10–50 km"),
    (50_000, 100_000, "50–100 km"),
    (100_000, 300_000, "100–300 km"),
)

HERE = Path(__file__).resolve().parent
DEFAULT_FIT_DIR = HERE / "fit"
DEFAULT_MODEL_PATH = DEFAULT_FIT_DIR / "dataset1_background_model.npz"
DEFAULT_OUTPUT_DIR = HERE / "gen"
DEFAULT_PREVIEW_MARGIN_M = 5_000.0
DEFAULT_PREVIEW_POINT_SIZE = 3.0


def _jsonable(value: Any) -> Any:
    """递归将 NumPy、Path 等对象转换为可写入 JSON 的值。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _parse_bool(value: str) -> bool:
    """将命令行中的 true/false 文本明确解析为布尔值。"""
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("应填写 true 或 false")


def _format_path_value(value: float | int) -> str:
    """将数值转换为紧凑且跨平台安全的目录名片段。"""
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def build_generation_directory_name(args: argparse.Namespace) -> str:
    """根据影响样本内容的关键参数构造批次配置目录名。

    B-random 场景中每个样本各自抽取五个背景旋钮，因此只在目录名中标记其模式；
    固定背景场景则把五个旋钮值写入目录名。样本数和随机种子也参与命名，避免不同
    批次意外写入同一个目录。
    """
    if args.random_knobs:
        background_name = "B-random"
    else:
        background_name = "B-fixed-" + "-".join(
            (
                f"L{_format_path_value(args.level)}",
                f"D{_format_path_value(args.decay)}",
                f"N{_format_path_value(args.near_field)}",
                f"G{_format_path_value(args.gain)}",
                f"C{_format_path_value(args.cluster)}",
            )
        )
    target_name = "-".join(
        (
            f"R{_format_path_value(args.range_min_m)}-{_format_path_value(args.range_max_m)}m",
            f"V{_format_path_value(args.max_speed_mps)}",
            f"A{_format_path_value(args.max_acceleration_mps2)}",
            f"C{_format_path_value(args.curve_strength)}",
            f"J{_format_path_value(args.measurement_jitter_std_m)}",
            f"K{_format_path_value(args.response_multiplier)}",
            "Q"
            f"{_format_path_value(args.min_injection_probability)}-"
            f"{_format_path_value(args.max_injection_probability)}",
        )
    )
    return "_".join(
        (
            f"F{args.frames}",
            f"N{args.samples}",
            f"S{args.seed}",
            background_name,
            f"T-{target_name}",
        )
    )


def resolve_generation_output_dir(
    output_root: Path,
    args: argparse.Namespace,
) -> Path:
    """在生成根目录下定位本批次关键参数对应的配置目录。"""
    return Path(output_root) / build_generation_directory_name(args)


def load_model(path: Path = DEFAULT_MODEL_PATH) -> dict[str, np.ndarray]:
    """读取 fit_dataset1.py 输出的 NPZ 模型，且不启用 pickle。"""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}

def save_packed_sample(
    path: Path,
    background: np.ndarray,
    observation: np.ndarray,
    metadata: dict[str, Any],
    target: dict[str, Any],
) -> None:
    """保存一个含背景、目标叠加矩阵和航迹真值的位打包样本。

    observation_packed 是轨迹检测算法的完整二值输入；background_only_packed
    仅用于背景质量复核和带真值的预览图复现。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    background = np.asarray(background, dtype=np.bool_)
    observation = np.asarray(observation, dtype=np.bool_)
    if background.shape != observation.shape:
        raise ValueError("background 与 observation 的形状必须一致。")

    background_probability_1m = np.asarray(
        metadata["target_probability_1m"], dtype=np.float32
    )
    compact_metadata = dict(metadata)
    compact_metadata.pop("target_probability_1m", None)
    compact_metadata.pop("target_probability_1km", None)
    compact_metadata.pop("frame_gains", None)

    np.savez_compressed(
        path,
        # 背景与完整观测矩阵均按距离维位打包：每帧 300000 个 0/1 bin 对应 37500 字节。
        background_only_packed=np.packbits(background, axis=1),
        observation_packed=np.packbits(observation, axis=1),
        # 生成前的逐 1 m 背景响应率；用于背景曲线验收与三联图左图。
        background_probability_1m=background_probability_1m,
        # 目标真值：潜在连续距离、实际注入距离、是否注入及相关运动/概率量。
        target_true_range_m=np.asarray(target["true_range_m"], dtype=np.float32),
        target_measured_bin=np.asarray(target["measured_bin"], dtype=np.int32),
        target_hit=np.asarray(target["hit"], dtype=np.bool_),
        target_hit_bin=np.asarray(target["hit_bin"], dtype=np.int32),
        target_velocity_mps=np.asarray(target["velocity_mps"], dtype=np.float32),
        target_acceleration_mps2=np.asarray(target["acceleration_mps2"], dtype=np.float32),
        target_p_background=np.asarray(target["p_background_on_track"], dtype=np.float32),
        target_p_on=np.asarray(target["p_on_track"], dtype=np.float32),
        target_injection_probability=np.asarray(
            target["injection_probability"], dtype=np.float32
        ),
        metadata_json=np.array(
            json.dumps(_jsonable(compact_metadata), ensure_ascii=False)
        ),
    )


def load_packed_sample(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """读取样本，返回背景、完整观测矩阵、元数据和目标真值。"""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        background = np.unpackbits(
            archive["background_only_packed"],
            axis=1,
            count=int(metadata["range_bins"]),
        ).astype(bool, copy=False)
        observation = np.unpackbits(
            archive["observation_packed"],
            axis=1,
            count=int(metadata["range_bins"]),
        ).astype(bool, copy=False)
        metadata["background_probability_1m"] = archive[
            "background_probability_1m"
        ].copy()
        target = {
            "true_range_m": archive["target_true_range_m"].copy(),
            "measured_bin": archive["target_measured_bin"].copy(),
            "hit": archive["target_hit"].copy(),
            "hit_bin": archive["target_hit_bin"].copy(),
            "velocity_mps": archive["target_velocity_mps"].copy(),
            "acceleration_mps2": archive["target_acceleration_mps2"].copy(),
            "p_background_on_track": archive["target_p_background"].copy(),
            "p_on_track": archive["target_p_on"].copy(),
            "injection_probability": archive[
                "target_injection_probability"
            ].copy(),
        }
    return background, observation, metadata, target


def profile_binary(background: np.ndarray) -> dict[str, Any]:
    """按真实数据的同一口径统计一个合成 bool 背景。"""
    if background.ndim != 2 or background.shape[1] != RANGE_BINS:
        raise ValueError(f"Expected shape (frames, {RANGE_BINS}), got {background.shape}")
    values = np.asarray(background, dtype=bool)
    frames = values.shape[0]
    frame_counts = values.sum(axis=1, dtype=np.int64)
    range_counts = values.sum(axis=0, dtype=np.int64)
    probability_by_range = range_counts / frames
    grouped_rates = range_counts.reshape(-1, PROFILE_WIDTH_M).sum(axis=1) / (
        frames * PROFILE_WIDTH_M
    )

    temporal_pairs = 0
    spatial_pairs = 0
    previous_column: np.ndarray | None = None
    for start in range(BACKGROUND_START_M, RANGE_BINS, CHUNK_RANGE_BINS):
        stop = min(start + CHUNK_RANGE_BINS, RANGE_BINS)
        block = values[:, start:stop]
        temporal_pairs += int(np.count_nonzero(block[1:, :] & block[:-1, :]))
        if previous_column is not None:
            spatial_pairs += int(np.count_nonzero(previous_column & block[:, 0]))
        spatial_pairs += int(np.count_nonzero(block[:, 1:] & block[:, :-1]))
        previous_column = block[:, -1].copy()

    background_counts = range_counts[BACKGROUND_START_M:]
    background_probability = probability_by_range[BACKGROUND_START_M:]
    expected_temporal_pairs = float(
        np.sum(background_counts * (background_counts - 1), dtype=np.float64)
        / frames
    )
    expected_spatial_pairs = float(
        frames * np.dot(background_probability[:-1], background_probability[1:])
    )
    mean_events = float(frame_counts.mean())
    return {
        "frame_counts": frame_counts,
        "grouped_rates": grouped_rates,
        "mean_events_per_frame": mean_events,
        "fano_factor": float(frame_counts.var(ddof=1) / mean_events),
        "temporal_pair_ratio": float(temporal_pairs / expected_temporal_pairs),
        "spatial_pair_ratio": float(spatial_pairs / expected_spatial_pairs),
        "region_rates": {
            label: float(range_counts[left:right].sum() / ((right - left) * frames))
            for left, right, label in REGIONS
        },
    }



def generate_target(
    p_bg_1m: np.ndarray,
    *,
    frames: int,
    range_min_m: float,
    range_max_m: float,
    max_speed_mps: float,
    max_acceleration_mps2: float,
    curve_strength: float,
    smooth_window_frames: int,
    measurement_jitter_std_m: float,
    response_multiplier: float,
    min_injection_probability: float,
    max_injection_probability: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """生成一条随机潜在航迹、目标注入事件和对应真值。

    先以平滑白噪声生成加速度，再递推得到限速轨迹。目标注入概率由参考背景
    响应率和 response_multiplier 换算得到，并最终裁剪到给定上下限。
    """
    p_bg_1m = np.asarray(p_bg_1m, dtype=np.float64)
    if p_bg_1m.shape != (RANGE_BINS,):
        raise ValueError(f"p_bg_1m 的形状必须为 ({RANGE_BINS},)。")
    if frames <= 0:
        raise ValueError("frames 必须为正整数。")
    if not 0.0 <= range_min_m < range_max_m <= RANGE_BINS:
        raise ValueError("目标距离范围必须满足 0 <= min < max <= RANGE_BINS。")
    if max_speed_mps <= 0.0 or max_acceleration_mps2 < 0.0:
        raise ValueError("速度上限必须为正，加速度尺度上限必须非负。")
    if smooth_window_frames <= 0:
        raise ValueError("smooth_window_frames 必须为正整数。")
    if measurement_jitter_std_m < 0.0:
        raise ValueError("measurement_jitter_std_m 必须非负。")
    if response_multiplier < 0.0:
        raise ValueError("response_multiplier 必须非负。")
    if not 0.0 <= min_injection_probability <= max_injection_probability <= 1.0:
        raise ValueError("目标注入概率上下限必须满足 0 <= min <= max <= 1。")

    # 生成低频加速度：长度为 W 的均值核会滤除高频帧间抖动。
    window = min(int(smooth_window_frames), frames)
    window = window if window % 2 == 1 else max(window - 1, 1)
    kernel = np.ones(window, dtype=np.float64) / window
    acceleration = np.convolve(rng.normal(size=frames), kernel, mode="same")
    acceleration /= max(float(np.max(np.abs(acceleration))), 1e-12)
    acceleration *= float(curve_strength) * float(max_acceleration_mps2)

    # 递推速度，并约束每一帧的速度绝对值不超过 max_speed_mps。
    velocity = np.empty(frames, dtype=np.float64)
    velocity[0] = rng.uniform(-0.8 * max_speed_mps, 0.8 * max_speed_mps)
    for frame_index in range(1, frames):
        velocity[frame_index] = np.clip(
            velocity[frame_index - 1]
            + acceleration[frame_index - 1] * FRAME_INTERVAL_S,
            -max_speed_mps,
            max_speed_mps,
        )

    # 依据相对位移选择合法起点，确保整条潜在轨迹落在给定距离范围内。
    displacement = np.r_[0.0, np.cumsum(velocity[:-1] * FRAME_INTERVAL_S)]
    start_low = range_min_m - float(displacement.min())
    start_high = range_max_m - float(displacement.max())
    if start_low > start_high:
        raise ValueError("设定距离域无法容纳当前随机轨迹。")
    true_range_m = rng.uniform(start_low, start_high) + displacement

    # 测距抖动仅影响实际注入的整数距离 bin，不改变潜在连续轨迹。
    measured_bin = np.rint(
        true_range_m + rng.normal(0.0, measurement_jitter_std_m, size=frames)
    ).astype(np.int64)
    measured_bin = np.clip(measured_bin, 0, RANGE_BINS - 1)

    p_background_on_track = p_bg_1m[measured_bin]
    candidate_probability = np.maximum(
        p_background_on_track,
        float(response_multiplier) * p_background_on_track,
    )

    # 反解“额外置 1”概率，并只由 min/max_injection_probability 统一裁剪。
    raw_injection_probability = (
        candidate_probability - p_background_on_track
    ) / (1.0 - p_background_on_track)
    injection_probability = np.clip(raw_injection_probability, 0.0, 1.0)
    injection_probability = np.clip(
        injection_probability,
        min_injection_probability,
        max_injection_probability,
    )

    # 每帧独立决定是否额外注入目标响应；与背景叠加由 overlay_target 完成。
    hit = rng.random(frames) < injection_probability
    hit_bin = np.where(hit, measured_bin, -1)
    p_on_track = p_background_on_track + (
        1.0 - p_background_on_track
    ) * injection_probability

    return {
        "true_range_m": true_range_m,
        "measured_bin": measured_bin,
        "hit": hit,
        "hit_bin": hit_bin,
        "velocity_mps": velocity,
        "acceleration_mps2": acceleration,
        "p_background_on_track": p_background_on_track,
        "p_on_track": p_on_track,
        "injection_probability": injection_probability,
        "max_step_m": float(max_speed_mps * FRAME_INTERVAL_S),
        "expected_extra_responses": float(
            np.sum((1.0 - p_background_on_track) * injection_probability)
        ),
    }


def overlay_target(background: np.ndarray, target: dict[str, Any]) -> np.ndarray:
    """将已注入的目标响应与背景做逻辑 OR，返回完整二值观测矩阵。"""
    observation = np.asarray(background, dtype=np.bool_).copy()
    hit = np.asarray(target["hit"], dtype=bool)
    hit_bin = np.asarray(target["hit_bin"], dtype=np.int64)
    if observation.shape != (hit.size, RANGE_BINS):
        raise ValueError("background 的形状必须与目标帧数和 RANGE_BINS 一致。")
    observation[np.flatnonzero(hit), hit_bin[hit]] = True
    return observation


def save_sample_preview(
    path: Path,
    background: np.ndarray,
    observation: np.ndarray,
    probability_1m: np.ndarray,
    target: dict[str, Any],
    margin_m: float = DEFAULT_PREVIEW_MARGIN_M,
    point_size: float = DEFAULT_PREVIEW_POINT_SIZE,
) -> None:
    """保存三联图：背景概率、未标注观测、带目标真值的局部时距图。"""
    if margin_m < 0.0 or point_size <= 0.0:
        raise ValueError("margin_m 必须非负，point_size 必须为正。")

    # 延迟导入，保证只调用背景函数时无需加载绘图库。
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    probability_1m = np.asarray(probability_1m, dtype=np.float64)
    frames = background.shape[0]
    if background.shape != observation.shape or probability_1m.shape != (RANGE_BINS,):
        raise ValueError("绘图输入的背景、观测矩阵或概率曲线形状不正确。")

    p_bg_1km = probability_1m.reshape(-1, PROFILE_WIDTH_M).mean(axis=1)
    centres_km = np.arange(p_bg_1km.size) + 0.5
    time_s = np.arange(frames) * FRAME_INTERVAL_S
    true_range_m = np.asarray(target["true_range_m"], dtype=np.float64)
    true_range_km = true_range_m / 1_000.0
    hit = np.asarray(target["hit"], dtype=bool)
    hit_range_km = np.asarray(target["hit_bin"], dtype=np.int64)[hit] / 1_000.0

    window_start = max(0, int(np.floor(true_range_m.min() - margin_m)))
    window_stop = min(RANGE_BINS, int(np.ceil(true_range_m.max() + margin_m + 1.0)))
    local_background = background[:, window_start:window_stop]
    local_observation = observation[:, window_start:window_stop]
    background_frames, background_bins = np.nonzero(local_background)
    observation_frames, observation_bins = np.nonzero(local_observation)
    local_ylim_km = (window_start / 1_000.0, window_stop / 1_000.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), constrained_layout=True)

    axes[0].semilogy(centres_km, p_bg_1km, color="#4c78a8", lw=1.2)
    axes[0].axvspan(
        true_range_km.min(),
        true_range_km.max(),
        color="#d62728",
        alpha=0.16,
        label="本次目标经过距离",
    )
    axes[0].set(
        title="拟合模型恢复的背景占据概率",
        xlabel="距离（km）",
        ylabel="每 1 m bin 的占据概率",
        xlim=(0, 300),
    )
    axes[0].legend(fontsize=8)

    axes[1].scatter(
        time_s[observation_frames],
        (window_start + observation_bins) / 1_000.0,
        s=point_size,
        marker="s",
        linewidths=0,
        alpha=0.8,
        color="#1f77b4",
        rasterized=True,
        label="二值响应（未标注）",
    )
    axes[1].set(
        title="局部二值时距图（目标未标注）",
        xlabel="时间（s）",
        ylabel="距离（km）",
        ylim=local_ylim_km,
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    axes[2].scatter(
        time_s[background_frames],
        (window_start + background_bins) / 1_000.0,
        s=point_size,
        marker="s",
        linewidths=0,
        alpha=0.8,
        color="#1f77b4",
        rasterized=True,
        label="背景响应",
    )
    axes[2].plot(
        time_s,
        true_range_km,
        color="#202124",
        lw=1.0,
        alpha=0.85,
        label="潜在轨迹",
    )
    axes[2].scatter(
        time_s[hit],
        hit_range_km,
        s=4,
        color="#d62728",
        zorder=3,
        label="实际目标响应",
    )
    axes[2].set(
        title="背景叠加后的局部二值时距图",
        xlabel="时间（s）",
        ylabel="距离（km）",
        ylim=local_ylim_km,
    )
    axes[2].legend(fontsize=8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def step_3_6_select_coefficients(
    model: dict[str, np.ndarray],
    level: float = 0.0,
    decay: float = 0.0,
    near_field: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """第 3.6 步：将三个归一化旋钮映射为基函数系数 theta。"""
    def _map_normalized_value(interval: np.ndarray, normalized: float) -> float:
        """将 [-1, 1] 内的旋钮值映射到拟合得到的数值区间。"""
        normalized = float(np.clip(normalized, -1.0, 1.0))
        midpoint = 0.5 * float(interval[0] + interval[1])
        half_width = 0.5 * float(interval[1] - interval[0])
        return midpoint + normalized * half_width

    # 三个旋钮独立确定三个基函数系数。
    knobs = np.clip(np.array([level, decay, near_field], dtype=np.float64), -1.0, 1.0)
    coefficient_ranges = model["coefficient_ranges"]
    coefficients = np.array([
        _map_normalized_value(coefficient_ranges[index], knobs[index])
        for index in range(3)
    ])
    metadata = {
        "normalized_knobs": {
            "level": float(knobs[0]),
            "decay": float(knobs[1]),
            "near_field": float(knobs[2]),
        },
        "coefficients": dict(
            zip(model["parameter_names"].tolist(), coefficients.tolist())
        ),
    }
    return coefficients, metadata


def step_3_7_restore_probability(
    model: dict[str, np.ndarray],
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """第 3.7 步：恢复 Bernoulli 概率，并由 1 km 插值至 1 m。返回的 p_theta(r) 尚未加入空间簇和帧级变化。"""
    def _build_envelope() -> tuple[np.ndarray, np.ndarray]:
        """由稳定真实样本的平滑曲线构建逐 1 km 的 a 坐标上下界。"""
        margin = np.log1p(float(model["expanded_fraction"]))    # 把概率扩展比例转换为坐标扩展值
        smoothed_curves = np.asarray(                           # 7 个真实样本高斯平滑后的概率坐标曲线
            model["sample_smoothed_log_intensity"],
            dtype=np.float64,
        )
        lower = np.min(smoothed_curves, axis=0) - margin        # 生成样本的逐距离包络 a 下界
        upper = np.max(smoothed_curves, axis=0) + margin        # 生成样本的逐距离包络 a 上界
        return lower, upper

    # 在 a 坐标构造曲线，并裁剪到从稳定真实样本现场计算的逐距离包络内。
    log_intensity_1km = model["base_log_intensity"] + model["basis"] @ coefficients # (300,) 构造的距离组响应概率坐标曲线
    envelope_lower, envelope_upper = _build_envelope()
    log_intensity_1km = np.clip(log_intensity_1km, envelope_lower, envelope_upper)

    # 在线性 a 坐标而非概率 p 坐标中，从 1 km 中心插值至每个 1 m bin。
    width_m = int(model["profile_width_m"])
    range_bins = int(model["range_bins"])
    centres_m = (np.arange(log_intensity_1km.size) + 0.5) * width_m
    log_intensity_1m = np.interp(
        np.arange(range_bins, dtype=np.float64),
        centres_m,
        log_intensity_1km,
        left=log_intensity_1km[0],
        right=log_intensity_1km[-1],
    )

    # H(a) = 1 - exp[-exp(a)] 从概率坐标 a 转换为合法概率 p。
    probability_1km = -np.expm1(-np.exp(log_intensity_1km))
    probability_1m = -np.expm1(-np.exp(log_intensity_1m))
    probability_1m[0] = 0.0
    return probability_1m, probability_1km


def step_3_8_prepare_spatial_clusters(
    model: dict[str, np.ndarray],
    probability_1m: np.ndarray,
    probability_1km: np.ndarray,
    cluster: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """第 3.8 步：标定 1--3 m 空间簇，并递推得到种子概率 q(r)。

    返回依次为嵌套延伸概率 (e_1,e_2,e_3)、基准种子概率 q(r)，以及目标的
    10 km+ 空间相邻响应对比值。本步骤所需的标定和递推函数均封装在本方法内。
    """
    def _map_normalized_value(interval: np.ndarray, normalized: float) -> float:
        """将 [-1, 1] 内的旋钮值映射到拟合得到的数值区间。"""
        normalized = float(np.clip(normalized, -1.0, 1.0))
        midpoint = 0.5 * float(interval[0] + interval[1])
        half_width = 0.5 * float(interval[1] - interval[0])
        return midpoint + normalized * half_width

    def _stationary_seed_probability(probability: np.ndarray, survival: np.ndarray) -> np.ndarray:
        """
            在 1 km 粗略平稳近似下（即假设某个 1 km 距离组附近的种子概率都相同），
            基于局部结构延伸概率 survival 求解 q_coarse(k)，保持延伸后响应边际概率 1.0 - probability 不变
        """
        # 二分搜索在各距离组设置局部关联种子的概率 q_coarse(k)，使得 1-p_theta(k)=(1-q_coarse(k))∏_d[1-e_d q_coarse(k-d)]。
        low = np.zeros_like(probability, dtype=np.float64)  # (300,) 二分搜索初始下界
        high = probability.astype(np.float64, copy=True)    # (300,) 二分搜索初始上界
        target_zero = 1.0 - probability                     # (300,) 需要保持的边际响应率

        # 二分搜索，找到一个合适的种子概率，使得延伸后边际概率接近目标值。
        for _ in range(45):
            midpoint = 0.5 * (low + high)

            # 在粗略平稳近似下，计算候选种子概率对应的延伸后边际概率。
            produced_zero = 1.0 - midpoint
            for extension_probability in survival:
                produced_zero *= 1.0 - extension_probability * midpoint
            
            # 根据延伸后边际概率与目标值的比较，调整二分搜索的上下界。
            need_more_seeds = produced_zero > target_zero
            low = np.where(need_more_seeds, midpoint, low)
            high = np.where(need_more_seeds, high, midpoint)
        return 0.5 * (low + high)

    def _coarse_spatial_pair_ratio(survival: np.ndarray) -> float:
        """估计候选延伸向量对应的 10 km+ 空间相邻对比值 R_sp。"""
        # 基于 1 km 距离组概率 p_theta(k) 和候选的局部结构延伸概率 (e_1, e_2, e_3) 求临时种子概率 q_coarse(k)
        # 使设置局部关联后的边际概率仍为 p_theta(k)
        coarse_seed_probability = _stationary_seed_probability(probability_1km, survival)
        
        # 评估这种子概率 q_coarse(k) 对应的空间相邻对比值 R_sp
        joint_probability = probability_1km - (1.0 - probability_1km) * coarse_seed_probability # 相邻 bin 同时响应概率的粗略近似
        start_group = BACKGROUND_START_M // PROFILE_WIDTH_M
        numerator = float(np.sum(joint_probability[start_group:]))          # 10 km 后，设置局部结构时，相邻 bin 同时响应概率和
        denominator = float(np.sum(probability_1km[start_group:] ** 2))     # 10 km 后，独立响应时，相邻 bin 同时响应概率和
        return numerator / denominator

    def _calibrate_survival(target_pair_ratio: float) -> np.ndarray:
        """二分搜索满足目标空间相邻对比值的 (e_1,e_2,e_3)。"""
        pattern = np.array([1.0, 0.55, 0.25], dtype=np.float64) # 局部关联概率模版
        low, high = 0.0, 0.02                                   # 搜索关联强度参数的初始值

        # 粗略搜索，找到一个足够大的 high，使得 _coarse_spatial_pair_ratio(high * pattern) >= target_pair_ratio。
        while _coarse_spatial_pair_ratio(high * pattern) < target_pair_ratio:
            high *= 2.0
            if high > 0.25:
                raise RuntimeError(f"Could not reach spatial pair ratio {target_pair_ratio}")
        
        # 二分搜索，找到一个合适的强度参数，使得 _coarse_spatial_pair_ratio(midpoint * pattern) 接近目标值。
        for _ in range(45):
            midpoint = 0.5 * (low + high)
            if _coarse_spatial_pair_ratio(midpoint * pattern) < target_pair_ratio:
                low = midpoint
            else:
                high = midpoint
        return 0.5 * (low + high) * pattern

    def _solve_seed_probability(survival: np.ndarray) -> np.ndarray:
        """在 1m 精度下，递推求解使局部延伸后边际概率保持为 p_theta(r) 的种子概率 q(r)"""
        seed_probability = np.zeros_like(probability_1m, dtype=np.float64)
        for range_index in range(probability_1m.size):
            inherited_zero = 1.0
            for distance, extension_probability in enumerate(survival, start=1):
                previous = range_index - distance
                if previous >= 0:
                    inherited_zero *= (
                        1.0 - extension_probability * seed_probability[previous]
                    )
            # 由 1-p(r)=(1-q(r))∏_d[1-e_d q(r-d)] 变形得到。
            candidate = 1.0 - (1.0 - probability_1m[range_index]) / inherited_zero
            seed_probability[range_index] = np.clip(
                candidate,
                0.0,
                probability_1m[range_index],
            )
        return seed_probability

    # 将归一化参数 cluster 映射回目标相邻bin响应对比值 R_sp=实际相邻距离 bin 同时响应的数量/假设各 bin 独立时预期的数量
    target_spatial_pair_ratio = _map_normalized_value(model["spatial_pair_range"], cluster)

    # 标定局部空间簇的延伸概率 e_1,e_2,e_3，使得 10 km+ 空间相邻对比值接近目标值
    survival = _calibrate_survival(target_spatial_pair_ratio)
    
    # 递推求解使延伸后边际概率保持为 p_theta(r) 的种子概率 q(r)
    seed_probability = _solve_seed_probability(survival)
    return survival, seed_probability, target_spatial_pair_ratio


def step_3_9_prepare_frame_adjustment(
    model: dict[str, np.ndarray],
    probability_1m: np.ndarray,
    seed_probability: np.ndarray,
    gain: float,
    frames: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """第 3.9 步：按目标 Fano 生成帧级系数 c_t，并准备 q_t(r) 的计算。

    返回 -log[1-q(r)]、逐帧系数 c_t，以及目标 Fano factor。实际的 q_t(r) 按帧块
    在第 3.10 步计算，以避免一次构造完整的“帧 × 距离”浮点概率矩阵。
    """
    def _map_normalized_value(interval: np.ndarray, normalized: float) -> float:
        """将 [-1, 1] 内的旋钮值映射到拟合得到的数值区间。"""
        normalized = float(np.clip(normalized, -1.0, 1.0))
        midpoint = 0.5 * float(interval[0] + interval[1])
        half_width = 0.5 * float(interval[1] - interval[0])
        return midpoint + normalized * half_width

    target_fano_factor = _map_normalized_value(model["fano_range"], gain)
    target_mean_events = float(np.sum(probability_1m))
    gain_variance = max((target_fano_factor - 1.0) / target_mean_events, 0.0)
    gain_sigma = float(np.sqrt(np.log1p(gain_variance)))
    seed_coordinate = -np.log1p(-seed_probability).astype(np.float32)
    frame_gains = np.exp(
        gain_sigma * rng.standard_normal(frames) - 0.5 * gain_sigma**2
    ).astype(np.float32)
    return seed_coordinate, frame_gains, target_fano_factor


def step_3_10_generate_binary_background(
    range_bins: int,
    frame_gains: np.ndarray,
    seed_coordinate: np.ndarray,
    survival: np.ndarray,
    rng: np.random.Generator,
    block_frames: int,
) -> np.ndarray:
    """第 3.10 步：按帧生成 Bernoulli 种子、空间延伸并合成为最终背景。"""
    if block_frames <= 0:
        raise ValueError("block_frames must be positive")

    frames = frame_gains.size
    background = np.empty((frames, range_bins), dtype=np.bool_)
    for start in range(0, frames, block_frames):
        stop = min(start + block_frames, frames)
        block_gain = frame_gains[start:stop, None]

        # 先施加 c_t 得到 q_t(r)，再独立采样 Bernoulli 种子 S_{t,r}。
        block_seed_probability = -np.expm1(-block_gain * seed_coordinate[None, :])
        seeds = (
            rng.random(block_seed_probability.shape, dtype=np.float32)
            < block_seed_probability
        )

        # 每个位置只生成一个 U；同一个 U 与 e_1>=e_2>=e_3 保证延伸嵌套。
        block = seeds.copy()
        extension_uniform = rng.random(seeds.shape, dtype=np.float32)
        for distance, extension_probability in enumerate(survival, start=1):
            block[:, distance:] |= (
                seeds[:, :-distance]
                & (extension_uniform[:, :-distance] < extension_probability)
            )
        block[:, 0] = False
        background[start:stop] = block
    return background


def generate_background(
    model: dict[str, np.ndarray],
    frames: int = 300,
    seed: int = 20260717,
    level: float = 0.0,
    decay: float = 0.0,
    near_field: float = 0.0,
    gain: float = 0.0,
    cluster: float = 0.0,
    block_frames: int = 16,
) -> tuple[np.ndarray, dict[str, Any]]:
    """使用已拟合模型生成一个“帧 × 距离”的 bool 背景数组。

    该函数只编排方案第 3.6--3.10 步；每个步骤的计算细节位于对应的步骤方法中。
    """
    if frames <= 0:
        raise ValueError("frames must be positive")
    rng = np.random.default_rng(seed)

    # 第 3.6 步：由三个场景旋钮选择基函数系数 theta。
    coefficients, curve_metadata = step_3_6_select_coefficients(model, level, decay, near_field,)

    # 第 3.7 步：恢复并插值得到基础距离响应率 p_theta(r)。
    probability_1m, probability_1km = step_3_7_restore_probability(model, coefficients)

    # 第 3.8 步：标定局部空间簇，并递推得到种子概率 q(r)。
    survival, seed_probability, target_spatial_pair_ratio = (
        step_3_8_prepare_spatial_clusters(
            model,
            probability_1m,
            probability_1km,
            cluster,
        )
    )

    # 第 3.9 步：生成帧级调节系数 c_t，准备按帧块计算 q_t(r)。
    seed_coordinate, frame_gains, target_fano_factor = (
        step_3_9_prepare_frame_adjustment(
            model,
            probability_1m,
            seed_probability,
            gain,
            frames,
            rng,
        )
    )

    # 第 3.10 步：生成最终“帧 × 距离”的 bool 背景矩阵。
    background = step_3_10_generate_binary_background(
        int(model["range_bins"]),
        frame_gains,
        seed_coordinate,
        survival,
        rng,
        block_frames,
    )

    metadata = {
        "model_version": str(model["model_version"]),
        "frames": int(frames),
        "range_bins": int(model["range_bins"]),
        "seed": int(seed),
        **curve_metadata,
        "normalized_knobs": {
            **curve_metadata["normalized_knobs"],
            "gain": float(np.clip(gain, -1.0, 1.0)),
            "cluster": float(np.clip(cluster, -1.0, 1.0)),
        },
        "target_fano_factor": float(target_fano_factor),
        "target_spatial_pair_ratio_10km_plus": float(target_spatial_pair_ratio),
        "cluster_survival_1_to_3m": survival.tolist(),
        "target_probability_1m": probability_1m,
        "target_probability_1km": probability_1km,
        "frame_gains": frame_gains,
    }
    return background, metadata


def generate_sample(
    model: dict[str, np.ndarray],
    output_dir: Path,
    index: int,
    frames: int,
    background_seed: int,
    target_seed: int,
    prefix: str,
    level: float,
    decay: float,
    near_field: float,
    gain: float,
    cluster: float,
    block_frames: int,
    validate: bool,
    target_config: dict[str, float | int],
    preview_margin_m: float = DEFAULT_PREVIEW_MARGIN_M,
    preview_point_size: float = DEFAULT_PREVIEW_POINT_SIZE,
) -> dict[str, Any]:
    """生成、保存一个含目标样本，并返回其轻量级清单记录。

    单进程与并行脚本共用此函数，保证两种入口的背景、目标、落盘字段和预览图
    完全一致。调用方负责预先确定该样本的五个背景旋钮及两个随机种子。
    """
    if preview_margin_m < 0.0 or preview_point_size <= 0.0:
        raise ValueError("预览图的距离余量必须非负，点大小必须为正。")

    output_dir = Path(output_dir)
    target_config = dict(target_config)

    # 先生成背景，再使用独立随机种子生成潜在航迹与目标注入事件。
    background, metadata = generate_background(
        model=model,
        frames=frames,
        seed=background_seed,
        level=float(level),
        decay=float(decay),
        near_field=float(near_field),
        gain=float(gain),
        cluster=float(cluster),
        block_frames=block_frames,
    )
    target = generate_target(
        np.asarray(metadata["target_probability_1m"], dtype=np.float64),
        frames=frames,
        rng=np.random.default_rng(target_seed),
        **target_config,
    )
    observation = overlay_target(background, target)

    # 每个样本拥有独立目录，数据与对应三联检查图始终成对保存。
    sample_dir = output_dir / f"{prefix}_{index:04d}"
    data_path = sample_dir / "data.npz"
    preview_path = sample_dir / "preview.png"
    metadata = {
        **metadata,
        "target": {
            "seed": target_seed,
            "config": target_config,
            "actual_injected_responses": int(np.count_nonzero(target["hit"])),
            "expected_extra_responses": float(target["expected_extra_responses"]),
        },
    }
    save_packed_sample(data_path, background, observation, metadata, target)
    save_sample_preview(
        preview_path,
        background,
        observation,
        np.asarray(metadata["target_probability_1m"], dtype=np.float64),
        target,
        margin_m=preview_margin_m,
        point_size=preview_point_size,
    )

    # 清单只保留复现、统计和定位样本所需的轻量级信息。
    record: dict[str, Any] = {
        "index": index,
        "directory": sample_dir.name,
        "data_file": str(data_path.relative_to(output_dir)),
        "preview_file": str(preview_path.relative_to(output_dir)),
        "background_seed": background_seed,
        "target_seed": target_seed,
        "normalized_knobs": metadata["normalized_knobs"],
        "coefficients": metadata["coefficients"],
        "target_fano_factor": metadata["target_fano_factor"],
        "target_spatial_pair_ratio_10km_plus": (
            metadata["target_spatial_pair_ratio_10km_plus"]
        ),
        "cluster_survival_1_to_3m": metadata["cluster_survival_1_to_3m"],
        "target": metadata["target"],
    }
    if validate:
        profile = profile_binary(background)
        record["observed"] = {
            "mean_events_per_frame": profile["mean_events_per_frame"],
            "fano_factor": profile["fano_factor"],
            "temporal_pair_ratio_10km_plus": profile["temporal_pair_ratio"],
            "spatial_pair_ratio_10km_plus": profile["spatial_pair_ratio"],
        }
    return record


def generate_batch(
    model: dict[str, np.ndarray],
    output_dir: Path,
    samples: int,
    frames: int,
    seed: int,
    prefix: str,
    level: float,
    decay: float,
    near_field: float,
    gain: float,
    cluster: float,
    random_knobs: bool,
    block_frames: int,
    validate: bool,
    target_config: dict[str, float | int],
    preview_margin_m: float = DEFAULT_PREVIEW_MARGIN_M,
    preview_point_size: float = DEFAULT_PREVIEW_POINT_SIZE,
) -> list[dict[str, Any]]:
    """批量生成含目标的样本，并返回可写入清单的轻量级记录。

    random_knobs=False 时，所有样本使用同一组连续旋钮，仅随机种子不同；
    random_knobs=True 时，五个旋钮独立从 [-1,1] 均匀抽取，对应文档的 B-random 场景。
    每个样本写入独立目录，其中含完整 NPZ 数据和三联预览图。
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    if preview_margin_m < 0.0 or preview_point_size <= 0.0:
        raise ValueError("预览图的距离余量必须非负，点大小必须为正。")

    target_config = dict(target_config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    knob_rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []

    for index in range(samples):
        if random_knobs:
            level_i, decay_i, near_field_i, gain_i, cluster_i = knob_rng.uniform(-1.0, 1.0, size=5)
        else:
            level_i, decay_i, near_field_i, gain_i, cluster_i = level, decay, near_field, gain, cluster
        background_seed = int(seed + index)
        target_seed = int(seed + 1_000_000 + index)

        record = generate_sample(
            model=model,
            output_dir=output_dir,
            index=index,
            frames=frames,
            background_seed=background_seed,
            target_seed=target_seed,
            prefix=prefix,
            level=float(level_i),
            decay=float(decay_i),
            near_field=float(near_field_i),
            gain=float(gain_i),
            cluster=float(cluster_i),
            block_frames=block_frames,
            validate=validate,
            target_config=target_config,
            preview_margin_m=preview_margin_m,
            preview_point_size=preview_point_size,
        )
        records.append(record)
    return records


def build_parser() -> argparse.ArgumentParser:
    """构建批量生成命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="生成根目录；其下会自动创建参数命名的批次目录")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--prefix", type=str, default="dataset1_synthetic")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--frames", type=int, default=300, help="每个样本的帧数")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--block-frames", type=int, default=16, help="每次生成的帧块数，限制峰值内存")
    parser.add_argument("--level", type=float, default=0.0, help="归一化范围 [-1,1]")
    parser.add_argument("--decay", type=float, default=0.0, help="归一化范围 [-1,1]")
    parser.add_argument("--near-field", type=float, default=0.0, help="归一化范围 [-1,1]")
    parser.add_argument("--gain", type=float, default=0.0, 
        help="归一化目标帧计数 Fano factor, 范围 [-1,1]"
    )
    parser.add_argument("--cluster", type=float, default=0.0, 
        help="归一化帧内相邻 1-3 m bin 的局部关联强度，范围 [-1,1]"
    )
    parser.add_argument("--random-knobs", type=_parse_bool, default=True, 
        help="为 True 时每个样本独立均匀抽取五个旋钮，构造 B-random 场景"
    )
    parser.add_argument("--validate", type=_parse_bool, default=True, 
        help="是否为每个样本额外计算并写入经验诊断量，批量较大时会较慢"
    )

    parser.add_argument("--preview-margin-m", type=float, default=5_000.0, help="三联预览图在航迹两端保留的距离余量（m）")
    parser.add_argument("--preview-point-size", type=float, default=3.0, help="局部时距图中方形二值响应点的面积（pt²）")
    target_group = parser.add_argument_group("目标合成参数")
    target_group.add_argument("--range-min-m", type=float, default=10_000.0, help="潜在轨迹允许的最小距离（m）")
    target_group.add_argument("--range-max-m", type=float, default=290_000.0, help="潜在轨迹允许的最大距离（m）")
    target_group.add_argument("--max-speed-mps", type=float, default=340.0, help="目标速度绝对值上限（m/s）")
    target_group.add_argument("--max-acceleration-mps2", type=float, default=6.0, help="加速度尺度上限（m/s²）")
    target_group.add_argument("--curve-strength", type=float, default=10.0, help="曲率强度；0 时生成严格直线")
    target_group.add_argument("--smooth-window-frames", type=int, default=31, help="加速度移动平均窗口（帧）")
    target_group.add_argument("--measurement-jitter-std-m", type=float, default=1.0, help="目标响应测距抖动的标准差（m）")
    target_group.add_argument("--response-multiplier", type=float, default=300.0, help="目标相对背景响应倍率 kappa")
    target_group.add_argument("--min-injection-probability", type=float, default=0.35, help="目标额外注入响应的逐帧概率下限")
    target_group.add_argument("--max-injection-probability", type=float, default=0.95, help="目标额外注入响应的逐帧概率上限")
    return parser


def target_config_from_args(args: argparse.Namespace) -> dict[str, float | int]:
    """从两个生成入口共用的命令行参数中整理目标合成配置。"""
    return {
        "range_min_m": args.range_min_m,
        "range_max_m": args.range_max_m,
        "max_speed_mps": args.max_speed_mps,
        "max_acceleration_mps2": args.max_acceleration_mps2,
        "curve_strength": args.curve_strength,
        "smooth_window_frames": args.smooth_window_frames,
        "measurement_jitter_std_m": args.measurement_jitter_std_m,
        "response_multiplier": args.response_multiplier,
        "min_injection_probability": args.min_injection_probability,
        "max_injection_probability": args.max_injection_probability,
    }


def build_generation_hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    """整理会影响样本内容、诊断或预览结果的完整批次超参数。"""
    return {
        "data_geometry": {
            "range_bins": RANGE_BINS,
            "frame_interval_s": FRAME_INTERVAL_S,
            "frames_per_sample": int(args.frames),
        },
        "batch": {
            "samples": int(args.samples),
            "seed": int(args.seed),
            "prefix": str(args.prefix),
        },
        "background": {
            "random_knobs": bool(args.random_knobs),
            "fixed_knobs": {
                "level": float(args.level),
                "decay": float(args.decay),
                "near_field": float(args.near_field),
                "gain": float(args.gain),
                "cluster": float(args.cluster),
            },
            "random_knob_distribution": (
                "independent_uniform[-1,1]"
                if args.random_knobs
                else None
            ),
            "block_frames": int(args.block_frames),
        },
        "target": target_config_from_args(args),
        "preview": {
            "margin_m": float(args.preview_margin_m),
            "background_point_size": float(args.preview_point_size),
        },
        "diagnostics": {"validate": bool(args.validate)},
    }


def main(argv: Iterable[str] | None = None) -> None:
    """读取模型、批量生成含随机目标的样本，并写出 JSON 清单。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    output_root = Path(args.output_dir)
    args.output_dir = resolve_generation_output_dir(args.output_dir, args)
    target_config = target_config_from_args(args)
    hyperparameters = build_generation_hyperparameters(args)
    model = load_model(args.model)
    records = generate_batch(
        model=model,
        output_dir=args.output_dir,
        samples=args.samples,
        frames=args.frames,
        seed=args.seed,
        prefix=args.prefix,
        level=args.level,
        decay=args.decay,
        near_field=args.near_field,
        gain=args.gain,
        cluster=args.cluster,
        random_knobs=args.random_knobs,
        block_frames=args.block_frames,
        validate=args.validate,
        target_config=target_config,
        preview_margin_m=args.preview_margin_m,
        preview_point_size=args.preview_point_size,
    )
    manifest_path = args.manifest or args.output_dir / "batch_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": str(Path(args.model).resolve()),
        "output_root": str(output_root.resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "samples": args.samples,
        "frames_per_sample": args.frames,
        "random_knobs": args.random_knobs,
        "target_config": target_config,
        "hyperparameters": hyperparameters,
        "records": records,
    }
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已生成 {len(records)} 个含目标二值样本目录：{args.output_dir}")
    print(f"批次清单：{manifest_path}")


if __name__ == "__main__":
    main()
