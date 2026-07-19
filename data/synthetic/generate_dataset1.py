#!/usr/bin/env python3
"""读取已拟合的 dataset1 模型，批量生成二值背景数据。

常用命令：

    # 用默认的中等场景生成 10 个样本；每个样本只因随机种子不同而不同。
    conda run -n linetracker-py311 python data/synthetic/generate_dataset1.py \
        --samples 10

    # 随机抽取五个连续旋钮，构造多样的 B-random 场景。
    conda run -n linetracker-py311 python data/synthetic/generate_dataset1.py \
        --samples 100 --random-knobs --seed 20260717

本脚本只读取拟合结果，不读取原始 dataset1 MAT 文件。所有输出始终为 0/1
布尔矩阵；概率坐标变量不表示返回幅值或累计光子数。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


RANGE_BINS = 300_000
PROFILE_WIDTH_M = 1_000
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


def load_model(path: Path = DEFAULT_MODEL_PATH) -> dict[str, np.ndarray]:
    """读取 fit_dataset1.py 输出的 NPZ 模型，且不启用 pickle。"""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}

def save_packed_background(
    path: Path,
    background: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """以 NPZ 位打包格式保存 bool 背景及其理论概率与紧凑元数据。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target_probability_1m = np.asarray(
        metadata["target_probability_1m"], dtype=np.float32
    )
    target_probability_1km = np.asarray(metadata["target_probability_1km"])
    frame_gains = np.asarray(metadata["frame_gains"])
    compact_metadata = dict(metadata)
    compact_metadata.pop("target_probability_1m", None)
    compact_metadata.pop("target_probability_1km", None)
    compact_metadata.pop("frame_gains", None)
    np.savez_compressed(
        path,
        background_packed=np.packbits(background, axis=1),  # 将每帧的 300000 个 0/1 bin 打包为 37500 个 uint8 字节
        # 保存生成前的 1 m 理论占据概率；质检时据此按 1 km 求均值，
        # 使理论曲线与二值背景的经验 1 km 占据率使用完全相同的统计口径。
        target_p_1m=target_probability_1m,
        # 保留原有的 1 km 曲线，供旧版读取程序或空间簇标定排查使用。
        target_p_1km=target_probability_1km,
        frame_gains=frame_gains,
        metadata_json=np.array(
            json.dumps(_jsonable(compact_metadata), ensure_ascii=False)
        ),
    )


def load_packed_background(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """读取位打包背景文件，并恢复“帧 × 距离”bool 数组。"""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        background = np.unpackbits(
            archive["background_packed"],
            axis=1,
            count=int(metadata["range_bins"]),
        ).astype(bool, copy=False)
        metadata["target_probability_1m"] = archive["target_p_1m"].copy()
        metadata["target_probability_1km"] = archive["target_p_1km"].copy()
        metadata["frame_gains"] = archive["frame_gains"].copy()
    return background, metadata


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
) -> list[dict[str, Any]]:
    """批量生成样本并返回可写入清单的轻量级记录。

    random_knobs=False 时，所有样本使用同一组连续旋钮，仅随机种子不同；
    random_knobs=True 时，五个旋钮独立从 [-1,1] 均匀抽取，对应文档的 B-random 场景。
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    knob_rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []

    for index in range(samples):
        if random_knobs:
            level_i, decay_i, near_field_i, gain_i, cluster_i = knob_rng.uniform(-1.0, 1.0, size=5)
        else:
            level_i, decay_i, near_field_i, gain_i, cluster_i = level, decay, near_field, gain, cluster
        sample_seed = int(seed + index)

        # 拟合背景
        background, metadata = generate_background(
            model=model,
            frames=frames,
            seed=sample_seed,
            level=float(level_i),
            decay=float(decay_i),
            near_field=float(near_field_i),
            gain=float(gain_i),
            cluster=float(cluster_i),
            block_frames=block_frames,
        )
        output_path = output_dir / f"{prefix}_{index:04d}.npz"
        save_packed_background(output_path, background, metadata)

        # 记录元信息
        record: dict[str, Any] = {
            "index": index,
            "file": output_path.name,
            "seed": sample_seed,
            "normalized_knobs": metadata["normalized_knobs"],
            "coefficients": metadata["coefficients"],
            "target_fano_factor": metadata["target_fano_factor"],
            "target_spatial_pair_ratio_10km_plus": (
                metadata["target_spatial_pair_ratio_10km_plus"]
            ),
            "cluster_survival_1_to_3m": metadata["cluster_survival_1_to_3m"],
        }
        if validate:
            profile = profile_binary(background)
            record["observed"] = {
                "mean_events_per_frame": profile["mean_events_per_frame"],
                "fano_factor": profile["fano_factor"],
                "temporal_pair_ratio_10km_plus": profile["temporal_pair_ratio"],
                "spatial_pair_ratio_10km_plus": profile["spatial_pair_ratio"],
            }
        records.append(record)
    return records


def build_parser() -> argparse.ArgumentParser:
    """构建批量生成命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--prefix", type=str, default="dataset1_synthetic")
    parser.add_argument("--samples", type=int, default=1)
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
    return parser

def main(argv: Iterable[str] | None = None) -> None:
    """读取模型、批量生成背景，并写出 JSON 清单。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    args.samples = 100
    args.random_knobs = True
    args.validate = True

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
    )
    manifest_path = args.manifest or args.output_dir / "batch_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": str(Path(args.model).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "samples": args.samples,
        "frames_per_sample": args.frames,
        "random_knobs": args.random_knobs,
        "records": records,
    }
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已生成 {len(records)} 个二值背景文件：{args.output_dir}")
    print(f"批次清单：{manifest_path}")


if __name__ == "__main__":
    main()
