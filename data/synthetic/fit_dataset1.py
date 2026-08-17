#!/usr/bin/env python3
"""从真实 dataset1 数据拟合连续可调的二值背景模型。

常用命令：

    conda run -n linetracker-py311 python data/synthetic/fit_dataset1.py

本脚本只读取稳定真实样本并输出模型 NPZ 与拟合摘要 JSON，不生成合成背景。
合成数据请使用同目录下的 generate_dataset1.py。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, cast

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d


MODEL_VERSION = "dataset1-binary-background-v1"
RANGE_BINS = 300_000        # 距离维度的总 bin 数
PROFILE_WIDTH_M = 1_000     # 逐距离响应率统计的分组宽度，单位为米。
BACKGROUND_START_M = 10_000 # 只在统计时间相邻对、空间相邻对时，忽略 10 km 以内近场区域。
CHUNK_RANGE_BINS = 5_000    # 读取大文件时的距离块大小，必须是 PROFILE_WIDTH_M 的整数倍。
STABLE_SAMPLE_NUMBERS = (1, 3, 4, 6, 7, 9, 10)
REGIONS = (                 # 计算各距离段的平均响应率，用于 JSON 摘要和检查。
    (1, 10_000, "0–10 km"),
    (10_000, 50_000, "10–50 km"),
    (50_000, 100_000, "50–100 km"),
    (100_000, 300_000, "100–300 km"),
)

HERE = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = HERE.parent / "raw" / "dataset1"
DEFAULT_FIT_DIR = HERE / "fit"
DEFAULT_MODEL_PATH = DEFAULT_FIT_DIR / "dataset1_background_model.npz"
DEFAULT_SUMMARY_PATH = DEFAULT_FIT_DIR / "dataset1_fit_summary.json"


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


def _find_hdf5_matrix(path: Path) -> tuple[str, tuple[int, int]]:
    """在 MAT 文件中定位唯一的二维 300 km 测量矩阵。"""
    candidates: list[tuple[str, tuple[int, int]]] = []
    with h5py.File(path, "r") as handle:
        def visitor(name: str, node: h5py.Dataset | h5py.Group) -> None:
            """收集距离维长度为 RANGE_BINS 的二维数据集。"""
            if isinstance(node, h5py.Dataset) and node.ndim == 2 and RANGE_BINS in node.shape:
                # node.ndim 已确认是二维；显式拆包以保留 tuple[int, int] 类型信息。
                rows, columns = (int(value) for value in node.shape)
                candidates.append((name, (rows, columns)))

        handle.visititems(visitor)
    if len(candidates) != 1:
        raise ValueError(f"Expected one {RANGE_BINS}-bin matrix in {path}, found {candidates}")
    return candidates[0]


def _sample_path(raw_root: Path, sample_number: int) -> Path:
    """定位某个编号 dataset1 样本对应的唯一 MAT 文件。"""
    matches = sorted((raw_root / str(sample_number)).glob("*.mat"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one .mat file under {raw_root / str(sample_number)}, found {matches}"
        )
    return matches[0]


def _contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    """将排序索引转换为闭区间连续段 (start, end)。"""
    if indices.size == 0:
        return []
    split_points = np.where(np.diff(indices) != 1)[0] + 1
    return [
        (int(group[0]), int(group[-1]))
        for group in np.split(indices, split_points)
    ]


def profile_real_sample(
    path: Path,
    sample_id: str,
    profile_width_m: int = PROFILE_WIDTH_M,
    chunk_range_bins: int = CHUNK_RANGE_BINS,
) -> dict[str, Any]:
    """流式统计一个真实样本的二值背景特征。

    对应方案第 3.1 节的逐 1 m 与 1 km 响应率统计，并同时收集第 3.8--3.9 节
    所需的 10 km+ 相邻对比值和帧计数波动。所有原始非零值都会变为 True；
    返回幅值大小不参与拟合，bin 0 固定硬件标记会被屏蔽。
    """
    if RANGE_BINS % profile_width_m:
        raise ValueError("profile_width_m must divide RANGE_BINS")
    if chunk_range_bins % profile_width_m:
        raise ValueError("chunk_range_bins must align with profile_width_m")
    if not h5py.is_hdf5(path):
        raise ValueError(f"dataset1 file is expected to be MATLAB v7.3/HDF5: {path}")

    variable, shape = _find_hdf5_matrix(path)
    K = RANGE_BINS // profile_width_m
    range_first = shape[0] == RANGE_BINS                # 原始矩阵是否已经是“距离 × 帧”；否则其形状为“帧 × 距离”，读取块时需转置。
    frames = shape[1] if range_first else shape[0]      # 总帧数 F
    frame_counts = np.zeros(frames, dtype=np.int64)     # (F,) 累计每一帧非零 bin 数 N_t
    range_counts = np.zeros(RANGE_BINS, dtype=np.int64) # (RANGE_BINS, ) 累计每个 bin 在全部帧中取值为 1 的次数。
    grouped_counts = np.zeros(K, dtype=np.int64)        # (K, ) 累计每个 1 km 距离组在全部帧中取值为 1 的次数。
    temporal_pairs = 0                                  # 10 km 以上、同 bin 相邻两帧中同时为 1 的观测对数。
    spatial_pairs = 0                                   # 10 km 以上、用帧相邻 bin 同时为 1 的观测对数。
    previous_background_row: np.ndarray | None = None   # 前一个距离块的最后一行，用于统计两个相邻读取块边界处的空间相邻对

    with h5py.File(path, "r") as handle:
        dataset = handle[variable]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"{path}: {variable!r} 不是 HDF5 数据集。")
        for start in range(0, RANGE_BINS, chunk_range_bins):
            stop = min(start + chunk_range_bins, RANGE_BINS)
            group_start = start // profile_width_m
            group_stop = stop // profile_width_m
            
            # 第 3.1 步：统一为“距离 × 帧”方向并二值化。
            if range_first:
                raw_block = np.asarray(dataset[start:stop, :])
            else:
                raw_block = np.asarray(dataset[:, start:stop]).T
            occupied = raw_block > 0        # (chunk_range_bins, frames)
            if start == 0:
                occupied = occupied.copy()
                occupied[0, :] = False

            # 累计逐距离、逐帧与 1 km 距离组的二值响应计数。
            frame_counts += occupied.sum(axis=0, dtype=np.int64)
            block_range_counts = occupied.sum(axis=1, dtype=np.int64)
            range_counts[start:stop] = block_range_counts
            grouped_counts[group_start:group_stop] = block_range_counts.reshape(
                -1,
                profile_width_m,
            ).sum(axis=1, dtype=np.int64)

            # 仅在 10 km+ 背景域统计时空相邻对，并处理相邻距离块的边界。
            local_start = max(0, BACKGROUND_START_M - start)
            background = occupied[local_start:, :]
            if background.size:
                temporal_pairs += int(np.count_nonzero(background[:, 1:] & background[:, :-1]))
                if previous_background_row is not None:
                    spatial_pairs += int(np.count_nonzero(previous_background_row & background[0]))
                spatial_pairs += int(np.count_nonzero(background[1:, :] & background[:-1, :]))
                previous_background_row = background[-1].copy()

    # 相邻对分母均以观测 p(r) 下的独立背景期望为准。
    probability_by_range = range_counts / frames                        # 每个 1 m bin 在全部帧中为 1 的经验概率 p(r)。
    background_counts = range_counts[BACKGROUND_START_M:]               # 仅保留 10 km 以上各 bin 的累计响应次数。
    background_probability = probability_by_range[BACKGROUND_START_M:]  # 10 km 以上的经验概率 p(r)。
    
    expected_temporal_pairs = float(np.sum(background_counts * (background_counts - 1), dtype=np.float64) / frames) # 按各 bin 的 p(r) 独立时，同一 bin 相邻帧同时为 1 的期望对数。
    temporal_pair_ratio = float(temporal_pairs / expected_temporal_pairs)                                           # 同一 bin 相邻帧同时为 1 的值与独立期望的比值。
    expected_spatial_pairs = float(frames * np.dot(background_probability[:-1], background_probability[1:]))    # 按相邻 bin 的 p(r) 独立时，同一帧相邻 bin 同时为 1 的期望对数。        
    spatial_pair_ratio = float(spatial_pairs / expected_spatial_pairs)                                          # 同一帧相邻 bin 同时为 1 的值与独立期望的比值。

    mean_events = float(frame_counts.mean())                    # 每帧平均响应 bin 数 E[N_t]。
    frame_variance = float(frame_counts.var(ddof=1))            # 帧响应数 N_t 的样本方差。
    fano = frame_variance / mean_events                         # 帧响应数的 Fano 系数 Var(N_t) / E[N_t]。
    median_count = float(np.median(frame_counts))               # 识别低计数连续帧时使用的基准中位数。

    return {
        "sample_id": sample_id,                     # 样本标识如 dataset1/1
        "frames": int(frames),                      # 样本总帧数 F；用于概率归一化和摘要。
        "grouped_counts": grouped_counts,           # 每个 1 km 距离组的响应次数；用于计算修正后的拟合概率。
        "grouped_rates": grouped_counts / (frames * profile_width_m),  # 每个 1 km 距离组的平均响应概率；用于保存真实曲线和检查。
        "mean_events_per_frame": mean_events,       # 每帧平均响应数；用于真实与合成数据对比。
        "fano_factor": float(fano),                 # 帧响应数 N_t 的 Fano 系数；用于标定生成时的 gain 范围。
        "temporal_pair_ratio": temporal_pair_ratio, # 10 km+ 同一 bin 相邻帧对 / 独立期望；用于时间独立性检查。
        "spatial_pair_ratio": spatial_pair_ratio,   # 10 km+ 相邻 bin 同帧对 / 独立期望；用于标定空间簇强度。
        "low_count_runs": _contiguous_runs(         # 帧响应数低于中位数 50% 的连续帧索引；用于识别异常样本。
            np.flatnonzero(frame_counts < 0.5 * median_count)
        ),  
        "region_rates": {                           # 各预设距离段内每个 1 m bin 的平均响应概率；用于分段对比。
            label: float(range_counts[left:right].sum() / ((right - left) * frames))
            for left, right, label in REGIONS
        },
    }


def _expanded_interval(
    values: np.ndarray,
    fraction: float,
    floor: float | None = None,
) -> np.ndarray:
    """以中点为中心扩展样本最小值—最大值区间。"""
    low = float(np.min(values))
    high = float(np.max(values))
    midpoint = 0.5 * (low + high)
    half_width = 0.5 * (high - low)
    if half_width == 0.0:
        half_width = max(abs(midpoint) * 0.05, 1e-6)
    interval = np.array(
        [
            midpoint - (1.0 + fraction) * half_width,
            midpoint + (1.0 + fraction) * half_width,
        ],
        dtype=np.float64,
    )
    if floor is not None:
        interval[0] = max(interval[0], floor)
    return interval


def _build_fit_summary(
    profiles: list[dict[str, Any]],
    model: dict[str, np.ndarray],
    raw_root: Path,
) -> dict[str, Any]:
    """整理拟合诊断指标，生成便于阅读和检查的 JSON 摘要。"""
    smoothed_curves = model["sample_smoothed_log_intensity"]
    base_curve = model["base_log_intensity"]
    basis = model["basis"]
    coefficients = model["sample_coefficients"]
    residual_rmse = np.sqrt(np.mean(
        (smoothed_curves - (base_curve + coefficients @ basis.T)) ** 2,
        axis=1,
    ))

    sample_metrics = []
    for index, profile in enumerate(profiles):
        sample_metrics.append({
            "sample": profile["sample_id"],
            "frames": profile["frames"],
            "mean_events_per_frame": profile["mean_events_per_frame"],
            "fano_factor": profile["fano_factor"],
            "temporal_pair_ratio_10km_plus": profile["temporal_pair_ratio"],
            "spatial_pair_ratio_10km_plus": profile["spatial_pair_ratio"],
            "low_count_runs": profile["low_count_runs"],
            "region_rates": profile["region_rates"],
            "coefficients": coefficients[index],
            "three_parameter_log_intensity_rmse": residual_rmse[index],
        })

    return {
        "model_version": MODEL_VERSION,
        "raw_root": raw_root,
        "fit_samples": model["sample_ids"].tolist(),
        "excluded_samples": {
            "dataset1/2": "byte-identical duplicate of dataset1/1",
            "dataset1/5": "reserved for low-count/dropout stress testing",
            "dataset1/8": "reserved for low-count/dropout stress testing",
        },
        "profile_width_m": int(model["profile_width_m"]),
        "expanded_fraction": float(model["expanded_fraction"]),
        "parameter_names": model["parameter_names"],
        "coefficient_ranges": model["coefficient_ranges"],
        "fano_range": model["fano_range"],
        "spatial_pair_range": model["spatial_pair_range"],
        "sample_metrics": sample_metrics,
    }


def fit_dataset1(
    raw_root: Path = DEFAULT_RAW_ROOT,
    expanded_fraction: float = 0.25,
    smoothing_sigma_km: float = 1.5,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """从稳定真实样本拟合连续可调的 dataset1 二值背景模型。

    对应方案第 3.1--3.6 节：统计响应率、映射到数值坐标 a、高斯平滑、提取
    跨样本中位数基准、拟合三个基函数，并保存生成步骤 3.7 所需的真实曲线与
    参数范围。变量名中的 intensity 只是历史命名，实际表示数值坐标 a。
    """
    raw_root = Path(raw_root).resolve()
    # 第 3.1 步：统计经验距离响应率，排除重复样本 2 和异常样本 5、8
    profiles = [
        profile_real_sample(
            _sample_path(raw_root, sample_number),
            sample_id=f"dataset1/{sample_number}",
        )
        for sample_number in STABLE_SAMPLE_NUMBERS
    ]

    frames = np.array([profile["frames"] for profile in profiles], dtype=np.int64)
    sample_ids = np.array([profile["sample_id"] for profile in profiles])
    grouped_rates = np.stack([profile["grouped_rates"] for profile in profiles])    # (7, 300) 每个样本的逐距离组平均响应率，用于保存真实曲线和检查。

    # 第 3.2 步：把概率映射到便于拟合的数值坐标
    grouped_counts = np.stack([profile["grouped_counts"] for profile in profiles])  # (7, 300) 各样本各距离组相应总数
    group_bin_totals = frames[:, None] * PROFILE_WIDTH_M                            # (7, 1)   各样本 1 km 距离组中参与统计的二值单元总数
    adjusted_rates = (grouped_counts + 0.5) / (group_bin_totals + 1.0)              # (7, 300) 各样本各距离组的经验响应概率，修正避免 log(0)。        
    raw_log_intensity = np.log(-np.log1p(-adjusted_rates))                          # (7, 300) 各样本各距离组的概率数值坐标 a，便于拟合

    # 第 3.3--3.4 步：沿距离组平滑，并取跨样本中位数，得到跨样本公共基准曲线。
    smoothed_log_intensity = gaussian_filter1d(                                     # (7, 300) 
        raw_log_intensity,
        sigma=smoothing_sigma_km,
        axis=1,
        mode="nearest",
    )
    base_log_intensity = np.median(smoothed_log_intensity, axis=0)                  # (300, ) 跨样本中位数基准曲线，控制概率衰减曲线总体形状

    # 第 3.5 步：设置整体、衰减和近场三个基函数的
    centres_km = np.arange(RANGE_BINS // PROFILE_WIDTH_M) + 0.5         # (300,) 各 1 km 距离组的中心距离
    level_basis = np.ones_like(centres_km)                              # (300,) b_0 常数基函数；用于整体抬高或降低概率曲线。
    decay_basis = 0.5 - centres_km / (RANGE_BINS / 1_000)               # (300,) b_1 远近衰减程度基函数，近场为正、远场为负；用于调节衰减程度。
    near_basis = np.exp(-0.5 * (centres_km / 8.0) ** 2)                 # (300,) b_2 进场抬升基函数，仅近场较大；用于调节 0–10 km 抬升。
    
    # 第 3.6 步：对每个样本最小二乘拟合三个基函数的系数
    basis = np.column_stack([level_basis, decay_basis, near_basis])     # (300, 3) 三个基函数组成的拟合设计矩阵。
    coefficients = np.stack([                                           # (7, 3) 每个稳定样本对应的 (theta_0, theta_1, theta_2) 拟合系数。
        np.linalg.lstsq(basis, curve - base_log_intensity, rcond=None)[0]
        for curve in smoothed_log_intensity
    ])  
    coefficient_ranges = np.stack([                                     # (3, 2) 三个系数各自的 [下界, 上界]；供生成器连续采样。
        _expanded_interval(coefficients[:, column], expanded_fraction)
        for column in range(coefficients.shape[1])
    ])  

    fano_values = np.array([profile["fano_factor"] for profile in profiles])                # (7,) 每个样本的帧响应数 Fano 系数，用于标定生成时的 gain 范围
    temporal_ratios = np.array([profile["temporal_pair_ratio"] for profile in profiles])    # (7,) 每个样本的 10 km+ 同一 bin 相邻帧对 / 独立期望，用于检查时间独立性
    spatial_ratios = np.array([profile["spatial_pair_ratio"] for profile in profiles])      # (7,) 每个样本的 10 km+ 相邻 bin 同帧对 / 独立期望，用于标定空间簇强度

    model = {
        "model_version": np.array(MODEL_VERSION),                       # 模型格式版本；写入生成数据的元信息
        "range_bins": np.array(RANGE_BINS, dtype=np.int64),             # 距离 bin 总数 R=300000
        "profile_width_m": np.array(PROFILE_WIDTH_M, dtype=np.int64),   # 距离组宽度，用于恢复 1 m 概率。
        "expanded_fraction": np.array(expanded_fraction),               # 包络和参数范围相对真实样本的外扩比例。
        "parameter_names": np.array(["level", "decay", "near_field"]),  # 三个基函数与 theta 系数的名称对应。
        "sample_ids": sample_ids,                                       # 参与拟合的稳定样本名称；用于摘要和检查图例。
        "sample_grouped_rates": grouped_rates,                          # (7, 300) 原始 1 km 响应率曲线；用于检查图。
        "sample_smoothed_log_intensity": smoothed_log_intensity,        # (7, 300) 平滑 a 曲线；用于生成器构建 3.7 包络。
        "sample_coefficients": coefficients,                            # (7, 3) 各样本拟合的 theta 系数；用于检查参数拟合范围。
        "base_log_intensity": base_log_intensity,                       # (300,) 跨样本中位数基准 a 曲线。
        "basis": basis,                                                 # (300, 3) 三个基函数曲线，用于由 theta 重建新曲线。
        "coefficient_ranges": coefficient_ranges,                       # (3, 2) 三个 theta 的连续采样区间。
        "fano_range": _expanded_interval(                               # gain 映射的目标 Fano 区间，用于在 3.9 步标定帧级系数 c_t 的波动强度。
            fano_values,
            expanded_fraction,
            floor=0.0,
        ),
        "temporal_pair_values": temporal_ratios,                        # 10 km+ 同一 bin 相邻帧对 / 独立期望，用于步骤 3.9 后的检查参考，确认 c_t 不额外引入帧间持续相关
        "spatial_pair_range": _expanded_interval(                       # cluster 映射的目标空间相邻对比值区间，用于在步骤 3.8 标定延伸概率 (e_1,e_2,e_3)
            spatial_ratios,
            expanded_fraction,
            floor=1.0,
        ),
    }

    summary = _build_fit_summary(profiles, model, raw_root)
    return model, summary


def save_model(
    model: dict[str, np.ndarray],
    summary: dict[str, Any],
    model_path: Path = DEFAULT_MODEL_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> None:
    """保存 NPZ 模型数组和便于阅读的 JSON 拟合摘要。"""
    model_path = Path(model_path)
    summary_path = Path(summary_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # NumPy 的类型桩无法表达 ``**model`` 这种同名数组的动态关键字参数；
    # 运行时仍按原样保存 NPZ 中的每个模型数组。
    np.savez_compressed(model_path, **cast(Any, model))
    summary_path.write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """构建仅用于拟合的命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--expanded-fraction", type=float, default=0.25,
        help="数值坐标包络的外扩比例 eta，默认 0.25，即低概率区约对应概率上浮 25% 或下探至原来的 1/1.25",
    )
    parser.add_argument("--smoothing-sigma-km", type=float, default=1.5, 
        help="高斯平滑的标准差，默认 1.5 km"                    
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    """读取真实数据、拟合模型并保存输出。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    model, summary = fit_dataset1(
        raw_root=args.raw_root,
        expanded_fraction=args.expanded_fraction,
        smoothing_sigma_km=args.smoothing_sigma_km,
    )
    save_model(model, summary, args.model, args.summary)
    print(f"已保存模型：{args.model}")
    print(f"已保存拟合摘要：{args.summary}")
    print("稳定样本：", ", ".join(model["sample_ids"].tolist()))


if __name__ == "__main__":
    main()
