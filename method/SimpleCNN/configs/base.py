"""SimpleCNN 的集中式配置、profile 加载和命令行覆盖工具。"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


METHOD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = METHOD_ROOT.parents[1]
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "data/synthetic/gen/F300_N200_S20260717_B-random_T-R10000-290000m-V340-A6-C10-J1-K300-Q0p35-0p95"
)


@dataclass
class SimpleCNNConfig:
    """第一版 SimpleCNN 的全部可调超参数。

    所有 batch size 都是“每 GPU”的 micro-batch size；训练总预算以
    ``total_optimizer_steps`` 为准，而不是 epoch 数。
    """

    # 实验与随机性
    profile: str = "simplecnn_v1"
    run_name: str = "simplecnn_v1"
    seed: int = 20260725
    output_root: Path = METHOD_ROOT / "runs"
    data_root: Path = DEFAULT_DATA_ROOT

    # 完整序列划分
    split_seed: int = 20260725
    train_fraction: float = 0.80
    val_fraction: float = 0.10
    test_fraction: float = 0.10

    # 分块与压缩观测
    frames_per_window: int = 20
    range_bins: int = 300_000
    block_width_m: int = 10_000
    spatial_step_m: int = 9_000
    packed_bitorder: str = "big"

    # 训练在线数据流
    batch_size_per_gpu: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    source_cache_size: int = 4
    source_positive_quota: int = 64
    positive_fraction: float = 0.25
    standard_positive_fraction: float = 0.30
    positive_margin_m: int = 100
    negative_guard_m: int = 100
    negative_local_span_m: int = 3_000
    negative_local_weight: float = 1.0 / 3.0
    negative_same_time_weight: float = 1.0 / 3.0
    negative_random_weight: float = 1.0 / 3.0
    counterfactual_negative_weight: float = 0.0

    # 固定标准网格验证
    validation_time_stride: int = 5
    eval_batch_size_per_gpu: int = 64

    # 网络结构（与 _doc/SimpleCNN.md 第 4 章一致）
    input_channels: int = 8
    hidden_dim: int = 256
    dropout: float = 0.10
    max_speed_per_frame_m: float = 17.0

    # 损失
    lambda_q: float = 1.0
    lambda_line: float = 1.0
    huber_delta_bins: float = 3.0

    # 优化器和按 step 调度
    total_optimizer_steps: int = 50_000
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate_ratio: float = 0.10
    warmup_ratio: float = 0.05
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    amp: str = "auto"  # auto / bf16 / fp16 / off
    compile_model: bool = False

    # 日志、验证和断点
    log_interval_steps: int = 20
    eval_interval_steps: int = 1_000
    checkpoint_interval_steps: int = 1_000
    keep_last_checkpoints: int = 2
    resume: Path | None = None

    # Weights & Biases
    wandb_enabled: bool = True
    wandb_project: str = "LineTracker-SimpleCNN"
    wandb_entity: str | None = None
    wandb_mode: str = "online"  # online / offline / disabled

    def validate(self) -> None:
        """在启动训练前尽早报告配置错误。"""
        if self.frames_per_window != 20:
            raise ValueError("SimpleCNN-v1 固定使用 20 帧输入。")
        if self.block_width_m != 10_000:
            raise ValueError("SimpleCNN-v1 固定使用 10000 m 距离块。")
        if self.block_width_m % self.input_channels != 0:
            raise ValueError("距离块宽度必须能被重排通道数整除。")
        if self.spatial_step_m <= 0 or self.spatial_step_m > self.block_width_m:
            raise ValueError("spatial_step_m 必须位于 (0, block_width_m]。")
        if self.batch_size_per_gpu < 2:
            raise ValueError("batch_size_per_gpu 至少为 2，才能同时容纳正负样本。")
        if not 0.0 < self.positive_fraction < 1.0:
            raise ValueError("positive_fraction 必须位于 (0, 1)。")
        if not 0.0 <= self.standard_positive_fraction <= 1.0:
            raise ValueError("standard_positive_fraction 必须位于 [0, 1]。")
        if self.validation_time_stride < 1:
            raise ValueError("validation_time_stride 的最小值为 1。")
        if self.total_optimizer_steps < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("训练步数和梯度累积步数必须为正。")
        if self.source_cache_size < 1 or self.source_positive_quota < 1:
            raise ValueError("source cache 和 source quota 必须为正。")
        if self.huber_delta_bins <= 0.0:
            raise ValueError("huber_delta_bins 必须为正。")
        if self.amp not in {"auto", "bf16", "fp16", "off"}:
            raise ValueError("amp 仅支持 auto、bf16、fp16 或 off。")
        if self.packed_bitorder not in {"big", "little"}:
            raise ValueError("packed_bitorder 仅支持 big 或 little。")
        weight_sum = (
            self.negative_local_weight
            + self.negative_same_time_weight
            + self.negative_random_weight
            + self.counterfactual_negative_weight
        )
        if weight_sum <= 0.0:
            raise ValueError("至少一种负样本来源的权重必须为正。")
        split_sum = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(split_sum - 1.0) > 1e-8:
            raise ValueError("train/val/test 三个比例之和必须为 1。")

    def as_serializable_dict(self) -> dict[str, Any]:
        """把 Path 等对象转换为可写入 JSON 的配置快照。"""

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))


def _coerce_override(current: Any, raw_value: str) -> Any:
    """按现有字段类型把 ``--set key=value`` 文本转换为对应 Python 类型。"""
    if isinstance(current, bool):
        normalized = raw_value.strip().lower()
        if normalized not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"布尔覆盖值无效：{raw_value}")
        return normalized in {"true", "1", "yes"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw_value)
    if isinstance(current, float):
        return float(raw_value)
    if isinstance(current, Path):
        return Path(raw_value).expanduser().resolve()
    if current is None:
        return None if raw_value.lower() == "none" else raw_value
    return raw_value


def apply_overrides(config: SimpleCNNConfig, assignments: list[str]) -> SimpleCNNConfig:
    """应用形如 ``--set learning_rate=1e-4`` 的扁平配置覆盖。"""
    known_fields = {item.name for item in fields(config)}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"覆盖参数必须采用 key=value 格式：{assignment}")
        key, raw_value = assignment.split("=", maxsplit=1)
        key = key.strip()
        if key not in known_fields:
            raise KeyError(f"未知配置字段：{key}")
        setattr(config, key, _coerce_override(getattr(config, key), raw_value))
    config.validate()
    return config


def load_profile(profile_name: str) -> SimpleCNNConfig:
    """加载 ``configs/<profile_name>.py`` 中的 ``get_config``。"""
    module = importlib.import_module(f"configs.{profile_name}")
    if not hasattr(module, "get_config"):
        raise AttributeError(f"configs.{profile_name} 未定义 get_config()。")
    config = module.get_config()
    if not isinstance(config, SimpleCNNConfig):
        raise TypeError("get_config() 必须返回 SimpleCNNConfig。")
    config.profile = profile_name
    config.validate()
    return config


def load_config_json(path: Path) -> SimpleCNNConfig:
    """从已经落盘的 ``resolved_config.json`` 恢复配置。"""
    values = json.loads(path.read_text(encoding="utf-8"))
    path_keys = {"output_root", "data_root", "resume"}
    for key in path_keys:
        if values.get(key) is not None:
            values[key] = Path(values[key])
    config = SimpleCNNConfig(**values)
    config.validate()
    return config


def save_config_json(config: SimpleCNNConfig, path: Path) -> None:
    """保存解析后的完整配置，便于恢复和可复现实验。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.as_serializable_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_entrypoint_args(description: str) -> argparse.Namespace:
    """训练和独立评估入口共享的基础命令行参数。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="simplecnn_v1", help="configs 下的 profile 名称。")
    parser.add_argument("--resume", type=Path, default=None, help="要恢复的 last.pt checkpoint。")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖 profile 字段，可重复使用，例如 --set batch_size_per_gpu=16。",
    )
    return parser.parse_args()
