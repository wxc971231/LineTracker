"""生成人类可读且可区分完整算法配置的实验名称。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from configs.base import SimpleCNNConfig


_CONFIG_DIGEST_EXCLUDED_FIELDS = {
    "profile",
    "run_name",
    "output_root",
    "data_root",
    "resume",
    "num_workers",
    "pin_memory",
    "log_interval_steps",
    "checkpoint_interval_steps",
    "keep_last_checkpoints",
    "wandb_project",
    "wandb_entity",
    "wandb_mode",
}


def _compact_count(value: int) -> str:
    """精确压缩整千数量；其他值保持原样，避免产生歧义。"""
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}k"
    return str(value)


def _scientific_token(value: float) -> str:
    """返回适合路径的紧凑科学计数法，如 ``3e-4``。"""
    mantissa, exponent = f"{value:.5e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def _decimal_token(value: float) -> str:
    """返回不含多余零的十进制路径片段。"""
    return format(value, ".6g").replace(".", "p")


def algorithm_config_digest(config: SimpleCNNConfig, length: int = 8) -> str:
    """计算排除本机路径、日志和纯 I/O 选项后的稳定配置摘要。"""
    if not 4 <= length <= 64:
        raise ValueError("配置摘要长度必须位于 [4, 64]。")
    values = config.as_serializable_dict()
    algorithm_values = {
        key: value
        for key, value in values.items()
        if key not in _CONFIG_DIGEST_EXCLUDED_FIELDS
    }
    negative_weight_names = (
        "negative_local_weight",
        "negative_same_time_weight",
        "negative_random_weight",
        "negative_partial_weight",
    )
    negative_weight_sum = sum(
        float(algorithm_values[name]) for name in negative_weight_names
    )
    if negative_weight_sum > 0.0:
        for name in negative_weight_names:
            algorithm_values[name] = (
                float(algorithm_values[name]) / negative_weight_sum
            )
    payload = json.dumps(
        algorithm_values,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def build_experiment_slug(config: SimpleCNNConfig, world_size: int) -> str:
    """组合数据上限、全局 batch、模型容量和完整配置短哈希。"""
    if world_size < 1:
        raise ValueError("world_size 必须为正整数。")
    global_batch_size = (
        config.batch_size_per_gpu
        * world_size
        * config.gradient_accumulation_steps
    )
    source_limit = (
        "all"
        if config.source_sample_limit == 0
        else _compact_count(config.source_sample_limit)
    )
    return (
        f"limit{source_limit}-gbs{global_batch_size}"
        f"-lr{_scientific_token(config.learning_rate)}"
        f"-pos{_decimal_token(config.positive_fraction * 100.0)}"
        f"-vs{config.validation_time_stride}-model{config.model_type}"
        f"-cfg{algorithm_config_digest(config)}"
    )


def build_wandb_run_name(config: SimpleCNNConfig, run_dir: Path) -> str:
    """让 W&B 名称与新目录可读信息一致，并兼容旧两级 run 目录。"""
    timestamp = run_dir.name
    if run_dir.parent.name == config.run_name:
        return f"{config.run_name}-{timestamp}"
    return f"{config.run_name}-{run_dir.parent.name}-{timestamp}"
