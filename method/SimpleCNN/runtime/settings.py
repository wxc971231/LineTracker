"""从本地 ``.env`` 读取与算法无关的运行时设置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from configs.base import SimpleCNNConfig


METHOD_ROOT = Path(__file__).resolve().parents[1]
_ACCELERATORS = {"auto", "cuda", "npu", "cpu"}
_BACKENDS = {"auto", "nccl", "hccl", "gloo"}
_AMP_MODES = {"auto", "fp16", "bf16", "off"}


@dataclass(frozen=True)
class RuntimeSettings:
    """仅描述当前机器、设备和 I/O；不包含模型或采样超参数。"""

    accelerator: str = "auto"
    distributed_backend: str = "auto"
    amp: str | None = None
    distributed_timeout_minutes: int = 360
    pin_memory: bool | None = None
    data_root: Path | None = None
    output_root: Path | None = None
    env_file: Path | None = None


def _clean_env_value(raw: str) -> str:
    """移除未加引号值的行尾注释，兼容 VS Code 对 envFile 的解析结果。"""
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value.split(" #", maxsplit=1)[0].rstrip()


def _parse_dotenv(path: Path) -> dict[str, str]:
    """解析本项目需要的简洁 KEY=VALUE 格式，不引入额外运行时依赖。"""
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_number} 不是 KEY=VALUE 格式。")
        key, value = (part.strip() for part in line.split("=", maxsplit=1))
        if not key.startswith("LT_"):
            continue
        values[key] = _clean_env_value(value)
    return values


def _read_env_file(path: Path | None) -> Path | None:
    """将 .env 中未在进程环境中定义的 LT_* 值写入当前进程。"""
    candidate = METHOD_ROOT / ".env" if path is None else path.expanduser().resolve()
    if not candidate.exists():
        if path is not None:
            raise FileNotFoundError(f"指定的运行时环境文件不存在：{candidate}")
        return None
    for key, value in _parse_dotenv(candidate).items():
        os.environ.setdefault(key, value)
    return candidate


def _optional_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    normalized = _clean_env_value(raw).lower()
    if normalized == "auto":
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 应为 auto、true 或 false。")


def _optional_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = _clean_env_value(raw)
    if not value:
        return None
    return Path(os.path.expandvars(value)).expanduser().resolve()


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(_clean_env_value(raw))
    except ValueError as error:
        raise ValueError(f"{name} 应为正整数。") from error
    if value < 1:
        raise ValueError(f"{name} 应为正整数。")
    return value


def _choice(name: str, allowed: set[str], default: str | None) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = _clean_env_value(raw).lower()
    if value not in allowed:
        raise ValueError(f"{name} 应为 {sorted(allowed)} 之一，实际为 {raw!r}。")
    return value


def load_runtime_settings(env_file: Path | None = None) -> RuntimeSettings:
    """读取运行时设置；已有环境变量优先于 .env，保护外部注入变量。"""
    loaded_file = _read_env_file(env_file)
    return RuntimeSettings(
        accelerator=_choice("LT_ACCELERATOR", _ACCELERATORS, "auto") or "auto",
        distributed_backend=_choice("LT_DISTRIBUTED_BACKEND", _BACKENDS, "auto") or "auto",
        amp=_choice("LT_AMP", _AMP_MODES, None),
        distributed_timeout_minutes=_positive_int("LT_DISTRIBUTED_TIMEOUT_MINUTES", 360),
        pin_memory=_optional_bool("LT_PIN_MEMORY"),
        data_root=_optional_path("LT_DATA_ROOT"),
        output_root=_optional_path("LT_OUTPUT_ROOT"),
        env_file=loaded_file,
    )


def apply_runtime_settings(config: SimpleCNNConfig, settings: RuntimeSettings) -> SimpleCNNConfig:
    """把环境相关覆盖值应用到实验配置；随后 CLI --set 仍可覆盖它们。"""
    if settings.amp is not None:
        config.amp = settings.amp
    if settings.pin_memory is not None:
        config.pin_memory = settings.pin_memory
    if settings.data_root is not None:
        config.data_root = settings.data_root
    if settings.output_root is not None:
        config.output_root = settings.output_root
    return config


def apply_device_defaults(
    config: SimpleCNNConfig,
    settings: RuntimeSettings,
    device_type: str,
) -> SimpleCNNConfig:
    """补齐未显式指定的设备相关设置；CLI ``--set`` 可在其后覆盖。"""
    if settings.pin_memory is None and device_type != "cuda":
        config.pin_memory = False
    return config
