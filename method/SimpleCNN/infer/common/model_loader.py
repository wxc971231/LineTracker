"""单进程推理使用的 checkpoint、配置和模型加载工具。

本模块不初始化 torch.distributed：流式推理的一个运行实例只持有一张
CUDA/NPU（或 CPU）设备。训练时保存的 resolved_config.json 是模型结构的
唯一来源；运行时 .env 只允许覆盖机器相关的 I/O、AMP 和设备默认值。
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


METHOD_ROOT = Path(__file__).resolve().parents[2]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from configs.base import SimpleCNNConfig, load_config_json
from runtime.settings import (
    RuntimeSettings,
    apply_device_defaults,
    apply_runtime_settings,
    load_runtime_settings,
)
from train.model import SimpleCNN
from utils.checkpoint import load_checkpoint


@dataclass(frozen=True)
class InferenceBundle:
    """一次推理所需的已验证模型、配置和 checkpoint 元数据。

    model 已经迁移到 device 并调用 eval()。调用方只需在
    torch.inference_mode() 中将输入 batch 放到同一设备即可。
    """

    model: SimpleCNN
    config: SimpleCNNConfig
    checkpoint: dict[str, Any]
    checkpoint_path: Path
    run_dir: Path
    resolved_config_path: Path
    device: torch.device
    runtime_settings: RuntimeSettings
    npu_checkpoint_shim_installed: bool


def install_torch_npu_checkpoint_shim() -> bool:
    """让没有 torch_npu 的 CPU/CUDA 环境读取 Ascend 保存的 checkpoint。

    真正安装了 torch_npu 的 Ascend 环境保持原生反序列化。否则注册一个仅在
    torch.load 反序列化阶段使用的最小替身，忽略 Ascend storage format；
    map_location 会把权重放到当前推理设备。
    """

    try:
        import torch_npu  # noqa: F401 - 原生 Ascend 环境不应替换真实模块。
    except ModuleNotFoundError:
        torch_npu = None  # type: ignore[assignment]
    else:
        return bool(getattr(torch_npu, "__linetracker_checkpoint_shim__", False))

    torch_npu_module = types.ModuleType("torch_npu")
    utils_module = types.ModuleType("torch_npu.utils")
    storage_module = types.ModuleType("torch_npu.utils.storage")
    npu_module = types.ModuleType("torch_npu.npu")
    format_module = types.ModuleType("torch_npu.npu._format")

    class Format:
        """仅承接 checkpoint 中的 Ascend 格式标记。"""

        def __init__(self, value: int) -> None:
            self.value = value

    def rebuild_npu_tensor(
        storage: Any,
        offset: int,
        size: Any,
        stride: Any,
        requires_grad: bool,
        hooks: Any,
        npu_format: Any,
    ) -> torch.Tensor:
        """以普通 PyTorch storage 重建张量，格式标记由 CPU/CUDA 忽略。"""

        del npu_format
        return torch._utils._rebuild_tensor_v2(
            storage, offset, size, stride, requires_grad, hooks
        )

    format_module.Format = Format
    storage_module._rebuild_npu_tensor = rebuild_npu_tensor
    torch_npu_module.utils = utils_module
    torch_npu_module.npu = npu_module
    torch_npu_module.__linetracker_checkpoint_shim__ = True
    utils_module.storage = storage_module
    npu_module._format = format_module
    sys.modules.update(
        {
            "torch_npu": torch_npu_module,
            "torch_npu.utils": utils_module,
            "torch_npu.utils.storage": storage_module,
            "torch_npu.npu": npu_module,
            "torch_npu.npu._format": format_module,
        }
    )
    return True


def resolve_run_dir(checkpoint_path: Path | str) -> Path:
    """从 <run>/checkpoints/<name>.pt 精确定位训练 run 目录。"""

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 文件不存在：{path}")
    path = path.resolve()
    if path.parent.name != "checkpoints":
        raise ValueError(
            "checkpoint 必须直接位于 <run>/checkpoints/ 目录下，"
            f"实际路径为：{path}"
        )
    if path.suffix != ".pt":
        raise ValueError(f"checkpoint 应为 .pt 文件，实际为：{path.name}")
    run_dir = path.parent.parent
    if not run_dir.is_dir():  # pragma: no cover - parent 存在时通常必然为目录。
        raise FileNotFoundError(f"无法从 checkpoint 定位 run 目录：{run_dir}")
    return run_dir


def load_resolved_config(run_dir: Path | str) -> SimpleCNNConfig:
    """读取 run 内保存的配置快照，不回退到当前 profile 默认值。"""

    resolved_path = Path(run_dir).expanduser().resolve() / "resolved_config.json"
    if not resolved_path.is_file():
        raise FileNotFoundError(
            "checkpoint 所属 run 缺少 resolved_config.json，无法安全恢复模型结构："
            f"{resolved_path}"
        )
    try:
        return load_config_json(resolved_path)
    except Exception as error:
        raise ValueError(f"无法读取 resolved_config.json：{resolved_path}") from error


def _npu_is_available() -> bool:
    """延迟导入 Ascend 依赖，避免 CUDA/CPU 环境强制安装 torch_npu。"""

    try:
        import torch_npu  # noqa: F401 - 导入会向 torch 注册 npu 后端。
    except ImportError:
        return False
    return hasattr(torch, "npu") and bool(torch.npu.is_available())


def _set_cuda_device(request: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(f"请求设备 {request!r}，但当前 PyTorch 未检测到 CUDA。")
    device = torch.device(request)
    if device.index is not None and not 0 <= device.index < torch.cuda.device_count():
        raise ValueError(
            f"请求的 CUDA 设备索引 {device.index} 越界；当前可见设备数为 "
            f"{torch.cuda.device_count()}。"
        )
    torch.cuda.set_device(device)
    return device


def _set_npu_device(request: str) -> torch.device:
    if not _npu_is_available():
        raise RuntimeError(
            f"请求设备 {request!r}，但当前环境未检测到可用 Ascend NPU。"
            "请确认 torch、torch_npu 与 CANN 版本匹配。"
        )
    import torch_npu  # noqa: F401 - 已由上面的可用性检查注册 torch.npu。

    device = torch.device(request)
    device_count = int(torch.npu.device_count())
    if device.index is not None and not 0 <= device.index < device_count:
        raise ValueError(
            f"请求的 NPU 设备索引 {device.index} 越界；当前可见设备数为 {device_count}。"
        )
    if hasattr(torch.npu, "config"):
        torch.npu.config.allow_internal_format = True
    torch.npu.set_device(device)
    return device


def resolve_inference_device(
    device: str | torch.device | None = "auto",
    *,
    runtime_settings: RuntimeSettings | None = None,
) -> torch.device:
    """按显式设备或 LT_ACCELERATOR 解析一张本地推理设备。

    显式 device 优先于 .env。auto 时保持训练侧相同的优先级：
    CUDA、Ascend NPU、CPU。此函数不创建进程组，也不读取 LOCAL_RANK。
    """

    settings = runtime_settings or RuntimeSettings()
    requested = settings.accelerator if device is None or str(device).lower() == "auto" else str(device)
    requested = requested.lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return _set_cuda_device("cuda")
        if _npu_is_available():
            return _set_npu_device("npu")
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" or requested.startswith("cuda:"):
        return _set_cuda_device(requested)
    if requested == "npu" or requested.startswith("npu:"):
        return _set_npu_device(requested)
    raise ValueError(
        "device 仅支持 auto、cpu、cuda[:index] 或 npu[:index]，"
        f"实际为 {device!r}。"
    )


def infer_model_type(checkpoint: Mapping[str, Any]) -> tuple[str, dict[str, torch.Tensor]]:
    """从 checkpoint 的首层通道识别 xn/n/s，并规范化可选 DDP 前缀。"""

    raw_state_dict = checkpoint.get("model_state")
    if not isinstance(raw_state_dict, Mapping):
        raise KeyError("checkpoint 缺少字典类型的 model_state。")
    state_dict = dict(raw_state_dict)
    if state_dict and all(str(name).startswith("module.") for name in state_dict):
        state_dict = {
            str(name).removeprefix("module."): value
            for name, value in state_dict.items()
        }
    first_layer = state_dict.get("features.0.0.weight")
    if not isinstance(first_layer, torch.Tensor):
        raise KeyError("checkpoint 缺少 features.0.0.weight，无法识别模型规格。")
    model_type_by_channels = {8: "xn", 16: "n", 24: "s"}
    channels = int(first_layer.shape[0])
    try:
        model_type = model_type_by_channels[channels]
    except KeyError as error:
        raise RuntimeError(f"不支持的首层输出通道数：{channels}。") from error
    return model_type, state_dict


def _normalize_data_root(data_root: Path | str) -> Path:
    return Path(data_root).expanduser().resolve()


def load_inference_bundle(
    checkpoint_path: Path | str,
    *,
    data_root: Path | str | None = None,
    device: str | torch.device | None = "auto",
    env_file: Path | str | None = None,
) -> InferenceBundle:
    """从训练 run 恢复单进程推理模型。

    resolved_config.json 决定网络结构和分块协议；运行时设置仅覆盖本机
    data_root、输出根、AMP 和 pin_memory。显式 data_root 优先级最高。
    为防止错配权重，该函数会验证 checkpoint 推断出的模型规格与配置一致。
    """

    raw_checkpoint_path = Path(checkpoint_path).expanduser()
    run_dir = resolve_run_dir(raw_checkpoint_path)
    checkpoint_path = raw_checkpoint_path.resolve()
    resolved_config_path = run_dir / "resolved_config.json"
    config = load_resolved_config(run_dir)

    normalized_env_file = None if env_file is None else Path(env_file)
    runtime_settings = load_runtime_settings(normalized_env_file)
    config = apply_runtime_settings(config, runtime_settings)
    if data_root is not None:
        config.data_root = _normalize_data_root(data_root)

    resolved_device = resolve_inference_device(device, runtime_settings=runtime_settings)
    config = apply_device_defaults(config, runtime_settings, resolved_device.type)
    config.validate()

    shim_installed = install_torch_npu_checkpoint_shim()
    try:
        checkpoint = load_checkpoint(checkpoint_path, resolved_device)
    except Exception as error:
        raise RuntimeError(f"无法加载 checkpoint：{checkpoint_path}") from error
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint 顶层应为 dict，实际为 {type(checkpoint).__name__}。")

    inferred_model_type, state_dict = infer_model_type(checkpoint)
    if config.model_type != inferred_model_type:
        raise RuntimeError(
            "checkpoint 权重与所属 resolved_config.json 的模型规格不一致："
            f"checkpoint={inferred_model_type!r}，config={config.model_type!r}。"
        )

    model = SimpleCNN(config).to(resolved_device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "checkpoint 参数形状与 resolved_config.json 不兼容；"
            "请确认 checkpoint 没有被放入错误的 run 目录。"
        ) from error
    model.eval()

    return InferenceBundle(
        model=model,
        config=config,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        resolved_config_path=resolved_config_path,
        device=resolved_device,
        runtime_settings=runtime_settings,
        npu_checkpoint_shim_installed=shim_installed,
    )
