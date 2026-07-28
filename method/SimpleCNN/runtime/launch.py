"""按当前可见 CUDA 或 Ascend NPU 数量启动单节点 torchrun。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from runtime.settings import load_runtime_settings
from utils.process_title import ensure_process_job_id, set_process_title


def _visible_accelerator_count(requested: str) -> int:
    """返回当前后端可见设备数；可见设备变量由运行环境先行过滤。"""
    import torch

    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.cuda.device_count()
    if requested in {"auto", "npu"}:
        try:
            import torch_npu  # noqa: F401 - 注册 torch.npu 后端。
        except ImportError:
            if requested == "npu":
                raise RuntimeError("LT_ACCELERATOR=npu，但未安装 torch_npu。") from None
        else:
            if hasattr(torch, "npu") and torch.npu.is_available():
                return torch.npu.device_count()
    if requested == "cuda":
        raise RuntimeError("LT_ACCELERATOR=cuda，但未检测到可用 CUDA 设备。")
    if requested == "npu":
        raise RuntimeError("LT_ACCELERATOR=npu，但未检测到可用 Ascend NPU。")
    raise RuntimeError("未检测到 CUDA 或 Ascend NPU，不能启动多卡训练。")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析训练脚本及其透传参数。"""
    parser = argparse.ArgumentParser(description="SimpleCNN 可见设备数自适应 torchrun 启动器")
    parser.add_argument("training_script", help="传给 torchrun 的训练脚本路径。")
    parser.add_argument("training_args", nargs=argparse.REMAINDER, help="训练脚本参数。")
    return parser.parse_args(argv)


def _option_value(arguments: Sequence[str], option: str) -> str | None:
    """同时识别 ``--option value`` 和 ``--option=value``。"""
    for index, argument in enumerate(arguments):
        if argument == option and index + 1 < len(arguments):
            return arguments[index + 1]
        prefix = f"{option}="
        if argument.startswith(prefix):
            return argument[len(prefix):]
    return None


def main(argv: Sequence[str] | None = None) -> None:
    """按 CUDA_VISIBLE_DEVICES 或 ASCEND_RT_VISIBLE_DEVICES 派生 nproc。"""
    args = parse_args(argv)
    script_stem = Path(args.training_script).stem
    mode = "eval" if "eval" in script_stem else "train"
    label = _option_value(args.training_args, "--config") or mode
    job_id = ensure_process_job_id()
    set_process_title(
        f"{mode}-launch",
        label=label,
        job_id=job_id,
        infer_rank=False,
    )
    settings = load_runtime_settings()
    nproc_per_node = _visible_accelerator_count(settings.accelerator)
    if nproc_per_node < 1:  # pragma: no cover - 后端可用时应至少暴露一张卡。
        raise RuntimeError("当前设备后端未暴露任何可用于训练的设备。")

    from torch.distributed.run import main as torchrun_main

    torchrun_main(
        [
            "--standalone",
            f"--nproc-per-node={nproc_per_node}",
            args.training_script,
            *args.training_args,
        ]
    )


if __name__ == "__main__":
    main()
