"""SimpleCNN 推理计算量的可复现静态估计。"""

from __future__ import annotations

from dataclasses import dataclass
import math

from torch import nn


@dataclass(frozen=True)
class ModelComplexity:
    """按单个局部块统计的模型规模；MACs 仅覆盖 Conv2d 与 Linear。"""

    parameter_count: int
    conv_linear_macs_per_block: int

    @property
    def conv_linear_flops_per_block(self) -> int:
        """采用 1 MAC = 2 FLOPs 的常用口径。"""
        return self.conv_linear_macs_per_block * 2


def _conv_output_size(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    return math.floor((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


def estimate_model_complexity(
    model: nn.Module,
    *,
    input_channels: int = 8,
    input_height: int = 20,
    input_width: int = 1250,
) -> ModelComplexity:
    """基于实际模块超参数估算单块 Conv2d/Linear MACs，不依赖设备 profiler。"""
    channels = int(input_channels)
    height = int(input_height)
    width = int(input_width)
    macs = 0

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            assert module.in_channels == channels, (
                "卷积层通道与推导输入不一致；当前复杂度估算仅支持顺序 SimpleCNN 特征提取器。"
            )
            kernel_h, kernel_w = module.kernel_size
            stride_h, stride_w = module.stride
            # ``Conv2d.padding`` 的类型也允许字符串（例如 "same"），而本模型
            # 的静态 MAC 推导只支持可解析为整数的显式零填充。
            if isinstance(module.padding, str):
                raise ValueError(f"不支持字符串卷积 padding：{module.padding!r}。")
            if isinstance(module.padding, int):
                padding_h = padding_w = module.padding
            else:
                padding_h, padding_w = module.padding
            dilation_h, dilation_w = module.dilation
            height = _conv_output_size(height, kernel_h, stride_h, padding_h, dilation_h)
            width = _conv_output_size(width, kernel_w, stride_w, padding_w, dilation_w)
            assert height >= 1 and width >= 1, "卷积输出空间尺寸无效。"
            macs += (
                height
                * width
                * module.out_channels
                * (module.in_channels // module.groups)
                * kernel_h
                * kernel_w
            )
            channels = module.out_channels
        elif isinstance(module, nn.Linear):
            macs += module.in_features * module.out_features

    return ModelComplexity(
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        conv_linear_macs_per_block=int(macs),
    )
