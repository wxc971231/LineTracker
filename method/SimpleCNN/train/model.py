"""文档第 4 章定义的二维 SimpleCNN，提供 xn/n/s 三种容量规格。"""

from __future__ import annotations

import torch
from torch import nn

from configs.base import SimpleCNNConfig


_FEATURE_CHANNELS_BY_MODEL_TYPE: dict[str, tuple[int, ...]] = {
    # xn 是 n 的 0.5 倍宽度，用于快速迭代或显存受限场景。
    "xn": (8, 16, 32, 32, 48, 48),
    # n 完全复现原始 SimpleCNN-v1 通道数。
    "n": (16, 32, 64, 64, 96, 96),
    # s 是 n 的 1.5 倍宽度；时间/空间下采样结构保持不变。
    "s": (24, 48, 96, 96, 144, 144),
}
_HIDDEN_DIM_RATIO_BY_MODEL_TYPE: dict[str, tuple[int, int]] = {
    "xn": (1, 2),
    "n": (1, 1),
    "s": (3, 2),
}


def _scaled_hidden_dim(base_hidden_dim: int, model_type: str) -> int:
    """按容量规格放大共享 MLP；奇数基数向上取整，确保不会缩小容量。"""
    numerator, denominator = _HIDDEN_DIM_RATIO_BY_MODEL_TYPE[model_type]
    return (base_hidden_dim * numerator + denominator - 1) // denominator


class SimpleCNN(nn.Module):
    """从 ``[B,8,20,1250]`` 原始二值重排张量回归 $(q,\rho,\nu)$。

    ``model_type=xn`` 将 n 的通道和共享 MLP 宽度减半；``model_type=n``
    保持原始基线；``model_type=s`` 将二者放大 1.5 倍。三种规格均不改变
    输入、下采样路径或输出接口。
    """

    def __init__(self, config: SimpleCNNConfig) -> None:
        super().__init__()
        self.block_width_m = float(config.block_width_m)
        self.max_speed_per_frame_m = float(config.max_speed_per_frame_m)
        self.model_type = config.model_type
        self.feature_channels = _FEATURE_CHANNELS_BY_MODEL_TYPE[self.model_type]
        self.hidden_dim = _scaled_hidden_dim(config.hidden_dim, self.model_type)

        def block(
            in_channels: int,
            out_channels: int,
            kernel: tuple[int, int],
            stride: tuple[int, int],
            padding: tuple[int, int],
        ) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        # 以下空间尺寸严格对应 _doc/SimpleCNN.md；xn/n/s 仅改变通道数：
        # [8,20,1250] → [C1,20,625] → [C2,20,313] → [C3,20,157]
        # → [C4,10,79] → [C5,5,40] → [C6,1,40]。
        channels = self.feature_channels
        self.features = nn.Sequential(
            block(config.input_channels, channels[0], (3, 5), (1, 2), (1, 2)),
            block(channels[0], channels[1], (3, 5), (1, 2), (1, 2)),
            block(channels[1], channels[2], (3, 5), (1, 2), (1, 2)),
            block(channels[2], channels[3], (3, 3), (2, 2), (1, 1)),
            block(channels[3], channels[4], (3, 3), (2, 2), (1, 1)),
            block(channels[4], channels[5], (5, 3), (1, 1), (0, 1)),
        )
        self.flatten = nn.Flatten()
        self.shared = nn.Sequential(
            nn.Linear(channels[-1] * 40, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )
        self.q_head = nn.Linear(self.hidden_dim, 1)
        self.rho_head = nn.Linear(self.hidden_dim, 1)
        self.nu_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        r"""输出 q logit、q 概率、中心距离和斜率。

        ``rho_m`` 以块内 m/bins 数值表示；当前 $1\,\mathrm{m/bin}$ 下，
        它可直接与逐帧距离 bin 标签比较。
        """
        if x.ndim != 4 or x.shape[1:] != (8, 20, 1250):
            raise ValueError(f"SimpleCNN 期望 [B,8,20,1250]，实际输入为 {tuple(x.shape)}。")
        features = self.shared(self.flatten(self.features(x)))
        q_logit = self.q_head(features).squeeze(-1)
        rho_m = self.block_width_m * torch.sigmoid(self.rho_head(features).squeeze(-1))
        nu_mpf = self.max_speed_per_frame_m * torch.tanh(self.nu_head(features).squeeze(-1))
        return {
            "q_logit": q_logit,
            "q": torch.sigmoid(q_logit),
            "rho_m": rho_m,
            "nu_mpf": nu_mpf,
        }
