"""文档第 4 章定义的经典二维 SimpleCNN。"""

from __future__ import annotations

import torch
from torch import nn

from configs.base import SimpleCNNConfig


class SimpleCNN(nn.Module):
    """从 ``[B,8,20,1250]`` 原始二值重排张量回归 $(q,\rho,\nu)$。"""

    def __init__(self, config: SimpleCNNConfig) -> None:
        super().__init__()
        self.block_width_m = float(config.block_width_m)
        self.max_speed_per_frame_m = float(config.max_speed_per_frame_m)

        def block(in_channels: int, out_channels: int, kernel: tuple[int, int], stride: tuple[int, int], padding: tuple[int, int]) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        # 以下特征尺寸严格对应 _doc/SimpleCNN.md：
        # [8,20,1250] → [16,20,625] → [32,20,313] → [64,20,157]
        # → [64,10,79] → [96,5,40] → [96,1,40]。
        self.features = nn.Sequential(
            block(config.input_channels, 16, (3, 5), (1, 2), (1, 2)),
            block(16, 32, (3, 5), (1, 2), (1, 2)),
            block(32, 64, (3, 5), (1, 2), (1, 2)),
            block(64, 64, (3, 3), (2, 2), (1, 1)),
            block(64, 96, (3, 3), (2, 2), (1, 1)),
            block(96, 96, (5, 3), (1, 1), (0, 1)),
        )
        self.flatten = nn.Flatten()
        self.shared = nn.Sequential(
            nn.Linear(96 * 1 * 40, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )
        self.q_head = nn.Linear(config.hidden_dim, 1)
        self.rho_head = nn.Linear(config.hidden_dim, 1)
        self.nu_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """输出 q logit、q 概率、中心距离和斜率。

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
