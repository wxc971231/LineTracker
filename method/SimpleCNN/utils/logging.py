"""rank 0 专用的 Weights & Biases 日志封装。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from configs.base import SimpleCNNConfig
from utils.distributed import DistributedContext
from utils.run_naming import build_wandb_run_name


_WANDB_PROXY_ENV_NAMES = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)


def _configure_direct_wandb_transport() -> None:
    """让 W&B 服务直连，不继承可能随 SSH 断开的本地代理。"""
    for name in _WANDB_PROXY_ENV_NAMES:
        os.environ.pop(name, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    os.environ.pop("WANDB__PROXIES", None)


@dataclass
class WandbLogger:


    """避免非主进程导入或写入 W&B 的轻量封装。"""

    run: Any | None = None

    @classmethod
    def create(
        cls,
        config: SimpleCNNConfig,
        context: DistributedContext,
        run_dir: Path,
        resume_id: str | None = None,
    ) -> "WandbLogger":
        """仅在 rank 0 且 wandb_mode 非 disabled 时初始化 W&B。"""
        if not context.is_main or config.wandb_mode == "disabled":
            return cls()

        try:
            import wandb
        except ImportError as error:  # pragma: no cover - 依赖错误由运行时明确报告
            raise RuntimeError("已启用 W&B，但当前环境未安装 wandb。") from error
        _configure_direct_wandb_transport()
        wandb_settings = wandb.Settings(
            http_proxy=None,
            https_proxy=None,
            x_proxies={},
        )

        wandb_dir = run_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=build_wandb_run_name(config, run_dir),
            dir=str(wandb_dir),
            config=config.as_serializable_dict(),
            mode=config.wandb_mode,
            settings=wandb_settings,
            id=resume_id,
            resume="allow" if resume_id else None,
        )
        return cls(run=run)

    @property
    def run_id(self) -> str | None:
        """返回可写入 checkpoint 的 W&B run id。"""
        return None if self.run is None else str(self.run.id)

    def log(self, values: dict[str, float | int], step: int) -> None:
        """写入同一 optimizer step 的标量。"""
        if self.run is not None:
            self.run.log(values, step=step)

    def finish(self) -> None:
        """结束主进程的 W&B run。"""
        if self.run is not None:
            self.run.finish()
