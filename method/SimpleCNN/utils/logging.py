"""rank 0 专用的 Weights & Biases 日志封装。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from configs.base import SimpleCNNConfig
from utils.distributed import DistributedContext
from utils.run_naming import build_wandb_run_name


@dataclass
class WandbLogger:
    """避免非主进程导入或写入 W&B 的轻量封装。"""

    run: Any | None = None

    @property
    def enabled(self) -> bool:
        """保持既有接口；是否启用直接由 run 是否存在推导。"""
        return self.run is not None

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

        wandb_dir = run_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=build_wandb_run_name(config, run_dir),
            dir=str(wandb_dir),
            config=config.as_serializable_dict(),
            mode=config.wandb_mode,
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
