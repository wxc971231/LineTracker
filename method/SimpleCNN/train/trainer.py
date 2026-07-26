"""按 optimizer step 训练 SimpleCNN，并支持 AMP、DDP、W&B 与断点恢复。"""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from configs.base import SimpleCNNConfig
from eval.evaluator import autocast_context, evaluate_model, move_batch_to_device
from eval.metrics import empty_metric_totals, metrics_from_totals
from train.losses import LossOutput, compute_losses
from train.model import SimpleCNN
from utils.checkpoint import atomic_torch_save, rotate_checkpoints
from utils.distributed import DistributedContext, barrier, rank_zero_print, reduce_sum
from utils.logging import WandbLogger


class Trainer:
    """SimpleCNN-v1 的单机单卡/多卡统一训练器。"""

    def __init__(
        self,
        config: SimpleCNNConfig,
        context: DistributedContext,
        run_dir: Path,
        train_loader: Iterator[dict[str, torch.Tensor]],
        validation_loader: Iterator[dict[str, torch.Tensor]],
        logger: WandbLogger,
        resume_payload: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.context = context
        self.run_dir = run_dir
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.logger = logger
        self.checkpoint_dir = run_dir / "checkpoints"
        self.global_step = 0
        self.best_validation_loss = float("inf")
        self.train_iterator = iter(train_loader)

        self.raw_model = SimpleCNN(config).to(context.device)
        if context.is_distributed:
            self.model: nn.Module = DistributedDataParallel(
                self.raw_model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            self.model = self.raw_model
        if config.compile_model:
            self.model = torch.compile(self.model)

        self.optimizer = torch.optim.AdamW(
            self.raw_model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
        self.amp_dtype, scaler_enabled = self._resolve_amp()
        self.scaler = self._create_scaler(scaler_enabled)
        self._set_learning_rate(self._learning_rate_for_step(0))

        if resume_payload is not None:
            self._restore(resume_payload)

    def _resolve_amp(self) -> tuple[torch.dtype | None, bool]:
        """根据配置和硬件能力选择 AMP 精度与 GradScaler 开关。"""
        if self.context.device.type != "cuda" or self.config.amp == "off":
            return None, False
        if self.config.amp == "fp16":
            return torch.float16, True
        if self.config.amp == "bf16":
            return torch.bfloat16, False
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False
        return torch.float16, True

    def _create_scaler(self, enabled: bool):
        """兼容不同 PyTorch 版本的 GradScaler 构造接口。"""
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError):  # pragma: no cover - 兼容旧版 PyTorch
            return torch.cuda.amp.GradScaler(enabled=enabled)

    def _autocast(self):
        """返回当前训练 step 的自动混合精度上下文。"""
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _learning_rate_for_step(self, step: int) -> float:
        """按 optimizer step 执行线性 warmup 后的余弦退火。"""
        total = self.config.total_optimizer_steps
        warmup = int(total * self.config.warmup_ratio)
        if warmup > 0 and step < warmup:
            return self.config.learning_rate * float(step + 1) / float(warmup)
        if total <= warmup:
            return self.config.learning_rate
        progress = min(1.0, max(0.0, (step - warmup) / float(total - warmup)))
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())
        ratio = self.config.min_learning_rate_ratio + (1.0 - self.config.min_learning_rate_ratio) * cosine
        return self.config.learning_rate * ratio

    def _set_learning_rate(self, learning_rate: float) -> None:
        """把当前 step 的学习率写入全部 optimizer 参数组。"""
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def _restore(self, payload: dict[str, Any]) -> None:
        """恢复模型、优化器、AMP scaler 和 step 计数；在线裁剪流从新随机位置继续。"""
        self.raw_model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if "scaler_state" in payload:
            self.scaler.load_state_dict(payload["scaler_state"])
        self.global_step = int(payload["global_step"])
        self.best_validation_loss = float(payload.get("best_validation_loss", float("inf")))
        self._set_learning_rate(self._learning_rate_for_step(self.global_step))
        rank_zero_print(self.context, f"从 step={self.global_step} 恢复训练。")

    def _snapshot_payload(self) -> dict[str, Any]:
        """组织可恢复训练所需的状态；数据流本身按源数据循环重新采样。"""
        return {
            "model_state": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "global_step": self.global_step,
            "best_validation_loss": self.best_validation_loss,
            "wandb_id": self.logger.run_id,
        }

    def _save_last_checkpoint(self) -> None:
        """由 rank 0 保存 last 与带 step 的历史 checkpoint。"""
        if not self.context.is_main:
            return
        payload = self._snapshot_payload()
        atomic_torch_save(payload, self.checkpoint_dir / "last.pt")
        atomic_torch_save(payload, self.checkpoint_dir / f"step_{self.global_step:07d}.pt")
        rotate_checkpoints(self.checkpoint_dir, self.config.keep_last_checkpoints)

    def _save_best_checkpoint(self, validation_loss: float) -> None:
        """验证总损失改善时更新 best.pt。"""
        if validation_loss >= self.best_validation_loss:
            return
        self.best_validation_loss = validation_loss
        if self.context.is_main:
            atomic_torch_save(self._snapshot_payload(), self.checkpoint_dir / "best.pt")
            rank_zero_print(self.context, f"更新最佳 checkpoint：val_loss={validation_loss:.6g}")

    @staticmethod
    def _loss_totals(losses: LossOutput, prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """转为与 eval.metrics 相同顺序的 float64 累加统计量。"""
        q_error = prediction["q"].float() - batch["q"].float()
        device = q_error.device
        return torch.stack(
            [
                losses.q_loss_sum.detach().double(),
                losses.q_count.detach().double(),
                losses.line_loss_sum.detach().double(),
                losses.line_count.detach().double(),
                losses.abs_error_sum.detach().double(),
                losses.squared_error_sum.detach().double(),
                losses.point_count.detach().double(),
                q_error.abs().sum().detach().double(),
                q_error.square().sum().detach().double(),
                torch.tensor(float(batch["x"].shape[0]), device=device, dtype=torch.float64),
                batch["is_positive"].sum().detach().double(),
            ]
        )

    def _next_train_batch(self) -> dict[str, torch.Tensor]:
        """无限数据流不会耗尽；此方法保留重建迭代器的兼容保护。"""
        try:
            return next(self.train_iterator)
        except StopIteration:  # pragma: no cover - 防御性分支
            self.train_iterator = iter(self.train_loader)
            return next(self.train_iterator)

    def _run_optimizer_step(self) -> tuple[torch.Tensor, float, float]:
        """执行一次包含若干 micro-batch 的 optimizer 更新。"""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = empty_metric_totals(self.context.device)
        start_time = time.perf_counter()
        accumulation = self.config.gradient_accumulation_steps

        for micro_step in range(accumulation):
            batch = move_batch_to_device(self._next_train_batch(), self.context.device)
            no_sync = (
                self.model.no_sync()
                if isinstance(self.model, DistributedDataParallel) and micro_step < accumulation - 1
                else nullcontext()
            )
            with no_sync:
                with self._autocast():
                    prediction = self.model(batch["x"])
                    losses = compute_losses(prediction, batch, self.config)
                    scaled_loss = losses.total / accumulation
                self.scaler.scale(scaled_loss).backward()
            totals += self._loss_totals(losses, prediction, batch)

        self.scaler.unscale_(self.optimizer)
        if self.config.grad_clip_norm > 0:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                self.raw_model.parameters(), self.config.grad_clip_norm
            )
            grad_norm = float(grad_norm_tensor.item())
        else:
            grad_norm = 0.0
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.global_step += 1
        learning_rate = self._learning_rate_for_step(self.global_step)
        self._set_learning_rate(learning_rate)
        elapsed = time.perf_counter() - start_time
        return totals, grad_norm, elapsed

    def _log_train_totals(self, totals: torch.Tensor, grad_norm: float, elapsed: float) -> None:
        """跨卡合并一个日志周期的训练统计，并由 rank 0 写入 W&B。"""
        totals = reduce_sum(totals, self.context)
        metrics = metrics_from_totals(totals, self.config, "train")
        global_samples = self.config.batch_size_per_gpu * self.context.world_size * self.config.gradient_accumulation_steps
        metrics.update(
            {
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                "train/grad_norm": grad_norm,
                "train/global_batch_size": global_samples,
                "train/steps_per_second": 1.0 / max(elapsed, 1e-12),
            }
        )
        if self.context.is_main:
            self.logger.log(metrics, step=self.global_step)
            rank_zero_print(
                self.context,
                f"step={self.global_step:7d} "
                f"loss={metrics['train/loss_total']:.5f} "
                f"q={metrics['train/loss_q']:.5f} "
                f"line={metrics['train/loss_line']:.5f} "
                f"lr={metrics['train/learning_rate']:.3e}",
            )

    def _validate(self) -> dict[str, float]:
        """在固定标准网格上评估，并更新 best checkpoint。"""
        metrics = evaluate_model(self.model, self.validation_loader, self.config, self.context, prefix="val")
        self._save_best_checkpoint(metrics["val/loss_total"])
        if self.context.is_main:
            self.logger.log(metrics, step=self.global_step)
            rank_zero_print(
                self.context,
                f"validation step={self.global_step} "
                f"loss={metrics['val/loss_total']:.5f} "
                f"point_mae={metrics['val/point_mae_bin']:.3f} bin",
            )
        return metrics

    def train(self) -> None:
        """按 ``total_optimizer_steps`` 训练，并周期性日志、验证和保存。"""
        log_totals = empty_metric_totals(self.context.device)
        log_elapsed = 0.0
        latest_grad_norm = 0.0
        while self.global_step < self.config.total_optimizer_steps:
            totals, latest_grad_norm, elapsed = self._run_optimizer_step()
            log_totals += totals
            log_elapsed += elapsed

            should_log = (
                self.global_step % self.config.log_interval_steps == 0
                or self.global_step == self.config.total_optimizer_steps
            )
            if should_log:
                self._log_train_totals(log_totals, latest_grad_norm, log_elapsed)
                log_totals.zero_()
                log_elapsed = 0.0

            should_evaluate = (
                self.global_step % self.config.eval_interval_steps == 0
                or self.global_step == self.config.total_optimizer_steps
            )
            if should_evaluate:
                self._validate()

            should_save = (
                self.global_step % self.config.checkpoint_interval_steps == 0
                or self.global_step == self.config.total_optimizer_steps
            )
            if should_save:
                self._save_last_checkpoint()
            barrier(self.context)
