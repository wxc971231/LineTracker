"""按 optimizer step 训练 SimpleCNN，并支持 AMP、DDP、W&B 与断点恢复。"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from configs.base import SimpleCNNConfig
from eval.evaluator import evaluate_model, move_batch_to_device
from eval.metrics import empty_metric_totals, metric_accumulator_dtype, metrics_from_totals
from train.losses import LossOutput, compute_losses
from train.model import SimpleCNN
from utils.checkpoint import atomic_torch_save, rotate_checkpoints
from utils.distributed import DistributedContext, broadcast_module_buffers, rank_zero_print, reduce_sum
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
        data_stream_generation: int = 0,
        validation_dataset_id: str = "unknown",
    ) -> None:
        self.config = config
        if data_stream_generation < 0:
            raise ValueError("data_stream_generation 不得为负。")
        self.context = context
        self.run_dir = run_dir
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.data_stream_generation = int(data_stream_generation)
        self.validation_dataset_id = str(validation_dataset_id)
        self.logger = logger
        self.checkpoint_dir = run_dir / "checkpoints"
        self.global_step = 0
        self.best_validation_loss = float("inf")
        self.best_validation_metric_kind = self._validation_metric_kind()
        self.requires_resume_validation_baseline = False
        self.train_iterator = iter(train_loader)

        self.raw_model = SimpleCNN(config).to(context.device)
        self.ddp_model: DistributedDataParallel | None = None
        if context.is_distributed:
            ddp_kwargs: dict[str, Any] = {
                # 模型包含 BatchNorm；每次 forward 前从 rank 0 同步 running
                # statistics，确保分布式验证使用一致的模型状态。
                "broadcast_buffers": True,
                "find_unused_parameters": False,
            }
            if context.device.type != "cpu":
                ddp_kwargs.update(
                    {
                        "device_ids": [context.local_rank],
                        "output_device": context.local_rank,
                    }
                )
            self.ddp_model = DistributedDataParallel(self.raw_model, **ddp_kwargs)
            self.model: nn.Module = self.ddp_model
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

    def _validation_metric_kind(self) -> str:
        """标识用于选择 best checkpoint 的可比较验证协议。"""
        if self.config.max_eval_batch_num == 0:
            return f"full_grid_v2:data={self.validation_dataset_id}"
        return (
            f"resampled_replacement_v1:data={self.validation_dataset_id}:seed={self.config.seed}:"
            f"batches={self.config.max_eval_batch_num}:batch={self.config.eval_batch_size_per_gpu}:"
            f"world={self.context.world_size}"
        )

    def _resolve_amp(self) -> tuple[torch.dtype | None, bool]:
        """根据配置和硬件能力选择 AMP 精度与 GradScaler 开关。"""
        device_type = self.context.device.type
        if device_type == "cpu" or self.config.amp == "off":
            return None, False
        if self.config.amp == "fp16":
            return torch.float16, True
        if self.config.amp == "bf16":
            return torch.bfloat16, False
        if device_type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16, False
        if device_type == "npu":
            try:
                if torch.npu.is_bf16_supported():
                    return torch.bfloat16, False
            except (AttributeError, RuntimeError):
                pass
        # 无 BF16 能力的 CUDA/NPU 后端回退到 FP16，并启用 GradScaler。
        return torch.float16, True

    def _create_scaler(self, enabled: bool):
        """兼容不同 PyTorch 版本的 GradScaler 构造接口。"""
        try:
            return torch.amp.GradScaler(self.context.device.type, enabled=enabled)
        except (AttributeError, TypeError):  # pragma: no cover - 兼容旧版 PyTorch
            if self.context.device.type == "npu":
                import torch_npu

                return torch_npu.npu.amp.GradScaler(enabled=enabled)
            return torch.cuda.amp.GradScaler(enabled=enabled)

    def _autocast(self):
        """返回当前训练 step 的自动混合精度上下文。"""
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.context.device.type, dtype=self.amp_dtype)

    def _learning_rate_for_step(self, step: int) -> float:
        """按 optimizer step 执行线性 warmup 后的余弦退火。"""
        total = self.config.total_optimizer_steps
        warmup = int(total * self.config.warmup_ratio)
        if warmup > 0 and step < warmup:
            return self.config.learning_rate * float(step + 1) / float(warmup)
        if total <= warmup:
            return self.config.learning_rate
        progress = min(1.0, max(0.0, (step - warmup) / float(total - warmup)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        ratio = self.config.min_learning_rate_ratio + (1.0 - self.config.min_learning_rate_ratio) * cosine
        return self.config.learning_rate * ratio

    def _set_learning_rate(self, learning_rate: float) -> None:
        """把当前 step 的学习率写入全部 optimizer 参数组。"""
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def _apply_configured_optimizer_hyperparameters(self) -> None:
        """恢复 state 后重新应用 resolved config，避免参数组继续使用旧实验设置。"""
        for group in self.optimizer.param_groups:
            group["betas"] = (self.config.adam_beta1, self.config.adam_beta2)
            group["eps"] = self.config.adam_eps
            group["weight_decay"] = self.config.weight_decay

    def _restore(self, payload: dict[str, Any]) -> None:
        """恢复模型、优化器、AMP scaler 和 step 计数；在线裁剪流从新随机位置继续。"""
        self.raw_model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self._apply_configured_optimizer_hyperparameters()
        if "scaler_state" in payload:
            self.scaler.load_state_dict(payload["scaler_state"])
        self.global_step = int(payload["global_step"])
        saved_metric_kind = payload.get("best_validation_metric_kind")
        if saved_metric_kind == self.best_validation_metric_kind:
            restored_best = float(payload.get("best_validation_loss", float("inf")))
            self.best_validation_loss = restored_best if math.isfinite(restored_best) else float("inf")
        else:
            self.best_validation_loss = float("inf")
            self.requires_resume_validation_baseline = True
            rank_zero_print(
                self.context,
                "checkpoint 的验证集或 best 验证协议已变化；"
                "将在继续训练前用当前 checkpoint 重新建立验证基线。",
            )
        self._set_learning_rate(self._learning_rate_for_step(self.global_step))
        rank_zero_print(
            self.context,
            f"从 step={self.global_step} 恢复训练，数据流代次={self.data_stream_generation}。",
        )

    def _snapshot_payload(self) -> dict[str, Any]:
        """组织可恢复训练所需状态，并记录下一次恢复所需的数据流代次。"""
        return {
            "model_state": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "global_step": self.global_step,
            "best_validation_loss": self.best_validation_loss,
            "best_validation_metric_kind": self.best_validation_metric_kind,
            "data_stream_generation": self.data_stream_generation,
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
        if not math.isfinite(validation_loss):
            rank_zero_print(self.context, f"跳过非有限验证损失：{validation_loss!r}")
            return
        if validation_loss >= self.best_validation_loss:
            return
        self.best_validation_loss = validation_loss
        if self.context.is_main:
            atomic_torch_save(self._snapshot_payload(), self.checkpoint_dir / "best.pt")
            rank_zero_print(self.context, f"更新最佳 checkpoint：val_loss={validation_loss:.6g}")

    @staticmethod
    def _loss_totals(
        losses: LossOutput,
        prediction: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """转为与 eval.metrics 相同顺序、兼容当前后端的累加统计量。"""
        q_error = prediction["q"].float() - batch["q"].float()
        device = q_error.device
        metric_dtype = metric_accumulator_dtype(device)
        return torch.stack(
            [
                losses.q_loss_sum.detach().to(dtype=metric_dtype),
                losses.q_count.detach().to(dtype=metric_dtype),
                losses.line_loss_sum.detach().to(dtype=metric_dtype),
                losses.line_count.detach().to(dtype=metric_dtype),
                losses.abs_error_sum.detach().to(dtype=metric_dtype),
                losses.squared_error_sum.detach().to(dtype=metric_dtype),
                losses.point_count.detach().to(dtype=metric_dtype),
                q_error.abs().sum().detach().to(dtype=metric_dtype),
                q_error.square().sum().detach().to(dtype=metric_dtype),
                torch.tensor(float(batch["x"].shape[0]), device=device, dtype=metric_dtype),
                batch["is_positive"].sum().detach().to(dtype=metric_dtype),
            ]
        )

    def _next_train_batch(self) -> dict[str, torch.Tensor]:
        """无限数据流不会耗尽；此方法保留重建迭代器的兼容保护。"""
        try:
            return next(self.train_iterator)
        except StopIteration:  # pragma: no cover - 防御性分支
            self.train_iterator = iter(self.train_loader)
            return next(self.train_iterator)

    def _run_optimizer_step(self) -> tuple[torch.Tensor, torch.Tensor, float, bool]:
        """执行一次包含若干 micro-batch 的 optimizer 更新。"""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = empty_metric_totals(self.context.device)
        start_time = time.perf_counter()
        accumulation = self.config.gradient_accumulation_steps

        for micro_step in range(accumulation):
            batch = move_batch_to_device(self._next_train_batch(), self.context.device)
            no_sync = (
                self.ddp_model.no_sync()
                if self.ddp_model is not None and micro_step < accumulation - 1
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
            grad_norm = grad_norm_tensor.detach()
        else:
            grad_norm = torch.zeros((), device=self.context.device)
        scale_before = float(self.scaler.get_scale())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after = float(self.scaler.get_scale())
        optimizer_updated = scale_after >= scale_before
        elapsed = time.perf_counter() - start_time
        if not optimizer_updated:
            rank_zero_print(
                self.context,
                "GradScaler 检测到非有限梯度，本次 optimizer 更新已跳过；"
                f"loss scale {scale_before:g} -> {scale_after:g}，global_step 保持不变。",
            )
            return totals, grad_norm, elapsed, False

        self.global_step += 1
        learning_rate = self._learning_rate_for_step(self.global_step)
        self._set_learning_rate(learning_rate)
        return totals, grad_norm, elapsed, True

    def _log_train_totals(
        self,
        totals: torch.Tensor,
        grad_norm: torch.Tensor | float,
        elapsed: float,
        step_count: int,
    ) -> None:
        """跨卡合并一个日志周期的训练统计，并由 rank 0 写入 W&B。"""
        totals = reduce_sum(totals, self.context)
        metrics = metrics_from_totals(totals, self.config, "train")
        grad_norm_value = (
            float(grad_norm.detach().item())
            if isinstance(grad_norm, torch.Tensor)
            else float(grad_norm)
        )
        global_samples = (
            self.config.batch_size_per_gpu
            * self.context.world_size
            * self.config.gradient_accumulation_steps
        )
        metrics.update(
            {
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                "train/grad_norm": grad_norm_value,
                "train/global_batch_size": global_samples,
                "train/steps_per_second": step_count / max(elapsed, 1e-12),
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
        """评估唯一的验证数据流，并用同一组指标选择 best checkpoint。"""
        broadcast_module_buffers(self.raw_model, self.context)
        metrics = evaluate_model(
            self.raw_model,
            self.validation_loader,
            self.config,
            self.context,
            prefix="val",
        )
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
        log_step_count = 0
        latest_grad_norm = torch.zeros((), device=self.context.device)
        if self.requires_resume_validation_baseline:
            rank_zero_print(
                self.context,
                f"在更新模型前评估 step={self.global_step} 的 checkpoint，"
                "作为当前验证集的新 best 基线。",
            )
            self._validate()
            self.requires_resume_validation_baseline = False

        while self.global_step < self.config.total_optimizer_steps:
            totals, latest_grad_norm, elapsed, optimizer_updated = self._run_optimizer_step()
            if not optimizer_updated:
                continue
            log_totals += totals
            log_elapsed += elapsed
            log_step_count += 1

            should_log = (
                self.global_step % self.config.log_interval_steps == 0
                or self.global_step == self.config.total_optimizer_steps
            )
            if should_log:
                self._log_train_totals(
                    log_totals,
                    latest_grad_norm,
                    log_elapsed,
                    log_step_count,
                )
                log_totals.zero_()
                log_elapsed = 0.0
                log_step_count = 0

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
