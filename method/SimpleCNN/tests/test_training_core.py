"""损失、吞吐统计和分布式 buffer 同步的 CPU 回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from configs.base import SimpleCNNConfig
from eval.metrics import METRIC_NAMES
from train.losses import compute_losses
from train.model import SimpleCNN
from train.trainer import Trainer, _configure_distributed_batch_norm
from utils.distributed import DistributedContext, broadcast_module_buffers


class LossTests(unittest.TestCase):
    """验证无分支 Huber 聚合的数值和梯度行为。"""

    def test_no_valid_line_block_keeps_zero_gradient_graph(self) -> None:
        config = SimpleCNNConfig()
        q_logit = torch.zeros(2, requires_grad=True)
        rho = torch.ones(2, requires_grad=True)
        nu = torch.ones(2, requires_grad=True)
        prediction = {"q_logit": q_logit, "rho_m": rho, "nu_mpf": nu}
        batch = {
            "q": torch.zeros(2),
            "q_valid": torch.ones(2, dtype=torch.bool),
            "I": torch.zeros((2, config.frames_per_window), dtype=torch.bool),
            "d": torch.full((2, config.frames_per_window), -1, dtype=torch.int16),
            "is_positive": torch.zeros(2, dtype=torch.bool),
        }

        losses = compute_losses(prediction, batch, config)
        losses.total.backward()

        self.assertEqual(losses.line_loss.item(), 0.0)
        self.assertIsNotNone(rho.grad)
        self.assertIsNotNone(nu.grad)
        self.assertTrue(torch.equal(rho.grad, torch.zeros_like(rho)))
        self.assertTrue(torch.equal(nu.grad, torch.zeros_like(nu)))


    def test_large_residuals_are_finite_for_all_model_dtypes(self) -> None:
        config = SimpleCNNConfig()
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                q_logit = torch.zeros(2, dtype=dtype, requires_grad=True)
                rho = torch.tensor([65_504.0, -65_504.0], dtype=dtype, requires_grad=True)
                nu = torch.zeros(2, dtype=dtype, requires_grad=True)
                prediction = {"q_logit": q_logit, "rho_m": rho, "nu_mpf": nu}
                batch = {
                    "q": torch.ones(2),
                    "q_valid": torch.ones(2, dtype=torch.bool),
                    "I": torch.ones((2, config.frames_per_window), dtype=torch.bool),
                    "d": torch.full((2, config.frames_per_window), 300_000, dtype=torch.int64),
                    "is_positive": torch.ones(2, dtype=torch.bool),
                }

                losses = compute_losses(prediction, batch, config)
                checked = (
                    losses.total,
                    losses.q_loss_sum,
                    losses.line_loss_sum,
                    losses.abs_error_sum,
                    losses.squared_error_sum,
                )
                for value in checked:
                    self.assertEqual(value.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(value).item())
                losses.total.backward()
                for tensor in (q_logit, rho, nu):
                    self.assertIsNotNone(tensor.grad)
                    self.assertTrue(torch.isfinite(tensor.grad).all().item())

    def test_q_loss_ignores_zero_evidence_sample(self) -> None:
        config = SimpleCNNConfig()
        q_logit = torch.tensor([10.0, 0.0], requires_grad=True)
        rho = torch.zeros(2, requires_grad=True)
        nu = torch.zeros(2, requires_grad=True)
        prediction = {"q_logit": q_logit, "rho_m": rho, "nu_mpf": nu}
        batch = {
            "q": torch.zeros(2),
            "q_valid": torch.tensor([False, True]),
            "I": torch.zeros((2, config.frames_per_window), dtype=torch.bool),
            "d": torch.full((2, config.frames_per_window), -1, dtype=torch.int64),
            "is_positive": torch.zeros(2, dtype=torch.bool),
        }

        losses = compute_losses(prediction, batch, config)
        losses.total.backward()

        self.assertEqual(losses.q_count.item(), 1.0)
        self.assertAlmostEqual(losses.q_loss.item(), 0.693147, places=5)
        self.assertEqual(q_logit.grad[0].item(), 0.0)

    def test_line_huber_is_divided_by_fixed_scale(self) -> None:
        config = SimpleCNNConfig(line_loss_scale=5_000.0, huber_delta_bins=3.0)
        q_logit = torch.zeros(1, requires_grad=True)
        rho = torch.tensor([100.0], requires_grad=True)
        nu = torch.zeros(1, requires_grad=True)
        prediction = {"q_logit": q_logit, "rho_m": rho, "nu_mpf": nu}
        batch = {
            "q": torch.ones(1),
            "q_valid": torch.ones(1, dtype=torch.bool),
            "I": torch.ones((1, config.frames_per_window), dtype=torch.bool),
            "d": torch.zeros((1, config.frames_per_window), dtype=torch.int64),
            "is_positive": torch.ones(1, dtype=torch.bool),
        }

        losses = compute_losses(prediction, batch, config)
        raw_huber = 3.0 * (100.0 - 1.5)
        self.assertAlmostEqual(losses.line_loss.item(), raw_huber / 5_000.0, places=6)



class _CaptureLogger:
    def __init__(self) -> None:
        self.metrics: dict[str, float] | None = None
        self.run_id = "test-run"

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.metrics = metrics


class _ScaleDroppingScaler:
    """模拟 GradScaler 因非有限梯度跳过 optimizer.step 并降低 scale。"""

    def __init__(self) -> None:
        self.scale_value = 1024.0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer) -> None:
        return None

    def step(self, optimizer) -> None:
        return None

    def update(self) -> None:
        self.scale_value /= 2.0

    def get_scale(self) -> float:
        return self.scale_value


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.value.expand(x.shape[0])
        return {
            "q_logit": value,
            "q": torch.sigmoid(value),
            "rho_m": value,
            "nu_mpf": value,
        }


class TrainerMetricTests(unittest.TestCase):
    """验证日志区间吞吐量使用真实 optimizer step 数。"""

    def test_steps_per_second_uses_interval_step_count(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.config = SimpleCNNConfig()
        trainer.context = DistributedContext(0, 0, 1, torch.device("cpu"), "gloo")
        trainer.logger = _CaptureLogger()
        trainer.global_step = 20
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        trainer.optimizer = torch.optim.SGD([parameter], lr=0.1)
        totals = torch.zeros(len(METRIC_NAMES), dtype=torch.float64)
        totals[1] = 1.0
        totals[METRIC_NAMES.index("block_count")] = 1.0

        with mock.patch("train.trainer.rank_zero_print"):
            trainer._log_train_totals(totals, grad_norm=0.0, elapsed=4.0, step_count=20)

        assert trainer.logger.metrics is not None
        self.assertEqual(trainer.logger.metrics["train/steps_per_second"], 5.0)


    def test_skipped_grad_scaler_update_does_not_advance_global_step(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.config = SimpleCNNConfig(batch_size_per_gpu=2)
        trainer.context = DistributedContext(0, 0, 1, torch.device("cpu"), "gloo")
        trainer.raw_model = _TinyModel()
        trainer.model = trainer.raw_model
        trainer.ddp_model = None
        trainer.optimizer = torch.optim.AdamW(trainer.raw_model.parameters(), lr=1e-3)
        trainer.scaler = _ScaleDroppingScaler()
        trainer.amp_dtype = None
        trainer.global_step = 7
        batch = {
            "x": torch.zeros((2, 1)),
            "q": torch.zeros(2),
            "I": torch.zeros((2, 20), dtype=torch.bool),
            "q_valid": torch.ones(2, dtype=torch.bool),
            "d": torch.full((2, 20), -1, dtype=torch.int64),
            "is_positive": torch.zeros(2, dtype=torch.bool),
        }
        trainer.train_loader = [batch]
        trainer.train_iterator = iter(trainer.train_loader)

        with mock.patch("train.trainer.rank_zero_print"):
            _, _, _, optimizer_updated = trainer._run_optimizer_step()

        self.assertFalse(optimizer_updated)
        self.assertEqual(trainer.global_step, 7)

    def test_successful_optimizer_update_advances_global_step(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.config = SimpleCNNConfig(batch_size_per_gpu=2)
        trainer.context = DistributedContext(0, 0, 1, torch.device("cpu"), "gloo")
        trainer.raw_model = _TinyModel()
        trainer.model = trainer.raw_model
        trainer.ddp_model = None
        trainer.optimizer = torch.optim.AdamW(trainer.raw_model.parameters(), lr=1e-3)
        trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
        trainer.amp_dtype = None
        trainer.global_step = 7
        batch = {
            "x": torch.zeros((2, 1)),
            "q": torch.zeros(2),
            "I": torch.zeros((2, 20), dtype=torch.bool),
            "q_valid": torch.ones(2, dtype=torch.bool),
            "d": torch.full((2, 20), -1, dtype=torch.int64),
            "is_positive": torch.zeros(2, dtype=torch.bool),
        }
        trainer.train_loader = [batch]
        trainer.train_iterator = iter(trainer.train_loader)

        _, _, _, optimizer_updated = trainer._run_optimizer_step()

        self.assertTrue(optimizer_updated)
        self.assertEqual(trainer.global_step, 8)

    def test_resume_baseline_runs_before_any_new_optimizer_step(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.config = SimpleCNNConfig(total_optimizer_steps=5)
        trainer.context = DistributedContext(0, 0, 1, torch.device("cpu"), "gloo")
        trainer.global_step = 5
        trainer.requires_resume_validation_baseline = True
        trainer._validate = mock.Mock()

        with mock.patch("train.trainer.rank_zero_print"):
            trainer.train()

        trainer._validate.assert_called_once_with()
        self.assertFalse(trainer.requires_resume_validation_baseline)

    def test_validation_metric_selects_best_checkpoint(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.config = SimpleCNNConfig()
        trainer.context = DistributedContext(0, 0, 1, torch.device("cpu"), "gloo")
        trainer.raw_model = torch.nn.Identity()
        trainer.validation_loader = object()
        trainer.logger = _CaptureLogger()
        trainer.global_step = 10
        trainer._save_best_checkpoint = mock.Mock()
        expected = {"val/loss_total": 10.0, "val/point_mae_bin": 3.0}
        with mock.patch("train.trainer.evaluate_model", return_value=expected) as evaluate:
            actual = trainer._validate()

        evaluate.assert_called_once()
        trainer._save_best_checkpoint.assert_called_once_with(10.0)
        self.assertEqual(actual, expected)


class DistributedBufferTests(unittest.TestCase):
    """验证 checkpoint 内容和分布式模型 buffer 同步。"""

    def test_restore_reapplies_resolved_optimizer_hyperparameters(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.config = SimpleCNNConfig(
            learning_rate=2e-3,
            adam_beta1=0.8,
            adam_beta2=0.88,
            adam_eps=1e-6,
            weight_decay=0.02,
        )
        trainer.context = DistributedContext(0, 0, 1, torch.device("cpu"), "gloo")
        trainer.raw_model = torch.nn.Linear(1, 1)
        trainer.optimizer = torch.optim.AdamW(
            trainer.raw_model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
        )
        saved_optimizer = trainer.optimizer.state_dict()
        trainer.scaler = mock.Mock()
        trainer.best_validation_metric_kind = "same-protocol"
        trainer.best_validation_loss = float("inf")
        trainer.requires_resume_validation_baseline = False
        trainer.data_stream_generation = 1
        payload = {
            "model_state": trainer.raw_model.state_dict(),
            "optimizer_state": saved_optimizer,
            "scaler_state": {},
            "global_step": 10,
            "best_validation_loss": 1.5,
            "best_validation_metric_kind": "same-protocol",
        }

        with mock.patch("train.trainer.rank_zero_print"):
            trainer._restore(payload)

        group = trainer.optimizer.param_groups[0]
        self.assertEqual(group["betas"], (0.8, 0.88))
        self.assertEqual(group["eps"], 1e-6)
        self.assertEqual(group["weight_decay"], 0.02)
        self.assertEqual(group["lr"], trainer._learning_rate_for_step(10))

    def test_validation_dataset_id_changes_best_protocol(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.config = SimpleCNNConfig()
        trainer.context = DistributedContext(0, 0, 1, torch.device("cpu"), "gloo")
        trainer.validation_dataset_id = "cache-a"
        first = trainer._validation_metric_kind()
        trainer.validation_dataset_id = "cache-b"
        second = trainer._validation_metric_kind()

        self.assertNotEqual(first, second)
        self.assertIn("cache-a", first)
        self.assertIn("cache-b", second)

    def test_changed_validation_dataset_requires_resume_baseline(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.config = SimpleCNNConfig()
        trainer.context = DistributedContext(0, 0, 1, torch.device("cpu"), "gloo")
        trainer.raw_model = torch.nn.Linear(1, 1)
        trainer.optimizer = torch.optim.AdamW(trainer.raw_model.parameters())
        trainer.scaler = mock.Mock()
        trainer.best_validation_metric_kind = "new-data-protocol"
        trainer.best_validation_loss = 0.5
        trainer.requires_resume_validation_baseline = False
        trainer.data_stream_generation = 2
        payload = {
            "model_state": trainer.raw_model.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict(),
            "scaler_state": {},
            "global_step": 20,
            "best_validation_loss": 0.1,
            "best_validation_metric_kind": "old-data-protocol",
        }

        with mock.patch("train.trainer.rank_zero_print"):
            trainer._restore(payload)

        self.assertTrue(trainer.requires_resume_validation_baseline)
        self.assertEqual(trainer.best_validation_loss, float("inf"))

    def test_checkpoint_snapshot_records_stream_and_metric_protocol(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.raw_model = torch.nn.Linear(1, 1)
        trainer.optimizer = torch.optim.SGD(trainer.raw_model.parameters(), lr=0.1)
        trainer.scaler = mock.Mock()
        trainer.scaler.state_dict.return_value = {"scale": 1.0}
        trainer.global_step = 123
        trainer.best_validation_loss = 0.5
        trainer.best_validation_metric_kind = "fixed-test"
        trainer.data_stream_generation = 4
        trainer.logger = _CaptureLogger()

        payload = trainer._snapshot_payload()

        self.assertEqual(payload["global_step"], 123)
        self.assertEqual(payload["data_stream_generation"], 4)
        self.assertEqual(payload["best_validation_metric_kind"], "fixed-test")
        self.assertEqual(payload["wandb_id"], "test-run")

    def test_all_model_buffers_are_broadcast(self) -> None:
        model = SimpleCNN(SimpleCNNConfig())
        context = DistributedContext(0, 0, 2, torch.device("cpu"), "gloo")
        with mock.patch("utils.distributed.dist.broadcast") as broadcast:
            broadcast_module_buffers(model, context)

        self.assertEqual(broadcast.call_count, len(tuple(model.buffers())))

    def test_accelerator_ddp_converts_batch_norm_without_changing_state_keys(self) -> None:
        model = SimpleCNN(SimpleCNNConfig())
        original_state_keys = tuple(model.state_dict())
        context = DistributedContext(0, 0, 2, torch.device("cuda", 0), "nccl")

        converted = _configure_distributed_batch_norm(model, context)

        self.assertTrue(
            any(isinstance(module, torch.nn.SyncBatchNorm) for module in converted.modules())
        )
        self.assertFalse(
            any(type(module) is torch.nn.BatchNorm2d for module in converted.modules())
        )
        self.assertEqual(tuple(converted.state_dict()), original_state_keys)

    def test_single_device_and_cpu_ddp_keep_regular_batch_norm(self) -> None:
        contexts = (
            DistributedContext(0, 0, 1, torch.device("cuda", 0), "nccl"),
            DistributedContext(0, 0, 2, torch.device("cpu"), "gloo"),
        )
        for context in contexts:
            with self.subTest(context=context):
                model = _configure_distributed_batch_norm(SimpleCNN(SimpleCNNConfig()), context)
                self.assertTrue(
                    any(type(module) is torch.nn.BatchNorm2d for module in model.modules())
                )
                self.assertFalse(
                    any(isinstance(module, torch.nn.SyncBatchNorm) for module in model.modules())
                )

class ModelCapacityTests(unittest.TestCase):
    """验证 n/s 容量规格只扩展网络宽度，保持统一模型接口。"""

    def test_n_and_s_use_expected_widths(self) -> None:
        normal = SimpleCNN(SimpleCNNConfig(model_type="n", hidden_dim=256))
        scaled = SimpleCNN(SimpleCNNConfig(model_type="s", hidden_dim=256))

        self.assertEqual(normal.feature_channels, (16, 32, 64, 64, 96, 96))
        self.assertEqual(normal.hidden_dim, 256)
        self.assertEqual(scaled.feature_channels, (24, 48, 96, 96, 144, 144))
        self.assertEqual(scaled.hidden_dim, 384)
        self.assertGreater(
            sum(parameter.numel() for parameter in scaled.parameters()),
            sum(parameter.numel() for parameter in normal.parameters()),
        )

    def test_invalid_model_type_is_rejected_by_config_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_type"):
            SimpleCNNConfig(model_type="large").validate()


if __name__ == "__main__":
    unittest.main()
