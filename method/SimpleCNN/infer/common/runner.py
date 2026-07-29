"""单进程流式推理共用的批构造、设备同步计时和模型前向。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
import torch

from configs.base import SimpleCNNConfig
from data.dataloader import PackedSource, rearrange_distance_channels
from eval.evaluator import autocast_context


@dataclass(frozen=True)
class BatchTiming:
    """一次候选块批量推理的实际工作量与关键耗时，单位为秒。"""

    preprocess_s: float
    model_s: float
    forward_calls: int
    blocks_evaluated: int


@dataclass(frozen=True)
class BatchPrediction:
    """一批距离块的三个模型输出及其对应的块起点。"""

    range_starts_m: np.ndarray
    q: np.ndarray
    rho_m: np.ndarray
    nu_mpf: np.ndarray
    timing: BatchTiming

    def __post_init__(self) -> None:
        count = int(self.range_starts_m.size)
        assert all(array.ndim == 1 and int(array.size) == count for array in (self.q, self.rho_m, self.nu_mpf)), "批量预测的候选数组必须是一维且长度一致。"


def synchronize_device(device: torch.device) -> None:
    """等待当前 CUDA/NPU 队列完成，使延迟统计反映真实执行时间。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        npu = getattr(torch, "npu", None)
        assert npu is not None, "当前 device 为 NPU，但 torch.npu 未注册。"
        npu.synchronize()


class ModelRunner:
    """把局部距离块拼为 batch 后执行一致的推理与计时。"""

    def __init__(
        self,
        model: torch.nn.Module,
        config: SimpleCNNConfig,
        device: torch.device,
        *,
        max_blocks_per_forward: int = 0,
    ) -> None:
        assert max_blocks_per_forward >= 0, "max_blocks_per_forward 必须为非负整数；0 表示不拆分 batch。"
        self.model = model
        self.config = config
        self.device = device
        self.max_blocks_per_forward = int(max_blocks_per_forward)
        self.model.eval()

    def _build_input(
        self,
        source: PackedSource,
        time_start: int,
        range_starts_m: Sequence[int],
    ) -> torch.Tensor:
        assert len(range_starts_m) > 0, "至少需要一个距离块才能推理。"
        arrays = [
            rearrange_distance_channels(
                source.extract_window(time_start, int(range_start), self.config),
                self.config,
            )
            for range_start in range_starts_m
        ]
        batch = np.stack(arrays, axis=0).astype(np.float32, copy=False)
        return torch.from_numpy(batch).to(self.device, non_blocking=self.config.pin_memory)

    @staticmethod
    def _as_numpy(prediction: dict[str, torch.Tensor], key: str) -> np.ndarray:
        value = prediction[key].detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float32).reshape(-1)

    @torch.inference_mode()
    def predict_blocks(
        self,
        source: PackedSource,
        time_start: int,
        range_starts_m: Sequence[int],
    ) -> BatchPrediction:
        """对给定时间窗和距离块集合前向；必要时按上限拆成多个小 batch。"""
        starts = np.asarray(tuple(int(item) for item in range_starts_m), dtype=np.int32)
        assert starts.ndim == 1 and starts.size > 0, "range_starts_m 必须为非空一维距离块起点序列。"
        maximum_start = self.config.range_bins - self.config.block_width_m
        assert not (np.any(starts < 0) or np.any(starts > maximum_start)), "距离块起点超出有效范围。"

        split_size = self.max_blocks_per_forward or int(starts.size)
        q_values: list[np.ndarray] = []
        rho_values: list[np.ndarray] = []
        nu_values: list[np.ndarray] = []
        preprocess_s = 0.0
        model_s = 0.0
        forward_calls = 0

        for chunk_start in range(0, int(starts.size), split_size):
            chunk = starts[chunk_start : chunk_start + split_size]
            phase_start = perf_counter()
            model_input = self._build_input(source, time_start, chunk.tolist())
            preprocess_s += perf_counter() - phase_start

            synchronize_device(self.device)
            model_start = perf_counter()
            with autocast_context(self.config, self.device):
                output = self.model(model_input)
            synchronize_device(self.device)
            model_s += perf_counter() - model_start
            forward_calls += 1

            q_values.append(self._as_numpy(output, "q"))
            rho_values.append(self._as_numpy(output, "rho_m"))
            nu_values.append(self._as_numpy(output, "nu_mpf"))

        timing = BatchTiming(
            preprocess_s=preprocess_s,
            model_s=model_s,
            forward_calls=forward_calls,
            blocks_evaluated=int(starts.size),
        )
        return BatchPrediction(
            range_starts_m=starts,
            q=np.concatenate(q_values),
            rho_m=np.concatenate(rho_values),
            nu_mpf=np.concatenate(nu_values),
            timing=timing,
        )

    def warmup(
        self,
        source: PackedSource,
        *,
        range_starts_m: Sequence[int],
    ) -> None:
        """预热动态方法可能使用的 batch 尺寸；预热不计入任何样本统计。"""
        starts = tuple(int(item) for item in range_starts_m)
        assert starts, "预热需要至少一个可用距离块。"
        for count in (1, 3, 5, 34):  # 覆盖全局与 L0/L1/L2 的实际 batch 尺寸。
            if count < 1:
                continue
            self.predict_blocks(source, 0, starts[: min(int(count), len(starts))])
