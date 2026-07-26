"""SimpleCNN 的集中式配置、profile 加载和命令行覆盖工具。"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = METHOD_ROOT.parents[1]
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "data/synthetic/gen/F300_N200_S20260717_B-random_T-R10000-290000m-V340-A6-C10-J1-K300-Q0p35-0p95"
)


@dataclass
class SimpleCNNConfig:
    """第一版 SimpleCNN 的全部可调超参数。

    所有 batch size 都是“每 GPU”的 micro-batch size；训练总预算以
    ``total_optimizer_steps`` 为准，而不是 epoch 数。
    """

    # 实验与随机性
    profile: str = "simplecnn_v1"                   # 当前加载的 configs profile 名称，用于写入实验配置快照。
    run_name: str = "simplecnn_v1"                  # 本次实验名；决定 runs/ 下的一级输出目录。
    seed: int = 42                                  # 全局随机种子；rank 和 DataLoader worker 会派生各自子种子。
    output_root: Path = METHOD_ROOT / "runs"        # 所有训练运行目录的根路径。
    data_root: Path = DEFAULT_DATA_ROOT             # 合成完整序列根目录；其下每个子目录含一份 data.npz。

    # 完整数据集划分
    split_seed: int = 42                            # 按完整序列随机划分 train/val/test 时使用的固定种子。
    source_sample_limit: int = 100                  # 划分前按稳定路径排序仅取前 N 份完整序列；0 表示使用 data_root 下全部序列。
    train_fraction: float = 0.80                    # 完整序列划入训练集的比例。
    val_fraction: float = 0.10                      # 完整序列划入验证集的比例。
    test_fraction: float = 0.10                     # 完整序列划入测试集的比例；三者之和必须为 1。

    # 分块与压缩观测
    frames_per_window: int = 20                     # 每个局部输入块的连续帧数；SimpleCNN-v1 固定为 10 km。
    range_bins: int = 300_000                       # 每帧总距离 bin 数；当前每个 bin 对应 1 m。
    block_width_m: int = 10_000                     # 单个局部块的距离宽度（m/bin）；SimpleCNN-v1 固定为 10 km。
    spatial_step_m: int = 9_000                     # 推理/验证标准分块的距离步进（m），相邻块重叠 1 km。
    packed_bitorder: str = "big"                    # observation_packed 的 bit 解包顺序，必须与合成器 packbits 一致。

    # 训练数据流
    num_workers: int = 0                            # 每个 rank 的 DataLoader 子进程数；0 表示在主训练进程中在线裁剪。
    pin_memory: bool = True                         # 是否锁页 CPU batch 内存，以加快向 CUDA 设备传输。
    source_cache_size: int = 4                      # 每个 rank/worker 同时缓存的完整 data.npz 序列数量。
    source_positive_quota: int = 64                 # 一份缓存序列累计贡献该数量正样本后，尝试替换为新序列。
    positive_fraction: float = 0.25                 # 每个训练 batch 的可见正样本比例；默认近似正负 1:3。
    standard_positive_fraction: float = 0.30        # 正样本中按推理标准 9 km 网格裁剪的比例，其余使用随机空间起点。
    positive_margin_m: int = 100                    # 正样本目标轨迹与块距离边界至少间隔的距离宽度（m）。
    negative_guard_m: int = 100                     # 普通负样本块与目标轨迹至少间隔的距离宽度（m）。
    negative_local_span_m: int = 3_000              # 局部困难负样本在可行区间内、贴近轨迹一侧的抽样跨度（m）。
    negative_local_weight: float = 1.0 / 3.0        # 局部困难负样本：同时间窗内，找到与目标轨迹至少相隔 negative_guard_m 的距离块，向远离轨迹方向随机偏移最多 negative_local_span_m
    negative_same_time_weight: float = 1.0 / 3.0    # 局部分层负样本：同时间窗内，距离安全区间中，先随机采样中心距离区间 [0,10), [10,50), [50,300)，再在距离区间内随机抽取负样本块
    negative_random_weight: float = 1.0 / 3.0       # 随机负样本：从随机序列、随机时间和随机距离采样负样本
    counterfactual_negative_weight: float = 0.0     # 反事实负样本：从 background_only_packed 取同位置纯背景反事实负样本的相对权重

    # 训练与验证设置
    batch_size_per_gpu: int = 32                    # 每张 GPU 每个 micro-batch 的局部块数量。
    eval_batch_size_per_gpu: int = 64               # 每张 GPU 在验证、测试时一次前向的标准块数量。
    validation_time_stride: int = 5                 # 固定验证网格在时间轴上的步进（帧）；最小可设为 1。

    # 网络结构（与 _doc/SimpleCNN.md 第 4 章一致）
    input_channels: int = 8                         # 距离轴按 mod 8 无损重排后的输入通道数。
    hidden_dim: int = 256                           # Flatten 后共享 MLP 隐层的特征维度。
    dropout: float = 0.10                           # 共享 MLP 隐层的 dropout 概率。
    max_speed_per_frame_m: float = 17.0             # 斜率头的绝对上限（m/frame），对应 340 m/s 和 50 ms 帧间隔。

    # 损失
    lambda_q: float = 1.0                           # 质量头 BCE 损失在总损失中的权重。
    lambda_line: float = 1.0                        # 逐帧响应距离 Huber 几何损失在总损失中的权重。
    huber_delta_bins: float = 3.0                   # Huber 二次段切换到线性段的阈值（bin，即 m）。

    # 优化器和按 step 调度
    total_optimizer_steps: int = 50_000             # 训练总 optimizer 更新次数；无限在线数据流不按 epoch 停止。
    gradient_accumulation_steps: int = 1            # 多少个 micro-batch 累积梯度后执行一次 optimizer 更新，等效训练 batch_size = batch_size_per_gpu * world_size * gradient_accumulation_steps。
    learning_rate: float = 3e-4                     # AdamW 峰值学习率；warmup 后从该值开始余弦衰减。
    min_learning_rate_ratio: float = 0.10           # 余弦衰减终点学习率相对于峰值学习率的比例。
    warmup_ratio: float = 0.05                      # 训练前该比例的 optimizer steps 采用线性 warmup。
    weight_decay: float = 1e-4                      # AdamW 权重衰减系数。
    adam_beta1: float = 0.9                         # AdamW 一阶动量的指数衰减系数。
    adam_beta2: float = 0.999                       # AdamW 二阶动量的指数衰减系数。
    adam_eps: float = 1e-8                          # AdamW 分母的数值稳定项。
    grad_clip_norm: float = 1.0                     # 全局梯度范数裁剪上限；设为非正值可关闭。
    amp: str = "auto"                               # 混合精度模式：auto / bf16 / fp16 / off。
    compile_model: bool = False                     # 是否用 torch.compile 包装模型；可提速但增加首次编译时间。

    # 日志、验证和断点
    log_interval_steps: int = 20                    # 每隔多少 optimizer steps 聚合并写入一次训练日志。
    eval_interval_steps: int = 1_000                # 每隔多少 optimizer steps 在固定验证网格上评估一次。
    checkpoint_interval_steps: int = 1_000          # 每隔多少 optimizer steps 保存一次 last 与带 step 的断点。
    keep_last_checkpoints: int = 2                  # 保留的最近 step checkpoint 数；last.pt 和 best.pt 始终保留。
    resume: Path | None = None                      # 要恢复训练的 last.pt 路径；None 表示从头开始训练。

    # Weights & Biases
    wandb_project: str = "LineTracker-SimpleCNN"    # W&B 项目名称。
    wandb_entity: str | None = None                 # W&B 团队/用户实体；None 时使用当前登录默认实体。
    wandb_mode: str = "online"                      # 唯一 W&B 开关：online / offline / disabled。

    def validate(self) -> None:
        """在启动训练前尽早报告配置错误。"""
        if self.frames_per_window != 20:
            raise ValueError("SimpleCNN-v1 固定使用 20 帧输入。")
        if self.block_width_m != 10_000:
            raise ValueError("SimpleCNN-v1 固定使用 10000 m 距离块。")
        if self.block_width_m % self.input_channels != 0:
            raise ValueError("距离块宽度必须能被重排通道数整除。")
        if self.spatial_step_m <= 0 or self.spatial_step_m > self.block_width_m:
            raise ValueError("spatial_step_m 必须位于 (0, block_width_m]。")
        if self.batch_size_per_gpu < 2:
            raise ValueError("batch_size_per_gpu 至少为 2，才能同时容纳正负样本。")
        if not 0.0 < self.positive_fraction < 1.0:
            raise ValueError("positive_fraction 必须位于 (0, 1)。")
        if not 0.0 <= self.standard_positive_fraction <= 1.0:
            raise ValueError("standard_positive_fraction 必须位于 [0, 1]。")
        if self.validation_time_stride < 1:
            raise ValueError("validation_time_stride 的最小值为 1。")
        if self.total_optimizer_steps < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("训练步数和梯度累积步数必须为正。")
        if self.source_cache_size < 1 or self.source_positive_quota < 1:
            raise ValueError("source cache 和 source quota 必须为正。")
        if self.source_sample_limit < 0:
            raise ValueError("source_sample_limit 必须为非负整数；0 表示使用全部序列。")
        if self.huber_delta_bins <= 0.0:
            raise ValueError("huber_delta_bins 必须为正。")
        if self.amp not in {"auto", "bf16", "fp16", "off"}:
            raise ValueError("amp 仅支持 auto、bf16、fp16 或 off。")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode 仅支持 online、offline 或 disabled。")
        if self.packed_bitorder not in {"big", "little"}:
            raise ValueError("packed_bitorder 仅支持 big 或 little。")
        weight_sum = (
            self.negative_local_weight
            + self.negative_same_time_weight
            + self.negative_random_weight
            + self.counterfactual_negative_weight
        )
        if weight_sum <= 0.0:
            raise ValueError("至少一种负样本来源的权重必须为正。")
        split_sum = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(split_sum - 1.0) > 1e-8:
            raise ValueError("train/val/test 三个比例之和必须为 1。")

    def as_serializable_dict(self) -> dict[str, Any]:
        """把 Path 等对象转换为可写入 JSON 的配置快照。"""

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))


def _coerce_override(current: Any, raw_value: str) -> Any:
    """按现有字段类型把 ``--set key=value`` 文本转换为对应 Python 类型。"""
    if isinstance(current, bool):
        normalized = raw_value.strip().lower()
        if normalized not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"布尔覆盖值无效：{raw_value}")
        return normalized in {"true", "1", "yes"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw_value)
    if isinstance(current, float):
        return float(raw_value)
    if isinstance(current, Path):
        return Path(raw_value).expanduser().resolve()
    if current is None:
        return None if raw_value.lower() == "none" else raw_value
    return raw_value


def apply_overrides(config: SimpleCNNConfig, assignments: list[str]) -> SimpleCNNConfig:
    """应用形如 ``--set learning_rate=1e-4`` 的扁平配置覆盖。"""
    known_fields = {item.name for item in fields(config)}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"覆盖参数必须采用 key=value 格式：{assignment}")
        key, raw_value = assignment.split("=", maxsplit=1)
        key = key.strip()
        if key not in known_fields:
            raise KeyError(f"未知配置字段：{key}")
        setattr(config, key, _coerce_override(getattr(config, key), raw_value))
    config.validate()
    return config


def load_profile(profile_name: str) -> SimpleCNNConfig:
    """加载 ``configs/<profile_name>.py`` 中的 ``get_config``。"""
    module = importlib.import_module(f"configs.{profile_name}")
    if not hasattr(module, "get_config"):
        raise AttributeError(f"configs.{profile_name} 未定义 get_config()。")
    config = module.get_config()
    if not isinstance(config, SimpleCNNConfig):
        raise TypeError("get_config() 必须返回 SimpleCNNConfig。")
    config.profile = profile_name
    config.validate()
    return config


def load_config_json(path: Path) -> SimpleCNNConfig:
    """从已经落盘的 ``resolved_config.json`` 恢复配置。"""
    values = json.loads(path.read_text(encoding="utf-8"))
    # 早期实验快照曾含有冗余的 wandb_enabled；现在统一由 wandb_mode 控制。
    values.pop("wandb_enabled", None)
    path_keys = {"output_root", "data_root", "resume"}
    for key in path_keys:
        if values.get(key) is not None:
            values[key] = Path(values[key])
    config = SimpleCNNConfig(**values)
    config.validate()
    return config


def save_config_json(config: SimpleCNNConfig, path: Path) -> None:
    """保存解析后的完整配置，便于恢复和可复现实验。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.as_serializable_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_entrypoint_args(
    description: str = "SimpleCNN DDP 训练",
    config: str = "simplecnn_v1",
    resume: Path | str | None = None,
    overrides: Sequence[str] | None = None,
) -> argparse.Namespace:
    """解析训练入口参数，并允许调用方提供无需命令行的默认值。

    ``config``、``resume`` 和 ``overrides`` 是传入 ``argparse`` 的默认值；
    如果实际命令行含有 ``--config``、``--resume`` 或 ``--set``，相应命令行值会覆盖这些默认值。
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=config,
        help="configs 下的 profile 名称。"
    )
    parser.add_argument("--resume", type=Path, default=None if resume is None else Path(resume),
        help="要恢复的 last.pt checkpoint。",
    )
    parser.add_argument("--set", action="append", default=list(overrides or ()), metavar="KEY=VALUE",
        help="覆盖 profile 字段，可重复使用，例如 --set batch_size_per_gpu=16。",
    )
    return parser.parse_args()
