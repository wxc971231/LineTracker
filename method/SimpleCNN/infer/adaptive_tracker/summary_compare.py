"""对比多个 schema v5 推理结果的轨迹质量。

本模块复用 :mod:`summary_single` 的单方案读取与校验逻辑。比较时仅使用所有
方案都通过校验的共同样本编号，以避免不同方案因缺失或损坏样本不同而产生不公平
的质量结论。

可直接调用::

    build_comparison([
        (Path("/path/to/run_a"), "adaptive_tracker_stride5"),
        (Path("/path/to/run_b"), "global_top1_stride5"),
    ])

也可在命令行重复传入 ``--item <run_dir> <method_dir>``。当没有命令行 ``--item``
时，``DEFAULT_COMPARISON_INPUTS`` 中的列表会被使用，便于在服务器上直接编辑运行。
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias, cast

if __package__:
    from . import summary_single
else:  # 支持从 adaptive_tracker 目录或项目根目录直接执行此文件。
    import summary_single


ComparisonItem: TypeAlias = tuple[Path | str, str]

# 服务器直接运行时可在此填写待比较方案；命令行 --item 会覆盖该列表。
DEFAULT_COMPARISON_INPUTS: list[ComparisonItem] = []
DEFAULT_SAMPLE_START: int | None = None
DEFAULT_SAMPLE_STOP: int | None = None
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "_output" / "compare"


def _nonnegative_int(value: str) -> int:
    """沿用单方案汇总器的样本编号约束。"""
    return summary_single._nonnegative_int(value)


def _entry_name(run_dir: Path, method_dir: str) -> str:
    """生成简短且能区分不同 run 的行名称。"""
    return f"{run_dir.name} / {method_dir}"


def _normalise_items(items: Sequence[ComparisonItem]) -> list[tuple[Path, str]]:
    """校验并规范化 ``[(run_dir, method_dir), ...]`` 输入列表。"""
    if not items:
        raise ValueError("至少需要提供一个 (run_dir, method_dir) 比较项。")

    normalised: list[tuple[Path, str]] = []
    seen: set[tuple[Path, str]] = set()
    for raw_run_dir, method_dir in items:
        run_dir = Path(raw_run_dir).expanduser().resolve()
        if not isinstance(method_dir, str) or not method_dir:
            raise ValueError(f"method_dir 必须是非空字符串，实际为 {method_dir!r}")
        # 复用单方案脚本的方法目录契约，尽早指出配置错误。
        summary_single._parse_method_directory(method_dir)
        key = (run_dir, method_dir)
        if key in seen:
            raise ValueError(f"重复的比较项：{run_dir} / {method_dir}")
        seen.add(key)
        normalised.append(key)
    return normalised


def _stats(samples: Sequence[Mapping[str, Any]], field: str) -> dict[str, float | int | None]:
    """按单方案脚本完全相同的统计定义重算某个共同样本指标。"""
    return summary_single._stats(float(sample[field]) for sample in samples)


def _quality_row(
    *,
    name: str,
    run_dir: Path,
    method_dir: str,
    samples: Sequence[Mapping[str, Any]],
    source_count_before_intersection: int,
    missing_count: int,
    invalid_count: int,
) -> dict[str, Any]:
    """汇总一项方案在共同样本上的质量、计算量、耗时与状态机可靠性。"""
    if not samples:
        raise ValueError(f"{name} 在共同样本集上为空，无法比较。")

    trajectory = {field: _stats(samples, field) for field in summary_single._TRAJECTORY_FIELDS}
    compute = {field: _stats(samples, field) for field in summary_single._COMPUTE_FIELDS}
    timing: dict[str, Any] = {
        field: _stats(samples, field) for field in summary_single._TIMING_FIELDS
    }
    logical_steps_total = sum(float(sample["logical_steps"]) for sample in samples)
    end_to_end_total_s = sum(float(sample["end_to_end_total_s"]) for sample in samples)
    timing["weighted_end_to_end_mean_ms"] = (
        None if logical_steps_total == 0.0 else 1_000.0 * end_to_end_total_s / logical_steps_total
    )
    timing["by_step_type"] = summary_single._step_timing_summary(samples)
    capture_successes = sum(float(sample["capture_success"]) for sample in samples)
    recapture_starts = sum(float(sample["recapture_start_count"]) for sample in samples)
    recapture_successes = sum(float(sample["recapture_success_count"]) for sample in samples)
    capture_delays = summary_single._stats(
        float(sample["first_capture_delay_frames"])
        for sample in samples
        if float(sample["capture_success"]) > 0.0
        and sample["first_capture_delay_frames"] is not None
    )

    return {
        "name": name,
        "run_dir": str(run_dir),
        "run_dir_relative": summary_single._project_relative_path(run_dir),
        "method_dir": method_dir,
        "valid_sample_count_before_intersection": source_count_before_intersection,
        "common_sample_count": len(samples),
        "missing_metrics_count": missing_count,
        "invalid_metrics_count": invalid_count,
        "trajectory": trajectory,
        "compute": compute,
        "timing": timing,
        "tracker": {
            "initial_capture_success_rate": capture_successes / len(samples),
            "first_capture_delay_frames_success_only": capture_delays,
            "recapture_start_count_total": int(round(recapture_starts)),
            "recapture_success_count_total": int(round(recapture_successes)),
            "recapture_success_rate_per_start": (
                None if recapture_starts == 0.0 else recapture_successes / recapture_starts
            ),
        },
    }


def build_comparison(
    items: Sequence[ComparisonItem],
    *,
    sample_start: int | None = None,
    sample_stop: int | None = None,
) -> dict[str, Any]:
    """读取多个单方案结果，并以共同有效样本为基准构建质量对比数据。"""
    normalised_items = _normalise_items(items)
    loaded: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []

    for run_dir, method_dir in normalised_items:
        name = _entry_name(run_dir, method_dir)
        try:
            summary, samples, missing_indices, invalid_samples = summary_single.build_summary(
                run_dir,
                method_dir,
                sample_start=sample_start,
                sample_stop=sample_stop,
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            # 其它方案仍可产生可读报告；配置级失败会在报告中完整保留。
            unavailable.append(
                {
                    "name": name,
                    "run_dir": str(run_dir),
                    "method_dir": method_dir,
                    "reason": str(error),
                }
            )
            continue
        loaded.append(
            {
                "name": name,
                "run_dir": run_dir,
                "method_dir": method_dir,
                "summary": summary,
                "samples": samples,
                "missing_indices": missing_indices,
                "invalid_samples": invalid_samples,
            }
        )

    if not loaded:
        reasons = "; ".join(f"{item['name']}: {item['reason']}" for item in unavailable)
        raise RuntimeError(f"没有可用于比较的方案。{reasons}")

    common_indices = set(int(sample["source_index"]) for sample in loaded[0]["samples"])
    for item in loaded[1:]:
        common_indices.intersection_update(int(sample["source_index"]) for sample in item["samples"])
    if not common_indices:
        raise ValueError("各方案之间不存在共同通过校验的样本编号，无法进行公平质量对比。")
    ordered_common_indices = sorted(common_indices)

    comparisons: list[dict[str, Any]] = []
    for item in loaded:
        sample_by_index = {int(sample["source_index"]): sample for sample in item["samples"]}
        common_samples = [sample_by_index[index] for index in ordered_common_indices]
        comparisons.append(
            _quality_row(
                name=str(item["name"]),
                run_dir=cast(Path, item["run_dir"]),
                method_dir=str(item["method_dir"]),
                samples=common_samples,
                source_count_before_intersection=len(cast(Sequence[Mapping[str, Any]], item["samples"])),
                missing_count=len(cast(Sequence[int], item["missing_indices"])),
                invalid_count=len(cast(Sequence[Mapping[str, Any]], item["invalid_samples"])),
            )
        )

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_source_index_range": {"start": sample_start, "stop": sample_stop},
        "common_source_indices": ordered_common_indices,
        "common_sample_count": len(ordered_common_indices),
        "comparisons": comparisons,
        "unavailable_configurations": unavailable,
    }


def _format_stat(
    row: Mapping[str, Any],
    metric: str,
    statistic: str,
    *,
    section: str = "trajectory",
) -> str:
    """读取任一统计分区的数值，并沿用单方案报告的格式。"""
    metrics = cast(Mapping[str, Mapping[str, Any]], row[section])
    return summary_single._format_number(metrics[metric][statistic])


def _build_report(comparison: Mapping[str, Any]) -> str:
    """将共同样本质量对比渲染为简洁 Markdown 文档。"""
    rows = cast(Sequence[Mapping[str, Any]], comparison["comparisons"])
    common_indices = cast(Sequence[int], comparison["common_source_indices"])
    unavailable = cast(Sequence[Mapping[str, str]], comparison["unavailable_configurations"])
    requested_range = summary_single._format_requested_sample_range(
        comparison["requested_source_index_range"]
    )

    lines = [
        "# SimpleCNN 推理轨迹质量对比",
        "",
        f"- 生成时间（UTC）：{comparison['generated_at_utc']}",
        f"- 样本编号范围（配置）：{requested_range}",
        f"- 公平比较样本：{comparison['common_sample_count']} 个共同通过校验的样本",
        f"- 共同样本编号：{min(common_indices)}–{max(common_indices)}",
        "- 下列质量指标均只基于共同样本计算；因此不会受到缺失或校验失败样本不一致的影响。",
        "",
        "## 输入方案与有效样本",
        "",
        "| 方案 | Run 目录 | 方法目录 | 共同样本前有效数 | 缺少 metrics | 校验失败 | 共同样本数 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | `{row['run_dir_relative']}` | `{row['method_dir']}` | "
            f"{row['valid_sample_count_before_intersection']} | {row['missing_metrics_count']} | "
            f"{row['invalid_metrics_count']} | {row['common_sample_count']} |"
        )

    lines.extend([
        "",
        "## 轨迹质量（共同样本）",
        "",
        "| 方案 | coverage 均值 | unreliable coverage 均值 | MAE 均值 (m) | MAE 中位数 (m) | MAE p95 (m) | RMSE 中位数 (m) | RMSE p95 (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['name']} | {_format_stat(row, 'coverage', 'mean')} | "
            f"{_format_stat(row, 'unreliable_coverage', 'mean')} | "
            f"{_format_stat(row, 'mae_m', 'mean')} | {_format_stat(row, 'mae_m', 'median')} | "
            f"{_format_stat(row, 'mae_m', 'p95')} | {_format_stat(row, 'rmse_m', 'median')} | "
            f"{_format_stat(row, 'rmse_m', 'p95')} |"
        )

    lines.extend([
        "",
        "## 实际目标响应帧质量（共同样本）",
        "",
        "| 方案 | hit coverage 均值 | hit MAE 均值 (m) | hit MAE 中位数 (m) | hit MAE p95 (m) | jump count 均值 | jump count p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['name']} | {_format_stat(row, 'hit_coverage', 'mean')} | "
            f"{_format_stat(row, 'hit_mae_m', 'mean')} | {_format_stat(row, 'hit_mae_m', 'median')} | "
            f"{_format_stat(row, 'hit_mae_m', 'p95')} | {_format_stat(row, 'jump_count', 'mean')} | "
            f"{_format_stat(row, 'jump_count', 'p95')} |"
        )

    lines.extend([
        "",
        "## 计算量（共同样本，逐样本统计）",
        "",
        "| 方案 | logical steps 均值 | logical steps p95 | blocks 均值 | blocks p95 | forward calls 均值 | forward calls p95 | FLOPs 均值 | FLOPs p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['name']} | {_format_stat(row, 'logical_steps', 'mean', section='compute')} | "
            f"{_format_stat(row, 'logical_steps', 'p95', section='compute')} | "
            f"{_format_stat(row, 'blocks_evaluated', 'mean', section='compute')} | "
            f"{_format_stat(row, 'blocks_evaluated', 'p95', section='compute')} | "
            f"{_format_stat(row, 'forward_calls', 'mean', section='compute')} | "
            f"{_format_stat(row, 'forward_calls', 'p95', section='compute')} | "
            f"{_format_stat(row, 'estimated_conv_linear_flops_total', 'mean', section='compute')} | "
            f"{_format_stat(row, 'estimated_conv_linear_flops_total', 'p95', section='compute')} |"
        )

    lines.extend([
        "",
        "## 端到端耗时（共同样本）",
        "",
        "| 方案 | 端到端总耗时均值 (s) | 端到端总耗时 p95 (s) | 预处理均值 (s) | 模型前向均值 (s) | 每逻辑步加权均值 (ms) | 样本内步均值 p95 (ms) | 样本内步 p95 的 p95 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        timing = cast(Mapping[str, Any], row["timing"])
        lines.append(
            f"| {row['name']} | {_format_stat(row, 'end_to_end_total_s', 'mean', section='timing')} | "
            f"{_format_stat(row, 'end_to_end_total_s', 'p95', section='timing')} | "
            f"{_format_stat(row, 'preprocess_total_s', 'mean', section='timing')} | "
            f"{_format_stat(row, 'model_total_s', 'mean', section='timing')} | "
            f"{summary_single._format_number(timing['weighted_end_to_end_mean_ms'])} | "
            f"{_format_stat(row, 'end_to_end_mean_ms', 'p95', section='timing')} | "
            f"{_format_stat(row, 'end_to_end_p95_ms', 'p95', section='timing')} |"
        )

    lines.extend([
        "",
        "## 逐逻辑步端到端耗时（共同样本 pooled）",
        "",
        "| 方案 | 步类型 | 步数 | 均值 (ms) | 中位数 (ms) | p95 (ms) | 最大 (ms) | 总耗时 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    step_order = {"CAPTURE": 0, "RECAPTURE": 1, "Track-L0": 2, "Track-L1": 3, "Track-L2": 4}
    all_step_types = {
        step_type
        for row in rows
        for step_type in cast(Mapping[str, Mapping[str, Any]], cast(Mapping[str, Any], row["timing"])["by_step_type"])
    }
    # 外层按步类型、内层按方案输出，便于同一种逻辑步横向比较不同模型。
    for step_type in sorted(all_step_types, key=lambda item: (step_order.get(item, len(step_order)), item)):
        for row in rows:
            timing = cast(Mapping[str, Any], row["timing"])
            by_step_type = cast(Mapping[str, Mapping[str, Any]], timing["by_step_type"])
            stat = by_step_type.get(step_type)
            if stat is None:
                continue
            lines.append(
                f"| {row['name']} | {step_type} | {summary_single._format_number(stat['count'], 0)} | "
                f"{summary_single._format_number(stat['mean'])} | {summary_single._format_number(stat['median'])} | "
                f"{summary_single._format_number(stat['p95'])} | {summary_single._format_number(stat['max'])} | "
                f"{summary_single._format_number(stat['total_ms'])} |"
            )

    lines.extend([
        "",
        "## 状态机可靠性（共同样本）",
        "",
        "| 方案 | 初始捕获成功率 | 首次捕获延迟中位数（帧） | 重捕获触发次数 | 重捕获成功率 |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in rows:
        tracker = cast(Mapping[str, Any], row["tracker"])
        delay = cast(Mapping[str, Any], tracker["first_capture_delay_frames_success_only"])
        lines.append(
            f"| {row['name']} | {summary_single._format_number(tracker['initial_capture_success_rate'])} | "
            f"{summary_single._format_number(delay['median'])} | "
            f"{summary_single._format_number(tracker['recapture_start_count_total'], 0)} | "
            f"{summary_single._format_number(tracker['recapture_success_rate_per_start'])} |"
        )

    if unavailable:
        lines.extend([
            "",
            "## 未纳入比较的配置",
            "",
            "| 方案 | 原因 |",
            "|---|---|",
        ])
        for item in unavailable:
            reason = item["reason"].replace("|", "\\|")
            lines.append(f"| {item['name']} | {reason} |")
    lines.append("")
    return "\n".join(lines)


def write_comparison(
    items: Sequence[ComparisonItem],
    output_dir: Path | str,
    *,
    sample_start: int | None = None,
    sample_stop: int | None = None,
) -> dict[str, Any]:
    """构建对比结果，并将 ``comparison.json`` 与 ``report.md`` 写入输出目录。"""
    comparison = build_comparison(
        items,
        sample_start=sample_start,
        sample_stop=sample_stop,
    )
    destination = Path(output_dir).expanduser().resolve()
    summary_single._atomic_json(destination / "comparison.json", comparison)
    summary_single._atomic_text(destination / "report.md", _build_report(comparison))
    return comparison


def build_parser() -> argparse.ArgumentParser:
    """定义可重复传入比较项的命令行接口。"""
    parser = argparse.ArgumentParser(description="对比多个 schema v5 SimpleCNN 推理结果的轨迹质量")
    parser.add_argument(
        "--item",
        nargs=2,
        action="append",
        metavar=("RUN_DIR", "METHOD_DIR"),
        help="重复传入：--item <run_dir> <method_dir>；未传入时使用 DEFAULT_COMPARISON_INPUTS",
    )
    parser.add_argument("--sample-start", type=_nonnegative_int, default=DEFAULT_SAMPLE_START)
    parser.add_argument("--sample-stop", type=_nonnegative_int, default=DEFAULT_SAMPLE_STOP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    """读取配置列表并生成质量对比文档。"""
    args = build_parser().parse_args()
    items: Sequence[ComparisonItem]
    if args.item:
        items = [(Path(run_dir), method_dir) for run_dir, method_dir in args.item]
    else:
        items = [
            (Path('/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/infer/_output/F300-N10k-S42--v2-s-best-s48000'), 'adaptive_tracker_stride5'),
            (Path('/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/infer/_output/F300-N10k-S42--v2-n-best-s42000'), 'adaptive_tracker_stride5'),
            (Path('/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/infer/_output/F300-N10k-S42--v2-xn-best-s18000'), 'adaptive_tracker_stride5'),   
        ]


    comparison = write_comparison(
        items,
        args.output_dir,
        sample_start=args.sample_start,
        sample_stop=args.sample_stop,
    )
    print(
        f"已生成 {len(comparison['comparisons'])} 个方案、"
        f"{comparison['common_sample_count']} 个共同有效样本的质量对比 -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
