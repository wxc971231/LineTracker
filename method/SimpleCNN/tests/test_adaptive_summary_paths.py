"""自适应跟踪汇总器的参数结果目录兼容性测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from infer.adaptive_tracker import summary_compare, summary_single  # noqa: E402


class AdaptiveSummaryPathTests(unittest.TestCase):
    def test_parameterised_method_directory_parses_method_and_stride(self) -> None:
        self.assertEqual(
            summary_single._parse_method_directory(
                "adaptive_tracker_stride5_captureq0.5_trackq0.5"
            ),
            ("adaptive_tracker", 5),
        )

    def test_compare_accepts_parameterised_method_directory(self) -> None:
        run_dir = Path("/tmp/F300-N10k-S42--v2-n-best-s42000")
        self.assertEqual(
            summary_compare._normalise_items(
                [(run_dir, "adaptive_tracker_stride5_captureq0.5_trackq0.5")]
            ),
            [(run_dir.resolve(), "adaptive_tracker_stride5_captureq0.5_trackq0.5")],
        )

    def test_compare_output_is_grouped_by_run_directory_name(self) -> None:
        run_dir = Path("/tmp/F300-N10k-S42--v2-n-best-s42000")
        self.assertEqual(
            summary_compare._comparison_output_directory(
                Path("/tmp/compare"),
                [(run_dir, "adaptive_tracker_stride5_captureq0.5_trackq0.5")],
            ),
            Path("/tmp/compare/F300-N10k-S42--v2-n-best-s42000"),
        )

    def test_compare_parser_accepts_repeated_run_and_method_directories(self) -> None:
        args = summary_compare.build_parser().parse_args(
            [
                "--item",
                "/tmp/F300-N10k-S42--v2-n-best-s42000",
                "adaptive_tracker_stride5_captureq0.5_trackq0.5",
                "--item",
                "/tmp/F300-N10k-S42--v2-n-best-s42000",
                "global_top1_stride5",
            ]
        )
        self.assertEqual(
            args.item,
            [
                [
                    "/tmp/F300-N10k-S42--v2-n-best-s42000",
                    "adaptive_tracker_stride5_captureq0.5_trackq0.5",
                ],
                ["/tmp/F300-N10k-S42--v2-n-best-s42000", "global_top1_stride5"],
            ],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
