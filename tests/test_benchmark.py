"""Integration test for the full TPU benchmark pipeline.

Tests that the benchmark runs end-to-end on small instances
and produces valid CSV output.
"""

import sys
import os
import csv
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tpu.benchmark import run_benchmark


def test_benchmark_csv_output():
    """Verify benchmark produces a valid CSV with expected columns."""
    print("test_benchmark_csv_output:")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "results.csv")

        instance_sizes = {
            "scheduling": [4],
            "coloring": [4],
            "partitioning": [4],
            "coverage": [4],
        }

        run_benchmark(
            instance_sizes=instance_sizes,
            output_path=output_path,
            num_trials=1,
            verbose=False,
        )

        # Verify CSV exists and has rows
        assert os.path.exists(output_path), f"CSV not found at {output_path}"

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0, "CSV has no data rows"

        expected_cols = {"problem", "size", "trial", "solver",
                         "runtime_seconds", "solution_quality", "metric_name"}
        actual_cols = set(rows[0].keys())
        assert expected_cols.issubset(actual_cols), \
            f"Missing columns: {expected_cols - actual_cols}"

        # Check all four problems are represented
        problems = {r["problem"] for r in rows}
        assert problems == {"scheduling", "coloring", "partitioning", "coverage"}, \
            f"Expected 4 problem types, got {problems}"

        print(f"  ✓ CSV has {len(rows)} rows across {len(problems)} problem types")
        print(f"  ✓ Columns: {actual_cols}")

        # Print a summary
        for r in rows:
            print(f"    {r['problem']:15s} {r['solver']:20s} "
                  f"{r['metric_name']:15s}={r['solution_quality']:>10s}  "
                  f"time={r['runtime_seconds']:>10s}s")

    print()


def test_benchmark_quick_mode():
    """Verify --quick mode runs without error."""
    print("test_benchmark_quick_mode:")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "quick_results.csv")

        # Only run scheduling and coloring for speed
        instance_sizes = {
            "scheduling": [4],
            "coloring": [4],
        }

        run_benchmark(
            instance_sizes=instance_sizes,
            output_path=output_path,
            num_trials=1,
            verbose=False,
        )

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0
        problems = {r["problem"] for r in rows}
        assert problems == {"scheduling", "coloring"}

        # Verify runtime is parseable as float
        for r in rows:
            runtime = float(r["runtime_seconds"])
            assert runtime >= 0, f"Negative runtime: {runtime}"

        print(f"  ✓ Quick mode OK: {len(rows)} rows")
        print()


if __name__ == "__main__":
    test_benchmark_csv_output()
    test_benchmark_quick_mode()
    print("All benchmark tests passed!")
