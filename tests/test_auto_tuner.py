"""Tests for the auto-tuning module."""

import os
import sys
import tempfile
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tpu.auto_tuner import AutoTuner, TUNING_SEARCH_SPACE


def test_auto_tuner_init():
    """Verify AutoTuner can be instantiated."""
    print("test_auto_tuner_init:")
    tuner = AutoTuner(seed=42)
    assert tuner is not None
    assert tuner.tuning_results == []
    print("  \u2713 AutoTuner created")
    print()


def test_auto_tuner_tune_fast():
    """Run a minimal tuning (grid search fallback, 2 trials) on scheduling/FEM."""
    print("test_auto_tuner_tune_fast:")
    tuner = AutoTuner(seed=42)
    result = tuner.tune("scheduling", "FEM", n_trials=2, size=4)
    assert result is not None
    assert "solver" in result
    assert result["solver"] == "FEM"
    assert result["problem_type"] == "scheduling"
    assert "best_objective" in result
    print(f"  \u2713 Tune result: objective={result['best_objective']:.4f}")
    print(f"  \u2713 Params: {result}")
    print()


def test_auto_tuner_tune_all_problems():
    """Quick-tune SBM on scheduling (1 trial, small instance)."""
    print("test_auto_tuner_tune_all_problems:")
    tuner = AutoTuner(seed=42)
    # Test SBM (fastest solver) on scheduling with size=4
    result = tuner.tune("scheduling", "SBM", n_trials=1, size=4)
    assert result is not None
    assert result["solver"] == "SBM"
    print(f"  \u2713 SBM on scheduling: objective={result['best_objective']:.4f}")
    print(f"  \u2713 Params: {result}")
    print()


def test_auto_tuner_save_and_load():
    """Verify tuned configs can be saved and loaded via get_best_config."""
    print("test_auto_tuner_save_and_load:")
    tuner = AutoTuner(seed=42)

    # Temporarily redirect tuned dir
    with tempfile.TemporaryDirectory() as tmpdir:
        # Monkey-patch the tuned dir
        import src.tpu.auto_tuner as at
        original_dir = at._TUNED_DIR
        at._TUNED_DIR = Path(tmpdir)

        try:
            result = tuner.tune("scheduling", "FEM", n_trials=2, size=4)
            tuner._save_best_config("FEM", "scheduling", result)

            loaded = AutoTuner.get_best_config("FEM", "scheduling")
            assert loaded is not None
            assert loaded["solver"] == "FEM"
            assert loaded["problem_type"] == "scheduling"
            print(f"  \u2713 Saved and loaded: {loaded}")
        finally:
            at._TUNED_DIR = original_dir

    print()


def test_tuning_search_space():
    """Verify search space contains expected solvers and params."""
    print("test_tuning_search_space:")
    for solver_name in ("FEM", "SBM", "QIS3"):
        assert solver_name in TUNING_SEARCH_SPACE, f"Missing {solver_name}"
        params = TUNING_SEARCH_SPACE[solver_name]
        assert len(params) > 0, f"Empty search space for {solver_name}"
        print(f"  \u2713 {solver_name}: {list(params.keys())}")
    print()


def test_tuning_results_csv():
    """Verify tuning results CSV is written correctly."""
    print("test_tuning_results_csv:")
    import csv
    tuner = AutoTuner(seed=42)
    result = tuner.tune("scheduling", "FEM", n_trials=2, size=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            tuner._save_results_csv()
            csv_path = Path("build") / "tuning_results.csv"
            assert csv_path.exists(), f"CSV not found: {csv_path}"
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) >= 2, f"Expected >=2 rows, got {len(rows)}"
            print(f"  \u2713 CSV has {len(rows)} rows")
        finally:
            os.chdir(original_cwd)
    print()


if __name__ == "__main__":
    test_auto_tuner_init()
    test_auto_tuner_tune_fast()
    test_auto_tuner_tune_all_problems()
    test_auto_tuner_save_and_load()
    test_tuning_search_space()
    test_tuning_results_csv()
    print("All auto-tuner tests passed!")
