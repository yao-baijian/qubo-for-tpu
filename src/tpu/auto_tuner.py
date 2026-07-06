"""Auto-tuning module for TPU QUBO solvers.

Searches for optimal hyperparameters for each solver on each problem type
using Optuna (preferred) or grid search fallback.
"""

from __future__ import annotations

import time
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Tuning search space ──────────────────────────────────────────────────

TUNING_SEARCH_SPACE: Dict[str, Dict[str, list]] = {
    "FEM": {
        "num_steps": [200, 500, 1000, 2000],
        "betamin": [0.001, 0.01, 0.05],
        "betamax": [0.2, 0.5, 1.0],
        "learning_rate": [0.05, 0.1, 0.2],
    },
    "SBM": {
        "num_iters": [200, 500, 1000, 2000],
        "dt": [0.05, 0.1, 0.2, 0.5],
    },
    "QIS3": {
        "num_iters": [200, 500, 1000],
        "dt": [0.05, 0.1, 0.2],
        "branch_depth": [0, 1, 2],
        "popsize": [5, 10, 20],
        "adaptive": [True, False],
    },
}

# Default config dirs
_CONFIG_DIR = Path.cwd() / "config"
_SRC_CONFIG_DIR = Path(__file__).resolve().parents[2] / "src" / "configs"
_TUNED_DIR = _CONFIG_DIR / "tuned"

# Problem sizes used during tuning (small enough to be fast)
_TUNING_SIZES = {
    "scheduling": 10,
    "coloring": 10,
    "partitioning": 10,
    "coverage": 10,
}


# ── Solver parameter mapping ─────────────────────────────────────────────

def _solver_defaults(solver_name: str) -> Dict[str, Any]:
    """Return default parameters for a solver from JSON config."""
    # Try config/ first, then src/configs/
    for cfg_dir in (_CONFIG_DIR, _SRC_CONFIG_DIR):
        path = cfg_dir / f"{solver_name.lower()}.json"
        if path.exists():
            with open(path) as f:
                cfg = json.load(f)
            # Strip non-parameter keys
            return {k: v for k, v in cfg.items()
                    if k not in ("description",)}
    return {}


# ── Optuna / Grid-search helper ──────────────────────────────────────────

_HAS_OPTUNA = False
try:
    import optuna
    _HAS_OPTUNA = True
except ImportError:
    optuna = None  # type: ignore


def _build_synthetic_instance(problem_type: str, size: int) -> Dict[str, Any]:
    """Build a small synthetic instance and return (metadata, Q, num_vars)."""
    from src.tpu.generators import (
        build_scheduling_qubo, build_coloring_qubo,
        build_partitioning_qubo, build_coverage_qubo,
    )

    if problem_type == "scheduling":
        num_proc = max(2, size // 5)
        time_hor = max(10, size * 2)
        inst = {
            "num_ops": size, "num_processors": num_proc,
            "time_horizon": time_hor,
            "exec_time": [random.uniform(1.0, 5.0) for _ in range(size)],
            "comm_cost": [[0.0] * size for _ in range(size)],
            "resource_demand": [random.uniform(0.5, 2.0) for _ in range(size)],
            "proc_capacity": [[random.uniform(4.0, 10.0) for _ in range(time_hor)]
                              for _ in range(num_proc)],
        }
        for u in range(size):
            for v in range(u + 1, size):
                if random.random() < 0.3:
                    w = random.uniform(0.5, 3.0)
                    inst["comm_cost"][u][v] = w
                    inst["comm_cost"][v][u] = w
        Q, n = build_scheduling_qubo(**inst)
        return inst, Q, n

    elif problem_type == "coloring":
        num_t = size
        max_c = max(3, num_t // 4)
        edges = []
        for u in range(num_t):
            for v in range(u + 1, num_t):
                if random.random() < 0.2:
                    edges.append((u, v))
        tsize = [random.uniform(1.0, 10.0) for _ in range(num_t)]
        cap = sum(tsize) / max_c * 1.5
        inst = {"num_tensors": num_t, "max_colors": max_c,
                "conflict_edges": edges, "tensor_size": tsize, "capacity": cap}
        Q, n = build_coloring_qubo(num_t, max_c, edges, tsize, capacity=cap)
        return inst, Q, n

    elif problem_type == "partitioning":
        num_o = size
        max_g = max(2, num_o // 10)
        ew = []
        for u in range(num_o):
            for v in range(u + 1, num_o):
                if random.random() < 0.3:
                    ew.append((u, v, random.uniform(0.1, 5.0)))
        oc = [random.uniform(1.0, 10.0) for _ in range(num_o)]
        inst = {"num_ops": num_o, "max_groups": max_g,
                "edge_weights": ew, "op_cost": oc}
        Q, n = build_partitioning_qubo(num_o, max_g, ew, oc)
        return inst, Q, n

    elif problem_type == "coverage":
        nt = size
        np_pts = size * 3
        cm = [[False] * np_pts for _ in range(nt)]
        for t in range(nt):
            ncov = random.randint(1, max(1, np_pts // 5))
            pts = random.sample(range(np_pts), min(ncov, np_pts))
            for p in pts:
                cm[t][p] = True
        ms = max(2, nt // 5)
        pw = [random.uniform(0.5, 2.0) for _ in range(np_pts)]
        inst = {"num_tests": nt, "num_points": np_pts,
                "coverage_matrix": cm, "max_select": ms,
                "point_weights": pw}
        Q, n = build_coverage_qubo(nt, np_pts, cm, ms, pw)
        return inst, Q, n

    raise ValueError(f"Unknown problem type: {problem_type}")


def _run_solver_with_params(
    solver_name: str, params: Dict[str, Any], Q, num_vars: int,
) -> float:
    """Run a solver with given params and return the QUBO objective value."""
    if solver_name == "FEM":
        from qubo_solver import FemSolver
        solver = FemSolver(
            num_trials=params.get("num_trials", 5),
            num_steps=params.get("num_steps", 500),
            anneal=params.get("anneal", "lin"),
            dev=params.get("dev", "cpu"),
            betamin=params.get("betamin", 0.01),
            betamax=params.get("betamax", 0.5),
            learning_rate=params.get("learning_rate", 0.1),
            manual_grad=params.get("manual_grad", False),
        )
    elif solver_name == "SBM":
        from qubo_solver import SbmSolver
        solver = SbmSolver(
            num_iters=params.get("num_iters", 500),
            dt=params.get("dt", 0.1),
            num_trials=params.get("num_trials", 5),
            lambda_balance=params.get("lambda_balance", 1.0),
        )
    elif solver_name == "QIS3":
        from qubo_solver import Qis3Solver
        solver = Qis3Solver(
            num_iters=params.get("num_iters", 500),
            dt=params.get("dt", 0.1),
            branch_depth=params.get("branch_depth", 1),
            popsize=params.get("popsize", 5),
            adaptive=params.get("adaptive", True),
        )
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    try:
        solution = solver.solve(Q, num_vars)
    except Exception:
        # Solver failed — return worst-case objective
        return 1e12
    # Compute QUBO objective: x^T Q x
    obj = 0.0
    for i, j, val in Q:
        if i == j:
            obj += val * solution[i]
        else:
            obj += val * solution[i] * solution[j]
    return obj


def _decode_metric(problem_type: str, solution, inst) -> float:
    """Decode the primary objective metric from a solution."""
    from src.tpu.benchmark import (
        _decode_scheduling, _decode_coloring,
        _decode_partitioning, _decode_coverage,
    )
    decoders = {
        "scheduling": _decode_scheduling,
        "coloring": _decode_coloring,
        "partitioning": _decode_partitioning,
        "coverage": _decode_coverage,
    }
    decoder = decoders.get(problem_type)
    if decoder is None:
        return 0.0
    metrics = decoder(solution, inst)
    if problem_type == "scheduling":
        return metrics.get("makespan", 0.0)
    elif problem_type == "coloring":
        return metrics.get("colors_used", 0.0)
    elif problem_type == "partitioning":
        return metrics.get("cut_weight", 0.0)
    elif problem_type == "coverage":
        return metrics.get("coverage_pct", 0.0)
    return 0.0


class AutoTuner:
    """Auto-tuner for QUBO solver hyperparameters.

    Usage::

        tuner = AutoTuner()
        best = tuner.tune("scheduling", "FEM", n_trials=10)
        tuner.run_full_tuning(n_trials_per_config=3)
        cfg = tuner.get_best_config("FEM", "scheduling")
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self._results: List[Dict[str, Any]] = []

    @property
    def tuning_results(self) -> List[Dict[str, Any]]:
        return list(self._results)

    # ── Core tuning method ────────────────────────────────────────────────

    def tune(
        self,
        problem_type: str,
        solver_name: str,
        n_trials: int = 20,
        size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Tune a solver on a problem type.

        Parameters
        ----------
        problem_type : str
            One of ``"scheduling"``, ``"coloring"``, ``"partitioning"``,
            ``"coverage"``.
        solver_name : str
            One of ``"FEM"``, ``"SBM"``, ``"QIS3"``.
        n_trials : int
            Number of hyperparameter trials.
        size : int or None
            Instance size for tuning (defaults to ``_TUNING_SIZES``).

        Returns
        -------
        dict
            Best hyperparameters found with ``"objective"`` key.
        """
        if size is None:
            size = _TUNING_SIZES.get(problem_type, 10)

        search_space = TUNING_SEARCH_SPACE.get(solver_name, {})
        if not search_space:
            logger.warning("No search space for %s", solver_name)
            return {"solver": solver_name, "problem_type": problem_type}

        best_params: Optional[Dict[str, Any]] = None
        best_obj = float("inf")

        if _HAS_OPTUNA:
            best_params, best_obj = self._tune_optuna(
                problem_type, solver_name, search_space, n_trials, size,
            )
        else:
            best_params, best_obj = self._tune_grid(
                problem_type, solver_name, search_space, n_trials, size,
            )

        result = {
            "solver": solver_name,
            "problem_type": problem_type,
            "best_objective": best_obj,
            **best_params,
        }
        logger.info(
            "Tuning %s on %s: best objective=%.4f, params=%s",
            solver_name, problem_type, best_obj, best_params,
        )
        return result

    # ── Optuna backend ────────────────────────────────────────────────────

    def _tune_optuna(
        self, problem_type, solver_name, search_space, n_trials, size,
    ) -> Tuple[Dict[str, Any], float]:
        def objective(trial):
            params = {}
            for param_name, values in search_space.items():
                if isinstance(values[0], bool):
                    params[param_name] = trial.suggest_categorical(
                        param_name, values
                    )
                elif all(isinstance(v, int) for v in values):
                    params[param_name] = trial.suggest_int(
                        param_name, min(values), max(values)
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name, float(min(values)), float(max(values))
                    )
            inst, Q, n = _build_synthetic_instance(problem_type, size)
            obj = _run_solver_with_params(solver_name, params, Q, n)
            self._results.append({
                "problem": problem_type,
                "solver": solver_name,
                "trial": trial.number,
                **params,
                "objective": obj,
            })
            return obj

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        study.optimize(objective, n_trials=n_trials)

        return study.best_params, study.best_value

    # ── Grid search fallback ──────────────────────────────────────────────

    def _tune_grid(
        self, problem_type, solver_name, search_space, n_trials, size,
    ) -> Tuple[Dict[str, Any], float]:
        import itertools

        keys = list(search_space.keys())
        value_lists = list(search_space.values())
        all_combos = list(itertools.product(*value_lists))
        random.shuffle(all_combos)
        combos = all_combos[:min(n_trials, len(all_combos))]

        best_params = dict(zip(keys, combos[0]))
        best_obj = float("inf")

        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                inst, Q, n = _build_synthetic_instance(problem_type, size)
                obj = _run_solver_with_params(solver_name, params, Q, n)
            except Exception:
                obj = 1e12
            self._results.append({
                "problem": problem_type,
                "solver": solver_name,
                **params,
                "objective": obj,
            })
            if obj < best_obj:
                best_obj = obj
                best_params = params

        return best_params, best_obj

    # ── Full tuning run ───────────────────────────────────────────────────

    def run_full_tuning(
        self,
        problem_types: Optional[List[str]] = None,
        solver_names: Optional[List[str]] = None,
        n_trials_per_config: int = 3,
    ) -> Dict[str, Dict[str, Any]]:
        """Run tuning for all (problem, solver) combinations.

        Saves best configs to ``config/tuned/{solver}_{problem}.json``.
        """
        if problem_types is None:
            problem_types = ["scheduling", "coloring", "partitioning", "coverage"]
        if solver_names is None:
            solver_names = list(TUNING_SEARCH_SPACE.keys())

        all_best: Dict[str, Dict[str, Any]] = {}
        for solver_name in solver_names:
            for problem_type in problem_types:
                best = self.tune(problem_type, solver_name, n_trials_per_config)
                key = f"{solver_name}_{problem_type}"
                all_best[key] = best
                self._save_best_config(solver_name, problem_type, best)

        # Save tuning results CSV
        self._save_results_csv()

        return all_best

    # ── Config save/load ──────────────────────────────────────────────────

    def _save_best_config(self, solver_name: str, problem_type: str, best: dict):
        """Save best config to ``config/tuned/{solver}_{problem}.json``."""
        _TUNED_DIR.mkdir(parents=True, exist_ok=True)
        path = _TUNED_DIR / f"{solver_name}_{problem_type}.json"
        with open(path, "w") as f:
            json.dump(best, f, indent=2)
        logger.info("Saved tuned config to %s", path)

    def _save_results_csv(self):
        """Save all tuning trial results to ``build/tuning_results.csv``."""
        import csv
        out_dir = Path.cwd() / "build"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "tuning_results.csv"

        if not self._results:
            return

        # Only write trial entries that have an "objective" key
        trial_rows = [r for r in self._results if "objective" in r]
        if not trial_rows:
            return

        fieldnames = list(trial_rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trial_rows)
        logger.info("Tuning results saved to %s (rows=%d)", path, len(trial_rows))

    @staticmethod
    def get_best_config(
        solver_name: str,
        problem_type: str,
    ) -> Dict[str, Any]:
        """Load the best tuned config for a (solver, problem) pair.

        Falls back to the default solver config if no tuned config exists.
        """
        path = _TUNED_DIR / f"{solver_name}_{problem_type}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)

        # Fall back to defaults
        return _solver_defaults(solver_name)


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="TPU Solver Auto-Tuner")
    parser.add_argument("--problem", type=str, default="scheduling",
                        help="Problem type to tune")
    parser.add_argument("--solvers", type=str, default="FEM,SBM,QIS3",
                        help="Comma-separated solver names")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of tuning trials per solver")
    args = parser.parse_args()

    tuner = AutoTuner()
    for solver_name in args.solvers.split(","):
        solver_name = solver_name.strip()
        result = tuner.tune(args.problem, solver_name, n_trials=args.trials)
        print(f"Best for {solver_name} on {args.problem}:")
        for k, v in result.items():
            print(f"  {k}: {v}")
