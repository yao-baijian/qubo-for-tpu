"""TPU Full-Stack Optimization Benchmark.

Generates problem instances of varying sizes, builds QUBO matrices,
solves them with FEM, SBM, and QIS3 solvers, compares against baseline
heuristics, and outputs a CSV summary.
"""

import time
import csv
import os
import sys
import random
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tpu.generators import (
    build_scheduling_qubo,
    build_coloring_qubo,
    build_partitioning_qubo,
    build_coverage_qubo,
)
from src.tpu.baselines import (
    list_scheduling,
    greedy_coloring,
    kl_partitioning,
    greedy_coverage,
)
from src.tpu.data_loader import load_problem_instances

import json

# ── Config-driven solver instantiation ───────────────────────────────────


def _config_dir() -> Path:
    """Return the config directory (config/ or src/configs/ as fallback)."""
    cfg = Path.cwd() / "config"
    if cfg.is_dir():
        return cfg
    return Path(__file__).resolve().parents[2] / "src" / "configs"


def load_solver_config(solver_name: str) -> dict:
    """Load solver config from JSON file.

    Reads ``config/{solver_name}.json`` or falls back to
    ``src/configs/{solver_name}.json``.
    """
    for base in (Path.cwd() / "config",
                  Path(__file__).resolve().parents[2] / "src" / "configs"):
        path = base / f"{solver_name.lower()}.json"
        if path.exists():
            with open(path) as f:
                cfg = json.load(f)
            return {k: v for k, v in cfg.items() if k != "description"}
    return {}


def instantiate_solver(
    solver_name: str,
    config_overrides: Optional[dict] = None,
) -> object:
    """Create a solver instance from config file + optional overrides.

    Parameters
    ----------
    solver_name : str
        One of ``"FEM"``, ``"SBM"``, ``"QIS3"``.
    config_overrides : dict or None
        Keys to override in the loaded config.

    Returns
    -------
    object
        A solver instance with a ``.solve(Q, num_vars)`` method.
    """
    cfg = load_solver_config(solver_name)
    if config_overrides:
        cfg.update(config_overrides)

    name = solver_name.upper()
    if name == "FEM":
        from qubo_solver import FemSolver
        return FemSolver(
            num_trials=cfg.get("num_trials", 5),
            num_steps=cfg.get("num_steps", 500),
            anneal=cfg.get("anneal", "lin"),
            dev=cfg.get("dev", "cpu"),
            betamin=cfg.get("betamin", 0.01),
            betamax=cfg.get("betamax", 0.5),
            learning_rate=cfg.get("learning_rate", 0.1),
            manual_grad=cfg.get("manual_grad", False),
            use_compile=cfg.get("use_compile", False),
        )
    elif name == "SBM":
        from qubo_solver import SbmSolver
        return SbmSolver(
            num_iters=cfg.get("num_iters", 500),
            dt=cfg.get("dt", 0.1),
            num_trials=cfg.get("num_trials", 5),
            lambda_balance=cfg.get("lambda_balance", 1.0),
            use_compile=cfg.get("use_compile", False),
        )
    elif name == "QIS3":
        from qubo_solver import Qis3Solver
        return Qis3Solver(
            num_iters=cfg.get("num_iters", 500),
            dt=cfg.get("dt", 0.1),
            branch_depth=cfg.get("branch_depth", 1),
            popsize=cfg.get("popsize", 5),
            adaptive=cfg.get("adaptive", True),
            device=cfg.get("device", "cpu"),
        )
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


# ═══════════════════════════════════════════════════════════════════════════
# Instance Generators
# ═══════════════════════════════════════════════════════════════════════════

def _make_scheduling_instance(num_ops: int):
    """Create a random scheduling problem instance."""
    num_processors = max(2, num_ops // 5)
    time_horizon = max(10, num_ops * 2)

    exec_time = [random.uniform(1.0, 5.0) for _ in range(num_ops)]
    comm_cost = [[0.0] * num_ops for _ in range(num_ops)]
    # Sparse communication edges
    for u in range(num_ops):
        for v in range(u + 1, num_ops):
            if random.random() < 0.3:
                w = random.uniform(0.5, 3.0)
                comm_cost[u][v] = w
                comm_cost[v][u] = w

    resource_demand = [random.uniform(0.5, 2.0) for _ in range(num_ops)]
    proc_capacity = [[random.uniform(4.0, 10.0) for _ in range(time_horizon)]
                     for _ in range(num_processors)]

    return {
        "num_ops": num_ops,
        "num_processors": num_processors,
        "time_horizon": time_horizon,
        "exec_time": exec_time,
        "comm_cost": comm_cost,
        "resource_demand": resource_demand,
        "proc_capacity": proc_capacity,
    }


def _make_coloring_instance(num_tensors: int):
    """Create a random memory coloring instance."""
    max_colors = max(3, num_tensors // 4)
    conflict_edges = []
    # Sparse conflict graph
    for u in range(num_tensors):
        for v in range(u + 1, num_tensors):
            if random.random() < 0.2:
                conflict_edges.append((u, v))
    tensor_size = [random.uniform(1.0, 10.0) for _ in range(num_tensors)]
    capacity = sum(tensor_size) / max_colors * 1.5  # generous capacity

    return {
        "num_tensors": num_tensors,
        "max_colors": max_colors,
        "conflict_edges": conflict_edges,
        "tensor_size": tensor_size,
        "capacity": capacity,
    }


def _make_partitioning_instance(num_ops: int):
    """Create a random operator fusion (partitioning) instance."""
    max_groups = max(2, num_ops // 10)
    edge_weights = []
    # Sparse edges
    for u in range(num_ops):
        for v in range(u + 1, num_ops):
            if random.random() < 0.3:
                w = random.uniform(0.1, 5.0)
                edge_weights.append((u, v, w))
    op_cost = [random.uniform(1.0, 10.0) for _ in range(num_ops)]

    return {
        "num_ops": num_ops,
        "max_groups": max_groups,
        "edge_weights": edge_weights,
        "op_cost": op_cost,
    }


def _make_coverage_instance(num_tests: int, num_points: Optional[int] = None):
    """Create a random test coverage instance."""
    if num_points is None:
        num_points = num_tests * 3
    coverage_matrix = [[False] * num_points for _ in range(num_tests)]
    # Each test covers a random subset of points
    for t in range(num_tests):
        n_covered = random.randint(1, max(1, num_points // 5))
        points = random.sample(range(num_points), min(n_covered, num_points))
        for p in points:
            coverage_matrix[t][p] = True
    max_select = max(2, num_tests // 5)
    point_weights = [random.uniform(0.5, 2.0) for _ in range(num_points)]

    return {
        "num_tests": num_tests,
        "num_points": num_points,
        "coverage_matrix": coverage_matrix,
        "max_select": max_select,
        "point_weights": point_weights,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Solution Decoders — convert binary QUBO output to human-readable metrics
# ═══════════════════════════════════════════════════════════════════════════

def _decode_scheduling(solution: List[int], inst: dict) -> dict:
    """Decode scheduling QUBO solution into metrics."""
    n_ops = inst["num_ops"]
    n_proc = inst["num_processors"]
    T = inst["time_horizon"]
    exec_time = inst["exec_time"]
    comm_cost = inst["comm_cost"]
    resource_demand = inst["resource_demand"]
    proc_capacity = inst["proc_capacity"]

    def idx(v, p, t):
        return (v * n_proc + p) * T + t

    makespan = 0
    assigned = 0
    # Parse assignment: (proc, time) per operation
    assign_proc = {}
    assign_time = {}
    for v in range(n_ops):
        for p in range(n_proc):
            for t in range(T):
                if solution[idx(v, p, t)] == 1:
                    makespan = max(makespan, t + 1)
                    assigned += 1
                    assign_proc[v] = p
                    assign_time[v] = t

    # ── Constraint 1: Unique assignment ───────────────────────────────
    unique_violations = 0
    for v in range(n_ops):
        count = sum(solution[idx(v, p, t)] for p in range(n_proc) for t in range(T))
        if count != 1:
            unique_violations += abs(count - 1)

    # ── Constraint 2: Dependencies ────────────────────────────────────
    dependency_violations = 0
    for u in range(n_ops):
        if u not in assign_time:
            continue
        for v in range(n_ops):
            if u == v or v not in assign_time:
                continue
            w = comm_cost[u][v]
            if w == 0:
                continue
            min_sep = exec_time[u] + w
            if assign_time[v] < assign_time[u] + min_sep:
                dependency_violations += 1

    # ── Constraint 3: Resource capacity ───────────────────────────────
    capacity_violations = 0
    for p in range(n_proc):
        for t in range(T):
            used = sum(
                resource_demand[v]
                for v in range(n_ops)
                if assign_proc.get(v) == p and assign_time.get(v) == t
            )
            if used > proc_capacity[p][t]:
                capacity_violations += 1

    return {
        "makespan": makespan,
        "assigned_ops": assigned // int(max(1, max(exec_time))),
        "unique_violations": unique_violations,
        "dependency_violations": dependency_violations,
        "capacity_violations": capacity_violations,
    }


def _decode_coloring(solution: List[int], inst: dict) -> dict:
    """Decode coloring QUBO solution into metrics."""
    K = inst["max_colors"]
    n_tensors = inst["num_tensors"]
    n_base = n_tensors * K
    colors_used = sum(solution[n_base + c] for c in range(K))

    # Parse assignment: color per tensor
    color_of = {}
    for v in range(n_tensors):
        for c in range(K):
            if solution[v * K + c] == 1:
                color_of[v] = c
                break

    # ── Constraint 1: Unique color ────────────────────────────────────
    unique_violations = 0
    for v in range(n_tensors):
        count = sum(solution[v * K + c] for c in range(K))
        if count != 1:
            unique_violations += abs(count - 1)

    # ── Constraint 2: Conflict ────────────────────────────────────────
    conflict_violations = 0
    for u, v in inst["conflict_edges"]:
        if u in color_of and v in color_of and color_of[u] == color_of[v]:
            conflict_violations += 1

    # ── Constraint 3: Capacity (if applicable) ────────────────────────
    capacity_violations = 0
    if "tensor_size" in inst and inst["tensor_size"] is not None \
       and "capacity" in inst and inst["capacity"] is not None:
        tensor_size = inst["tensor_size"]
        cap = inst["capacity"]
        for c in range(K):
            total = sum(
                tensor_size[v] for v in range(n_tensors)
                if color_of.get(v) == c
            )
            if total > cap:
                capacity_violations += 1

    return {
        "colors_used": int(colors_used),
        "unique_violations": unique_violations,
        "conflict_violations": conflict_violations,
        "capacity_violations": capacity_violations,
    }


def _decode_partitioning(solution: List[int], inst: dict) -> dict:
    """Decode partitioning QUBO solution into metrics."""
    G = inst["max_groups"]
    n_ops = inst["num_ops"]
    op_cost = inst["op_cost"]

    # Parse assignment: group per operation
    group_of = {}
    for v in range(n_ops):
        for g in range(G):
            if solution[v * G + g] == 1:
                group_of[v] = g
                break

    # ── Constraint 1: Unique group ────────────────────────────────────
    unique_violations = 0
    for v in range(n_ops):
        count = sum(solution[v * G + g] for g in range(G))
        if count != 1:
            unique_violations += abs(count - 1)

    # Compute cut weight
    cut = 0.0
    for u, v, w in inst["edge_weights"]:
        if u in group_of and v in group_of and group_of[u] != group_of[v]:
            cut += w

    # Compute load imbalance
    group_cost = [0.0] * G
    for v in range(n_ops):
        if v in group_of:
            group_cost[group_of[v]] += op_cost[v]
    avg_load = sum(op_cost) / G
    max_imbalance = max(abs(c - avg_load) for c in group_cost) / avg_load if avg_load > 0 else 0

    # ── Constraint 2: Load balancing violation ────────────────────────
    # Count groups whose load deviates more than 50% from average
    imbalance_violations = sum(
        1 for c in group_cost
        if avg_load > 0 and abs(c - avg_load) / avg_load > 0.5
    )

    return {
        "cut_weight": cut,
        "max_imbalance": max_imbalance,
        "unique_violations": unique_violations,
        "imbalance_violations": imbalance_violations,
    }


def _decode_coverage(solution: List[int], inst: dict) -> dict:
    """Decode coverage QUBO solution into metrics."""
    n_tests = inst["num_tests"]
    n_pts = inst["num_points"]
    coverage_matrix = inst["coverage_matrix"]
    max_select = inst["max_select"]

    selected = [solution[t] for t in range(n_tests)]
    covered = [solution[n_tests + p] for p in range(n_pts)]
    n_selected = sum(selected)
    n_covered = sum(covered)

    # Compute actual coverage (points covered by selected tests)
    actual_covered = [False] * n_pts
    for t in range(n_tests):
        if selected[t]:
            for p in range(n_pts):
                if coverage_matrix[t][p]:
                    actual_covered[p] = True
    coverage_pct = sum(actual_covered) / n_pts * 100 if n_pts > 0 else 0

    # ── Constraint 1: Implication (x_t → y_p) ─────────────────────────
    # If test t is selected, point p covered by t must have y_p = 1
    implication_violations = 0
    for t in range(n_tests):
        if selected[t]:
            for p in range(n_pts):
                if coverage_matrix[t][p] and not covered[p]:
                    implication_violations += 1

    # ── Constraint 2: False positive (y_p → ∃ t covering p) ──────────
    false_positives = 0
    for p in range(n_pts):
        if covered[p]:
            covered_by_any = any(coverage_matrix[t][p] and selected[t]
                                 for t in range(n_tests))
            if not covered_by_any:
                false_positives += 1

    # ── Constraint 3: Cardinality (sum x_t == K) ──────────────────────
    cardinality_violations = abs(n_selected - max_select)

    return {
        "tests_selected": n_selected,
        "points_covered": n_covered,
        "coverage_pct": coverage_pct,
        "implication_violations": implication_violations,
        "false_positives": false_positives,
        "cardinality_violations": cardinality_violations,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Violation reporting helper
# ═══════════════════════════════════════════════════════════════════════════

VIOLATION_FIELDS = {
    "scheduling": ["unique_violations", "dependency_violations", "capacity_violations"],
    "coloring": ["unique_violations", "conflict_violations", "capacity_violations"],
    "partitioning": ["unique_violations", "imbalance_violations"],
    "coverage": ["implication_violations", "false_positives", "cardinality_violations"],
}


def _format_violations(problem_name: str, metrics: dict) -> str:
    """Format per-constraint violation counts as a human-readable string."""
    parts = []
    for field in VIOLATION_FIELDS.get(problem_name, []):
        count = metrics.get(field, 0)
        icon = "\u2713" if count == 0 else "\u2717"
        parts.append(f"{icon} {field}={count}")
    return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark runners
# ═══════════════════════════════════════════════════════════════════════════

def _run_solver(solver_fn: Callable, Q, num_vars) -> Tuple[List[int], float]:
    """Run a solver and return (solution, runtime_seconds)."""
    t0 = time.perf_counter()
    solution = solver_fn(Q, num_vars)
    dt = time.perf_counter() - t0
    return solution, dt


def _solve_with(name: str, solver: Callable, Q, num_vars):
    """Wrapper for solver dispatch."""
    return _run_solver(solver, Q, num_vars)


def run_benchmark(
    instance_sizes: Optional[Dict[str, List[int]]] = None,
    data_source: str = "synthetic",
    data_path: Optional[str] = None,
    output_path: str = "build/tpu_benchmark_results.csv",
    num_trials: int = 1,
    verbose: bool = True,
    use_gurobi: bool = False,
    use_tuned: bool = False,
    gurobi_time_limit: float = 30.0,
    config_overrides: Optional[Dict[str, dict]] = None,
    compress: bool = False,
):
    """Run the full TPU benchmark suite.

    Parameters
    ----------
    instance_sizes : dict or None
        Mapping from problem name to list of sizes, e.g.:
        {"scheduling": [10, 50, 100], ...}
        Only used when ``data_source="synthetic"``.
    data_source : str
        One of ``"synthetic"``, ``"tpugraphs"``, ``"hlo_dump"``, ``"mlperf"``.
    data_path : str or None
        Path to data (required for ``tpugraphs`` and ``hlo_dump``).
    output_path : str
        CSV output path.
    num_trials : int
        Number of random trials per instance size.
    verbose : bool
        Print progress to stdout.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Solver names — instantiated per-instance via config
    solver_names = ["FEM", "SBM", "QIS3"]

    # Optional Gurobi
    if use_gurobi:
        try:
            from src.tpu.gurobi_solver import GurobiSolver, is_gurobi_available
            if is_gurobi_available():
                solver_names.append("Gurobi")
            elif verbose:
                print("  [Gurobi not installed, skipping]")
        except ImportError:
            if verbose:
                print("  [Gurobi not installed, skipping]")

    # Pre-load tuned configs if requested
    tuned_configs: Dict[str, dict] = {}
    if use_tuned:
        try:
            from src.tpu.auto_tuner import AutoTuner
            for problem_name in ("scheduling", "coloring", "partitioning", "coverage"):
                for sn in ("FEM", "SBM", "QIS3"):
                    tuned_configs[f"{sn}_{problem_name}"] = \
                        AutoTuner.get_best_config(sn, problem_name)
        except ImportError:
            if verbose:
                print("  [AutoTuner module not available, using defaults]")

    # Baseline mapping
    baselines = {
        "scheduling": ("list_scheduling", list_scheduling),
        "coloring": ("greedy_coloring", greedy_coloring),
        "partitioning": ("kl_partitioning", kl_partitioning),
        "coverage": ("greedy_coverage", greedy_coverage),
    }

    # Decoder mapping
    decoders = {
        "scheduling": _decode_scheduling,
        "coloring": _decode_coloring,
        "partitioning": _decode_partitioning,
        "coverage": _decode_coverage,
    }

    # Collect all violation field names that appear across problem types
    all_violation_fields = sorted(set(
        f for fields in VIOLATION_FIELDS.values() for f in fields
    ))

    fieldnames = [
        "problem", "size", "trial", "solver",
        "runtime_seconds", "solution_quality", "metric_name",
        "optimality_gap",
    ] + all_violation_fields

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        max_vars_gurobi = 5000  # only run Gurobi on small instances

        if data_source == "synthetic" and instance_sizes is not None:
            _run_synthetic_benchmark(
                writer, instance_sizes, solver_names, baselines, decoders,
                num_trials, verbose, use_gurobi, tuned_configs,
                gurobi_time_limit, max_vars_gurobi, config_overrides,
            )
        elif data_source != "synthetic":
            _run_data_source_benchmark(
                writer, data_source, data_path, solver_names, baselines,
                decoders, verbose, use_gurobi, tuned_configs,
                gurobi_time_limit, max_vars_gurobi, config_overrides,
                compress=compress,
            )
        else:
            raise ValueError(
                "instance_sizes required when data_source='synthetic'"
            )

    if verbose:
        print(f"\nResults written to: {output_path}")


def _run_synthetic_benchmark(writer, instance_sizes, solver_names, baselines,
                              decoders, num_trials, verbose,
                              use_gurobi, tuned_configs,
                              gurobi_time_limit, max_vars_gurobi,
                              config_overrides):
    """Run benchmark on synthetic instances (original code path)."""
    for problem_name, sizes in instance_sizes.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Benchmarking: {problem_name}")
            print(f"{'='*60}")

        for size in sizes:
            for trial in range(num_trials):
                if verbose:
                    print(f"  Size={size}, Trial={trial+1}/{num_trials}")

                # ── Generate instance ──────────────────────────────────
                random.seed(trial * 1000 + size)
                np.random.seed(trial * 1000 + size)

                if problem_name == "scheduling":
                    inst = _make_scheduling_instance(size)
                    Q, num_vars = build_scheduling_qubo(**inst)
                elif problem_name == "coloring":
                    inst = _make_coloring_instance(size)
                    Q, num_vars = build_coloring_qubo(
                        inst["num_tensors"], inst["max_colors"],
                        inst["conflict_edges"], inst["tensor_size"],
                        capacity=inst["capacity"],
                    )
                elif problem_name == "partitioning":
                    inst = _make_partitioning_instance(size)
                    Q, num_vars = build_partitioning_qubo(
                        inst["num_ops"], inst["max_groups"],
                        inst["edge_weights"], inst["op_cost"],
                    )
                elif problem_name == "coverage":
                    inst = _make_coverage_instance(size)
                    Q, num_vars = build_coverage_qubo(
                        inst["num_tests"], inst["num_points"],
                        inst["coverage_matrix"], inst["max_select"],
                        inst["point_weights"],
                    )
                else:
                    raise ValueError(f"Unknown problem: {problem_name}")

                if verbose:
                    print(f"    Variables: {num_vars}, Q entries: {len(Q)}")

                _run_solvers_and_baselines(
                    writer, problem_name, inst, size, trial,
                    Q, num_vars, solver_names, baselines, decoders, verbose,
                    use_gurobi, tuned_configs, gurobi_time_limit,
                    max_vars_gurobi, config_overrides,
                )


def _run_data_source_benchmark(writer, data_source, data_path,
                                 solver_names, baselines, decoders, verbose,
                                 use_gurobi, tuned_configs,
                                 gurobi_time_limit, max_vars_gurobi,
                                 config_overrides, compress=False):
    """Run benchmark on instances loaded from a data source."""
    instances = load_problem_instances(
        source=data_source,
        source_path=data_path,
        max_instances=100,
        compress=compress,
    )

    generator_map = {
        "scheduling": build_scheduling_qubo,
        "coloring": build_coloring_qubo,
        "partitioning": build_partitioning_qubo,
        "coverage": build_coverage_qubo,
    }

    for idx, entry in enumerate(instances):
        problem_name = entry["problem_type"]
        meta = entry["metadata"]

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Instance {idx+1}/{len(instances)}: {problem_name}")
            print(f"{'='*60}")

        # Build QUBO from metadata
        gen_fn = generator_map.get(problem_name)
        if gen_fn is None:
            if verbose:
                print(f"  Skipping unsupported problem type: {problem_name}")
            continue

        try:
            if problem_name == "scheduling":
                Q, num_vars = gen_fn(**meta)
            elif problem_name == "coloring":
                Q, num_vars = gen_fn(
                    meta["num_tensors"], meta["max_colors"],
                    meta["conflict_edges"], meta.get("tensor_size"),
                    capacity=meta.get("capacity"),
                )
            elif problem_name == "partitioning":
                Q, num_vars = gen_fn(
                    meta["num_ops"], meta["max_groups"],
                    meta["edge_weights"], meta["op_cost"],
                )
            elif problem_name == "coverage":
                Q, num_vars = gen_fn(
                    meta["num_tests"], meta["num_points"],
                    meta["coverage_matrix"], meta["max_select"],
                    meta.get("point_weights"),
                )
            else:
                continue
        except Exception as e:
            if verbose:
                print(f"  Skipping instance {idx}: QUBO build failed ({e})")
            continue

        if verbose:
            print(f"    Variables: {num_vars}, Q entries: {len(Q)}")

        _run_solvers_and_baselines(
            writer, problem_name, meta, idx, 0,
            Q, num_vars, solver_names, baselines, decoders, verbose,
            use_gurobi, tuned_configs, gurobi_time_limit,
            max_vars_gurobi, config_overrides,
        )


def _run_solvers_and_baselines(writer, problem_name, inst, size, trial,
                                 Q, num_vars, solver_names, baselines,
                                 decoders, verbose,
                                 use_gurobi=False, tuned_configs=None,
                                 gurobi_time_limit=30.0,
                                 max_vars_gurobi=5000,
                                 config_overrides=None):
    """Run all solvers and baselines on a single instance and write results."""
    if tuned_configs is None:
        tuned_configs = {}
    if config_overrides is None:
        config_overrides = {}

    # Collect all violation field names (module-level VIOLATION_FIELDS)
    all_violation_fields = sorted(set(
        f for fields in VIOLATION_FIELDS.values() for f in fields
    ))

    gurobi_obj: Optional[float] = None

    # ── Run QUBO solvers ───────────────────────────────────────────────
    for solver_name in solver_names:
        if solver_name == "Gurobi":
            if num_vars > max_vars_gurobi:
                if verbose:
                    print(f"    Gurobi: skipped ({num_vars} vars > {max_vars_gurobi} limit)")
                continue
            try:
                from src.tpu.gurobi_solver import GurobiSolver
                solver = GurobiSolver(time_limit=gurobi_time_limit, verbose=False)
            except ImportError:
                if verbose:
                    print("    Gurobi: not installed, skipping")
                continue
        else:
            # Config-driven instantiation
            overrides = config_overrides.get(solver_name, {})
            # Optional tuned config
            tuned_key = f"{solver_name}_{problem_name}"
            if tuned_configs and tuned_key in tuned_configs:
                tuned_cfg = tuned_configs[tuned_key]
                # Merge tuned over defaults (tuned values take priority)
                for k, v in tuned_cfg.items():
                    if k not in ("solver", "problem_type", "best_objective"):
                        overrides.setdefault(k, v)
            solver = instantiate_solver(solver_name, config_overrides=overrides)

        try:
            solution, runtime = _solve_with(
                solver_name, solver.solve, Q, num_vars
            )
            metrics = decoders[problem_name](solution, inst)

            quality, metric_name = _extract_metric(problem_name, metrics)

            # Compute optimality gap if Gurobi ran
            gap_str = ""
            if solver_name != "Gurobi" and gurobi_obj is not None and gurobi_obj != 0:
                gap = (quality - gurobi_obj) / abs(gurobi_obj)
                gap_str = f"{gap:.4f}"
            elif solver_name == "Gurobi":
                gurobi_obj = quality
                gap_str = "0.0000"

            # Build row with violation fields
            row = {
                "problem": problem_name,
                "size": size,
                "trial": trial,
                "solver": solver_name,
                "runtime_seconds": f"{runtime:.6f}",
                "solution_quality": f"{quality:.4f}",
                "metric_name": metric_name,
                "optimality_gap": gap_str,
            }
            for vf in all_violation_fields:
                row[vf] = metrics.get(vf, "")
            writer.writerow(row)

            if verbose:
                extra = f"  gap={gap_str}" if gap_str else ""
                viol_str = _format_violations(problem_name, metrics)
                print(f"    {solver_name}: {metric_name}={quality:.4f}, "
                      f"time={runtime:.4f}s{extra}")
                print(f"      violations: {viol_str}")
        except Exception as e:
            if verbose:
                print(f"    {solver_name}: FAILED ({e})")
            row = {
                "problem": problem_name,
                "size": size,
                "trial": trial,
                "solver": solver_name,
                "runtime_seconds": "ERROR",
                "solution_quality": str(e),
                "metric_name": "error",
                "optimality_gap": "",
            }
            for vf in all_violation_fields:
                row[vf] = ""
            writer.writerow(row)

    # ── Run baseline ───────────────────────────────────────────────────
    baseline_name, baseline_fn = baselines[problem_name]
    try:
        t0 = time.perf_counter()
        if problem_name == "scheduling":
            baseline_sol = baseline_fn(**inst)
        elif problem_name == "coloring":
            baseline_sol = baseline_fn(
                inst["num_tensors"], inst["max_colors"],
                inst["conflict_edges"], inst["tensor_size"],
                inst["capacity"],
            )
        elif problem_name == "partitioning":
            baseline_sol = baseline_fn(
                inst["num_ops"], inst["max_groups"],
                inst["edge_weights"], inst["op_cost"],
            )
        elif problem_name == "coverage":
            baseline_sol = baseline_fn(
                inst["num_tests"], inst["num_points"],
                inst["coverage_matrix"], inst["max_select"],
                inst["point_weights"],
            )
        baseline_time = time.perf_counter() - t0
        metrics = decoders[problem_name](baseline_sol, inst)

        quality, metric_name = _extract_metric(problem_name, metrics)

        # Baseline gap vs Gurobi
        gap_str = ""
        if gurobi_obj is not None and gurobi_obj != 0:
            gap = (quality - gurobi_obj) / abs(gurobi_obj)
            gap_str = f"{gap:.4f}"

        row = {
            "problem": problem_name,
            "size": size,
            "trial": trial,
            "solver": baseline_name,
            "runtime_seconds": f"{baseline_time:.6f}",
            "solution_quality": f"{quality:.4f}",
            "metric_name": metric_name,
            "optimality_gap": gap_str,
        }
        for vf in all_violation_fields:
            row[vf] = metrics.get(vf, "")
        writer.writerow(row)

        if verbose:
            extra = f"  gap={gap_str}" if gap_str else ""
            viol_str = _format_violations(problem_name, metrics)
            print(f"    {baseline_name}: {metric_name}={quality:.4f}, "
                  f"time={baseline_time:.4f}s{extra}")
            print(f"      violations: {viol_str}")
    except Exception as e:
        if verbose:
            print(f"    {baseline_name}: FAILED ({e})")
        row = {
            "problem": problem_name,
            "size": size,
            "trial": trial,
            "solver": baseline_name,
            "runtime_seconds": "ERROR",
            "solution_quality": str(e),
            "metric_name": "error",
            "optimality_gap": "",
        }
        for vf in all_violation_fields:
            row[vf] = ""
        writer.writerow(row)


def _extract_metric(problem_name: str, metrics: dict):
    """Extract the primary metric name and value from a decoded metrics dict."""
    if problem_name == "scheduling":
        return metrics["makespan"], "makespan"
    elif problem_name == "coloring":
        return metrics["colors_used"], "colors_used"
    elif problem_name == "partitioning":
        return metrics["cut_weight"], "cut_weight"
    elif problem_name == "coverage":
        return metrics["coverage_pct"], "coverage_pct"
    else:
        return 0.0, "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TPU Full-Stack Optimization Benchmark")
    parser.add_argument("--output", default="build/tpu_benchmark_results.csv",
                        help="Output CSV path")
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of random trials per size")
    parser.add_argument("--sizes", type=str, default="10,50",
                        help="Comma-separated instance sizes (synthetic only)")
    parser.add_argument("--problems", type=str,
                        default="scheduling,coloring,partitioning,coverage",
                        help="Comma-separated problem names (synthetic only)")
    parser.add_argument("--quick", action="store_true",
                        help="Run only the smallest sizes for a quick test")
    parser.add_argument("--data-source", type=str, default="synthetic",
                        choices=["synthetic", "tpugraphs", "hlo_dump", "mlperf"],
                        help="Data source for problem instances")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to data directory (required for tpugraphs/hlo_dump)")
    parser.add_argument("--gurobi", action="store_true",
                        help="Run Gurobi exact solver on small instances")
    parser.add_argument("--gurobi-time-limit", type=float, default=30.0,
                        help="Time limit for Gurobi solver (seconds)")
    parser.add_argument("--tune", action="store_true",
                        help="Run auto-tuning before benchmark")
    parser.add_argument("--tune-trials", type=int, default=5,
                        help="Number of tuning trials per solver (used with --tune)")
    parser.add_argument("--compress", action="store_true",
                        help="Apply degree-1 chain reduction to scheduling graphs")
    args = parser.parse_args()

    # Optional auto-tuning
    if args.tune:
        try:
            from src.tpu.auto_tuner import AutoTuner
            tuner = AutoTuner()
            tuner.run_full_tuning(n_trials_per_config=args.tune_trials)
            print("Auto-tuning complete. Using tuned configs for benchmark.")
        except Exception as e:
            print(f"Auto-tuning failed: {e}")

    if args.data_source == "synthetic":
        if args.quick:
            size_map = {
                "scheduling": [10],
                "coloring": [10],
                "partitioning": [10],
                "coverage": [10],
            }
        else:
            sizes = [int(s) for s in args.sizes.split(",")]
            size_map = {p: sizes for p in args.problems.split(",")}

        run_benchmark(
            instance_sizes=size_map,
            data_source="synthetic",
            output_path=args.output,
            num_trials=args.trials,
            verbose=True,
            use_gurobi=args.gurobi,
            use_tuned=args.tune,
            gurobi_time_limit=args.gurobi_time_limit,
            compress=args.compress,
        )
    else:
        run_benchmark(
            instance_sizes=None,
            data_source=args.data_source,
            data_path=args.data_path,
            output_path=args.output,
            verbose=True,
            use_gurobi=args.gurobi,
            use_tuned=args.tune,
            gurobi_time_limit=args.gurobi_time_limit,
            compress=args.compress,
        )
