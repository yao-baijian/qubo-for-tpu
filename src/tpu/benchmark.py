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
from src.fem import FemSolver
from src.sbm import SbmSolver
from src.qis3 import Qis3Solver


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

    def idx(v, p, t):
        return (v * n_proc + p) * T + t

    makespan = 0
    assigned = 0
    for v in range(n_ops):
        for p in range(n_proc):
            for t in range(T):
                if solution[idx(v, p, t)] == 1:
                    makespan = max(makespan, t + 1)
                    assigned += 1
    # Feasibility checks
    conflicts = 0
    for v in range(n_ops):
        count = sum(solution[idx(v, p, t)] for p in range(n_proc) for t in range(T))
        if count != 1:
            conflicts += abs(count - 1)
    return {
        "makespan": makespan,
        "assigned_ops": assigned // int(max(1, max(inst["exec_time"]))),
        "unique_violations": conflicts,
    }


def _decode_coloring(solution: List[int], inst: dict) -> dict:
    """Decode coloring QUBO solution into metrics."""
    K = inst["max_colors"]
    n_base = inst["num_tensors"] * K
    colors_used = sum(solution[n_base + c] for c in range(K))
    conflict_violations = 0
    for u, v in inst["conflict_edges"]:
        for c in range(K):
            if solution[u * K + c] == 1 and solution[v * K + c] == 1:
                conflict_violations += 1
    return {
        "colors_used": int(colors_used),
        "conflict_violations": conflict_violations,
    }


def _decode_partitioning(solution: List[int], inst: dict) -> dict:
    """Decode partitioning QUBO solution into metrics."""
    G = inst["max_groups"]
    n_ops = inst["num_ops"]

    # Compute cut weight
    cut = 0.0
    group_of = {}
    for v in range(n_ops):
        for g in range(G):
            if solution[v * G + g] == 1:
                group_of[v] = g
                break

    for u, v, w in inst["edge_weights"]:
        if u in group_of and v in group_of and group_of[u] != group_of[v]:
            cut += w

    # Compute load imbalance
    group_cost = [0.0] * G
    for v in range(n_ops):
        if v in group_of:
            group_cost[group_of[v]] += inst["op_cost"][v]
    avg_load = sum(inst["op_cost"]) / G
    max_imbalance = max(abs(c - avg_load) for c in group_cost) / avg_load if avg_load > 0 else 0

    # Unique group violations
    violations = 0
    for v in range(n_ops):
        count = sum(solution[v * G + g] for g in range(G))
        if count != 1:
            violations += abs(count - 1)

    return {
        "cut_weight": cut,
        "max_imbalance": max_imbalance,
        "unique_violations": violations,
    }


def _decode_coverage(solution: List[int], inst: dict) -> dict:
    """Decode coverage QUBO solution into metrics."""
    n_tests = inst["num_tests"]
    n_pts = inst["num_points"]

    selected = [solution[t] for t in range(n_tests)]
    covered = [solution[n_tests + p] for p in range(n_pts)]
    n_selected = sum(selected)
    n_covered = sum(covered)

    # Compute actual coverage (points covered by selected tests)
    actual_covered = [False] * n_pts
    for t in range(n_tests):
        if selected[t]:
            for p in range(n_pts):
                if inst["coverage_matrix"][t][p]:
                    actual_covered[p] = True
    coverage_pct = sum(actual_covered) / n_pts * 100 if n_pts > 0 else 0

    # False positive: y_p=1 but no selected test covers it
    false_positives = 0
    for p in range(n_pts):
        if covered[p]:
            covered_by_any = any(inst["coverage_matrix"][t][p] and selected[t]
                                 for t in range(n_tests))
            if not covered_by_any:
                false_positives += 1

    return {
        "tests_selected": n_selected,
        "points_covered": n_covered,
        "coverage_pct": coverage_pct,
        "false_positives": false_positives,
    }


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

    # Solver factory functions
    solvers = {
        "FEM": lambda: FemSolver(num_trials=5, num_steps=500, anneal="lin", dev="cpu"),
        "SBM": lambda: SbmSolver(num_iters=500, dt=0.1, num_trials=5),
        "QIS3": lambda: Qis3Solver(num_iters=500, dt=0.1, branch_depth=1, popsize=5),
    }

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

    fieldnames = [
        "problem", "size", "trial", "solver",
        "runtime_seconds", "solution_quality", "metric_name",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if data_source == "synthetic" and instance_sizes is not None:
            _run_synthetic_benchmark(
                writer, instance_sizes, solvers, baselines, decoders,
                num_trials, verbose,
            )
        elif data_source != "synthetic":
            _run_data_source_benchmark(
                writer, data_source, data_path, solvers, baselines, decoders,
                verbose,
            )
        else:
            raise ValueError(
                "instance_sizes required when data_source='synthetic'"
            )

    if verbose:
        print(f"\nResults written to: {output_path}")


def _run_synthetic_benchmark(writer, instance_sizes, solvers, baselines,
                              decoders, num_trials, verbose):
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
                    Q, num_vars, solvers, baselines, decoders, verbose,
                )


def _run_data_source_benchmark(writer, data_source, data_path,
                                 solvers, baselines, decoders, verbose):
    """Run benchmark on instances loaded from a data source."""
    instances = load_problem_instances(
        source=data_source,
        source_path=data_path,
        max_instances=100,
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
            Q, num_vars, solvers, baselines, decoders, verbose,
        )


def _run_solvers_and_baselines(writer, problem_name, inst, size, trial,
                                 Q, num_vars, solvers, baselines,
                                 decoders, verbose):
    """Run all solvers and baselines on a single instance and write results."""
    # ── Run solvers ────────────────────────────────────────────────────
    for solver_name, solver_factory in solvers.items():
        solver = solver_factory()
        try:
            solution, runtime = _solve_with(
                solver_name, solver.solve, Q, num_vars
            )
            metrics = decoders[problem_name](solution, inst)

            quality, metric_name = _extract_metric(problem_name, metrics)

            writer.writerow({
                "problem": problem_name,
                "size": size,
                "trial": trial,
                "solver": solver_name,
                "runtime_seconds": f"{runtime:.6f}",
                "solution_quality": f"{quality:.4f}",
                "metric_name": metric_name,
            })

            if verbose:
                print(f"    {solver_name}: {metric_name}={quality:.4f}, "
                      f"time={runtime:.4f}s")
        except Exception as e:
            if verbose:
                print(f"    {solver_name}: FAILED ({e})")
            writer.writerow({
                "problem": problem_name,
                "size": size,
                "trial": trial,
                "solver": solver_name,
                "runtime_seconds": "ERROR",
                "solution_quality": str(e),
                "metric_name": "error",
            })

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

        writer.writerow({
            "problem": problem_name,
            "size": size,
            "trial": trial,
            "solver": baseline_name,
            "runtime_seconds": f"{baseline_time:.6f}",
            "solution_quality": f"{quality:.4f}",
            "metric_name": metric_name,
        })

        if verbose:
            print(f"    {baseline_name}: {metric_name}={quality:.4f}, "
                  f"time={baseline_time:.4f}s")
    except Exception as e:
        if verbose:
            print(f"    {baseline_name}: FAILED ({e})")
        writer.writerow({
            "problem": problem_name,
            "size": size,
            "trial": trial,
            "solver": baseline_name,
            "runtime_seconds": "ERROR",
            "solution_quality": str(e),
            "metric_name": "error",
        })


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
    args = parser.parse_args()

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
        )
    else:
        run_benchmark(
            instance_sizes=None,
            data_source=args.data_source,
            data_path=args.data_path,
            output_path=args.output,
            verbose=True,
        )
