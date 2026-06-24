#!/usr/bin/env python3
"""TPU Full-Stack Pipeline: End-to-end optimization on real TpuGraphs data.

Reads NLP and XLA graphs from ``benchmarks/v0/npz/``, runs all four
optimization stages (scheduling, coloring, partitioning, coverage) using
both the standard QUBO solver interface and the FEM mean-field interface,
and compares results.
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tpu.data_loader import load_tpugraphs_npz
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
from src.tpu.benchmark import (
    _decode_scheduling, _decode_coloring,
    _decode_partitioning, _decode_coverage,
    _extract_metric,
)
from src.fem import FemSolver
from src.sbm import SbmSolver
from src.qis3 import Qis3Solver
from src.tpu.fem_problem import (
    scheduling_to_fem_problem,
    coloring_to_fem_problem,
    partitioning_to_fem_problem,
    coverage_to_fem_problem,
    TpuFemSolver,
)

# ── Globals (set by CLI) ──────────────────────────────────────────────────
DEVICE = "cpu"

# ── Paths ─────────────────────────────────────────────────────────────────
NPZ_ROOT = PROJECT_ROOT / "benchmarks" / "v0" / "npz"
BUILD_DIR = PROJECT_ROOT / "build"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

_CONFIG_DIR = PROJECT_ROOT / "config"
_SRC_CONFIG_DIR = PROJECT_ROOT / "src" / "configs"
_TUNED_DIR = _CONFIG_DIR / "tuned"


def _ensure_configs():
    """Copy default configs from src/configs/ to config/, overwriting stale ones."""
    import json, shutil
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("fem", "sbm", "qis3"):
        src = _SRC_CONFIG_DIR / f"{name}.json"
        dst = _CONFIG_DIR / f"{name}.json"
        if not src.exists():
            continue
        # Always overwrite if src is newer or dst is missing/stale
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(str(src), str(dst))
            print(f"[config] Synced: {dst}")


def _load_solver_config(solver_name: str, problem_type: str, dev: str) -> dict:
    """Load solver config: tuned > config/ > src/configs/.

    Returns raw config dict without injecting device (handled by caller).
    """
    import json
    cfg: dict = {}

    # 1. Try tuned config first
    tuned_path = _TUNED_DIR / f"{solver_name}_{problem_type}.json"
    if tuned_path.exists():
        with open(tuned_path) as f:
            tuned = json.load(f)
        for k in ("num_steps", "num_iters", "dt", "betamin", "betamax",
                   "learning_rate", "branch_depth", "popsize", "adaptive",
                   "anneal", "num_trials"):
            if k in tuned:
                cfg[k] = tuned[k]
        return cfg

    # 2. Fall back to config/ then src/configs/
    for base in (_CONFIG_DIR, _SRC_CONFIG_DIR):
        path = base / f"{solver_name.lower()}.json"
        if path.exists():
            with open(path) as f:
                cfg = json.load(f)
            return {k: v for k, v in cfg.items() if k != "description"}

    return cfg

# ═══════════════════════════════════════════════════════════════════════════
# 1. Data Discovery
# ═══════════════════════════════════════════════════════════════════════════

def discover_npz_files() -> Dict[str, List[Path]]:
    """Discover all .npz files, grouped by domain (nlp / xla)."""
    groups: Dict[str, List[Path]] = defaultdict(list)
    if not NPZ_ROOT.is_dir():
        print(f"[pipeline] NPZ root not found: {NPZ_ROOT}")
        return groups

    # Layout collection: nlp and xla
    layout_dir = NPZ_ROOT / "layout"
    if layout_dir.is_dir():
        for domain in ("nlp", "xla"):
            domain_dir = layout_dir / domain
            if domain_dir.is_dir():
                for fpath in sorted(domain_dir.rglob("*.npz")):
                    groups[domain].append(fpath)

    print(f"[pipeline] Discovered files: "
          f"nlp={len(groups.get('nlp', []))}, "
          f"xla={len(groups.get('xla', []))}")
    return groups


# ═══════════════════════════════════════════════════════════════════════════
# 2. Stage Runners
# ═══════════════════════════════════════════════════════════════════════════

def _solve_qubo(
    solver_name: str, Q, num_vars: int, dev: str, problem_type: str,
) -> Tuple[List[int], float]:
    """Run a QUBO-path solver with config-driven instantiation."""
    cfg = _load_solver_config(solver_name, problem_type, dev)

    if solver_name == "FEM":
        params = {
            "num_trials": cfg.get("num_trials", 5),
            "num_steps": cfg.get("num_steps", 500),
            "anneal": cfg.get("anneal", "lin"),
            "dev": cfg.get("dev", dev),
            "betamin": cfg.get("betamin", 0.01),
            "betamax": cfg.get("betamax", 0.5),
            "learning_rate": cfg.get("learning_rate", 0.1),
        }
        solver = FemSolver(**params)
    elif solver_name == "SBM":
        params = {
            "num_iters": cfg.get("num_iters", 500),
            "dt": cfg.get("dt", 0.1),
            "num_trials": cfg.get("num_trials", 5),
            "device": cfg.get("device", dev),
        }
        solver = SbmSolver(**params)
    elif solver_name == "QIS3":
        params = {
            "num_iters": cfg.get("num_iters", 500),
            "dt": cfg.get("dt", 0.1),
            "branch_depth": cfg.get("branch_depth", 1),
            "popsize": cfg.get("popsize", 5),
            "adaptive": cfg.get("adaptive", True),
            "device": cfg.get("device", dev),
        }
        solver = Qis3Solver(**params)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    t0 = time.perf_counter()
    sol = solver.solve(Q, num_vars)
    dt = time.perf_counter() - t0
    return sol, dt


def _solve_fem_meanfield(
    problem_type: str, metadata: dict, dev: str,
) -> Tuple[List[int], float]:
    """Run the FEM mean-field path, return (solution, runtime_s)."""
    builder_map = {
        "scheduling": scheduling_to_fem_problem,
        "coloring": coloring_to_fem_problem,
        "partitioning": partitioning_to_fem_problem,
        "coverage": coverage_to_fem_problem,
    }
    builder = builder_map.get(problem_type)
    if builder is None:
        raise ValueError(f"No FEM mean-field builder for {problem_type}")

    # Load FEM config for mean-field params
    cfg = _load_solver_config("FEM", problem_type, dev)

    with_meta = {k: v for k, v in metadata.items()
                 if k not in ("problem_type",)}
    prob = builder(**with_meta)
    solver = TpuFemSolver(
        prob,
        num_trials=cfg.get("num_trials", 5),
        num_steps=cfg.get("num_steps", 500),
        anneal=cfg.get("anneal", "lin"),
        betamin=cfg.get("betamin", 0.01),
        betamax=cfg.get("betamax", 0.5),
        learning_rate=cfg.get("learning_rate", 0.1),
        dev=dev,
    )
    t0 = time.perf_counter()
    sol = solver.solve()
    dt = time.perf_counter() - t0
    return sol, dt


def _run_baseline(problem_type: str, metadata: dict) -> Tuple[List[int], float]:
    """Run the baseline heuristic, return (solution, runtime_s)."""
    t0 = time.perf_counter()
    if problem_type == "scheduling":
        sol = list_scheduling(**metadata)
    elif problem_type == "coloring":
        sol = greedy_coloring(
            metadata["num_tensors"], metadata["max_colors"],
            metadata["conflict_edges"],
            metadata.get("tensor_size"),
            metadata.get("capacity"),
        )
    elif problem_type == "partitioning":
        sol = kl_partitioning(
            metadata["num_ops"], metadata["max_groups"],
            metadata["edge_weights"], metadata["op_cost"],
        )
    elif problem_type == "coverage":
        sol = greedy_coverage(
            metadata["num_tests"], metadata["num_points"],
            metadata["coverage_matrix"], metadata["max_select"],
            metadata.get("point_weights"),
        )
    else:
        raise ValueError(f"Unknown problem: {problem_type}")
    dt = time.perf_counter() - t0
    return sol, dt


# ═══════════════════════════════════════════════════════════════════════════
# 3. Per-Stage Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def _build_qubo(problem_type: str, metadata: dict):
    """Build QUBO from metadata using the standard generators."""
    if problem_type == "scheduling":
        return build_scheduling_qubo(**metadata)
    elif problem_type == "coloring":
        return build_coloring_qubo(
            metadata["num_tensors"], metadata["max_colors"],
            metadata["conflict_edges"], metadata.get("tensor_size"),
            capacity=metadata.get("capacity"),
        )
    elif problem_type == "partitioning":
        return build_partitioning_qubo(
            metadata["num_ops"], metadata["max_groups"],
            metadata["edge_weights"], metadata["op_cost"],
        )
    elif problem_type == "coverage":
        return build_coverage_qubo(
            metadata["num_tests"], metadata["num_points"],
            metadata["coverage_matrix"], metadata["max_select"],
            metadata.get("point_weights"),
        )
    raise ValueError(f"Unknown problem: {problem_type}")


def _decoder_for(problem_type: str):
    """Return the decode function for a problem type."""
    return {
        "scheduling": _decode_scheduling,
        "coloring": _decode_coloring,
        "partitioning": _decode_partitioning,
        "coverage": _decode_coverage,
    }[problem_type]


def run_stage(
    writer,
    problem_type: str,
    metadata: dict,
    source_label: str,
    solver_names: List[str],
    use_fem_mf: bool = True,
    max_nodes_qubo: int = 50,
    dev: str = "cpu",
):
    """Run a single optimization stage with all solvers.

    Parameters
    ----------
    writer : csv.DictWriter
    problem_type : str
        One of ``"scheduling"``, ``"coloring"``, ``"partitioning"``, ``"coverage"``.
    metadata : dict
        Problem metadata compatible with the generator.
    source_label : str
        Human-readable label (e.g. ``"nlp/bert"``).
    solver_names : list of str
        QUBO-path solvers to run.
    use_fem_mf : bool
        Also run the FEM mean-field path.
    max_nodes_qubo : int
        Max nodes for QUBO building (large graphs can explode).
    """
    decoder = _decoder_for(problem_type)

    # Cap large graphs for QUBO building
    num_ops = metadata.get("num_ops", metadata.get("num_tensors", 0))
    if num_ops > max_nodes_qubo:
        print(f"    [skip] too large ({num_ops} nodes > {max_nodes_qubo} limit)")
        return

    # Build QUBO
    try:
        Q, num_vars = _build_qubo(problem_type, metadata)
    except Exception as e:
        print(f"    [skip] QUBO build failed: {e}")
        return

    if num_vars > 5000:
        print(f"    [skip] QUBO too large: {num_vars} vars")
        return

    results: List[dict] = []

    # ── QUBO-path solvers ────────────────────────────────────────────
    for solver_name in solver_names:
        try:
            sol, runtime = _solve_qubo(solver_name, Q, num_vars, dev, problem_type)
            metrics = decoder(sol, metadata)
            quality, mname = _extract_metric(problem_type, metrics)
            results.append({
                "solver": solver_name,
                "path": "qubo",
                "quality": quality,
                "metric": mname,
                "runtime": runtime,
                "constraint_ok": _check_constraints(problem_type, metrics),
            })
        except Exception as e:
            print(f"    {solver_name} QUBO: FAILED ({e})")
            results.append({
                "solver": solver_name, "path": "qubo",
                "quality": -1, "metric": "error", "runtime": -1,
                "constraint_ok": False,
            })

    # ── FEM mean-field path ──────────────────────────────────────────
    if use_fem_mf:
        try:
            sol_mf, runtime_mf = _solve_fem_meanfield(
                problem_type, metadata, dev,
            )
            metrics_mf = decoder(sol_mf, metadata)
            quality_mf, mname_mf = _extract_metric(problem_type, metrics_mf)
            results.append({
                "solver": "FEM-MF",
                "path": "meanfield",
                "quality": quality_mf,
                "metric": mname_mf,
                "runtime": runtime_mf,
                "constraint_ok": _check_constraints(problem_type, metrics_mf),
            })
        except Exception as e:
            print(f"    FEM-MF meanfield: FAILED ({e})")
            results.append({
                "solver": "FEM-MF", "path": "meanfield",
                "quality": -1, "metric": "error", "runtime": -1,
                "constraint_ok": False,
            })

    # ── Baseline ─────────────────────────────────────────────────────
    try:
        sol_bl, runtime_bl = _run_baseline(problem_type, metadata)
        metrics_bl = decoder(sol_bl, metadata)
        quality_bl, mname_bl = _extract_metric(problem_type, metrics_bl)
        results.append({
            "solver": "Baseline",
            "path": "heuristic",
            "quality": quality_bl,
            "metric": mname_bl,
            "runtime": runtime_bl,
            "constraint_ok": _check_constraints(problem_type, metrics_bl),
        })
    except Exception as e:
        print(f"    Baseline: FAILED ({e})")
        results.append({
            "solver": "Baseline", "path": "heuristic",
            "quality": -1, "metric": "error", "runtime": -1,
            "constraint_ok": False,
        })

    # ── Write results ────────────────────────────────────────────────
    for r in results:
        writer.writerow({
            "source": source_label,
            "problem": problem_type,
            "solver": r["solver"],
            "path": r["path"],
            "quality": f"{r['quality']:.4f}",
            "metric": r["metric"],
            "runtime_s": f"{r['runtime']:.6f}",
            "constraints_ok": r["constraint_ok"],
            "num_vars": num_vars,
        })

    # ── Print summary line ───────────────────────────────────────────
    best = min((r for r in results if r["quality"] >= 0),
               key=lambda x: (x["quality"], x["runtime"]),
               default=None)
    if best:
        print(f"    \u2713 {problem_type:12s} | {source_label:30s} | "
              f"best={best['solver']:8s} | {best['metric']}={best['quality']:.2f} | "
              f"{num_vars:5d} vars | {len(results)} solvers")


def _check_constraints(problem_type: str, metrics: dict) -> bool:
    """Check if constraints are satisfied (True = feasible)."""
    if problem_type == "scheduling":
        return metrics.get("unique_violations", 0) == 0
    elif problem_type == "coloring":
        return metrics.get("conflict_violations", 0) == 0
    elif problem_type == "partitioning":
        return metrics.get("unique_violations", 0) == 0
    elif problem_type == "coverage":
        return metrics.get("false_positives", 0) == 0
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 4. Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def _detect_device(dev_arg: str) -> str:
    """Resolve device string: auto-detect CUDA, fall back to cpu."""
    if dev_arg == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                print(f"[device] CUDA detected: {torch.cuda.get_device_name(0)}")
                return "cuda"
            else:
                print("[device] CUDA not available, using cpu")
                return "cpu"
        except Exception:
            return "cpu"
    return dev_arg


def main():
    parser = argparse.ArgumentParser(
        description="TPU Full-Stack Pipeline: End-to-end optimization"
    )
    parser.add_argument("--max-files", type=int, default=5,
                        help="Max .npz files per domain")
    parser.add_argument("--max-nodes", type=int, default=100,
                        help="Max nodes per graph (subsampling)")
    parser.add_argument("--max-nodes-qubo", type=int, default=50,
                        help="Max nodes for QUBO building")
    parser.add_argument("--output", type=str,
                        default=f"build/tpu_pipeline_{datetime.now():%Y%m%d_%H%M%S}.csv",
                        help="Output CSV path")
    parser.add_argument("--solvers", type=str,
                        default="FEM,SBM,QIS3",
                        help="Comma-separated QUBO solvers to run")
    parser.add_argument("--no-fem-mf", action="store_true",
                        help="Skip FEM mean-field path")
    parser.add_argument("--stages", type=str,
                        default="scheduling,coloring,partitioning,coverage",
                        help="Comma-separated optimization stages")
    parser.add_argument("--dev", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device for solver execution (auto=detect CUDA)")
    args = parser.parse_args()

    solver_names = [s.strip() for s in args.solvers.split(",")]
    stages = [s.strip() for s in args.stages.split(",")]

    # ── Device detection ─────────────────────────────────────────────
    global DEVICE
    DEVICE = _detect_device(args.dev)

    # ── Ensure configs exist ─────────────────────────────────────────
    _ensure_configs()
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  TPU Full-Stack Pipeline")
    print(f"  Device: {DEVICE} | Max files: {args.max_files} | Max nodes: {args.max_nodes}")
    print(f"  Stages: {stages}")
    print(f"  Solvers: {solver_names} {'+ FEM-MF' if not args.no_fem_mf else ''}")
    print(f"  Config: tuned={_TUNED_DIR.exists() and any(_TUNED_DIR.iterdir())}")
    print("=" * 70)

    # ── Discover data ─────────────────────────────────────────────────
    npz_groups = discover_npz_files()
    all_instances: List[Tuple[str, str, dict]] = []  # (domain, label, metadata)

    for domain in ("nlp", "xla"):
        files = npz_groups.get(domain, [])[:args.max_files]
        for fpath in files:
            # Build a short label
            rel = fpath.relative_to(NPZ_ROOT)
            label = str(rel.with_suffix(""))  # e.g. "layout/nlp/default/train/bert..."
            label = label.replace("layout/", "").replace("default/", "")

            parsed = load_tpugraphs_npz(str(fpath), max_nodes=args.max_nodes)
            if parsed is None:
                continue
            metadata = parsed["metadata"]
            all_instances.append((domain, label, metadata))

    print(f"\n[pipeline] Loaded {len(all_instances)} instances total\n")

    if not all_instances:
        print("[pipeline] No instances loaded — check data paths.")
        return

    # ── Write CSV header ──────────────────────────────────────────────
    fieldnames = [
        "source", "problem", "solver", "path",
        "quality", "metric", "runtime_s", "constraints_ok", "num_vars",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # ── Run each instance through each stage ──────────────────────
        for domain, label, metadata in all_instances:
            source_label = f"{domain}/{label[:40]}"
            num_ops = metadata.get("num_ops", 0)

            print(f"\n  [{domain}] {label[:60]}")
            print(f"  {'─' * 60}")

            for stage in stages:
                if stage == "scheduling":
                    run_stage(
                        writer, "scheduling", metadata, source_label,
                        solver_names, use_fem_mf=not args.no_fem_mf,
                        max_nodes_qubo=args.max_nodes_qubo,
                        dev=DEVICE,
                    )
                elif stage in ("coloring", "partitioning", "coverage"):
                    derived = _derive_metadata(stage, metadata)
                    if derived:
                        run_stage(
                            writer, stage, derived, source_label,
                            solver_names, use_fem_mf=not args.no_fem_mf,
                            max_nodes_qubo=args.max_nodes_qubo,
                            dev=DEVICE,
                        )
            f.flush()  # flush per-instance for crash safety

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete. Results: {output_path}")
    print(f"{'=' * 70}")
    _print_summary(output_path)


def _derive_metadata(stage: str, scheduling_meta: dict) -> Optional[dict]:
    """Derive a coloring/partitioning/coverage instance from scheduling metadata.

    Uses the graph's node and edge structure to create related problems.
    """
    import random
    n = scheduling_meta.get("num_ops", 0)
    if n < 4:
        return None

    if stage == "coloring":
        # Conflict graph from communication edges
        comm = scheduling_meta.get("comm_cost", [[0.0] * n for _ in range(n)])
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if comm[u][v] > 0:
                    edges.append((u, v))
        K = max(3, n // 4)
        return {
            "num_tensors": n,
            "max_colors": K,
            "conflict_edges": edges,
            "tensor_size": [random.uniform(1.0, 10.0) for _ in range(n)],
            "capacity": n * 5.0 / K * 1.5,
        }

    elif stage == "partitioning":
        # Edge weights from communication matrix
        comm = scheduling_meta.get("comm_cost", [[0.0] * n for _ in range(n)])
        ew = []
        for u in range(n):
            for v in range(u + 1, n):
                if comm[u][v] > 0:
                    ew.append((u, v, comm[u][v]))
        G = max(2, n // 10)
        return {
            "num_ops": n,
            "max_groups": G,
            "edge_weights": ew,
            "op_cost": [random.uniform(1.0, 10.0) for _ in range(n)],
        }

    elif stage == "coverage":
        nt = max(3, n // 2)
        np_pts = n * 2
        cm = [[False] * np_pts for _ in range(nt)]
        for t in range(nt):
            for p in range(t * 2, min((t + 1) * 2, np_pts)):
                cm[t][p] = True
        return {
            "num_tests": nt,
            "num_points": np_pts,
            "coverage_matrix": cm,
            "max_select": max(2, nt // 5),
            "point_weights": [1.0] * np_pts,
        }

    return None


def _print_summary(csv_path: Path):
    """Print a human-readable summary from the CSV."""
    import csv
    if not csv_path.exists():
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    # Group by (source, problem)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["source"], r["problem"])].append(r)

    print("\n  Per-instance results summary:")
    print(f"  {'Source':45s} {'Problem':12s} {'Solver':12s} {'Quality':10s} {'Runtime':10s}")
    print(f"  {'-' * 45} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 10}")
    for key in sorted(groups):
        src, prob = key
        for r in groups[key]:
            if r["solver"] == "Baseline":
                continue  # skip baseline in summary
            qual = r["quality"]
            rt = r["runtime_s"]
            if qual != "-1.0000":
                print(f"  {src[:44]:45s} {prob:12s} {r['solver']:12s} "
                      f"{qual:>10s} {rt:>10s}s")

    # Aggregated stats
    print("\n  Aggregated (across all instances):")
    agg = defaultdict(lambda: {"qual": [], "rt": []})
    for r in rows:
        if r["quality"] != "-1.0000" and r["metric"] != "error":
            key = (r["solver"], r["problem"])
            agg[key]["qual"].append(float(r["quality"]))
            agg[key]["rt"].append(float(r["runtime_s"]))

    print(f"  {'Solver':12s} {'Problem':12s} {'Avg Quality':12s} {'Avg Runtime':12s}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")
    for (sv, pr), vals in sorted(agg.items()):
        if vals["qual"]:
            print(f"  {sv:12s} {pr:12s} "
                  f"{np.mean(vals['qual']):12.4f} {np.mean(vals['rt']):12.6f}s")


if __name__ == "__main__":
    main()
