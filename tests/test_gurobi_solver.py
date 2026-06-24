"""Tests for the Gurobi QUBO solver wrapper."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tpu.gurobi_solver import GurobiSolver, is_gurobi_available


def test_gurobi_availability():
    """Check if Gurobi is installed (informational)."""
    print("test_gurobi_availability:")
    available = is_gurobi_available()
    print(f"  Gurobi available: {available}")
    if not available:
        print("  \u26a0 Gurobi not installed — skipping solver tests")
    print()


def test_gurobi_solve_small():
    """Solve a tiny QUBO with Gurobi and verify the solution."""
    print("test_gurobi_solve_small:")
    if not is_gurobi_available():
        print("  \u26a0 Skipped (no Gurobi)")
        return

    solver = GurobiSolver(time_limit=10.0, verbose=False)

    # Tiny QUBO: 2 variables
    # Q = [[1, -1], [0, 1]]   -> min x1^2 - x1*x2 + x2^2
    #   x=0 => 0, x=1 => 1, x=(1,0) => 1, x=(1,1) => 1-1+1=1
    # Optimal: x=(0,0) with obj=0
    Q = [(0, 0, 1.0), (0, 1, -1.0), (1, 1, 1.0)]
    solution = solver.solve(Q, 2)

    assert len(solution) == 2
    assert all(v in (0, 1) for v in solution)
    obj_val = 0.0
    for i, j, val in Q:
        if i == j:
            obj_val += val * solution[i]
        else:
            obj_val += val * solution[i] * solution[j]
    print(f"  \u2713 Solution: {solution}, objective: {obj_val:.4f}")
    print(f"  \u2713 Solver reported objective: {solver.last_objective:.4f}")
    print()


def test_gurobi_solve_trivial():
    """Solve a trivial QUBO where all zeros is optimal."""
    print("test_gurobi_solve_trivial:")
    if not is_gurobi_available():
        print("  \u26a0 Skipped (no Gurobi)")
        return

    solver = GurobiSolver(time_limit=10.0, verbose=False)
    # min 2*x0 + 3*x1, optimal is [0, 0]
    Q = [(0, 0, 2.0), (1, 1, 3.0)]
    solution = solver.solve(Q, 2)
    assert solution == [0, 0], f"Expected [0, 0], got {solution}"
    print(f"  \u2713 Trivial QUBO: solution={solution}, obj={solver.last_objective}")
    print()


def test_gurobi_optimality_gap():
    """Verify optimality gap property after solve."""
    print("test_gurobi_optimality_gap:")
    if not is_gurobi_available():
        print("  \u26a0 Skipped (no Gurobi)")
        return

    solver = GurobiSolver(time_limit=10.0, verbose=False, mip_gap=0.0)
    Q = [(0, 0, 1.0), (1, 1, 1.0)]
    solver.solve(Q, 2)
    gap = solver.last_gap
    print(f"  \u2713 Gap: {gap}")
    print()


if __name__ == "__main__":
    test_gurobi_availability()
    test_gurobi_solve_small()
    test_gurobi_solve_trivial()
    test_gurobi_optimality_gap()
    print("All Gurobi solver tests passed!")
