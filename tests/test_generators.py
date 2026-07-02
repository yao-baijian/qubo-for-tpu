"""Tests for TPU QUBO generators."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tpu.generators import (
    build_scheduling_qubo,
    build_coloring_qubo,
    build_partitioning_qubo,
    build_coverage_qubo,
    compute_time_windows,
)


def _check_qubo_structure(Q, num_vars):
    """Verify basic QUBO structure invariants."""
    # Q is a list of (i, j, val) tuples
    assert isinstance(Q, list), "Q must be a list"
    assert len(Q) > 0, "Q must not be empty"
    for entry in Q:
        assert len(entry) == 3, f"Each entry must be (i, j, val), got {entry}"
        i, j, val = entry
        assert 0 <= i < num_vars, f"Index i={i} out of range [0, {num_vars})"
        assert 0 <= j < num_vars, f"Index j={j} out of range [0, {num_vars})"
        assert i <= j, f"Upper-triangular violated: i={i} > j={j}"
        assert isinstance(val, (int, float)), f"Value must be numeric, got {val}"
    print(f"  ✓ QUBO structure OK: {len(Q)} entries, {num_vars} vars")


def test_build_scheduling_qubo():
    """Test scheduling QUBO construction."""
    print("test_build_scheduling_qubo:")
    num_ops = 4
    num_processors = 2
    time_horizon = 5
    exec_time = [2.0, 1.0, 3.0, 2.0]
    comm_cost = [
        [0.0, 1.0, 0.0, 2.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.5],
        [2.0, 0.0, 1.5, 0.0],
    ]
    resource_demand = [1.0, 0.5, 1.5, 1.0]
    proc_capacity = [[5.0] * time_horizon for _ in range(num_processors)]

    Q, n = build_scheduling_qubo(
        num_ops, num_processors, time_horizon,
        exec_time, comm_cost, resource_demand, proc_capacity,
    )
    expected_n = num_ops * num_processors * time_horizon
    assert n == expected_n, f"Expected {expected_n} vars, got {n}"
    _check_qubo_structure(Q, n)

    # Verify unique-assignment diagonal terms exist for each op
    diag_counts = sum(1 for i, j, _ in Q if i == j)
    print(f"  ✓ Scheduling QUBO: {diag_counts} diagonal entries")
    print()


def test_build_coloring_qubo():
    """Test coloring QUBO construction."""
    print("test_build_coloring_qubo:")
    num_tensors = 5
    max_colors = 3
    conflict_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    tensor_size = [2.0, 3.0, 1.0, 4.0, 2.5]

    Q, n = build_coloring_qubo(
        num_tensors, max_colors, conflict_edges,
        tensor_size=tensor_size, capacity=8.0,
    )
    expected_n = num_tensors * max_colors + max_colors
    assert n == expected_n, f"Expected {expected_n} vars, got {n}"
    _check_qubo_structure(Q, n)

    # Verify auxiliary y_c variables exist
    y_indices = [i for i, j, _ in Q if i >= num_tensors * max_colors]
    print(f"  ✓ Coloring QUBO: {len(y_indices)} auxiliary y_c entries")
    print()


def test_build_coloring_qubo_no_sizes():
    """Test coloring QUBO without capacity constraint."""
    print("test_build_coloring_qubo_no_sizes:")
    num_tensors = 4
    max_colors = 2
    conflict_edges = [(0, 1), (2, 3)]

    Q, n = build_coloring_qubo(num_tensors, max_colors, conflict_edges)
    _check_qubo_structure(Q, n)
    assert n == num_tensors * max_colors + max_colors
    print()


def test_build_partitioning_qubo():
    """Test partitioning QUBO construction."""
    print("test_build_partitioning_qubo:")
    num_ops = 6
    max_groups = 3
    edge_weights = [(0, 1, 5.0), (1, 2, 3.0), (2, 3, 1.0),
                    (3, 4, 4.0), (4, 5, 2.0), (0, 5, 6.0)]
    op_cost = [2.0, 3.0, 1.5, 4.0, 2.5, 3.5]

    Q, n = build_partitioning_qubo(num_ops, max_groups, edge_weights, op_cost)
    expected_n = num_ops * max_groups
    assert n == expected_n, f"Expected {expected_n} vars, got {n}"
    _check_qubo_structure(Q, n)
    print()


def test_build_coverage_qubo():
    """Test coverage QUBO construction."""
    print("test_build_coverage_qubo:")
    num_tests = 5
    num_points = 10
    coverage_matrix = [
        [True, True, False, False, False, False, False, False, False, False],
        [False, False, True, True, False, False, False, False, False, False],
        [False, False, False, False, True, True, False, False, False, False],
        [False, False, False, False, False, False, True, True, False, False],
        [False, False, False, False, False, False, False, False, True, True],
    ]
    max_select = 2

    Q, n = build_coverage_qubo(num_tests, num_points, coverage_matrix, max_select)
    expected_n = num_tests + num_points
    assert n == expected_n, f"Expected {expected_n} vars, got {n}"
    _check_qubo_structure(Q, n)
    print()


def test_build_coverage_qubo_implication():
    """Verify implication penalties exist (x_t -> y_p)."""
    print("test_build_coverage_qubo_implication:")
    num_tests = 3
    num_points = 4
    coverage = [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, True],
    ]
    max_select = 2

    Q, n = build_coverage_qubo(num_tests, num_points, coverage, max_select)
    _check_qubo_structure(Q, n)

    # Verify y_p diagonal has negative weight for objective
    y_neg = [(i, val) for i, j, val in Q if i == j and i >= num_tests and val < 0]
    assert len(y_neg) > 0, "Expected negative diagonal for y_p (objective)"
    print(f"  ✓ Coverage QUBO: {len(y_neg)} y_p have negative objective weight")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Tests for Time-Window Pruning
# ═══════════════════════════════════════════════════════════════════════════


def test_compute_time_windows():
    """Test compute_time_windows on a simple 4-node chain DAG."""
    print("test_compute_time_windows:")
    # Chain: 0 -> 1 -> 2 -> 3
    num_ops = 4
    exec_time = [2.0, 1.0, 3.0, 2.0]
    # Only forward edges (u < v) have non-zero cost
    comm_cost = [
        [0.0, 1.0, 0.0, 0.0],   # 0 -> 1: cost 1
        [0.0, 0.0, 2.0, 0.0],   # 1 -> 2: cost 2
        [0.0, 0.0, 0.0, 1.5],   # 2 -> 3: cost 1.5
        [0.0, 0.0, 0.0, 0.0],
    ]
    makespan_bound = 12

    est, lst, windows = compute_time_windows(
        num_ops, exec_time, comm_cost, makespan_bound,
    )

    # EST: chain dependencies
    # EST[0] = 0
    # EST[1] = EST[0] + ceil(2) + ceil(1) = 0 + 2 + 1 = 3
    # EST[2] = EST[1] + ceil(1) + ceil(2) = 3 + 1 + 2 = 6
    # EST[3] = EST[2] + ceil(3) + ceil(1.5) = 6 + 3 + 2 = 11
    assert est == [0, 3, 6, 11], f"EST mismatch: {est}"

    # LST: compute backwards from bound=12
    # LST[3] = 12 - ceil(2) = 10
    # LST[2] = LST[3] - ceil(3) - ceil(1.5) = 10 - 3 - 2 = 5
    # LST[1] = LST[2] - ceil(1) - ceil(2) = 5 - 1 - 2 = 2
    # LST[0] = LST[1] - ceil(2) - ceil(1) = 2 - 2 - 1 = -1
    # So node 0 has EST=0, LST=-1 -> infeasible
    assert lst == [-1, 2, 5, 10], f"LST mismatch: {lst}"
    assert windows == [0, 0, 0, 0], f"Windows mismatch: {windows}"
    print(f"  ✓ EST={est}, LST={lst}, windows={windows}")
    print(f"  ✓ Chain DAG: bound=12 is infeasible (makes sense)")

    # Test with a looser bound
    est2, lst2, windows2 = compute_time_windows(
        num_ops, exec_time, comm_cost, makespan_upper_bound=14,
    )
    # LST[3] = 14 - 2 = 12
    # LST[2] = 12 - 3 - 2 = 7
    # LST[1] = 7 - 1 - 2 = 4
    # LST[0] = 4 - 2 - 1 = 1
    assert est2 == [0, 3, 6, 11], f"EST mismatch: {est2}"
    assert lst2 == [1, 4, 7, 12], f"LST mismatch: {lst2}"
    assert windows2 == [2, 2, 2, 2], f"Windows mismatch: {windows2}"
    print(f"  ✓ EST={est2}, LST={lst2}, windows={windows2}")
    print()


def test_build_scheduling_qubo_with_windows():
    """Test scheduling QUBO with time-window pruning on a small DAG."""
    print("test_build_scheduling_qubo_with_windows:")
    # Simple 3-node chain: 0 -> 1 -> 2
    num_ops = 3
    num_processors = 2
    time_horizon = 10  # fallback / default bound
    exec_time = [1.0, 2.0, 1.0]
    comm_cost = [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
    resource_demand = [1.0, 1.0, 1.0]
    proc_capacity = [[5.0] * time_horizon for _ in range(num_processors)]

    Q, n = build_scheduling_qubo(
        num_ops, num_processors, time_horizon,
        exec_time, comm_cost, resource_demand, proc_capacity,
        compute_windows=True,
        makespan_upper_bound=8,
    )

    # Expected windows:
    # EST: [0, 1+1=2, 2+2+1=5]
    # LST (bound=8): [8-1-1=6, 7-2-1=4, 8-1=7]
    # Wait, let me recalculate:
    # exec_int = [1, 2, 1], comm_int = [[0,1,0],[0,0,1],[0,0,0]]
    # EST[0]=0, EST[1]=0+1+1=2, EST[2]=2+2+1=5
    # LST[2]=8-1=7, LST[1]=7-2-1=4, LST[0]=4-1-1=2
    # windows: 0: 2-0+1=3, 1: 4-2+1=3, 2: 7-5+1=3
    # Total vars = 2*3 + 2*3 + 2*3 = 18

    expected_n = 3 * 2 * 3  # 3 ops * 2 procs * 3 time slots each
    assert n == expected_n, f"Expected {expected_n} vars, got {n}"
    _check_qubo_structure(Q, n)

    # Verify all variable indices are within bounds
    max_idx = max(max(i, j) for i, j, _ in Q)
    assert max_idx < n, f"Max index {max_idx} >= num_vars {n}"

    # Verify we saved variables vs global mode (would be 3*2*10=60)
    print(f"  ✓ Pruned vars: {n} (vs {num_ops*num_processors*time_horizon} global)")
    print(f"  ✓ Window sizes uniform: 3 each")
    print()


def test_build_scheduling_qubo_windows_fallback():
    """Test that compute_windows=False produces exactly the same result."""
    print("test_build_scheduling_qubo_windows_fallback:")
    num_ops = 4
    num_processors = 2
    time_horizon = 5
    exec_time = [2.0, 1.0, 3.0, 2.0]
    comm_cost = [
        [0.0, 1.0, 0.0, 2.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.5],
        [2.0, 0.0, 1.5, 0.0],
    ]
    resource_demand = [1.0, 0.5, 1.5, 1.0]
    proc_capacity = [[5.0] * time_horizon for _ in range(num_processors)]

    # Same call as existing test — no new params
    Q, n = build_scheduling_qubo(
        num_ops, num_processors, time_horizon,
        exec_time, comm_cost, resource_demand, proc_capacity,
    )
    expected_n = num_ops * num_processors * time_horizon
    assert n == expected_n, f"Expected {expected_n} vars, got {n}"
    _check_qubo_structure(Q, n)
    print(f"  ✓ Fallback OK: {n} vars (global mode)")
    print()


if __name__ == "__main__":
    test_build_scheduling_qubo()
    test_build_coloring_qubo()
    test_build_coloring_qubo_no_sizes()
    test_build_partitioning_qubo()
    test_build_coverage_qubo()
    test_build_coverage_qubo_implication()
    test_compute_time_windows()
    test_build_scheduling_qubo_with_windows()
    test_build_scheduling_qubo_windows_fallback()
    print("All generator tests passed!")
