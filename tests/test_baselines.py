"""Tests for TPU baseline heuristics."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tpu.baselines import (
    list_scheduling,
    greedy_coloring,
    kl_partitioning,
    greedy_coverage,
)


def _check_binary_vector(vec: list, length: int, name: str):
    """Verify a binary vector of expected length."""
    assert isinstance(vec, list), f"{name} must be a list"
    assert len(vec) == length, f"{name} length: expected {length}, got {len(vec)}"
    assert all(v in (0, 1) for v in vec), f"{name} must be binary (0/1)"
    print(f"  ✓ {name}: length={length}, sum={sum(vec)}")


def test_list_scheduling():
    """Test list scheduling baseline."""
    print("test_list_scheduling:")
    num_ops = 4
    num_processors = 2
    time_horizon = 10
    exec_time = [2.0, 1.0, 3.0, 2.0]
    comm_cost = [
        [0.0, 1.0, 0.0, 2.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.5],
        [2.0, 0.0, 1.5, 0.0],
    ]
    resource_demand = [1.0, 0.5, 1.5, 1.0]
    proc_capacity = [[5.0] * time_horizon for _ in range(num_processors)]

    sol = list_scheduling(
        num_ops, num_processors, time_horizon,
        exec_time, comm_cost, resource_demand, proc_capacity,
    )
    expected_len = num_ops * num_processors * time_horizon
    _check_binary_vector(sol, expected_len, "list_scheduling")

    # At least some ops should be scheduled
    assert sum(sol) > 0, "Expected at least one op scheduled"
    print()


def test_greedy_coloring():
    """Test greedy coloring baseline."""
    print("test_greedy_coloring:")
    num_tensors = 6
    max_colors = 4
    conflict_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    tensor_size = [2.0, 3.0, 1.0, 4.0, 2.5, 1.5]

    sol = greedy_coloring(
        num_tensors, max_colors, conflict_edges,
        tensor_size=tensor_size, capacity=8.0,
    )
    expected_len = num_tensors * max_colors + max_colors
    _check_binary_vector(sol, expected_len, "greedy_coloring")

    # Each tensor should be assigned exactly one color
    K = max_colors
    for v in range(num_tensors):
        assigned = sum(sol[v * K + c] for c in range(K))
        assert assigned <= 1, f"Tensor {v} assigned to {assigned} colors"

    print()


def test_greedy_coloring_no_sizes():
    """Test greedy coloring without sizes."""
    print("test_greedy_coloring_no_sizes:")
    sol = greedy_coloring(4, 3, [(0, 1), (2, 3)])
    expected_len = 4 * 3 + 3
    _check_binary_vector(sol, expected_len, "greedy_coloring")
    print()


def test_kl_partitioning():
    """Test KL partitioning baseline."""
    print("test_kl_partitioning:")
    num_ops = 8
    max_groups = 3
    edge_weights = [
        (0, 1, 5.0), (1, 2, 3.0), (2, 3, 1.0), (3, 4, 4.0),
        (4, 5, 2.0), (5, 6, 6.0), (6, 7, 3.0), (0, 7, 7.0),
        (1, 6, 2.0), (2, 5, 1.0),
    ]
    op_cost = [2.0, 3.0, 1.5, 4.0, 2.5, 3.5, 1.0, 2.5]

    sol = kl_partitioning(num_ops, max_groups, edge_weights, op_cost)
    expected_len = num_ops * max_groups
    _check_binary_vector(sol, expected_len, "kl_partitioning")

    # Each op assigned to exactly one group
    G = max_groups
    for v in range(num_ops):
        assigned = sum(sol[v * G + g] for g in range(G))
        assert assigned == 1, f"Op {v} assigned to {assigned} groups"
    print()


def test_greedy_coverage():
    """Test greedy coverage baseline."""
    print("test_greedy_coverage:")
    num_tests = 5
    num_points = 8
    coverage = [
        [True, True, False, False, False, False, False, False],
        [False, False, True, True, False, False, False, False],
        [False, False, False, False, True, True, False, False],
        [False, False, False, False, False, False, True, True],
        [True, False, True, False, True, False, True, False],
    ]
    max_select = 3

    sol = greedy_coverage(num_tests, num_points, coverage, max_select)
    expected_len = num_tests + num_points
    _check_binary_vector(sol, expected_len, "greedy_coverage")

    # At most max_select tests selected
    selected = sum(sol[:num_tests])
    assert selected <= max_select, f"Selected {selected} tests, max {max_select}"
    print()


if __name__ == "__main__":
    test_list_scheduling()
    test_greedy_coloring()
    test_greedy_coloring_no_sizes()
    test_kl_partitioning()
    test_greedy_coverage()
    print("All baseline tests passed!")
