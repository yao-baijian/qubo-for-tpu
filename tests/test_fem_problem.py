"""Tests for FEM mean-field problem interface."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tpu.fem_problem import (
    MeanFieldProblem,
    TpuFemSolver,
    scheduling_to_fem_problem,
    coloring_to_fem_problem,
    partitioning_to_fem_problem,
    coverage_to_fem_problem,
)


def _check_binary_vector(vec, length):
    assert isinstance(vec, list), f"Expected list, got {type(vec)}"
    assert len(vec) == length, f"Expected length {length}, got {len(vec)}"
    assert all(v in (0, 1) for v in vec), f"Non-binary values: {set(vec)}"
    print(f"  \u2713 Binary vector: length={length}, sum={sum(vec)}")


def test_mean_field_problem_init():
    """Verify MeanFieldProblem construction."""
    print("test_mean_field_problem_init:")
    prob = MeanFieldProblem(num_vars=10)
    assert prob.num_vars == 10
    assert prob.num_terms == 0
    print("  \u2713 MeanFieldProblem created with 0 terms")
    print()


def test_mean_field_problem_add_terms():
    """Verify adding energy and constraint terms."""
    print("test_mean_field_problem_add_terms:")
    prob = MeanFieldProblem(num_vars=5)
    J = torch.eye(5)
    prob.add_energy("test_energy", J, weight=2.0)
    prob.add_constraint(
        "test_constraint",
        expected_func=lambda p: p.sum(dim=1),
        grad_func=lambda p: torch.ones_like(p),
        weight=3.0,
    )
    assert prob.num_terms == 2
    desc = prob.describe()
    assert "test_energy" in desc
    assert "test_constraint" in desc
    print(f"  \u2713 Terms: {prob.num_terms}")
    print(desc)
    print()


def test_mean_field_problem_compute():
    """Verify compute_expected and compute_grad on a simple problem."""
    print("test_mean_field_problem_compute:")
    prob = MeanFieldProblem(num_vars=3)
    J = torch.tensor([[2.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
    prob.add_energy("simple", J, weight=1.0)

    # Test with known p values
    p = torch.tensor([[[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]]])  # (1, 3, 2)
    # p1 = [0.8, 0.5, 0.1]
    # E = 2*0.8^2 + 1*0.5^2 + 1*0.1^2 = 1.28 + 0.25 + 0.01 = 1.54
    energy = prob.compute_expected(p)
    expected_val = 2.0 * 0.8**2 + 1.0 * 0.5**2 + 1.0 * 0.1**2
    assert abs(energy[0].item() - expected_val) < 1e-4, \
        f"Expected {expected_val:.4f}, got {energy[0].item():.4f}"
    print(f"  \u2713 Expected energy: {energy[0].item():.4f} (expected {expected_val:.4f})")

    grad = prob.compute_grad(p)
    assert grad.shape == p.shape
    print(f"  \u2713 Gradient shape: {grad.shape}")
    print()


def test_tpu_fem_solver_scheduling():
    """Solve a tiny scheduling problem via TpuFemSolver."""
    print("test_tpu_fem_solver_scheduling:")
    prob = scheduling_to_fem_problem(
        num_ops=4, num_processors=2, time_horizon=5,
        exec_time=[2.0, 1.0, 3.0, 2.0],
        comm_cost=[
            [0.0, 1.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.5],
            [2.0, 0.0, 1.5, 0.0],
        ],
        resource_demand=[1.0, 0.5, 1.5, 1.0],
        proc_capacity=[[5.0] * 5 for _ in range(2)],
    )
    assert prob.num_vars == 40
    print(f"  \u2713 Problem: {prob.num_vars} vars, {prob.num_terms} terms")

    solver = TpuFemSolver(prob, num_trials=3, num_steps=200, dev="cpu")
    solution = solver.solve()
    _check_binary_vector(solution, 40)
    print()


def test_tpu_fem_solver_coloring():
    """Solve a tiny coloring problem via TpuFemSolver."""
    print("test_tpu_fem_solver_coloring:")
    prob = coloring_to_fem_problem(
        num_tensors=4, max_colors=2,
        conflict_edges=[(0, 1), (2, 3)],
        tensor_size=[2.0, 3.0, 1.0, 4.0],
        capacity=6.0,
    )
    print(f"  \u2713 Problem: {prob.num_vars} vars, {prob.num_terms} terms")
    solver = TpuFemSolver(prob, num_trials=3, num_steps=200, dev="cpu")
    solution = solver.solve()
    n_base = 4 * 2
    expected_len = n_base + 2  # num_tensors * K + K
    _check_binary_vector(solution, expected_len)
    print()


def test_tpu_fem_solver_partitioning():
    """Solve a tiny partitioning problem via TpuFemSolver."""
    print("test_tpu_fem_solver_partitioning:")
    prob = partitioning_to_fem_problem(
        num_ops=4, max_groups=2,
        edge_weights=[(0, 1, 3.0), (1, 2, 2.0), (2, 3, 1.0)],
        op_cost=[2.0, 3.0, 1.5, 2.5],
    )
    print(f"  \u2713 Problem: {prob.num_vars} vars, {prob.num_terms} terms")
    solver = TpuFemSolver(prob, num_trials=3, num_steps=200, dev="cpu")
    solution = solver.solve()
    _check_binary_vector(solution, 8)
    print()


def test_tpu_fem_solver_coverage():
    """Solve a tiny coverage problem via TpuFemSolver."""
    print("test_tpu_fem_solver_coverage:")
    prob = coverage_to_fem_problem(
        num_tests=3, num_points=4,
        coverage_matrix=[
            [True, True, False, False],
            [False, False, True, True],
            [True, False, True, False],
        ],
        max_select=2,
    )
    print(f"  \u2713 Problem: {prob.num_vars} vars, {prob.num_terms} terms")
    solver = TpuFemSolver(prob, num_trials=3, num_steps=200, dev="cpu")
    solution = solver.solve()
    _check_binary_vector(solution, 7)
    print()


def test_describe():
    """Verify the describe() method returns a readable summary."""
    print("test_describe:")
    prob = scheduling_to_fem_problem(
        num_ops=4, num_processors=2, time_horizon=5,
        exec_time=[2.0, 1.0, 3.0, 2.0],
        comm_cost=[[0.0] * 4 for _ in range(4)],
        resource_demand=[1.0] * 4,
        proc_capacity=[[5.0] * 5 for _ in range(2)],
    )
    desc = prob.describe()
    assert "unique_assignment" in desc
    assert "dependency" in desc
    assert "makespan" in desc
    assert "resource_capacity" in desc
    print(f"  \u2713 Describe output:")
    for line in desc.split("\n"):
        print(f"    {line}")
    print()


if __name__ == "__main__":
    test_mean_field_problem_init()
    test_mean_field_problem_add_terms()
    test_mean_field_problem_compute()
    test_describe()
    test_tpu_fem_solver_scheduling()
    test_tpu_fem_solver_coloring()
    test_tpu_fem_solver_partitioning()
    test_tpu_fem_solver_coverage()
    print("All FEM problem tests passed!")
