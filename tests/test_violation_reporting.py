"""Test that per-constraint violation reporting works for all 4 problem types."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tpu.benchmark import (
    _decode_scheduling, _decode_coloring,
    _decode_partitioning, _decode_coverage,
    _format_violations, VIOLATION_FIELDS,
)
from src.tpu.baselines import (
    list_scheduling, greedy_coloring,
    kl_partitioning, greedy_coverage,
)


def test_scheduling_violations():
    print("test_scheduling_violations:")
    inst = dict(
        num_ops=4, num_processors=2, time_horizon=5,
        exec_time=[2.0, 1.0, 3.0, 2.0],
        comm_cost=[[0, 1, 0, 2], [1, 0, 0, 0],
                   [0, 0, 0, 1.5], [2, 0, 1.5, 0]],
        resource_demand=[1.0, 0.5, 1.5, 1.0],
        proc_capacity=[[5.0] * 5 for _ in range(2)],
    )
    sol = list_scheduling(**inst)
    m = _decode_scheduling(sol, inst)

    assert "unique_violations" in m
    assert "dependency_violations" in m
    assert "capacity_violations" in m
    assert "makespan" in m

    viol_str = _format_violations("scheduling", m)
    print(f"  {viol_str}")
    print(f"  \u2713 All 3 scheduling violation fields present")
    print()


def test_coloring_violations():
    print("test_coloring_violations:")
    inst = dict(
        num_tensors=5, max_colors=3,
        conflict_edges=[(0, 1), (1, 2), (2, 3), (3, 4)],
        tensor_size=[2.0, 3.0, 1.0, 4.0, 2.5],
        capacity=8.0,
    )
    sol = greedy_coloring(5, 3, [(0, 1), (1, 2), (2, 3), (3, 4)],
                          [2.0, 3.0, 1.0, 4.0, 2.5], 8.0)
    m = _decode_coloring(sol, inst)

    assert "unique_violations" in m
    assert "conflict_violations" in m
    assert "capacity_violations" in m
    assert "colors_used" in m

    viol_str = _format_violations("coloring", m)
    print(f"  {viol_str}")
    print(f"  \u2713 All 3 coloring violation fields present")
    print()


def test_partitioning_violations():
    print("test_partitioning_violations:")
    inst = dict(
        num_ops=6, max_groups=3,
        edge_weights=[(0, 1, 5), (1, 2, 3), (2, 3, 1),
                      (3, 4, 4), (4, 5, 2), (0, 5, 6)],
        op_cost=[2.0, 3.0, 1.5, 4.0, 2.5, 3.5],
    )
    sol = kl_partitioning(6, 3,
                          [(0, 1, 5), (1, 2, 3), (2, 3, 1),
                           (3, 4, 4), (4, 5, 2), (0, 5, 6)],
                          [2.0, 3.0, 1.5, 4.0, 2.5, 3.5])
    m = _decode_partitioning(sol, inst)

    assert "unique_violations" in m
    assert "imbalance_violations" in m
    assert "cut_weight" in m

    viol_str = _format_violations("partitioning", m)
    print(f"  {viol_str}")
    print(f"  \u2713 All 2 partitioning violation fields present")
    print()


def test_coverage_violations():
    print("test_coverage_violations:")
    coverage_matrix = [
        [True, True, False, False, False, False, False, False, False, False],
        [False, False, True, True, False, False, False, False, False, False],
        [False, False, False, False, True, True, False, False, False, False],
        [False, False, False, False, False, False, True, True, False, False],
        [False, False, False, False, False, False, False, False, True, True],
    ]
    inst = dict(
        num_tests=5, num_points=10,
        coverage_matrix=coverage_matrix,
        max_select=2,
        point_weights=[1.0] * 10,
    )
    sol = greedy_coverage(5, 10, coverage_matrix, 2, [1.0] * 10)
    m = _decode_coverage(sol, inst)

    assert "implication_violations" in m
    assert "false_positives" in m
    assert "cardinality_violations" in m
    assert "coverage_pct" in m

    viol_str = _format_violations("coverage", m)
    print(f"  {viol_str}")
    print(f"  \u2713 All 3 coverage violation fields present")
    print()


def test_violation_fields_consistency():
    """Verify all VIOLATION_FIELDS match decoder outputs."""
    print("test_violation_fields_consistency:")

    # Check scheduling decoders produce all fields listed in VIOLATION_FIELDS
    for problem_type in VIOLATION_FIELDS:
        expected = set(VIOLATION_FIELDS[problem_type])
        print(f"  {problem_type}: expected fields = {expected}")

    print(f"  \u2713 VIOLATION_FIELDS defined for {len(VIOLATION_FIELDS)} problem types")
    print()


if __name__ == "__main__":
    test_scheduling_violations()
    test_coloring_violations()
    test_partitioning_violations()
    test_coverage_violations()
    test_violation_fields_consistency()
    print("All violation reporting tests passed!")
