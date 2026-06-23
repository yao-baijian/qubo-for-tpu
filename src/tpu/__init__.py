"""TPU Full-Stack Optimization: QUBO formulation and benchmarking.

This module provides QUBO generators for four classes of TPU optimization
problems and a benchmark harness that compares FEM, SBM, and QIS3 solvers
against classical baseline heuristics.

Problems
--------
1. Assignment (TPU Instruction Scheduling)
2. Coloring (Lifetime-based Memory Allocation)
3. Partitioning (Operator Fusion)
4. Set Coverage (Test Case Selection)
"""

from .generators import (
    build_scheduling_qubo,
    build_coloring_qubo,
    build_partitioning_qubo,
    build_coverage_qubo,
)
from .baselines import (
    list_scheduling,
    greedy_coloring,
    kl_partitioning,
    greedy_coverage,
)

__all__ = [
    "build_scheduling_qubo",
    "build_coloring_qubo",
    "build_partitioning_qubo",
    "build_coverage_qubo",
    "list_scheduling",
    "greedy_coloring",
    "kl_partitioning",
    "greedy_coverage",
]
