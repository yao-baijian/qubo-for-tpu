"""Gurobi-based exact QUBO solver.

Provides a ``GurobiSolver`` class with the standard
``solve(Q, num_vars)`` interface.  Falls back gracefully if ``gurobipy``
is not installed.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False
    gp = None  # type: ignore
    GRB = None  # type: ignore


class GurobiSolver:
    """Exact QUBO solver using Gurobi MIQP.

    Solves ``min x^T Q x`` with ``x_i in {0, 1}``.

    Usage::

        from src.tpu.gurobi_solver import GurobiSolver

        solver = GurobiSolver(time_limit=30.0)
        solution = solver.solve(Q, num_vars)  # returns List[int]
    """

    def __init__(
        self,
        time_limit: float = 30.0,
        mip_gap: float = 0.0,
        verbose: bool = False,
        **kwargs,
    ):
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.verbose = verbose
        self.extra_kwargs = kwargs
        self._last_objective: Optional[float] = None
        self._last_gap: Optional[float] = None

    @property
    def last_objective(self) -> Optional[float]:
        """Objective value of the last solved instance."""
        return self._last_objective

    @property
    def last_gap(self) -> Optional[float]:
        """Optimality gap (0 = proven optimal) from the last solve."""
        return self._last_gap

    def solve(self, Q, num_vars: int) -> List[int]:
        """Solve a QUBO problem via Gurobi MIQP.

        Parameters
        ----------
        Q : list of (int, int, float)
            Sparse upper-triangular QUBO matrix.
        num_vars : int
            Number of binary variables.

        Returns
        -------
        list of int
            Binary solution vector (0 or 1).
        """
        if not _HAS_GUROBI:
            raise ImportError(
                "gurobipy is not installed. Install with: pip install gurobipy"
            )

        model = gp.Model("qubo")
        if not self.verbose:
            model.Params.OutputFlag = 0

        model.Params.TimeLimit = self.time_limit
        model.Params.MIPGap = self.mip_gap
        for k, v in self.extra_kwargs.items():
            setattr(model.Params, k, v)

        # Variables
        x = model.addVars(num_vars, vtype=GRB.BINARY, name="x")

        # Objective: sum_{i,j} Q_ij * x_i * x_j
        obj = gp.QuadExpr()
        for i, j, val in Q:
            if i == j:
                obj += val * x[i]
            else:
                obj += val * x[i] * x[j]
        model.setObjective(obj, GRB.MINIMIZE)

        model.optimize()

        # Extract solution
        status = model.Status
        solution = [0] * num_vars
        if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
            for i in range(num_vars):
                solution[i] = int(round(x[i].X))

        self._last_objective = model.ObjVal
        self._last_gap = model.MIPGap if model.SolCount > 0 else None

        model.close()
        return solution


def is_gurobi_available() -> bool:
    """Check whether Gurobi is installed and importable."""
    return _HAS_GUROBI
