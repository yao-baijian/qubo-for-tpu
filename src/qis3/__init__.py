"""QIS3 (Quantum-Inspired Solver v3) package."""

from .qis3 import QIS3
import numpy as np


class Qis3Solver:
    """QIS3-based QUBO solver with standard solve(Q, num_vars) interface.

    Converts QUBO to Ising model internally and uses the QIS3 solver
    (SB + Branch & Bound + adaptive perturbation).

    Usage::
        from src.qis3 import Qis3Solver

        solver = Qis3Solver(num_iters=1000, dt=0.1)
        solution = solver.solve(Q, num_vars)
    """

    def __init__(self, num_iters: int = 1000, dt: float = 0.1,
                 sb_type: str = 'bsb', branch_depth: int = 2,
                 popsize: int = 10, adaptive: bool = True,
                 device: str = 'cpu'):
        self.num_iters = num_iters
        self.dt = dt
        self.sb_type = sb_type
        self.branch_depth = branch_depth
        self.popsize = popsize
        self.adaptive = adaptive
        self.device = device

    def solve(self, Q, num_vars):
        """Solve a QUBO problem via QIS3.

        Parameters
        ----------
        Q : list of (int, int, float)
            Sparse upper-triangular QUBO matrix.
        num_vars : int
            Number of binary variables.

        Returns
        -------
        list of int
            Binary solution vector of length num_vars (values 0 or 1).
        """
        import torch

        # Build symmetric QUBO matrix
        Q_mat = torch.zeros(num_vars, num_vars)
        for i, j, val in Q:
            Q_mat[i, j] = val
            if i != j:
                Q_mat[j, i] = val

        # Convert QUBO to Ising
        # x^T Q x = 1/4 * s^T Q s + 1/2 * sum_i (sum_j Q_ij) s_i + const
        # QIS3 uses Ising energy E = -1/2 * s^T J s - h^T s
        # Setting J = -Q/2, we get:
        # E_ising = -1/2 * s^T (-Q/2) s = 1/4 * s^T Q s
        # For the linear terms, we need h_i = -sum_j Q_ij / 4
        # But QIS3 expects J only (sym, diag=0) without h.
        # We absorb h into J by modifying diagonal of J.
        # Actually QIS3 expects the full Ising coupling, not just off-diagonal.

        J_ising = -Q_mat / 2.0

        qis3 = QIS3(
            J=J_ising,
            sb_type=self.sb_type,
            num_iters=self.num_iters,
            dt=self.dt,
            branch_depth=self.branch_depth,
            popsize=self.popsize,
            adaptive=self.adaptive,
            device=self.device,
        )
        best_spins, _ = qis3.solve()  # spins are +1 or -1

        # Convert Ising spins to QUBO binary: x = (s + 1) / 2
        solution = ((best_spins + 1) / 2).astype(int).tolist()
        return solution