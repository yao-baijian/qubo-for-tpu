"""SBM (Simulated Bifurcation Machine) solver package."""

from .sbm import bsb_torch_batch, bsb_bmincut_batch
import numpy as np


class SbmSolver:
    """SBM-based QUBO solver with standard solve(Q, num_vars) interface.

    Converts QUBO to Ising model internally and uses the SBM solver.

    Usage::
        from src.sbm import SbmSolver

        solver = SbmSolver(num_iters=1000, dt=0.1)
        solution = solver.solve(Q, num_vars)
    """

    def __init__(self, num_iters: int = 1000, dt: float = 0.1,
                 num_trials: int = 10, device: str = 'cpu',
                 lambda_balance: float = 1.0, use_compile: bool = False):
        self.num_iters = num_iters
        self.dt = dt
        self.num_trials = num_trials
        self.device = device
        self.lambda_balance = lambda_balance
        self.use_compile = use_compile

    def solve(self, Q, num_vars):
        """Solve a QUBO problem via SBM.

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

        # Convert QUBO to Ising: x = (s+1)/2
        # x^T Q x = 1/4 * s^T Q s + 1/2 * sum_i (sum_j Q_ij) s_i + const
        # Ising energy: E = -1/2 * s^T J s
        # So J = -Q/2, h_i = -sum_j Q_ij / 4 (linear field)
        J_ising = -Q_mat / 2.0
        # SBM bsb_bmincut_batch works with J directly (no h field needed for
        # balanced cut, but for general QUBO we incorporate the linear terms)
        # We'll use bsb_torch_batch which handles general J.

        init_x = 2 * torch.rand(self.num_trials, num_vars, device=self.device) - 1
        init_y = torch.zeros(self.num_trials, num_vars, device=self.device)

        energies, solutions, _ = bsb_torch_batch(
            J_ising, init_x, init_y, self.num_iters, self.dt,
        )

        # energies shape: (num_trials, num_iters)
        # solutions shape: (num_trials, num_vars)
        # Find best trial
        final_energies = energies[:, -1]
        best_trial = int(torch.argmin(final_energies))
        best_spins = solutions[best_trial].cpu().numpy()  # +1 or -1

        # Convert Ising spins to QUBO binary: x = (s + 1) / 2
        solution = ((best_spins + 1) / 2).astype(int).tolist()
        return solution