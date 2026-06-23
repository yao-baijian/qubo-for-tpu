from .interface import FEM
from .utils import read_graph
from .problem import expected_qubo, manual_grad_qubo, infer_qubo


class FemSolver:
    """FEM-based QUBO solver with standard solve(Q, num_vars) interface.

    Usage::
        from src.fem import FemSolver

        solver = FemSolver(num_trials=10, num_steps=1000)
        solution = solver.solve(Q, num_vars)
    """

    def __init__(self, num_trials: int = 10, num_steps: int = 1000,
                 anneal: str = 'lin', dev: str = 'cpu',
                 manual_grad: bool = False, use_compile: bool = False,
                 learning_rate: float = 0.1, betamin: float = 0.01,
                 betamax: float = 0.5, **kwargs):
        self.num_trials = num_trials
        self.num_steps = num_steps
        self.anneal = anneal
        self.dev = dev
        self.manual_grad = manual_grad
        self.use_compile = use_compile
        self.learning_rate = learning_rate
        self.betamin = betamin
        self.betamax = betamax
        self.extra_kwargs = kwargs

    def solve(self, Q, num_vars):
        """Solve a QUBO problem.

        Parameters
        ----------
        Q : list of (int, int, float)
            Sparse upper-triangular QUBO matrix. Each tuple (i, j, val)
            with i <= j represents Q[i,j] = val.
        num_vars : int
            Number of binary variables.

        Returns
        -------
        list of int
            Binary solution vector of length num_vars (values 0 or 1).
        """
        import torch

        # Build symmetric coupling matrix
        J = torch.zeros(num_vars, num_vars)
        for i, j, val in Q:
            J[i, j] = val
            if i != j:
                J[j, i] = val

        num_interactions = len(Q)

        # Use the 'customize' problem type with QUBO expectation/grad/infer
        case = FEM.from_couplings(
            'customize', num_vars, num_interactions, J,
            customize_expected_func=_qubo_expected,
            customize_grad_func=_qubo_grad,
            customize_infer_func=_qubo_infer,
        )
        case.set_up_solver(
            self.num_trials, self.num_steps,
            betamin=self.betamin, betamax=self.betamax,
            anneal=self.anneal, dev=self.dev,
            q=2, manual_grad=self.manual_grad,
            use_compile=self.use_compile,
            learning_rate=self.learning_rate,
            **self.extra_kwargs,
        )
        configs, results = case.solve()
        best_idx = torch.argwhere(results == results.min()).reshape(-1)[0]
        best_config = configs[best_idx]  # (num_vars, 2) one-hot
        # Column 1 is the probability of being 1
        solution = best_config[:, 1].cpu().numpy().astype(int).tolist()
        return solution


def _reshape_p(p, J):
    """Reshape p to (batch, N, q) if it got flattened by inference_value's vstack."""
    import torch
    if p.dim() == 2:
        # inference_value does vstack: (batch*N, q) → reshape back
        N = J.shape[0]
        batch = p.shape[0] // N
        p = p.reshape(batch, N, p.shape[1])
    return p


def _qubo_expected(J, p):
    """Custom expectation for QUBO with p shape (batch, N, q=2)."""
    import torch
    p = _reshape_p(p, J)
    p1 = p[..., 1]  # probability of being 1, shape (batch, N)
    return torch.bmm(
        (p1 @ J).reshape(-1, 1, J.shape[0]),
        p1.reshape(-1, p1.shape[1], 1),
    ).reshape(-1)


def _qubo_grad(J, p):
    """Custom gradient for QUBO with p shape (batch, N, q=2)."""
    import torch
    p = _reshape_p(p, J)
    p1 = p[..., 1]  # (batch, N)
    # Gradient of p1^T J p1 w.r.t. h (where p1 = sigmoid(h))
    grad_val = 2 * (p1 @ J) * p1 * (1 - p1)  # (batch, N)
    grad = torch.zeros_like(p)
    grad[..., 1] = grad_val
    grad[..., 0] = -grad_val
    return grad


def _qubo_infer(J, p):
    """Custom inference for QUBO with p shape (batch, N, q=2)."""
    import torch
    p = _reshape_p(p, J)
    config = torch.zeros_like(p)
    config[..., 1] = (p[..., 1] > 0.5).float()
    config[..., 0] = 1 - config[..., 1]
    p1 = config[..., 1]
    val = torch.bmm(
        (p1 @ J).reshape(-1, 1, J.shape[0]),
        p1.reshape(-1, p1.shape[1], 1),
    ).reshape(-1)
    return config, val