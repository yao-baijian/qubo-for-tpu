"""FEM Mean-Field Problem Interface.

Unlike SBM and QIS3 (which flatten everything into a single QUBO matrix),
FEM natively supports **multi-objective mean-field optimization** where
the total energy is a sum of separate terms:

    H = λ₁·E₁ + λ₂·E₂ + ... + μ₁·C₁ + μ₂·C₂ + ...

- Each **E** term represents a quadratic energy defined by a coupling matrix.
- Each **C** term represents an independent constraint (function-based).
- All terms are combined and fed directly into the FEM optimizer.

Usage::

    from src.tpu.fem_problem import MeanFieldProblem, TpuFemSolver

    # Build problem with separate energy/constraint terms
    prob = MeanFieldProblem(num_vars=100)
    prob.add_energy("assignment", J_unique, weight=10.0)
    prob.add_energy("dependency", J_dep, weight=10.0)
    prob.add_constraint("capacity", my_expected, my_grad, weight=10.0)

    # Solve using FEM mean-field optimization
    solver = TpuFemSolver(prob, num_trials=5, num_steps=500)
    solution = solver.solve()
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


# ── Energy and Constraint Term ────────────────────────────────────────────


class EnergyTerm:
    """A quadratic energy term: ``E(x) = w · x^T @ J @ x``.

    Parameters
    ----------
    name : str
        Label for debugging.
    coupling : torch.Tensor
        Symmetric coupling matrix of shape ``(N, N)``.
    weight : float
        Scalar multiplier (e.g., penalty weight λ).
    """

    def __init__(self, name: str, coupling: torch.Tensor, weight: float = 1.0):
        self.name = name
        self.coupling = coupling
        self.weight = weight


class ConstraintTerm:
    """An independent constraint term: ``C(x) = w · f(x)``.

    Parameters
    ----------
    name : str
        Label for debugging.
    expected_func : callable
        ``expected_func(p1) -> scalar`` — expectation of f under marginal p1.
        ``p1`` has shape ``(batch, N)`` (prob of each variable being 1).
    grad_func : callable
        ``grad_func(p1) -> (batch, N)`` — gradient w.r.t. the marginal h
        (the logit parameter), where ``p1 = sigmoid(h)``.
    weight : float
        Scalar multiplier (e.g., penalty weight μ).
    """

    def __init__(
        self,
        name: str,
        expected_func: Callable,
        grad_func: Callable,
        weight: float = 1.0,
    ):
        self.name = name
        self.expected_func = expected_func
        self.grad_func = grad_func
        self.weight = weight


# ═══════════════════════════════════════════════════════════════════════════
# Mean-Field Problem Builder
# ═══════════════════════════════════════════════════════════════════════════


class MeanFieldProblem:
    """A TPU problem defined as a sum of energy and constraint terms.

    The total objective is::

        H(p) = sum_i  λ_i · E[J_i](p)  +  sum_j  μ_j · C_j(p)

    where each term contributes its expectation under the mean-field
    product distribution ``p = sigmoid(h)``.

    Parameters
    ----------
    num_vars : int
        Number of binary variables.
    """

    def __init__(self, num_vars: int):
        self.num_vars = num_vars
        self._energies: List[EnergyTerm] = []
        self._constraints: List[ConstraintTerm] = []

    # ── Term registration ────────────────────────────────────────────────

    def add_energy(
        self,
        name: str,
        coupling: torch.Tensor,
        weight: float = 1.0,
    ) -> EnergyTerm:
        """Add a quadratic energy term.

        Parameters
        ----------
        name : str
            A label (e.g. ``"unique_assignment"``, ``"dependency"``).
        coupling : torch.Tensor
            Symmetric coupling matrix ``(N, N)``.
        weight : float
            Penalty / objective weight λ.
        """
        term = EnergyTerm(name, coupling, weight)
        self._energies.append(term)
        return term

    def add_constraint(
        self,
        name: str,
        expected_func: Callable,
        grad_func: Callable,
        weight: float = 1.0,
    ) -> ConstraintTerm:
        """Add an independent constraint term.

        Parameters
        ----------
        name : str
            A label (e.g. ``"capacity"``).
        expected_func : (batch, N) -> (batch,)
            Expectation of the constraint under marginal p1.
        grad_func : (batch, N) -> (batch, N)
            Gradient w.r.t. the logit parameter h.
        weight : float
            Penalty weight μ.
        """
        term = ConstraintTerm(name, expected_func, grad_func, weight)
        self._constraints.append(term)
        return term

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def num_terms(self) -> int:
        """Total number of registered terms (energies + constraints)."""
        return len(self._energies) + len(self._constraints)

    def describe(self) -> str:
        """Return a human-readable description of the problem."""
        lines = [f"MeanFieldProblem(num_vars={self.num_vars})"]
        for et in self._energies:
            lines.append(f"  Energy  [{et.name:20s}]  λ={et.weight:.2f}  "
                         f"J shape {tuple(et.coupling.shape)}")
        for ct in self._constraints:
            lines.append(f"  Constraint [{ct.name:20s}]  μ={ct.weight:.2f}")
        return "\n".join(lines)

    # ── Combined expectation and gradient ─────────────────────────────────

    def compute_expected(self, p) -> torch.Tensor:
        """Compute total expected energy ``H(p)``.

        Parameters
        ----------
        p : torch.Tensor
            Marginal probability of shape ``(batch, N, q=2)`` (softmax output)
            or ``(batch, N)`` (sigmoid output, binary case).

        Returns
        -------
        torch.Tensor
            Total energy of shape ``(batch,)``.
        """
        device = p.device
        dtype = p.dtype

        # Extract probability of being 1
        if p.dim() == 3:
            p1 = p[..., 1]  # (batch, N)
        else:
            p1 = p

        total = torch.zeros(p1.shape[0], device=device, dtype=dtype)

        # Energy terms: E = w * p1^T @ J @ p1
        for et in self._energies:
            J = et.coupling.to(device=device, dtype=dtype)
            energy = torch.bmm(
                (p1 @ J).reshape(-1, 1, self.num_vars),
                p1.reshape(-1, self.num_vars, 1),
            ).reshape(-1)
            total = total + et.weight * energy

        # Constraint terms
        for ct in self._constraints:
            val = ct.expected_func(p1)
            total = total + ct.weight * val

        return total

    def compute_grad(self, p) -> torch.Tensor:
        """Compute total gradient ``dH/dh``.

        Parameters
        ----------
        p : torch.Tensor
            Marginal probability (same shape as ``compute_expected``).

        Returns
        -------
        torch.Tensor
            Gradient of the same shape as ``p``.
        """
        device = p.device
        dtype = p.dtype

        if p.dim() == 3:
            batch, N, q = p.shape
            p1 = p[..., 1]
        else:
            batch, N = p.shape
            p1 = p

        # Derivative dp1/dh = p1 * (1 - p1)  (for sigmoid parameterization)
        dp1_dh = p1 * (1 - p1)  # (batch, N)
        # mask to prevent NaN: clamp small values
        dp1_dh = torch.clamp(dp1_dh, min=1e-12)

        # Gradient accumulator in the p1 space
        grad_p1 = torch.zeros(batch, N, device=device, dtype=dtype)

        # Energy terms: dE/dp1 = 2 * J @ p1  (per batch row)
        # Then dE/dh = dE/dp1 * dp1/dh
        for et in self._energies:
            J = et.coupling.to(device=device, dtype=dtype)
            de_dp1 = 2.0 * (p1 @ J)  # (batch, N)
            grad_p1 = grad_p1 + et.weight * de_dp1

        # Constraint terms
        for ct in self._constraints:
            dc_dh = ct.grad_func(p1)  # (batch, N)
            # grad_func returns gradient w.r.t. h directly
            grad_p1 = grad_p1 + ct.weight * dc_dh / torch.clamp(dp1_dh, min=1e-12)

        # Chain rule: dH/dh = dH/dp1 * dp1/dh
        grad_h = grad_p1 * dp1_dh  # (batch, N)

        # Project back to the full parameterization
        if p.dim() == 3:
            grad = torch.zeros(batch, N, q, device=device, dtype=dtype)
            grad[..., 1] = grad_h
            grad[..., 0] = -grad_h
            return grad
        else:
            return grad_h


# ═══════════════════════════════════════════════════════════════════════════
# FEM Solver for Mean-Field Problems
# ═══════════════════════════════════════════════════════════════════════════


class TpuFemSolver:
    """FEM-based solver for :class:`MeanFieldProblem` instances.

    Unlike the generic ``FemSolver.solve(Q, num_vars)`` which flattens
    everything into a single QUBO matrix, this solver leverages FEM's
    native ability to handle **separate energy and constraint terms**
    through multi-objective mean-field optimization.

    Usage::

        prob = MeanFieldProblem(num_vars=100)
        prob.add_energy("assignment", J_assign, weight=10.0)
        prob.add_constraint("capacity", cap_expected, cap_grad, weight=10.0)

        solver = TpuFemSolver(prob, num_steps=500, num_trials=5)
        solution = solver.solve()  # List[int]
    """

    def __init__(
        self,
        problem: MeanFieldProblem,
        num_trials: int = 10,
        num_steps: int = 1000,
        anneal: str = "lin",
        betamin: float = 0.01,
        betamax: float = 0.5,
        learning_rate: float = 0.1,
        dev: str = "cpu",
        manual_grad: bool = True,
        use_compile: bool = False,
        h_factor: float = 0.01,
        seed: int = 1,
        **kwargs,
    ):
        self.problem = problem
        self.num_trials = num_trials
        self.num_steps = num_steps
        self.anneal = anneal
        self.betamin = betamin
        self.betamax = betamax
        self.learning_rate = learning_rate
        self.dev = dev
        self.manual_grad = manual_grad
        self.use_compile = use_compile
        self.h_factor = h_factor
        self.seed = seed
        self.extra_kwargs = kwargs

    def solve(self) -> List[int]:
        """Solve the mean-field problem using FEM optimization.

        Returns
        -------
        list of int
            Binary solution vector of length ``num_vars`` (values 0 or 1).
        """
        N = self.problem.num_vars
        torch.manual_seed(self.seed)

        # Initialise parameters h
        h = self.h_factor * torch.randn(
            self.num_trials, N, 2, device=self.dev,
        )
        h.requires_grad = not self.manual_grad

        # Optimiser
        opt = torch.optim.Adam([h], lr=self.learning_rate)

        # Annealing schedule
        from math import log
        if self.anneal == "lin":
            betas = torch.linspace(self.betamin, self.betamax, self.num_steps)
        elif self.anneal == "exp":
            betas = torch.exp(
                torch.linspace(log(self.betamin), log(self.betamax), self.num_steps)
            )
        elif self.anneal == "inverse":
            betas = 1 / torch.linspace(self.betamax, self.betamin, self.num_steps)
        else:
            betas = torch.linspace(self.betamin, self.betamax, self.num_steps)
        betas = betas.to(self.dev)

        # Mean-field iteration
        for step in range(self.num_steps):
            p = torch.softmax(h, dim=2)  # (batch, N, 2)
            opt.zero_grad()

            energy = self.problem.compute_expected(p)
            entropy = self._entropy(p)
            free_energy = energy - entropy / betas[step]

            if self.manual_grad:
                h.grad = (
                    self.problem.compute_grad(p)
                    - self._entropy_grad(p) / betas[step]
                )
            else:
                free_energy.backward(gradient=torch.ones_like(free_energy))

            opt.step()

        # Infer final configuration
        p_final = torch.softmax(h, dim=2)
        config = torch.zeros_like(p_final)
        config[..., 1] = (p_final[..., 1] > 0.5).float()
        config[..., 0] = 1 - config[..., 1]

        # Find best trial
        p1 = config[..., 1]
        energies = self.problem.compute_expected(p_final)
        best_idx = int(torch.argmin(energies))
        solution = p1[best_idx].cpu().numpy().astype(int).tolist()
        return solution

    # ── Entropy helpers ──────────────────────────────────────────────────

    @staticmethod
    def _entropy(p: torch.Tensor) -> torch.Tensor:
        """Entropy ``-sum p*log(p)`` for shape ``(batch, N, q)``."""
        return -(p * torch.log(torch.clamp(p, min=1e-12))).sum((1, 2))

    @staticmethod
    def _entropy_grad(p: torch.Tensor) -> torch.Tensor:
        """Gradient of entropy w.r.t. the softmax parameter ``h``."""
        log_p = torch.log(torch.clamp(p, min=1e-12))
        return -p * (log_p - (p * log_p).sum(2, keepdim=True))


# ═══════════════════════════════════════════════════════════════════════════
# Builders: Convert QUBO tuples to MeanFieldProblem for each TPU type
# ═══════════════════════════════════════════════════════════════════════════


def scheduling_to_fem_problem(
    num_ops: int,
    num_processors: int,
    time_horizon: int,
    exec_time: List[float],
    comm_cost: List[List[float]],
    resource_demand: List[float],
    proc_capacity: List[List[float]],
    lambda1: float = 10.0,
    lambda2: float = 10.0,
    lambda3: float = 10.0,
) -> MeanFieldProblem:
    """Build a :class:`MeanFieldProblem` for TPU scheduling.

    The objective is decomposed into separate energy/constraint terms:

    - **E1** (Unique Assignment):  λ₁ · (Σx − 1)²  →  coupling from xᵢxⱼ cross terms
    - **E2** (Dependency):         λ₂ · penalty for violating data dependencies
    - **C1** (Resource Capacity):  λ₃ · max(0, used − cap)²  → independent per (p,t)
    - **Objective** (Makespan):    Σ t · x  → diagonal coupling

    This decomposition allows the FEM mean-field optimizer to handle each
    term naturally, without flattening into a single QUBO matrix.
    """
    n = num_ops * num_processors * time_horizon

    def idx(v: int, p: int, t: int) -> int:
        return (v * num_processors + p) * time_horizon + t

    # ── E1: Unique Assignment coupling ────────────────────────────────────
    J_assign = torch.zeros(n, n)
    for v in range(num_ops):
        vars_v = [(p, t) for p in range(num_processors) for t in range(time_horizon)]
        for a in range(len(vars_v)):
            p1, t1 = vars_v[a]
            i1 = idx(v, p1, t1)
            J_assign[i1, i1] += -1.0  # diagonal: -x (from (sum-1)^2 = sum^2 - 2sum + 1)
            for b in range(a + 1, len(vars_v)):
                p2, t2 = vars_v[b]
                i2 = idx(v, p2, t2)
                J_assign[i1, i2] += 2.0  # off-diagonal: 2*x_i*x_j
                J_assign[i2, i1] += 2.0

    # ── E2: Dependency coupling ───────────────────────────────────────────
    J_dep = torch.zeros(n, n)
    for u in range(num_ops):
        for v in range(num_ops):
            if u == v:
                continue
            w = comm_cost[u][v]
            if w == 0:
                continue
            min_sep = exec_time[u] + w
            for p_u in range(num_processors):
                for t_u in range(time_horizon):
                    i_u = idx(u, p_u, t_u)
                    for p_v in range(num_processors):
                        for t_v in range(time_horizon):
                            if t_v < t_u + min_sep:
                                i_v = idx(v, p_v, t_v)
                                J_dep[i_u, i_v] += 1.0
                                J_dep[i_v, i_u] += 1.0  # symmetric

    # ── E3: Makespan objective (diagonal) ─────────────────────────────────
    J_obj = torch.zeros(n, n)
    for v in range(num_ops):
        for p in range(num_processors):
            for t in range(time_horizon):
                J_obj[idx(v, p, t), idx(v, p, t)] += float(t)

    # ── C1: Resource capacity constraint (per processor-time slot) ───────
    def cap_expected(p1):
        """Expected capacity violation."""
        total = torch.zeros(p1.shape[0], device=p1.device, dtype=p1.dtype)
        for p in range(num_processors):
            for t in range(time_horizon):
                cap = proc_capacity[p][t]
                used = torch.zeros_like(total)
                for v in range(num_ops):
                    i = idx(v, p, t)
                    used = used + resource_demand[v] * p1[:, i]
                diff = used - cap
                # Quadratic penalty: sum max(0, diff)^2
                # Soft version: diff^2 (always penalized)
                total = total + diff * diff
        return total

    def cap_grad(p1):
        """Gradient of capacity constraint w.r.t. h."""
        batch = p1.shape[0]
        grad = torch.zeros(batch, n, device=p1.device, dtype=p1.dtype)
        for p in range(num_processors):
            for t in range(time_horizon):
                cap = proc_capacity[p][t]
                used = torch.zeros(batch, device=p1.device, dtype=p1.dtype)
                for v in range(num_ops):
                    i = idx(v, p, t)
                    used = used + resource_demand[v] * p1[:, i]
                diff = used - cap
                for v in range(num_ops):
                    i = idx(v, p, t)
                    # d/dh of diff^2 = 2 * diff * d(diff)/dh
                    # d(diff)/dh = r_v * dp1/dh = r_v * p1*(1-p1)
                    grad[:, i] = grad[:, i] + 2.0 * diff * resource_demand[v]
        return grad

    # ── Build problem ─────────────────────────────────────────────────────
    prob = MeanFieldProblem(n)
    prob.add_energy("unique_assignment", J_assign, weight=lambda1)
    prob.add_energy("dependency", J_dep, weight=lambda2)
    prob.add_energy("makespan", J_obj, weight=1.0)
    prob.add_constraint("resource_capacity", cap_expected, cap_grad, weight=lambda3)
    return prob


def coloring_to_fem_problem(
    num_tensors: int,
    max_colors: int,
    conflict_edges: List[Tuple[int, int]],
    tensor_size: Optional[List[float]] = None,
    lambda1: float = 10.0,
    lambda2: float = 10.0,
    lambda3: float = 10.0,
    capacity: Optional[float] = None,
) -> MeanFieldProblem:
    """Build a :class:`MeanFieldProblem` for memory coloring.

    Terms:
    - **E1** (Unique Color):  λ₁ · (Σ x − 1)² per tensor
    - **E2** (Conflict):      λ₂ · xᵤ xᵥ for conflicting (u,v) sharing a color
    - **C1** (Minimize colors): auxiliary y_c with x ≤ y link
    - **C2** (Capacity):       λ₄ · capacity check per color
    """
    K = max_colors
    n_base = num_tensors * K
    n = n_base + K  # + K auxiliary y_c vars

    def idx_tv(t: int, c: int) -> int:
        return t * K + c

    def idx_y(c: int) -> int:
        return n_base + c

    J_unique = torch.zeros(n, n)
    J_conflict = torch.zeros(n, n)
    J_link = torch.zeros(n, n)

    # E1: Unique color
    for v in range(num_tensors):
        for a in range(K):
            i1 = idx_tv(v, a)
            J_unique[i1, i1] += -1.0
            for b in range(a + 1, K):
                i2 = idx_tv(v, b)
                J_unique[i1, i2] += 2.0
                J_unique[i2, i1] += 2.0

    # E2: Conflict
    for u, v in conflict_edges:
        for c in range(K):
            i1 = idx_tv(u, c)
            i2 = idx_tv(v, c)
            J_conflict[i1, i2] += 1.0
            J_conflict[i2, i1] += 1.0

    # Link: x ≤ y  → λ₃ · x · (1 − y) = λ₃ · (x − xy)
    # Diagonal on x: +λ₃; off-diagonal (x, y): −λ₃
    for v in range(num_tensors):
        for c in range(K):
            i_x = idx_tv(v, c)
            i_y = idx_y(c)
            J_link[i_x, i_x] += 1.0  # λ₃ scales this
            J_link[i_x, i_y] += -1.0
            J_link[i_y, i_x] += -1.0

    # Objective: minimize sum y_c
    J_obj = torch.zeros(n, n)
    for c in range(K):
        J_obj[idx_y(c), idx_y(c)] += 1.0

    prob = MeanFieldProblem(n)
    prob.add_energy("unique_color", J_unique, weight=lambda1)
    prob.add_energy("conflict", J_conflict, weight=lambda2)
    prob.add_energy("link_x_le_y", J_link, weight=lambda3)
    prob.add_energy("minimize_colors", J_obj, weight=1.0)

    # Capacity constraint (if sizes provided)
    if tensor_size is not None and capacity is not None:
        def cap_expected(p1):
            total = torch.zeros(p1.shape[0], device=p1.device, dtype=p1.dtype)
            for c in range(K):
                used = torch.zeros_like(total)
                for v in range(num_tensors):
                    used = used + tensor_size[v] * p1[:, idx_tv(v, c)]
                diff = used - capacity
                total = total + diff * diff
            return total

        def cap_grad(p1):
            batch = p1.shape[0]
            grad = torch.zeros(batch, n, device=p1.device, dtype=p1.dtype)
            for c in range(K):
                used = torch.zeros(batch, device=p1.device, dtype=p1.dtype)
                for v in range(num_tensors):
                    used = used + tensor_size[v] * p1[:, idx_tv(v, c)]
                diff = used - capacity
                for v in range(num_tensors):
                    i = idx_tv(v, c)
                    grad[:, i] = grad[:, i] + 2.0 * diff * tensor_size[v]
            return grad

        prob.add_constraint("capacity", cap_expected, cap_grad, weight=lambda1)
        # Use lambda4 if provided; re-use lambda1 as default for capacity

    return prob


def partitioning_to_fem_problem(
    num_ops: int,
    max_groups: int,
    edge_weights: List[Tuple[int, int, float]],
    op_cost: List[float],
    lambda1: float = 10.0,
    lambda2: float = 10.0,
) -> MeanFieldProblem:
    """Build a :class:`MeanFieldProblem` for operator fusion.

    Terms:
    - **E1** (Unique Group):  λ₁ · (Σ x − 1)² per op
    - **E2** (Cut):           −w_uv · x_u x_v  (minimise inter-group edges)
    - **C1** (Load balance):  λ₂ · (load − L_avg)² per group
    """
    G = max_groups
    n = num_ops * G

    def idx(v: int, g: int) -> int:
        return v * G + g

    J_unique = torch.zeros(n, n)
    J_cut = torch.zeros(n, n)

    # E1: Unique group
    for v in range(num_ops):
        for a in range(G):
            i1 = idx(v, a)
            J_unique[i1, i1] += -1.0
            for b in range(a + 1, G):
                i2 = idx(v, b)
                J_unique[i1, i2] += 2.0
                J_unique[i2, i1] += 2.0

    # E2: Cut minimisation
    for u, v, w in edge_weights:
        for g in range(G):
            i1 = idx(u, g)
            i2 = idx(v, g)
            J_cut[i1, i2] += -w
            J_cut[i2, i1] += -w

    prob = MeanFieldProblem(n)
    prob.add_energy("unique_group", J_unique, weight=lambda1)
    prob.add_energy("cut", J_cut, weight=1.0)

    # C1: Load balancing
    L_avg = sum(op_cost) / G

    def balance_expected(p1):
        total = torch.zeros(p1.shape[0], device=p1.device, dtype=p1.dtype)
        for g in range(G):
            load = torch.zeros_like(total)
            for v in range(num_ops):
                load = load + op_cost[v] * p1[:, idx(v, g)]
            diff = load - L_avg
            total = total + diff * diff
        return total

    def balance_grad(p1):
        batch = p1.shape[0]
        grad = torch.zeros(batch, n, device=p1.device, dtype=p1.dtype)
        for g in range(G):
            load = torch.zeros(batch, device=p1.device, dtype=p1.dtype)
            for v in range(num_ops):
                load = load + op_cost[v] * p1[:, idx(v, g)]
            diff = load - L_avg
            for v in range(num_ops):
                i = idx(v, g)
                grad[:, i] = grad[:, i] + 2.0 * diff * op_cost[v]
        return grad

    prob.add_constraint("load_balance", balance_expected, balance_grad, weight=lambda2)
    return prob


def coverage_to_fem_problem(
    num_tests: int,
    num_points: int,
    coverage_matrix: List[List[bool]],
    max_select: int,
    point_weights: Optional[List[float]] = None,
    lambda1: float = 10.0,
    lambda2: float = 10.0,
    lambda3: float = 10.0,
) -> MeanFieldProblem:
    """Build a :class:`MeanFieldProblem` for test coverage.

    Terms:
    - **E1** (Implication):  λ₁ · x · (1 − y)  (if test covers point)
    - **E2** (No false pos): λ₂ · y · (1 − Σ x)
    - **E3** (Cardinality):  λ₃ · (Σ x − K)²
    - **Objective**:         −w_p · y_p  (maximise weighted coverage)
    """
    n = num_tests + num_points
    if point_weights is None:
        point_weights = [1.0] * num_points

    J_impl = torch.zeros(n, n)
    J_fp = torch.zeros(n, n)
    J_card = torch.zeros(n, n)
    J_obj = torch.zeros(n, n)

    # E1: Implication
    for t in range(num_tests):
        for p in range(num_points):
            if coverage_matrix[t][p]:
                i_x, i_y = t, num_tests + p
                J_impl[i_x, i_x] += 1.0
                J_impl[i_x, i_y] += -1.0
                J_impl[i_y, i_x] += -1.0

    # E2: Prevent false positives
    for p in range(num_points):
        i_y = num_tests + p
        J_fp[i_y, i_y] += 1.0
        covering = [t for t in range(num_tests) if coverage_matrix[t][p]]
        for t in covering:
            i_x = t
            J_fp[i_x, i_y] += -1.0
            J_fp[i_y, i_x] += -1.0

    # E3: Cardinality — λ₃ · (Σ x − K)²
    for t1 in range(num_tests):
        J_card[t1, t1] += -2.0 * max_select  # from -2λK·x
        for t2 in range(t1 + 1, num_tests):
            J_card[t1, t2] += 2.0  # from 2λ·x·x
            J_card[t2, t1] += 2.0

    # Objective: -w_p · y_p
    for p in range(num_points):
        i_y = num_tests + p
        J_obj[i_y, i_y] += -point_weights[p]

    prob = MeanFieldProblem(n)
    prob.add_energy("implication", J_impl, weight=lambda1)
    prob.add_energy("no_false_positives", J_fp, weight=lambda2)
    prob.add_energy("cardinality", J_card, weight=lambda3)
    prob.add_energy("coverage_objective", J_obj, weight=1.0)
    return prob
