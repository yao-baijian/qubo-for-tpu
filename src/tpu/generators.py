"""QUBO generators for four TPU optimization problems.

Each function returns a tuple (Q, num_vars) where Q is a list of
(i, j, value) tuples representing an upper-triangular sparse QUBO matrix,
and num_vars is the number of binary variables.

General Rule
------------
For an upper-triangular sparse matrix Q, represent it as a list of tuples
(i, j, value) where i <= j.  value on diagonal is the linear coefficient
(since x_i^2 = x_i).  value on off-diagonal (i < j) is the quadratic
coefficient for x_i * x_j.
"""

from typing import List, Tuple, Optional

# Type alias for sparse QUBO entries
QuboMatrix = List[Tuple[int, int, float]]


def _add_off_diag(Q: QuboMatrix, i: int, j: int, val: float):
    """Add an off-diagonal entry ensuring upper-triangular (i <= j)."""
    if i == j:
        Q.append((i, i, val))
    elif i < j:
        Q.append((i, j, val))
    else:
        Q.append((j, i, val))


# ═══════════════════════════════════════════════════════════════════════════
# 1. Assignment (TPU Instruction Scheduling)
# ═══════════════════════════════════════════════════════════════════════════

def build_scheduling_qubo(
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
) -> Tuple[QuboMatrix, int]:
    """Build QUBO for TPU instruction scheduling (assignment problem).

    Indexing
    --------
    idx(v, p, t) = (v * num_processors + p) * time_horizon + t
    Total vars n = num_ops * num_processors * time_horizon

    Parameters
    ----------
    num_ops : int
        Number of operations to schedule.
    num_processors : int
        Number of available processors.
    time_horizon : int
        Number of time steps.
    exec_time : list of float
        exec_time[v] = execution time of operation v.
    comm_cost : list of list of float
        comm_cost[u][v] = communication cost between op u and op v.
    resource_demand : list of float
        resource_demand[v] = resource units required by op v.
    proc_capacity : list of list of float
        proc_capacity[p][t] = capacity of processor p at time t.
    lambda1, lambda2, lambda3 : float
        Penalty weights for constraints.

    Returns
    -------
    Q : list of (int, int, float)
        Sparse upper-triangular QUBO matrix.
    num_vars : int
        Number of binary variables.
    """
    n = num_ops * num_processors * time_horizon
    Q: QuboMatrix = []
    _add = Q.append

    def idx(v: int, p: int, t: int) -> int:
        return (v * num_processors + p) * time_horizon + t

    # ── Step A: Unique Assignment (λ1) ────────────────────────────────────
    # penalty = λ1 * (sum_{p,t} x - 1)^2
    for v in range(num_ops):
        vars_v = [(p, t) for p in range(num_processors) for t in range(time_horizon)]
        m = len(vars_v)
        for a in range(m):
            p1, t1 = vars_v[a]
            i1 = idx(v, p1, t1)
            # Diagonal: -2*λ1 (from expanding (sum-1)^2 = sum^2 - 2*sum + 1)
            # sum^2 contributes +1 on diagonal, -2*sum contributes -2 -> net -2*λ1
            _add((i1, i1, -lambda1))
            for b in range(a + 1, m):
                p2, t2 = vars_v[b]
                i2 = idx(v, p2, t2)
                # Off-diagonal: +2*λ1 for cross terms from sum^2
                _add((i1, i2, 2.0 * lambda1))

    # ── Step B: Dependencies (λ2) ─────────────────────────────────────────
    # For edge (u,v), if t_v < t_u + exec_time[u] + comm_cost[u][v], penalize.
    for u in range(num_ops):
        for v in range(num_ops):
            if u == v:
                continue
            w = comm_cost[u][v]
            if w == 0:
                continue
            min_separation = exec_time[u] + w
            for p_u in range(num_processors):
                for t_u in range(time_horizon):
                    i_u = idx(u, p_u, t_u)
                    for p_v in range(num_processors):
                        for t_v in range(time_horizon):
                            if t_v < t_u + min_separation:
                                i_v = idx(v, p_v, t_v)
                                _add_off_diag(Q, i_u, i_v, lambda2)

    # ── Step C: Resource Capacity (λ3) ────────────────────────────────────
    # For each (p, t), used = sum_v r_v * x_{v,p,t}
    # Penalty = λ3 * max(0, used - cap)^2 = λ3 * (used - cap)^2 when used > cap
    # We use a soft penalty: λ3 * (sum_v r_v * x_{v,p,t} - cap)^2
    for p in range(num_processors):
        for t in range(time_horizon):
            cap = proc_capacity[p][t]
            vars_pt = [v for v in range(num_ops)]
            for a in range(len(vars_pt)):
                v1 = vars_pt[a]
                i1 = idx(v1, p, t)
                # Diagonal: -2*λ3*cap*r_v
                _add((i1, i1, -2.0 * lambda3 * cap * resource_demand[v1]))
                for b in range(a + 1, len(vars_pt)):
                    v2 = vars_pt[b]
                    i2 = idx(v2, p, t)
                    # Off-diagonal: +2*λ3*r_v1*r_v2
                    _add((i1, i2, 2.0 * lambda3 * resource_demand[v1] * resource_demand[v2]))

    # ── Step D: Objective (Makespan surrogate) ────────────────────────────
    # Add t to diagonal to encourage early scheduling
    for v in range(num_ops):
        for p in range(num_processors):
            for t in range(time_horizon):
                i = idx(v, p, t)
                _add((i, i, float(t)))

    return Q, n


# ═══════════════════════════════════════════════════════════════════════════
# 2. Coloring (Lifetime-based Memory Allocation)
# ═══════════════════════════════════════════════════════════════════════════

def build_coloring_qubo(
    num_tensors: int,
    max_colors: int,
    conflict_edges: List[Tuple[int, int]],
    tensor_size: Optional[List[float]] = None,
    lambda1: float = 10.0,
    lambda2: float = 10.0,
    lambda3: float = 10.0,
    lambda4: float = 10.0,
    capacity: Optional[float] = None,
) -> Tuple[QuboMatrix, int]:
    """Build QUBO for lifetime-based memory allocation (coloring problem).

    Indexing
    --------
    idx(v, c) = v * K + c  for tensor v, color c.
    n = num_tensors * K + K (latter K are auxiliary y_c variables).

    Parameters
    ----------
    num_tensors : int
        Number of tensors to assign.
    max_colors : int
        Maximum number of colors (memory blocks) K.
    conflict_edges : list of (int, int)
        Pairs of tensors that cannot share the same memory block.
    tensor_size : list of float or None
        size[v] = memory size of tensor v (optional, for capacity constraint).
    lambda1, lambda2, lambda3, lambda4 : float
        Penalty weights.
    capacity : float or None
        Capacity per memory block (required if tensor_size given).

    Returns
    -------
    Q : list of (int, int, float)
        Sparse upper-triangular QUBO matrix.
    num_vars : int
        Number of binary variables.
    """
    K = max_colors
    n_base = num_tensors * K
    n = n_base + K  # auxiliary y_c vars
    Q: QuboMatrix = []
    _add = Q.append

    def idx_tv(t: int, c: int) -> int:
        return t * K + c

    def idx_y(c: int) -> int:
        return n_base + c

    # ── Step A: Unique Color (λ1) ─────────────────────────────────────────
    for v in range(num_tensors):
        for a in range(K):
            i1 = idx_tv(v, a)
            _add((i1, i1, -lambda1))
            for b in range(a + 1, K):
                i2 = idx_tv(v, b)
                _add((i1, i2, 2.0 * lambda1))

    # ── Step B: Conflict (λ2) ─────────────────────────────────────────────
    for u, v in conflict_edges:
        for c in range(K):
            i1 = idx_tv(u, c)
            i2 = idx_tv(v, c)
            _add_off_diag(Q, i1, i2, lambda2)

    # ── Step C: Minimize Colors — Auxiliary y_c variables ─────────────────
    # Link x_{v,c} <= y_c: Penalty λ3 * x * (1 - y)
    # = λ3 * (x - x*y)  → diagonal +λ3, off-diagonal -λ3
    for v in range(num_tensors):
        for c in range(K):
            i_x = idx_tv(v, c)
            i_y = idx_y(c)
            _add((i_x, i_x, lambda3))
            # i_x < i_y always since x indices < y indices
            _add_off_diag(Q, i_x, i_y, -lambda3)

    # Objective: minimize sum of y_c
    for c in range(K):
        i = idx_y(c)
        _add((i, i, 1.0))

    # ── Step D: Capacity (λ4, if sizes provided) ──────────────────────────
    if tensor_size is not None and capacity is not None:
        for c in range(K):
            vars_c = list(range(num_tensors))
            for a in range(len(vars_c)):
                v1 = vars_c[a]
                i1 = idx_tv(v1, c)
                _add((i1, i1, -2.0 * lambda4 * capacity * tensor_size[v1]))
                for b in range(a + 1, len(vars_c)):
                    v2 = vars_c[b]
                    i2 = idx_tv(v2, c)
                    _add((i1, i2, 2.0 * lambda4 * tensor_size[v1] * tensor_size[v2]))

    return Q, n


# ═══════════════════════════════════════════════════════════════════════════
# 3. Partitioning (Operator Fusion)
# ═══════════════════════════════════════════════════════════════════════════

def build_partitioning_qubo(
    num_ops: int,
    max_groups: int,
    edge_weights: List[Tuple[int, int, float]],
    op_cost: List[float],
    lambda1: float = 10.0,
    lambda2: float = 10.0,
) -> Tuple[QuboMatrix, int]:
    """Build QUBO for operator fusion (graph partitioning).

    Indexing
    --------
    idx(v, g) = v * G + g
    n = num_ops * G

    Parameters
    ----------
    num_ops : int
        Number of operations to partition.
    max_groups : int
        Maximum number of groups (G).
    edge_weights : list of (int, int, float)
        Edge weights between operations (u, v, w_uv).
    op_cost : list of float
        op_cost[v] = computational cost of op v.
    lambda1, lambda2 : float
        Penalty weights.

    Returns
    -------
    Q : list of (int, int, float)
        Sparse upper-triangular QUBO matrix.
    num_vars : int
        Number of binary variables.
    """
    G = max_groups
    n = num_ops * G
    Q: QuboMatrix = []
    _add = Q.append

    def idx(v: int, g: int) -> int:
        return v * G + g

    # ── Step A: Unique Group (λ1) ─────────────────────────────────────────
    for v in range(num_ops):
        for a in range(G):
            i1 = idx(v, a)
            _add((i1, i1, -lambda1))
            for b in range(a + 1, G):
                i2 = idx(v, b)
                _add((i1, i2, 2.0 * lambda1))

    # ── Step B: Cut Minimization ──────────────────────────────────────────
    # Minimize - sum_{(u,v)} w_uv * sum_g z_{u,g} * z_{v,g}
    for u, v, w in edge_weights:
        for g in range(G):
            i1 = idx(u, g)
            i2 = idx(v, g)
            _add_off_diag(Q, i1, i2, -w)

    # ── Step C: Load Balancing (λ2) ───────────────────────────────────────
    L_avg = sum(op_cost) / G
    for g in range(G):
        vars_g = list(range(num_ops))
        for a in range(len(vars_g)):
            v1 = vars_g[a]
            i1 = idx(v1, g)
            _add((i1, i1, -2.0 * lambda2 * L_avg * op_cost[v1]))
            for b in range(a + 1, len(vars_g)):
                v2 = vars_g[b]
                i2 = idx(v2, g)
                _add((i1, i2, 2.0 * lambda2 * op_cost[v1] * op_cost[v2]))

    return Q, n


# ═══════════════════════════════════════════════════════════════════════════
# 4. Set Coverage (Test Case Selection)
# ═══════════════════════════════════════════════════════════════════════════

def build_coverage_qubo(
    num_tests: int,
    num_points: int,
    coverage_matrix: List[List[bool]],
    max_select: int,
    point_weights: Optional[List[float]] = None,
    lambda1: float = 10.0,
    lambda2: float = 10.0,
    lambda3: float = 10.0,
) -> Tuple[QuboMatrix, int]:
    """Build QUBO for test case selection (set coverage).

    Indexing
    --------
    Indices 0..num_tests-1 for x_t (test t selected).
    Indices num_tests..num_tests+num_points-1 for y_p (point p covered).

    Parameters
    ----------
    num_tests : int
        Number of available tests.
    num_points : int
        Number of coverage points.
    coverage_matrix : list of list of bool
        coverage_matrix[t][p] = True if test t covers point p.
    max_select : int
        Maximum (or exact) number of tests to select (K).
    point_weights : list of float or None
        Weight of each coverage point (default: 1.0 each).
    lambda1, lambda2, lambda3 : float
        Penalty weights.

    Returns
    -------
    Q : list of (int, int, float)
        Sparse upper-triangular QUBO matrix.
    num_vars : int
        Number of binary variables.
    """
    n = num_tests + num_points
    Q: QuboMatrix = []
    _add = Q.append

    if point_weights is None:
        point_weights = [1.0] * num_points

    # ── Step A: Implication (λ1) ──────────────────────────────────────────
    # If test t covers point p, enforce y_p >= x_t.
    # Penalty λ1 * x_t * (1 - y_p) = λ1 * (x_t - x_t * y_p)
    # Diagonal: +λ1 on x_t; Off-diagonal (x_t, y_p): -λ1
    for t in range(num_tests):
        for p in range(num_points):
            if coverage_matrix[t][p]:
                i_x = t
                i_y = num_tests + p
                _add((i_x, i_x, lambda1))
                # i_x < i_y always
                _add_off_diag(Q, i_x, i_y, -lambda1)

    # ── Step B: Prevent false positives (λ2) ──────────────────────────────
    # Penalty λ2 * y_p * (1 - sum_{t covers p} x_t)
    # = λ2 * (y_p - sum_t y_p * x_t)
    # Diagonal on y_p: +λ2
    # Off-diagonal (y_p, x_t): -λ2
    for p in range(num_points):
        i_y = num_tests + p
        _add((i_y, i_y, lambda2))
        for t in range(num_tests):
            if coverage_matrix[t][p]:
                i_x = t
                _add_off_diag(Q, i_x, i_y, -lambda2)

    # ── Step C: Objective — maximize covered points ───────────────────────
    # Add -w_p to diagonal of y_p (minimize negative weight = maximize coverage)
    for p in range(num_points):
        i_y = num_tests + p
        _add((i_y, i_y, -point_weights[p]))

    # ── Step D: Cardinality (λ3) — enforce exactly K tests ────────────────
    # λ3 * (sum_t x_t - K)^2 = λ3 * (sum_t x_t)^2 - 2*λ3*K*sum_t x_t + λ3*K^2
    # Diagonal: -2*λ3*K
    # Off-diagonal (t1 < t2): +2*λ3
    for t1 in range(num_tests):
        _add((t1, t1, -2.0 * lambda3 * max_select))
        for t2 in range(t1 + 1, num_tests):
            _add((t1, t2, 2.0 * lambda3))

    return Q, n
