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
import math
import warnings
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
# Helper: Time-Window Pruning
# ═══════════════════════════════════════════════════════════════════════════

def compute_time_windows(
    num_ops: int,
    exec_time: List[float],
    comm_cost: List[List[float]],
    makespan_upper_bound: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Compute earliest / latest start times for each operation.

    Assumes nodes are numbered in topological order (as produced by
    TpuGraphs ``edge_index``), so edges go from lower-index nodes to
    higher-index nodes.

    Parameters
    ----------
    num_ops : int
        Number of operations.
    exec_time : list of float
        exec_time[v] = execution time of operation v.
    comm_cost : list of list of float
        comm_cost[u][v] = communication cost between u and v.
        Non-zero values indicate a dependency edge u -> v.
    makespan_upper_bound : int
        Upper bound on the schedule makespan (in time steps).

    Returns
    -------
    est : list of int
        Earliest start time for each operation.
    lst : list of int
        Latest start time for each operation.
    windows : list of int
        Window size (lst[v] - est[v] + 1) for each operation.
    """
    # Convert float times to integer time steps (ceiling)
    exec_int = [int(math.ceil(et)) for et in exec_time]
    comm_int = [
        [int(math.ceil(c)) if c > 0 else 0 for c in row]
        for row in comm_cost
    ]

    est = [0] * num_ops
    lst = [0] * num_ops

    # ── Forward pass (topological order) ──────────────────────────────
    # EST[v] = max(EST[u] + exec_time[u] + comm_cost[u][v]) over all
    #          predecessors u with comm_cost[u][v] > 0
    for v in range(num_ops):
        max_est = 0
        for u in range(v):
            w = comm_int[u][v]
            if w > 0:
                candidate = est[u] + exec_int[u] + w
                if candidate > max_est:
                    max_est = candidate
        est[v] = max_est

    # ── Backward pass (reverse topological order) ─────────────────────
    # LST[exit] = makespan_upper_bound - exec_time[exit]
    # LST[u] = min(LST[v] - exec_time[u] - comm_cost[u][v]) over all
    #          successors v with comm_cost[u][v] > 0
    for v in range(num_ops - 1, -1, -1):
        # Find the minimum among all successors
        min_lst = makespan_upper_bound - exec_int[v]
        for u in range(v + 1, num_ops):
            w = comm_int[v][u]
            if w > 0:
                candidate = lst[u] - exec_int[v] - w
                if candidate < min_lst:
                    min_lst = candidate
        lst[v] = min_lst

    windows = [lst[v] - est[v] + 1 for v in range(num_ops)]

    return est, lst, windows


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
    compute_windows: bool = False,
    makespan_upper_bound: Optional[int] = None,
) -> Tuple[QuboMatrix, int]:
    """Build QUBO for TPU instruction scheduling (assignment problem).

    When ``compute_windows=True``, time-window pruning reduces variables
    by limiting each operation to its feasible time window
    [EST[v], LST[v]] instead of the full [0, time_horizon).

    Indexing (global time)
    ----------------------
    idx(v, p, t) = (v * num_processors + p) * time_horizon + t
    Total vars n = num_ops * num_processors * time_horizon

    Indexing (pruned windows)
    -------------------------
    Each node v has a window of ``window_sizes[v] = LST[v] - EST[v] + 1``
    time slots.  Offsets are pre-computed so that:
        idx(v, p, t) = offset[v] + p * window_sizes[v] + (t - EST[v])
    Total vars n = sum_v num_processors * window_sizes[v]

    Parameters
    ----------
    num_ops : int
        Number of operations to schedule.
    num_processors : int
        Number of available processors.
    time_horizon : int
        Number of time steps (used directly when ``compute_windows=False``,
        otherwise as default for ``makespan_upper_bound``).
    exec_time : list of float
        exec_time[v] = execution time of operation v.
    comm_cost : list of list of float
        comm_cost[u][v] = communication cost between op u and op v.
        Non-zero values indicate a dependency edge u -> v.
    resource_demand : list of float
        resource_demand[v] = resource units required by op v.
    proc_capacity : list of list of float
        proc_capacity[p][t] = capacity of processor p at time t.
    lambda1, lambda2, lambda3 : float
        Penalty weights for constraints.
    compute_windows : bool
        If True, use time-window pruning to reduce variables.
    makespan_upper_bound : int or None
        Upper bound on makespan for LST computation.
        If None and ``compute_windows=True``, defaults to ``time_horizon``.

    Returns
    -------
    Q : list of (int, int, float)
        Sparse upper-triangular QUBO matrix.
    num_vars : int
        Number of binary variables.
    """
    # ── Set up time windows or global horizon ──────────────────────────────
    if compute_windows:
        if makespan_upper_bound is None:
            makespan_upper_bound = time_horizon

        est, lst, window_sizes = compute_time_windows(
            num_ops, exec_time, comm_cost, makespan_upper_bound,
        )

        # Warn if any window is empty (infeasible)
        for v in range(num_ops):
            if est[v] > lst[v]:
                warnings.warn(
                    f"Operation {v}: EST={est[v]} > LST={lst[v]}, "
                    f"schedule infeasible under makespan bound {makespan_upper_bound}",
                )

        # Pre-compute offset array for dynamic index mapping
        offsets = [0] * num_ops
        for v in range(1, num_ops):
            offsets[v] = offsets[v - 1] + num_processors * window_sizes[v - 1]

        total_vars = offsets[-1] + num_processors * window_sizes[-1]

        def idx(v: int, p: int, t: int) -> int:
            return offsets[v] + p * window_sizes[v] + (t - est[v])

        def time_range(v: int):
            """Iterable of valid time slots for operation v."""
            return range(est[v], lst[v] + 1)
    else:
        total_vars = num_ops * num_processors * time_horizon

        def idx(v: int, p: int, t: int) -> int:
            return (v * num_processors + p) * time_horizon + t

        def time_range(v: int):
            """Iterable of valid time slots for operation v (global)."""
            return range(time_horizon)

    n = total_vars
    Q: QuboMatrix = []
    _add = Q.append

    # ── Step A: Unique Assignment (λ1) ────────────────────────────────────
    # penalty = λ1 * (sum_{p,t} x - 1)^2
    for v in range(num_ops):
        vars_v = [(p, t) for p in range(num_processors) for t in time_range(v)]
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
                for t_u in time_range(u):
                    i_u = idx(u, p_u, t_u)
                    for p_v in range(num_processors):
                        for t_v in time_range(v):
                            if t_v < t_u + min_separation:
                                i_v = idx(v, p_v, t_v)
                                _add_off_diag(Q, i_u, i_v, lambda2)

    # ── Step C: Resource Capacity (λ3) ────────────────────────────────────
    # For each (p, t), used = sum_v r_v * x_{v,p,t}
    # We use a soft penalty: λ3 * (sum_v r_v * x_{v,p,t} - cap)^2
    for p in range(num_processors):
        for t in range(time_horizon):
            cap = proc_capacity[p][t]
            # Only consider ops whose window includes this time t
            vars_pt = [v for v in range(num_ops)
                       if not compute_windows or (est[v] <= t <= lst[v])]
            for a in range(len(vars_pt)):
                v1 = vars_pt[a]
                if compute_windows and not (est[v1] <= t <= lst[v1]):
                    continue
                i1 = idx(v1, p, t)
                # Diagonal: -2*λ3*cap*r_v
                _add((i1, i1, -2.0 * lambda3 * cap * resource_demand[v1]))
                for b in range(a + 1, len(vars_pt)):
                    v2 = vars_pt[b]
                    if compute_windows and not (est[v2] <= t <= lst[v2]):
                        continue
                    i2 = idx(v2, p, t)
                    # Off-diagonal: +2*λ3*r_v1*r_v2
                    _add((i1, i2, 2.0 * lambda3 * resource_demand[v1] * resource_demand[v2]))

    # ── Step D: Objective (Makespan surrogate) ────────────────────────────
    # Add t to diagonal to encourage early scheduling
    for v in range(num_ops):
        for p in range(num_processors):
            for t in time_range(v):
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
