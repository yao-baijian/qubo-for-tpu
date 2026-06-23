"""Baseline heuristic algorithms for TPU optimization problems.

Each function returns a solution in the same format as the corresponding
QUBO solver output (list of ints representing binary/assignment decisions).
"""

from typing import List, Tuple, Optional
import heapq
import random
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════
# 1. List Scheduling (Critical Path Priority)
# ═══════════════════════════════════════════════════════════════════════════

def list_scheduling(
    num_ops: int,
    num_processors: int,
    time_horizon: int,
    exec_time: List[float],
    comm_cost: List[List[float]],
    resource_demand: List[float],
    proc_capacity: List[List[float]],
) -> List[int]:
    """List scheduling heuristic with critical-path priority.

    Returns a binary assignment vector of length
    num_ops * num_processors * time_horizon, where 1 indicates
    operation v is scheduled on processor p at time t.

    Uses critical path as priority: operations on the longest path
    get scheduled first.
    """
    n = num_ops * num_processors * time_horizon

    def idx(v: int, p: int, t: int) -> int:
        return (v * num_processors + p) * time_horizon + t

    # ── Compute critical path priorities (latest possible start time) ─────
    # We use a simple longest-path-to-exit ranking.
    # Build reverse adjacency
    successors = [[] for _ in range(num_ops)]
    in_degree = [0] * num_ops
    for u in range(num_ops):
        for v in range(num_ops):
            if u != v and comm_cost[u][v] > 0:
                successors[u].append(v)
                in_degree[v] += 1

    # Topological-like longest path (critical path length to exit)
    # Use DP: dist[v] = exec_time[v] + max_{v->w} (comm_cost[v][w] + dist[w])
    dist = [0.0] * num_ops
    order = list(range(num_ops))
    # Process in reverse topological order
    for v in reversed(order):
        max_succ = 0.0
        for w in successors[v]:
            max_succ = max(max_succ, comm_cost[v][w] + dist[w])
        dist[v] = exec_time[v] + max_succ

    # Priority = dist[v] (higher = more critical)
    ops_by_priority = sorted(range(num_ops), key=lambda v: -dist[v])

    # ── Schedule greedily ─────────────────────────────────────────────────
    # Track processor availability and resource usage
    proc_available_at = [0] * num_processors  # earliest available time
    cpu_usage = [[0.0] * time_horizon for _ in range(num_processors)]
    schedule = {}  # (op -> (proc, start_time))

    for v in ops_by_priority:
        # Find the earliest (processor, time) that satisfies:
        # 1. processor is idle for exec_time[v] consecutive steps
        # 2. resource demand fits within capacity
        # 3. dependencies are satisfied
        best_p = 0
        best_t = time_horizon

        # Compute earliest start time from data dependencies
        dep_ready = 0
        for u in range(num_ops):
            if u != v and comm_cost[u][v] > 0 and u in schedule:
                p_u, t_u = schedule[u]
                dep_ready = max(dep_ready, int(t_u + exec_time[u] + comm_cost[u][v]))

        for p in range(num_processors):
            start = max(dep_ready, proc_available_at[p])
            # Clamp to time_horizon - exec_time[v]
            max_start = max(time_horizon - int(exec_time[v]), 0)
            if start > max_start:
                continue
            # Check resource capacity for consecutive time steps
            feasible = True
            for dt in range(int(exec_time[v])):
                t_check = start + dt
                if t_check >= time_horizon:
                    feasible = False
                    break
                if cpu_usage[p][t_check] + resource_demand[v] > proc_capacity[p][t_check]:
                    feasible = False
                    break
            if feasible and start < best_t:
                best_p = p
                best_t = start

        if best_t < time_horizon:
            # Assign the operation
            schedule[v] = (best_p, best_t)
            proc_available_at[best_p] = best_t + int(exec_time[v])
            for dt in range(int(exec_time[v])):
                cpu_usage[best_p][best_t + dt] += resource_demand[v]

    # ── Build binary output vector ────────────────────────────────────────
    solution = [0] * n
    for v, (p, t) in schedule.items():
        for dt in range(int(exec_time[v])):
            if t + dt < time_horizon:
                solution[idx(v, p, t + dt)] = 1
    return solution


# ═══════════════════════════════════════════════════════════════════════════
# 2. Greedy Coloring (First-Fit by Degree)
# ═══════════════════════════════════════════════════════════════════════════

def greedy_coloring(
    num_tensors: int,
    max_colors: int,
    conflict_edges: List[Tuple[int, int]],
    tensor_size: Optional[List[float]] = None,
    capacity: Optional[float] = None,
) -> List[int]:
    """Greedy first-fit coloring, sorting vertices by degree (descending).

    Returns a binary assignment vector of length num_tensors * K + K,
    matching the QUBO variable ordering used in build_coloring_qubo.
    """
    K = max_colors
    n_base = num_tensors * K
    n = n_base + K
    solution = [0] * n

    # Build adjacency for conflict graph
    adj = [[] for _ in range(num_tensors)]
    for u, v in conflict_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Sort vertices by degree (descending) — Welsh-Powell strategy
    order = sorted(range(num_tensors), key=lambda v: -len(adj[v]))

    # Track which colors are used (for auxiliary y_c vars)
    color_used = [False] * K
    # Track total size per color
    color_size = [0.0] * K

    # Greedy assignment
    assignment = {}
    for v in order:
        # Find forbidden colors from neighbors already colored
        forbidden = set()
        for u in adj[v]:
            if u in assignment:
                forbidden.add(assignment[u])

        # Pick first feasible color
        chosen = None
        for c in range(K):
            if c in forbidden:
                continue
            if tensor_size is not None and capacity is not None:
                if color_size[c] + tensor_size[v] > capacity:
                    continue
            chosen = c
            break

        if chosen is not None:
            assignment[v] = chosen
            color_used[chosen] = True
            if tensor_size is not None:
                color_size[chosen] += tensor_size[v]

    # Build binary solution matching QUBO indexing
    for v, c in assignment.items():
        solution[v * K + c] = 1
    for c in range(K):
        if color_used[c]:
            solution[n_base + c] = 1

    return solution


# ═══════════════════════════════════════════════════════════════════════════
# 3. Kernighan-Lin Partitioning (random balanced init)
# ═══════════════════════════════════════════════════════════════════════════

def kl_partitioning(
    num_ops: int,
    max_groups: int,
    edge_weights: List[Tuple[int, int, float]],
    op_cost: List[float],
    max_passes: int = 10,
) -> List[int]:
    """Kernighan-Lin style partitioning with random balanced initialization.

    Returns a binary assignment vector of length num_ops * G, matching
    the QUBO indexing in build_partitioning_qubo.

    For k-way partitioning, applies recursive bisection.
    """
    G = max_groups
    n = num_ops * G

    if G < 2:
        solution = [0] * n
        for v in range(num_ops):
            solution[v * G + 0] = 1
        return solution

    # ── Build adjacency ───────────────────────────────────────────────────
    adj = defaultdict(dict)
    for u, v, w in edge_weights:
        adj[u][v] = adj[u].get(v, 0) + w
        adj[v][u] = adj[v].get(u, 0) + w

    # ── Recursive bisection ───────────────────────────────────────────────
    def _bisect(vertices: List[int], num_parts: int) -> List[int]:
        """Partition `vertices` into `num_parts` groups. Returns list of group ids."""
        if num_parts == 1:
            return [0] * len(vertices)

        # Split into two groups via KL
        nv = len(vertices)
        half = nv // 2

        # Random balanced init
        part = [0] * nv
        indices = list(range(nv))
        random.shuffle(indices)
        for i in indices[:half]:
            part[i] = 0
        for i in indices[half:]:
            part[i] = 1

        # KL refinement passes
        for _ in range(max_passes):
            improved = False
            # Compute gain for each vertex
            gains = []
            for i in range(nv):
                v = vertices[i]
                internal = 0.0
                external = 0.0
                for u, w in adj.get(v, {}).items():
                    if u in vertices:
                        j = vertices.index(u)
                        if part[j] == part[i]:
                            internal += w
                        else:
                            external += w
                gain = external - internal  # Gain of moving
                gains.append((gain, i))

            # Sort by gain descending
            gains.sort(key=lambda x: -x[0])

            # Try swapping best pairs
            swapped = [False] * nv
            for gain, i in gains:
                if swapped[i]:
                    continue
                # Find a vertex from the other side to swap with
                for gain_j, j in gains:
                    if not swapped[j] and part[j] != part[i]:
                        combined_gain = gain + gain_j
                        # Account for edge between i and j
                        vi, vj = vertices[i], vertices[j]
                        if vj in adj.get(vi, {}):
                            combined_gain -= 2 * adj[vi][vj]  # Already counted twice
                        if combined_gain > 0:
                            part[i], part[j] = part[j], part[i]
                            swapped[i] = swapped[j] = True
                            improved = True
                            break
            if not improved:
                break

        # Recursively bisect each half
        group0 = [vertices[i] for i in range(nv) if part[i] == 0]
        group1 = [vertices[i] for i in range(nv) if part[i] == 1]
        k0 = num_parts // 2
        k1 = num_parts - k0
        part0 = _bisect(group0, k0)
        part1 = _bisect(group1, k1)

        result = [0] * nv
        # Map results back
        idx0, idx1 = 0, 0
        for i in range(nv):
            if part[i] == 0:
                result[i] = part0[idx0]
                idx0 += 1
            else:
                result[i] = k0 + part1[idx1]
                idx1 += 1
        return result

    all_vertices = list(range(num_ops))
    group_ids = _bisect(all_vertices, G)

    # Build binary solution
    solution = [0] * n
    for v, g in enumerate(group_ids):
        solution[v * G + int(g)] = 1
    return solution


# ═══════════════════════════════════════════════════════════════════════════
# 4. Greedy Max-Coverage
# ═══════════════════════════════════════════════════════════════════════════

def greedy_coverage(
    num_tests: int,
    num_points: int,
    coverage_matrix: List[List[bool]],
    max_select: int,
    point_weights: Optional[List[float]] = None,
) -> List[int]:
    """Greedy max-coverage test selection.

    Iteratively selects the test that covers the most uncovered
    (weighted) points, up to K tests.

    Returns a binary vector of length num_tests + num_points, matching
    the QUBO indexing in build_coverage_qubo.
    """
    n = num_tests + num_points

    if point_weights is None:
        point_weights = [1.0] * num_points

    # For each test, compute the set of points it covers (as indices)
    test_points = []
    for t in range(num_tests):
        pts = [p for p in range(num_points) if coverage_matrix[t][p]]
        test_points.append(pts)

    # Greedy selection
    selected = [False] * num_tests
    covered = [False] * num_points
    remaining_weight = sum(point_weights)

    for _ in range(max_select):
        best_test = -1
        best_gain = 0.0
        for t in range(num_tests):
            if selected[t]:
                continue
            gain = sum(point_weights[p] for p in test_points[t] if not covered[p])
            if gain > best_gain:
                best_gain = gain
                best_test = t
        if best_test < 0 or best_gain <= 0:
            break
        selected[best_test] = True
        for p in test_points[best_test]:
            covered[p] = True

    # Build binary solution
    solution = [0] * n
    for t in range(num_tests):
        if selected[t]:
            solution[t] = 1
    for p in range(num_points):
        if covered[p]:
            solution[num_tests + p] = 1
    return solution
