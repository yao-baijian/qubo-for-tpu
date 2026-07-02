"""Data loader for TPU benchmark — loads problem instances from various sources.

Sources
-------
- **synthetic**:   Randomly generated instances (fallback; matches existing tests).
- **tpugraphs**:   TpuGraphs ``.npz`` dataset from ``benchmarks/v0/npz/``.
- **hlo_dump**:    XLA HLO dump files (*.txt / *.hlo).
- **mlperf**:      Synthetic DAGs that mimic MLPerf model topologies.

All loaders return metadata dicts compatible with the generators in
:mod:`src.tpu.generators` and the baselines in :mod:`src.tpu.baselines`.
"""

from __future__ import annotations

import os
import pathlib
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Hardware Configuration
# ---------------------------------------------------------------------------

TPU_V3_CONFIG = {
    "num_processors": 4,          # cores per TPU v3 chip
    "peak_tops": 92.0,            # tera-operations/second (peak)
    "effective_tops": 70.0,       # sustained throughput for FLOPs→time
    "sram_capacity_mb": 32,       # on-chip SRAM per core (MB)
    "hbm_bandwidth_gbps": 900,    # HBM bandwidth (GB/s)
    "ici_bandwidth_gbps": 100,    # inter-chip interconnect (GB/s)
    "num_banks": 16,              # SRAM banks for coloring constraints
}

# ---------------------------------------------------------------------------
# Opcode → name mapping (XLA hlo_opcode.h enums, subset relevant to FLOPs)
# ---------------------------------------------------------------------------

OPCODE_NAMES: Dict[int, str] = {
    2:   "constant",
    4:   "get-tuple-element",
    5:   "convolution",
    12:  "broadcast",
    13:  "reshape",
    19:  "reverse",
    20:  "concatenate",
    24:  "slice",
    25:  "dynamic-slice",
    26:  "dynamic-update-slice",
    31:  "tuple",
    32:  "reduce",
    34:  "pad",
    37:  "transpose",
    40:  "gather",
    45:  "dot",
    47:  "scatter",
    50:  "reduce-window",
    52:  "sort",
    54:  "copy",
    57:  "add",
    58:  "divide",
    59:  "parameter",
    60:  "multiply",
    61:  "subtract",
    63:  "maximum",
    66:  "minimum",
    70:  "reshape",
    72:  "convolution",
    75:  "logistic",
    77:  "exponential",
    81:  "sqrt",
    83:  "negate",
    84:  "abs",
    87:  "tanh",
    89:  "clamp",
    95:  "multiply",
    98:  "and",
    100: "or",
    102: "shift-left",
    107: "reduce",
}

# Opcodes that perform significant compute.
_COMPUTE_OPS = {5, 45, 72, 26, 32, 50, 57, 60, 61, 87, 75, 77, 81}

_MLPERF_MODELS = [
    "resnet50",
    "bert",
    "ssd-mobilenet",
    "3d-unet",
    "rnnt",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. TPU v3 Execution Time Estimator
# ═══════════════════════════════════════════════════════════════════════════


def estimate_exec_time(node_opcode: int, node_feat: np.ndarray) -> float:
    """Estimate execution time (ns) from HLO node opcode and features.

    Parameters
    ----------
    node_opcode : int
        Opcode integer from the ``node_opcode`` array.
    node_feat : np.ndarray
        140-dim feature vector for this node.

    Returns
    -------
    float
        Estimated execution time in nanoseconds.
    """
    volume = max(1.0, float(node_feat[28]))
    elem_type_idx = int(np.argmax(node_feat[2:21])) if node_feat[2:21].sum() > 0 else 4
    elem_bytes = [1, 1, 2, 2, 4, 4, 8, 8, 2, 2, 4, 8, 8, 16, 1, 1, 1, 1, 1]
    elem_size = elem_bytes[min(elem_type_idx, len(elem_bytes) - 1)]
    data_volume = volume * elem_size

    if node_opcode == 45:  # dot (matmul)
        dims = node_feat[21:27]
        non_zero = dims[dims > 0]
        if len(non_zero) >= 3:
            m, k, n_val = int(non_zero[0]), int(non_zero[1]), int(non_zero[2])
            flops = 2.0 * m * k * n_val
        elif len(non_zero) == 2:
            flops = 2.0 * float(non_zero[0] * non_zero[1])
        else:
            flops = 2.0 * data_volume
    elif node_opcode in (5, 72):  # convolution
        dims = node_feat[21:27]
        non_zero = dims[dims > 0]
        flops = 2.0 * float(np.prod(non_zero)) if len(non_zero) >= 4 else 2.0 * data_volume
    elif node_opcode in (57, 60, 61, 63, 66, 81, 87, 75, 77):
        flops = float(volume) * 2.0
    elif node_opcode in (32, 107):  # reduce
        flops = float(volume) * 1.0
    elif node_opcode == 26:
        flops = float(volume) * 0.5
    else:
        flops = float(volume) * 0.1

    exec_time_ns = (flops / (TPU_V3_CONFIG["effective_tops"] * 1e12)) * 1e9
    return max(exec_time_ns, 0.1)


# ═══════════════════════════════════════════════════════════════════════════
# 2. ASAP Scheduler for Lifetime Inference
# ═══════════════════════════════════════════════════════════════════════════


def infer_lifetimes(
    edge_index: np.ndarray,
    exec_time: np.ndarray,
    num_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, float, float, float]]]:
    """Infer tensor lifetimes using ASAP (As-Soon-As-Possible) scheduling.

    Returns
    -------
    start_time, finish_time : ndarray
        ASAP start/finish time for each node.
    tensor_intervals : list of (src, dst, start, end, size_proxy)
    """
    successors = [[] for _ in range(num_nodes)]
    predecessors = [[] for _ in range(num_nodes)]
    in_degree = [0] * num_nodes
    for u, v in edge_index:
        u, v = int(u), int(v)
        successors[u].append(v)
        predecessors[v].append(u)
        in_degree[v] += 1

    topo_order = []
    queue = [i for i in range(num_nodes) if in_degree[i] == 0]
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    start_time = np.zeros(num_nodes, dtype=np.float64)
    finish_time = np.zeros(num_nodes, dtype=np.float64)
    for u in topo_order:
        if predecessors[u]:
            start_time[u] = max(finish_time[p] for p in predecessors[u])
        finish_time[u] = start_time[u] + exec_time[u]

    tensor_intervals = []
    for u, v in edge_index:
        u, v = int(u), int(v)
        tensor_intervals.append((
            u, v, float(finish_time[u]), float(start_time[v]), float(exec_time[u]),
        ))
    return start_time, finish_time, tensor_intervals


# ═══════════════════════════════════════════════════════════════════════════
# 3. Communication Cost Calculator
# ═══════════════════════════════════════════════════════════════════════════


def compute_comm_cost(edge_size: float, hops: int = 1) -> float:
    """Compute communication cost (ns) from tensor size and interconnect hops."""
    bw_bytes_per_ns = TPU_V3_CONFIG["ici_bandwidth_gbps"] * 1e9 / 8
    if bw_bytes_per_ns <= 0:
        return 0.0
    return (edge_size / bw_bytes_per_ns) * hops


# ═══════════════════════════════════════════════════════════════════════════
# 4. TpuGraphs .npz Loader
# ═══════════════════════════════════════════════════════════════════════════


def load_tpugraphs_npz(
    npz_path: str, max_nodes: Optional[int] = None,
    compress: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load a single TpuGraphs ``.npz`` file and convert to metadata.

    Expected keys: ``node_opcode`` (n,), ``node_feat`` (n,140),
    ``edge_index`` (m,2).

    Parameters
    ----------
    npz_path : str
        Path to the ``.npz`` file.
    max_nodes : int or None
        If set, subsample to at most this many nodes (for quick testing).
    compress : bool
        If True, apply degree-1 chain reduction to shrink the graph.

    Returns metadata dict with ``problem_type`` and ``metadata`` compatible
    with :func:`build_scheduling_qubo`.
    """
    if not os.path.isfile(npz_path):
        return None
    try:
        data = np.load(npz_path)
    except Exception as exc:
        print(f"[data] Failed to load {npz_path}: {exc}")
        return None

    required = {"node_opcode", "node_feat", "edge_index"}
    if not required.issubset(data.keys()):
        return None

    node_opcode = np.asarray(data["node_opcode"]).ravel()
    node_feat = np.asarray(data["node_feat"])
    edge_index = np.asarray(data["edge_index"])
    num_nodes = len(node_opcode)

    if node_feat.shape[0] != num_nodes:
        return None

    # Optional subsampling for large graphs
    if max_nodes is not None and num_nodes > max_nodes:
        rng = np.random.default_rng(42)
        keep = rng.choice(num_nodes, size=max_nodes, replace=False)
        keep_set = set(keep)
        keep_idx = {int(old): new for new, old in enumerate(keep)}
        node_opcode = node_opcode[keep]
        node_feat = node_feat[keep]
        # Filter edges to only include kept nodes
        mask = np.isin(edge_index[:, 0], keep) & np.isin(edge_index[:, 1], keep)
        edge_index = edge_index[mask]
        # Remap indices
        edge_index = np.array([
            [keep_idx[int(u)], keep_idx[int(v)]]
            for u, v in edge_index
        ], dtype=np.int64)
        num_nodes = max_nodes

    exec_time = np.array([
        estimate_exec_time(int(oc), node_feat[i])
        for i, oc in enumerate(node_opcode)
    ])

    _, finish_time, tensor_intervals = infer_lifetimes(
        edge_index, exec_time, num_nodes,
    )
    makespan = float(finish_time.max())
    time_horizon = max(10, int(makespan * 1.5 / 1000) + 1)
    num_processors = TPU_V3_CONFIG["num_processors"]

    comm_cost = [[0.0] * num_nodes for _ in range(num_nodes)]
    for src, dst, _, _, _ in tensor_intervals:
        vol = max(1.0, float(node_feat[src, 28]))
        elem_type_idx = int(np.argmax(node_feat[src, 2:21])) \
            if node_feat[src, 2:21].sum() > 0 else 4
        elem_bytes = [1, 1, 2, 2, 4, 4, 8, 8, 2, 2, 4, 8, 8, 16, 1, 1, 1, 1, 1]
        tensor_size_bytes = vol * elem_bytes[min(elem_type_idx, len(elem_bytes) - 1)]
        cost = compute_comm_cost(tensor_size_bytes, hops=1)
        comm_cost[src][dst] = cost
        comm_cost[dst][src] = cost

    resource_demand = []
    for i in range(num_nodes):
        vol = max(1.0, float(node_feat[i, 28]))
        resource_demand.append(max(0.5, min(3.0, vol / 1e6)))

    proc_capacity = [[10.0] * time_horizon for _ in range(num_processors)]

    metadata = {
        "num_ops": num_nodes,
        "num_processors": num_processors,
        "time_horizon": time_horizon,
        "exec_time": exec_time.tolist(),
        "comm_cost": comm_cost,
        "resource_demand": resource_demand,
        "proc_capacity": proc_capacity,
    }

    if compress:
        metadata = compress_graph(metadata)
        # Update _num_nodes to reflect compressed size
        num_nodes = metadata["num_ops"]

    return {
        "problem_type": "scheduling",
        "metadata": metadata,
        "_source": str(npz_path),
        "_num_nodes": num_nodes,
        "_num_edges": edge_index.shape[0],
        "_makespan_estimate_ns": makespan,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Batch Loader for Benchmark
# ═══════════════════════════════════════════════════════════════════════════


def load_tpugraphs_batch(
    data_dir: str,
    max_files: Optional[int] = None,
    max_nodes: Optional[int] = None,
    compress: bool = False,
) -> List[Dict[str, Any]]:
    """Load multiple ``.npz`` files from a TpuGraphs directory tree.

    Layout: ``benchmarks/v0/npz/{collection}/{config}/{split}/{hash}.npz``

    Parameters
    ----------
    data_dir : str
        Root directory with ``.npz`` files.
    max_files : int or None
        Limit on number of files to load.
    max_nodes : int or None
        If set, subsample each graph to at most this many nodes.
    compress : bool
        If True, apply degree-1 chain reduction to each graph.
    """
    results: List[Dict[str, Any]] = []
    p_dir = pathlib.Path(data_dir)
    if not p_dir.is_dir():
        print(f"[data] Directory not found: {data_dir}")
        return results

    for fpath in sorted(p_dir.rglob("*.npz")):
        if max_files is not None and len(results) >= max_files:
            break
        parsed = load_tpugraphs_npz(str(fpath), max_nodes=max_nodes, compress=compress)
        if parsed is not None:
            results.append(parsed)

    print(f"[data] Loaded {len(results)} TpuGraphs instances from {data_dir}")
    return results


# ---------------------------------------------------------------------------
# B. XLA HLO Dump Parser
# ---------------------------------------------------------------------------


def _parse_hlo_text(text: str) -> Optional[Dict[str, Any]]:
    """Parse raw HLO text content into scheduling metadata."""
    instruction_pattern = re.compile(
        r"%([a-zA-Z_][a-zA-Z0-9_.-]*)\s*=\s*"
        r"([\w\[\],\s]+)\s+"
        r"([a-zA-Z][a-zA-Z_-]*)\s*"
        r"\((.*?)\)"
    )

    nodes: List[Dict[str, Any]] = []
    edges: List[Tuple[str, str]] = []
    name_to_index: Dict[str, int] = {}

    for match in instruction_pattern.finditer(text):
        name = match.group(1)
        shape_str = match.group(2).strip()
        op_type = match.group(3).lower()
        operands_str = match.group(4)
        dims = [int(d) for d in re.findall(r"\d+", shape_str)]
        volume = max(1, np.prod(dims)) if dims else 1

        flops_map = {
            "dot": volume * 2, "convolution": volume * 2,
            "add": volume, "multiply": volume, "subtract": volume,
        }
        flops = flops_map.get(op_type, volume * 0.1)
        exec_time_ns = (flops / (TPU_V3_CONFIG["effective_tops"] * 1e12)) * 1e9
        exec_time_ns = max(exec_time_ns, 0.1)

        idx = len(nodes)
        name_to_index[name] = idx
        nodes.append({
            "name": name,
            "op_type": op_type,
            "shape_dims": dims,
            "exec_time": exec_time_ns,
        })

        operand_names = re.findall(r"%([a-zA-Z_][a-zA-Z0-9_.-]*)", operands_str)
        for oname in operand_names:
            edges.append((oname, name))

    if not nodes:
        return None

    num_ops = len(nodes)
    num_processors = TPU_V3_CONFIG["num_processors"]
    time_horizon = max(10, int(sum(n["exec_time"] for n in nodes) * 1.5 / 1000) + 1)
    exec_time_list = [n["exec_time"] for n in nodes]

    comm_cost = [[0.0] * num_ops for _ in range(num_ops)]
    for src_name, dst_name in edges:
        if src_name in name_to_index and dst_name in name_to_index:
            u = name_to_index[src_name]
            v = name_to_index[dst_name]
            comm_cost[u][v] = 1.0
            comm_cost[v][u] = 1.0

    resource_demand = [1.0] * num_ops
    proc_capacity = [[10.0] * time_horizon for _ in range(num_processors)]

    return {
        "problem_type": "scheduling",
        "metadata": {
            "num_ops": num_ops,
            "num_processors": num_processors,
            "time_horizon": time_horizon,
            "exec_time": exec_time_list,
            "comm_cost": comm_cost,
            "resource_demand": resource_demand,
            "proc_capacity": proc_capacity,
        },
        "_nodes": nodes,
        "_edges": edges,
    }


def parse_hlo_dump(dump_dir: str) -> List[Dict[str, Any]]:
    """Parse all ``*.txt`` / ``*.hlo`` files in a directory as HLO dumps."""
    results: List[Dict[str, Any]] = []
    p_dir = pathlib.Path(dump_dir)
    if not p_dir.is_dir():
        print(f"[data] HLO dump directory not found: {dump_dir}")
        return results

    for fpath in sorted(p_dir.glob("*")):
        if fpath.suffix.lower() not in (".txt", ".hlo", ".hlo.pb.txt", ".pbtxt"):
            continue
        if not fpath.is_file():
            continue
        try:
            text = fpath.read_text(encoding="utf-8")
            parsed = _parse_hlo_text(text)
            if parsed is not None:
                parsed["_source"] = str(fpath)
                results.append(parsed)
        except Exception as exc:
            print(f"[data] Skipping {fpath}: {exc}")

    print(f"[data] Parsed {len(results)} HLO graphs from {dump_dir}")
    return results


# ---------------------------------------------------------------------------
# C. MLPerf Model List & Synthetic Generator
# ---------------------------------------------------------------------------


def get_mlperf_model_list() -> List[str]:
    """Return list of known MLPerf model names."""
    return list(_MLPERF_MODELS)


def _make_resnet50_like() -> Dict[str, Any]:
    """Synthetic DAG mimicking ResNet50."""
    num_ops = 54
    exec_time = [random.uniform(1.0, 5.0) for _ in range(num_ops)]
    comm_cost = [[0.0] * num_ops for _ in range(num_ops)]
    stages = [(1, 3), (3, 5), (4, 7), (4, 7), (6, 13),
              (8, 19), (10, 29), (12, 41), (2, 51)]
    for si, (width, end) in enumerate(stages):
        for j in range(end - width, end):
            if j > 0:
                w = random.uniform(0.5, 2.0)
                comm_cost[j - 1][j] = w
                comm_cost[j][j - 1] = w
        if si > 0:
            prev_end = stages[si - 1][1]
            sw = random.uniform(0.1, 1.0)
            comm_cost[prev_end - 1][end - width] = sw
            comm_cost[end - width][prev_end - 1] = sw
    num_processors = TPU_V3_CONFIG["num_processors"]
    time_horizon = max(10, int(sum(exec_time) * 1.5))
    resource_demand = [1.0] * num_ops
    proc_capacity = [[10.0] * time_horizon for _ in range(num_processors)]
    return {"problem_type": "scheduling", "metadata": {
        "num_ops": num_ops, "num_processors": num_processors,
        "time_horizon": time_horizon, "exec_time": exec_time,
        "comm_cost": comm_cost, "resource_demand": resource_demand,
        "proc_capacity": proc_capacity,
    }}


def _make_bert_like() -> Dict[str, Any]:
    """Synthetic DAG mimicking BERT."""
    num_ops = 48
    exec_time = [random.uniform(2.0, 8.0) for _ in range(num_ops)]
    comm_cost = [[0.0] * num_ops for _ in range(num_ops)]
    for layer in range(12):
        base = layer * 4
        for j in range(4):
            if base + j > 0:
                w = random.uniform(0.5, 3.0)
                comm_cost[base + j - 1][base + j] = w
                comm_cost[base + j][base + j - 1] = w
        if layer > 0:
            pb = (layer - 1) * 4
            sw = random.uniform(0.2, 1.5)
            comm_cost[pb + 1][base] = sw
            comm_cost[base][pb + 1] = sw
    num_processors = TPU_V3_CONFIG["num_processors"]
    time_horizon = max(10, int(sum(exec_time) * 1.5))
    resource_demand = [random.uniform(0.5, 2.0) for _ in range(num_ops)]
    proc_capacity = [[10.0] * time_horizon for _ in range(num_processors)]
    return {"problem_type": "scheduling", "metadata": {
        "num_ops": num_ops, "num_processors": num_processors,
        "time_horizon": time_horizon, "exec_time": exec_time,
        "comm_cost": comm_cost, "resource_demand": resource_demand,
        "proc_capacity": proc_capacity,
    }}


def _make_ssd_mobilenet_like() -> Dict[str, Any]:
    """Synthetic DAG mimicking SSD-MobileNet."""
    num_ops = 36
    exec_time = [random.uniform(0.5, 4.0) for _ in range(num_ops)]
    comm_cost = [[0.0] * num_ops for _ in range(num_ops)]
    for i in range(1, num_ops):
        if random.random() < 0.4:
            w = random.uniform(0.3, 2.0)
            comm_cost[i - 1][i] = w
            comm_cost[i][i - 1] = w
        if i > 2 and random.random() < 0.2:
            w = random.uniform(0.1, 1.0)
            comm_cost[i - 3][i] = w
            comm_cost[i][i - 3] = w
    num_processors = TPU_V3_CONFIG["num_processors"]
    time_horizon = max(10, int(sum(exec_time) * 1.5))
    resource_demand = [random.uniform(0.5, 1.5) for _ in range(num_ops)]
    proc_capacity = [[10.0] * time_horizon for _ in range(num_processors)]
    return {"problem_type": "scheduling", "metadata": {
        "num_ops": num_ops, "num_processors": num_processors,
        "time_horizon": time_horizon, "exec_time": exec_time,
        "comm_cost": comm_cost, "resource_demand": resource_demand,
        "proc_capacity": proc_capacity,
    }}


def _make_3d_unet_like() -> Dict[str, Any]:
    """Synthetic DAG mimicking 3D-UNet."""
    num_ops = 42
    exec_time = [random.uniform(1.0, 6.0) for _ in range(num_ops)]
    comm_cost = [[0.0] * num_ops for _ in range(num_ops)]
    for i in range(1, 21):
        w = random.uniform(0.5, 2.0)
        comm_cost[i - 1][i] = w
        comm_cost[i][i - 1] = w
    for i in range(25, 42):
        w = random.uniform(0.5, 2.0)
        comm_cost[i - 1][i] = w
        comm_cost[i][i - 1] = w
    for skip in range(5):
        di = skip * 4
        ui = 41 - skip * 4
        w = random.uniform(0.3, 1.5)
        comm_cost[di][ui] = w
        comm_cost[ui][di] = w
    num_processors = TPU_V3_CONFIG["num_processors"]
    time_horizon = max(10, int(sum(exec_time) * 1.5))
    resource_demand = [random.uniform(0.5, 2.0) for _ in range(num_ops)]
    proc_capacity = [[10.0] * time_horizon for _ in range(num_processors)]
    return {"problem_type": "scheduling", "metadata": {
        "num_ops": num_ops, "num_processors": num_processors,
        "time_horizon": time_horizon, "exec_time": exec_time,
        "comm_cost": comm_cost, "resource_demand": resource_demand,
        "proc_capacity": proc_capacity,
    }}


def _make_rnnt_like() -> Dict[str, Any]:
    """Synthetic DAG mimicking RNN-T."""
    num_ops = 30
    exec_time = [random.uniform(1.0, 5.0) for _ in range(num_ops)]
    comm_cost = [[0.0] * num_ops for _ in range(num_ops)]
    for i in range(1, 15):
        w = random.uniform(0.5, 2.5)
        comm_cost[i - 1][i] = w
        comm_cost[i][i - 1] = w
    for i in range(16, 25):
        w = random.uniform(0.5, 2.0)
        comm_cost[i - 1][i] = w
        comm_cost[i][i - 1] = w
    for i in range(26, 30):
        w = random.uniform(0.5, 1.5)
        comm_cost[i - 1][i] = w
        comm_cost[i][i - 1] = w
    comm_cost[14][25] = 3.0
    comm_cost[25][14] = 3.0
    comm_cost[24][25] = 2.0
    comm_cost[25][24] = 2.0
    num_processors = TPU_V3_CONFIG["num_processors"]
    time_horizon = max(10, int(sum(exec_time) * 1.5))
    resource_demand = [1.0] * num_ops
    proc_capacity = [[10.0] * time_horizon for _ in range(num_processors)]
    return {"problem_type": "scheduling", "metadata": {
        "num_ops": num_ops, "num_processors": num_processors,
        "time_horizon": time_horizon, "exec_time": exec_time,
        "comm_cost": comm_cost, "resource_demand": resource_demand,
        "proc_capacity": proc_capacity,
    }}


_MLPERF_GENERATORS = {
    "resnet50": _make_resnet50_like,
    "bert": _make_bert_like,
    "ssd-mobilenet": _make_ssd_mobilenet_like,
    "3d-unet": _make_3d_unet_like,
    "rnnt": _make_rnnt_like,
}


def generate_from_mlperf_model(model_name: str) -> Optional[Dict[str, Any]]:
    """Generate a synthetic HLO-like DAG for a given MLPerf model."""
    key = model_name.lower().replace("-", "").replace("_", "")
    for known_key, gen_fn in _MLPERF_GENERATORS.items():
        if known_key.replace("-", "").replace("_", "") == key:
            return gen_fn()
    print(f"[data] Unknown MLPerf model: {model_name}")
    return None


# ---------------------------------------------------------------------------
# D. Graph Compression (Chain Reduction)
# ---------------------------------------------------------------------------


def compress_graph(metadata: dict) -> dict:
    """Compress a scheduling metadata dict by folding degree-1 nodes.

    A *degree-1* node is a node with exactly 1 predecessor and 1 successor
    in the directed DAG (edges ``u->v`` where ``comm_cost[u][v] > 0`` and
    nodes are assumed to be in topological order).  Such a node is folded
    into its predecessor:

    * ``exec_time[predecessor] += exec_time[v]``
    * ``comm_cost[predecessor][successor] = comm_cost[predecessor][v]
      + comm_cost[v][successor]`` (set symmetrically)
    * Node ``v`` is removed and all remaining nodes are re-indexed.

    The process repeats until no degree-1 nodes remain (max 10 iterations).

    Parameters
    ----------
    metadata : dict
        Scheduling metadata dict with keys ``num_ops``, ``exec_time``,
        ``comm_cost``, ``resource_demand``, ``proc_capacity``, etc.

    Returns
    -------
    dict
        New metadata dict with the same structure but potentially fewer
        nodes.  ``time_horizon`` and ``proc_capacity`` are **not** updated
        (the caller should adjust if needed).
    """
    import copy

    meta = copy.deepcopy(metadata)
    num_ops = meta["num_ops"]
    exec_time = list(meta["exec_time"])
    # Convert comm_cost to a mutable list-of-lists
    comm_cost = [list(row) for row in meta["comm_cost"]]
    resource_demand = list(meta["resource_demand"])

    # Helper: compute in/out degree for each node (directed, topological)
    def _compute_degrees(n):
        in_deg = [0] * n
        out_deg = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                w = comm_cost[u][v]
                if w > 0:
                    out_deg[u] += 1
                    in_deg[v] += 1
        return in_deg, out_deg

    for iteration in range(10):
        in_deg, out_deg = _compute_degrees(num_ops)

        # Find degree-1 nodes (exactly 1 predecessor and 1 successor)
        degree1 = [
            v for v in range(num_ops)
            if in_deg[v] == 1 and out_deg[v] == 1
        ]
        if not degree1:
            break

        # Keep track of which nodes are still alive
        alive = [True] * num_ops

        for v in degree1:
            if not alive[v]:
                continue  # already folded by a previous iteration
            # Find predecessor u (u < v with comm_cost[u][v] > 0)
            u = None
            for cand in range(v):
                if alive[cand] and comm_cost[cand][v] > 0:
                    u = cand
                    break
            # Find successor w (w > v with comm_cost[v][w] > 0)
            w = None
            for cand in range(v + 1, num_ops):
                if alive[cand] and comm_cost[v][cand] > 0:
                    w = cand
                    break
            if u is None or w is None:
                continue

            # Fold v into predecessor u
            exec_time[u] += exec_time[v]

            # Update edge weight: u -> w
            new_weight = comm_cost[u][v] + comm_cost[v][w]
            comm_cost[u][w] = new_weight
            comm_cost[w][u] = new_weight

            # Zero out edges to/from v
            for x in range(num_ops):
                comm_cost[x][v] = 0.0
                comm_cost[v][x] = 0.0

            # Merge resource demand (take the max as a conservative bound)
            resource_demand[u] = max(resource_demand[u], resource_demand[v])
            resource_demand[v] = 0.0

            alive[v] = False

        # ── Re-index: remove dead nodes ───────────────────────────────
        old_to_new = {}
        new_idx = 0
        for old in range(num_ops):
            if alive[old]:
                old_to_new[old] = new_idx
                new_idx += 1

        new_n = new_idx
        new_exec = [0.0] * new_n
        new_comm = [[0.0] * new_n for _ in range(new_n)]
        new_demand = [0.0] * new_n

        for old, new in old_to_new.items():
            new_exec[new] = exec_time[old]
            new_demand[new] = resource_demand[old]
            for old2, new2 in old_to_new.items():
                if old < old2:
                    w = comm_cost[old][old2]
                    if w > 0:
                        new_comm[new][new2] = w
                        new_comm[new2][new] = w

        exec_time = new_exec
        comm_cost = new_comm
        resource_demand = new_demand
        num_ops = new_n
    else:
        # Loop completed without break (max iterations reached)
        pass

    # ── Update metadata ───────────────────────────────────────────────
    meta["num_ops"] = num_ops
    meta["exec_time"] = exec_time
    meta["comm_cost"] = comm_cost
    meta["resource_demand"] = resource_demand
    # time_horizon is intentionally left unchanged; the caller can adjust
    # proc_capacity is left unchanged (its time dimension still matches the
    # original time_horizon)

    return meta


# ---------------------------------------------------------------------------
# E. Unified Loader Interface
# ---------------------------------------------------------------------------


def _make_synthetic_instance(problem_type: str, size: int) -> Dict[str, Any]:
    """Create a single random synthetic instance."""
    random.seed()

    if problem_type == "scheduling":
        num_processors = TPU_V3_CONFIG["num_processors"]
        time_horizon = max(10, size * 2)
        exec_time = [random.uniform(1.0, 5.0) for _ in range(size)]
        comm_cost = [[0.0] * size for _ in range(size)]
        for u in range(size):
            for v in range(u + 1, size):
                if random.random() < 0.3:
                    w_val = random.uniform(0.5, 3.0)
                    comm_cost[u][v] = w_val
                    comm_cost[v][u] = w_val
        resource_demand = [random.uniform(0.5, 2.0) for _ in range(size)]
        proc_capacity = [[random.uniform(4.0, 10.0) for _ in range(time_horizon)]
                         for _ in range(num_processors)]
        return {"problem_type": "scheduling", "metadata": {
            "num_ops": size, "num_processors": num_processors,
            "time_horizon": time_horizon, "exec_time": exec_time,
            "comm_cost": comm_cost, "resource_demand": resource_demand,
            "proc_capacity": proc_capacity,
        }}

    elif problem_type == "coloring":
        num_tensors = size
        max_colors = max(3, num_tensors // 4)
        conflict_edges = []
        for u in range(num_tensors):
            for v in range(u + 1, num_tensors):
                if random.random() < 0.2:
                    conflict_edges.append((u, v))
        tensor_size = [random.uniform(1.0, 10.0) for _ in range(num_tensors)]
        capacity = sum(tensor_size) / max_colors * 1.5
        return {
            "problem_type": "coloring",
            "metadata": {
                "num_tensors": num_tensors,
                "max_colors": max_colors,
                "conflict_edges": conflict_edges,
                "tensor_size": tensor_size,
                "capacity": capacity,
            },
        }

    elif problem_type == "partitioning":
        num_ops = size
        max_groups = max(2, num_ops // 10)
        edge_weights = []
        for u in range(num_ops):
            for v in range(u + 1, num_ops):
                if random.random() < 0.3:
                    w = random.uniform(0.1, 5.0)
                    edge_weights.append((u, v, w))
        op_cost = [random.uniform(1.0, 10.0) for _ in range(num_ops)]
        return {
            "problem_type": "partitioning",
            "metadata": {
                "num_ops": num_ops,
                "max_groups": max_groups,
                "edge_weights": edge_weights,
                "op_cost": op_cost,
            },
        }

    elif problem_type == "coverage":
        num_tests = size
        num_points = size * 3
        coverage_matrix = [[False] * num_points for _ in range(num_tests)]
        for t in range(num_tests):
            n_covered = random.randint(1, max(1, num_points // 5))
            points = random.sample(range(num_points), min(n_covered, num_points))
            for p in points:
                coverage_matrix[t][p] = True
        max_select = max(2, num_tests // 5)
        point_weights = [random.uniform(0.5, 2.0) for _ in range(num_points)]
        return {
            "problem_type": "coverage",
            "metadata": {
                "num_tests": num_tests,
                "num_points": num_points,
                "coverage_matrix": coverage_matrix,
                "max_select": max_select,
                "point_weights": point_weights,
            },
        }

    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


def load_problem_instances(
    source: str = "synthetic",
    source_path: Optional[str] = None,
    problem_type: Optional[str] = None,
    max_instances: int = 100,
    sizes: Optional[List[int]] = None,
    compress: bool = False,
) -> List[Dict[str, Any]]:
    """Load problem instances from a given source.

    Parameters
    ----------
    source : str
        One of ``"synthetic"``, ``"tpugraphs"``, ``"hlo_dump"``, ``"mlperf"``.
    source_path : str or None
        Path to data directory / file (required for ``tpugraphs`` and
        ``hlo_dump``).
    problem_type : str or None
        If given, filter to this problem type (``"scheduling"``,
        ``"coloring"``, ``"partitioning"``, ``"coverage"``).
        For real-world sources only scheduling is supported.
    max_instances : int
        Maximum number of instances to return.
    sizes : list of int or None
        Instance sizes to generate (only used when ``source="synthetic"``).
    compress : bool
        If True and the source is scheduling data, apply degree-1 chain
        reduction to shrink the graph.

    Returns
    -------
    list of dict
        Each dict has ``problem_type`` and ``metadata`` keys, where
        ``metadata`` is compatible with the generator functions in
        :mod:`src.tpu.generators`.
    """
    instances: List[Dict[str, Any]] = []

    if source == "synthetic":
        if sizes is None:
            sizes = [10, 50]
        all_problems = ["scheduling", "coloring", "partitioning", "coverage"]
        if problem_type:
            all_problems = [p for p in all_problems if p == problem_type]
        for size in sizes:
            for pt in all_problems:
                if len(instances) >= max_instances:
                    break
                inst = _make_synthetic_instance(pt, size)
                # Compress only scheduling instances
                if compress and pt == "scheduling":
                    inst["metadata"] = compress_graph(inst["metadata"])
                instances.append(inst)

    elif source == "tpugraphs":
        if source_path is None:
            # Default: look in benchmarks/v0/npz/
            source_path = str(
                pathlib.Path(__file__).resolve().parents[2] / "benchmarks" / "v0" / "npz"
            )
        if not os.path.isdir(source_path):
            print(f"[data] TpuGraphs data dir not found: {source_path}")
            return instances
        instances = load_tpugraphs_batch(
            source_path, max_files=max_instances, compress=compress,
        )

    elif source == "hlo_dump":
        if source_path is None:
            print("[data] source_path required for hlo_dump source")
            return instances
        parsed_list = parse_hlo_dump(source_path)
        for parsed in parsed_list:
            if len(instances) >= max_instances:
                break
            if problem_type and parsed.get("problem_type") != problem_type:
                continue
            if compress and parsed.get("problem_type") == "scheduling":
                parsed = dict(parsed)
                parsed["metadata"] = compress_graph(parsed["metadata"])
            instances.append(parsed)

    elif source == "mlperf":
        models = get_mlperf_model_list()
        for model_name in models:
            if len(instances) >= max_instances:
                break
            parsed = generate_from_mlperf_model(model_name)
            if parsed is not None:
                if problem_type and parsed.get("problem_type") != problem_type:
                    continue
                if compress and parsed.get("problem_type") == "scheduling":
                    parsed = dict(parsed)
                    parsed["metadata"] = compress_graph(parsed["metadata"])
                instances.append(parsed)

    else:
        raise ValueError(f"Unknown data source: {source}")

    print(f"[data] Loaded {len(instances)} instances from source='{source}'")
    return instances
