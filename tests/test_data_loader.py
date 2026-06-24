"""Tests for TPU data loader module.

Covers synthetic, HLO dump, MLPerf, and TpuGraphs .npz data sources.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tpu.data_loader import (
    TPU_V3_CONFIG,
    estimate_exec_time,
    infer_lifetimes,
    compute_comm_cost,
    load_tpugraphs_npz,
    load_tpugraphs_batch,
    parse_hlo_dump,
    load_problem_instances,
    generate_from_mlperf_model,
    get_mlperf_model_list,
)
from src.tpu.generators import (
    build_scheduling_qubo,
    build_coloring_qubo,
    build_partitioning_qubo,
    build_coverage_qubo,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _check_qubo_structure(Q, num_vars):
    """Verify basic QUBO structure invariants."""
    assert isinstance(Q, list), "Q must be a list"
    assert len(Q) > 0, "Q must not be empty"
    for entry in Q:
        assert len(entry) == 3, f"Each entry must be (i, j, val), got {entry}"
        i, j, val = entry
        assert 0 <= i < num_vars, f"Index i={i} out of range [0, {num_vars})"
        assert 0 <= j < num_vars, f"Index j={j} out of range [0, {num_vars})"
        assert i <= j, f"Upper-triangular violated: i={i} > j={j}"
        assert isinstance(val, (int, float)), f"Value must be numeric, got {val}"
    print(f"  \u2713 QUBO structure OK: {len(Q)} entries, {num_vars} vars")


def _check_metadata_structure(entry: dict, expected_type: str):
    """Verify a loaded metadata entry has the expected fields."""
    assert "problem_type" in entry, "Missing 'problem_type'"
    assert entry["problem_type"] == expected_type
    assert "metadata" in entry, "Missing 'metadata'"
    meta = entry["metadata"]
    assert isinstance(meta, dict)

    if expected_type == "scheduling":
        for key in ("num_ops", "num_processors", "time_horizon",
                     "exec_time", "comm_cost", "resource_demand",
                     "proc_capacity"):
            assert key in meta, f"Missing scheduling key: {key}"
        assert len(meta["exec_time"]) == meta["num_ops"]
        assert len(meta["comm_cost"]) == meta["num_ops"]
        assert len(meta["proc_capacity"]) == meta["num_processors"]
    elif expected_type == "coloring":
        for key in ("num_tensors", "max_colors", "conflict_edges"):
            assert key in meta
    elif expected_type == "partitioning":
        for key in ("num_ops", "max_groups", "edge_weights", "op_cost"):
            assert key in meta
    elif expected_type == "coverage":
        for key in ("num_tests", "num_points", "coverage_matrix", "max_select"):
            assert key in meta
    print(f"  \u2713 Metadata structure OK: {expected_type}")


# ═════════════════════════════════════════════════════════════════════════
# 1. Core Pipeline Tests (estimate_exec_time, infer_lifetimes, comm_cost)
# ═════════════════════════════════════════════════════════════════════════


def test_estimate_exec_time():
    """Verify estimate_exec_time on known opcodes."""
    print("test_estimate_exec_time:")
    feat = np.zeros(140)
    feat[28] = 1000.0
    feat[5] = 1.0
    for oc, name in [(45, "DOT"), (57, "ADD"), (5, "CONV")]:
        t = estimate_exec_time(oc, feat)
        assert t > 0, f"{name}: expected positive time, got {t}"
        print(f"  \u2713 {name}({oc}) = {t:.2f} ns")
    assert estimate_exec_time(2, feat) > 0  # constant
    print()


def test_infer_lifetimes():
    """Verify ASAP scheduler on a simple DAG."""
    print("test_infer_lifetimes:")
    ei = np.array([[0, 1], [1, 2], [0, 2]])
    et = np.array([10.0, 20.0, 5.0])
    st, ft, intervals = infer_lifetimes(ei, et, 3)
    assert st[0] == 0.0
    assert st[1] == 10.0
    assert st[2] == 30.0
    assert ft[2] == 35.0
    assert len(intervals) == 3
    print(f"  \u2713 start={st}, finish={ft}, intervals={len(intervals)}")
    print()


def test_compute_comm_cost():
    """Verify communication cost calculation."""
    print("test_compute_comm_cost:")
    cost = compute_comm_cost(1024, hops=1)
    assert cost > 0, "Expected positive cost"
    cost2 = compute_comm_cost(1024, hops=2)
    assert cost2 == 2 * cost, "Doubling hops should double cost"
    print(f"  \u2713 1024 bytes, 1 hop: {cost:.6f} ns")
    print()


# ═════════════════════════════════════════════════════════════════════════
# 2. TpuGraphs .npz Tests
# ═════════════════════════════════════════════════════════════════════════

def _find_npz_sample():
    """Find a small .npz file in benchmarks/v0/npz/ for testing."""
    search_path = ROOT / "benchmarks" / "v0" / "npz"
    if not search_path.is_dir():
        return None
    for fpath in sorted(search_path.rglob("*.npz")):
        return str(fpath)
    return None


def test_load_tpugraphs_npz():
    """Load a real .npz file with max_nodes limit."""
    print("test_load_tpugraphs_npz:")
    sample = _find_npz_sample()
    if sample is None:
        print("  \u26a0 Skipped (no .npz files found)")
        return

    result = load_tpugraphs_npz(sample, max_nodes=50)
    assert result is not None
    assert result["problem_type"] == "scheduling"
    _check_metadata_structure(result, "scheduling")
    meta = result["metadata"]
    assert meta["num_ops"] <= 50
    print(f"  \u2713 Loaded {meta['num_ops']} ops from {Path(sample).name}")
    print()


def test_load_tpugraphs_npz_qubo():
    """Verify .npz metadata is compatible with build_scheduling_qubo."""
    print("test_load_tpugraphs_npz_qubo:")
    sample = _find_npz_sample()
    if sample is None:
        print("  \u26a0 Skipped (no .npz files found)")
        return

    result = load_tpugraphs_npz(sample, max_nodes=30)
    assert result is not None
    meta = result["metadata"]
    Q, n = build_scheduling_qubo(**meta)
    _check_qubo_structure(Q, n)
    expected_n = meta["num_ops"] * meta["num_processors"] * meta["time_horizon"]
    assert n == expected_n, f"Expected {expected_n} vars, got {n}"
    print(f"  \u2713 {meta['num_ops']} ops -> {n} QUBO vars, {len(Q)} entries")
    print()


def test_load_tpugraphs_batch():
    """Load a batch of .npz files from benchmarks directory."""
    print("test_load_tpugraphs_batch:")
    npz_dir = str(ROOT / "benchmarks" / "v0" / "npz")
    if not os.path.isdir(npz_dir):
        print("  \u26a0 Skipped (no npz dir)")
        return

    instances = load_tpugraphs_batch(npz_dir, max_files=3, max_nodes=50)
    assert len(instances) > 0
    assert instances[0]["problem_type"] == "scheduling"
    print(f"  \u2713 Loaded {len(instances)} instances")
    print()


def test_load_problem_instances_tpugraphs():
    """Load via the unified interface with tpugraphs source."""
    print("test_load_problem_instances_tpugraphs:")
    npz_dir = str(ROOT / "benchmarks" / "v0" / "npz")
    if not os.path.isdir(npz_dir):
        print("  \u26a0 Skipped (no npz dir)")
        return

    insts = load_problem_instances(
        source="tpugraphs", source_path=npz_dir, max_instances=2,
    )
    assert len(insts) >= 1, f"Expected >=1 instances, got {len(insts)}"
    for inst in insts:
        assert inst["problem_type"] == "scheduling"
    print(f"  \u2713 Loaded {len(insts)} instances from tpugraphs source")
    print()


# ═════════════════════════════════════════════════════════════════════════
# 3. HLO Dump Tests
# ═════════════════════════════════════════════════════════════════════════


def test_parse_hlo_dump():
    """Parse a directory containing HLO text files."""
    print("test_parse_hlo_dump:")
    hlo_snippets = [
        """
HloModule matmul
ENTRY main {
  %a = f32[16,16] parameter()
  %b = f32[16,16] parameter()
  %mm = f32[16,16] dot(%a, %b)
  ROOT %out = f32[16,16] tuple(%mm)
}""",
        """
HloModule conv
ENTRY main {
  %x = f32[8,224,224,3] parameter()
  %w = f32[3,3,3,64] parameter()
  %conv = f32[8,224,224,64] convolution(%x, %w)
  %b = f32[64] parameter()
  %add = f32[8,224,224,64] add(%conv, %b)
  ROOT %out = f32[8,224,224,64] tuple(%add)
}""",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, snippet in enumerate(hlo_snippets):
            fp = os.path.join(tmpdir, f"module_{i}.txt")
            with open(fp, "w") as f:
                f.write(snippet)

        results = parse_hlo_dump(tmpdir)
        assert len(results) == 2
        for r in results:
            _check_metadata_structure(r, "scheduling")

        types0 = [n["op_type"] for n in results[0].get("_nodes", [])]
        assert "dot" in types0
        types1 = [n["op_type"] for n in results[1].get("_nodes", [])]
        assert "convolution" in types1
        print(f"  \u2713 Parsed {len(results)} HLO files")
    print()


def test_load_hlo_dump_integration():
    """End-to-end HLO dump loading + QUBO build."""
    print("test_load_hlo_dump_integration:")
    hlo_text = """
HloModule test
ENTRY main {
  %x = f32[8,8] parameter()
  %y = f32[8,8] parameter()
  %mm = f32[8,8] dot(%x, %y)
  ROOT %t = f32[8,8] tuple(%mm)
}"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = os.path.join(tmpdir, "test.hlo")
        with open(fp, "w") as f:
            f.write(hlo_text)

        insts = load_problem_instances(source="hlo_dump", source_path=tmpdir)
        assert len(insts) == 1
        _check_metadata_structure(insts[0], "scheduling")
        meta = insts[0]["metadata"]
        Q, n = build_scheduling_qubo(**meta)
        _check_qubo_structure(Q, n)
        print(f"  \u2713 {meta['num_ops']} ops -> {n} vars")
    print()


# ═════════════════════════════════════════════════════════════════════════
# 4. Synthetic Instance Tests
# ═════════════════════════════════════════════════════════════════════════


def test_load_problem_instances_synthetic():
    """Verify synthetic loader returns all 4 problem types."""
    print("test_load_problem_instances_synthetic:")
    insts = load_problem_instances(source="synthetic", sizes=[4])
    assert len(insts) == 4
    types = {i["problem_type"] for i in insts}
    assert types == {"scheduling", "coloring", "partitioning", "coverage"}
    for inst in insts:
        _check_metadata_structure(inst, inst["problem_type"])
    print(f"  \u2713 {len(insts)} instances across {len(types)} types")
    print()


def test_load_problem_instances_filtered():
    """Verify problem_type filtering."""
    print("test_load_problem_instances_filtered:")
    insts = load_problem_instances(
        source="synthetic", sizes=[4, 8], problem_type="scheduling",
    )
    assert len(insts) == 2
    for inst in insts:
        assert inst["problem_type"] == "scheduling"
    print(f"  \u2713 Filtered to {len(insts)} scheduling instances")
    print()


def test_unified_metadata_compatibility():
    """Synthetic metadata → QUBO generators."""
    print("test_unified_metadata_compatibility:")
    gen_map = {
        "scheduling": build_scheduling_qubo,
        "coloring": build_coloring_qubo,
        "partitioning": build_partitioning_qubo,
        "coverage": build_coverage_qubo,
    }
    insts = load_problem_instances(source="synthetic", sizes=[4])
    for inst in insts:
        pt = inst["problem_type"]
        meta = inst["metadata"]
        fn = gen_map[pt]
        if pt == "scheduling":
            Q, n = fn(**meta)
        elif pt == "coloring":
            Q, n = fn(meta["num_tensors"], meta["max_colors"],
                       meta["conflict_edges"], meta.get("tensor_size"),
                       capacity=meta.get("capacity"))
        elif pt == "partitioning":
            Q, n = fn(meta["num_ops"], meta["max_groups"],
                       meta["edge_weights"], meta["op_cost"])
        elif pt == "coverage":
            Q, n = fn(meta["num_tests"], meta["num_points"],
                       meta["coverage_matrix"], meta["max_select"],
                       meta.get("point_weights"))
        _check_qubo_structure(Q, n)
    print(f"  \u2713 All {len(insts)} types QUBO-compatible")
    print()


# ═════════════════════════════════════════════════════════════════════════
# 5. MLPerf Tests
# ═════════════════════════════════════════════════════════════════════════


def test_mlperf_model_list():
    """Verify MLPerf model list."""
    print("test_mlperf_model_list:")
    models = get_mlperf_model_list()
    assert len(models) == 5
    print(f"  \u2713 {models}")
    print()


def test_mlperf_generators():
    """Each MLPerf model generates valid QUBO-compatible metadata."""
    print("test_mlperf_generators:")
    for model in get_mlperf_model_list():
        result = generate_from_mlperf_model(model)
        assert result is not None
        _check_metadata_structure(result, "scheduling")
        meta = result["metadata"]
        Q, n = build_scheduling_qubo(**meta)
        _check_qubo_structure(Q, n)
        print(f"  \u2713 {model}: {meta['num_ops']} ops, {n} vars")
    print()


def test_mlperf_unknown_model():
    """Unknown model returns None."""
    print("test_mlperf_unknown_model:")
    assert generate_from_mlperf_model("nonexistent") is None
    print("  \u2713 None returned")
    print()


# ═════════════════════════════════════════════════════════════════════════
# Run all
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Core pipeline
    test_estimate_exec_time()
    test_infer_lifetimes()
    test_compute_comm_cost()

    # TpuGraphs .npz
    test_load_tpugraphs_npz()
    test_load_tpugraphs_npz_qubo()
    test_load_tpugraphs_batch()
    test_load_problem_instances_tpugraphs()

    # HLO dump
    test_parse_hlo_dump()
    test_load_hlo_dump_integration()

    # Synthetic
    test_load_problem_instances_synthetic()
    test_load_problem_instances_filtered()
    test_unified_metadata_compatibility()

    # MLPerf
    test_mlperf_model_list()
    test_mlperf_generators()
    test_mlperf_unknown_model()

    print("All data loader tests passed!")
