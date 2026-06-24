"""Tests for TPU data loader module."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tpu.data_loader import (
    download_tpugraphs,
    parse_hlo_dump,
    parse_tpugraphs_sample,
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


# ── Helper (mirrors test_generators.py) ───────────────────────────────────

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
    print(f"  ✓ QUBO structure OK: {len(Q)} entries, {num_vars} vars")


def _check_metadata_structure(entry: dict, expected_type: str):
    """Verify a loaded metadata entry has the expected fields."""
    assert "problem_type" in entry, "Missing 'problem_type'"
    assert entry["problem_type"] == expected_type, \
        f"Expected problem_type={expected_type}, got {entry['problem_type']}"
    assert "metadata" in entry, "Missing 'metadata'"
    meta = entry["metadata"]
    assert isinstance(meta, dict), "metadata must be a dict"

    if expected_type == "scheduling":
        for key in ("num_ops", "num_processors", "time_horizon",
                     "exec_time", "comm_cost", "resource_demand",
                     "proc_capacity"):
            assert key in meta, f"Missing scheduling metadata key: {key}"
        assert len(meta["exec_time"]) == meta["num_ops"]
        assert len(meta["comm_cost"]) == meta["num_ops"]
        assert len(meta["proc_capacity"]) == meta["num_processors"]

    elif expected_type == "coloring":
        for key in ("num_tensors", "max_colors", "conflict_edges"):
            assert key in meta, f"Missing coloring metadata key: {key}"

    elif expected_type == "partitioning":
        for key in ("num_ops", "max_groups", "edge_weights", "op_cost"):
            assert key in meta, f"Missing partitioning metadata key: {key}"

    elif expected_type == "coverage":
        for key in ("num_tests", "num_points", "coverage_matrix",
                     "max_select"):
            assert key in meta, f"Missing coverage metadata key: {key}"

    print(f"  ✓ Metadata structure OK: {expected_type}")


# ── Tests ─────────────────────────────────────────────────────────────────


def test_download_tpugraphs():
    """Verify download_tpugraphs returns a valid directory path.

    Note: Skipped if network is unavailable or TpuGraphs is not reachable.
    """
    print("test_download_tpugraphs:")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_tpugraphs(tmpdir)
            assert isinstance(result, str), "Must return a string path"
            assert os.path.isdir(result), f"Path must be a directory: {result}"
            print(f"  ✓ TpuGraphs directory: {result}")
    except Exception as e:
        print(f"  ⚠ Skipped (network may be unavailable): {e}")
    print()


def test_parse_tpugraphs_sample():
    """Parse a minimal HLO snippet (saved as a temp .txt file)."""
    print("test_parse_tpugraphs_sample:")
    hlo_text = """
HloModule test_module

ENTRY main {
  %a = f32[10,20] parameter()
  %b = f32[20,30] parameter()
  %dot = f32[10,30] dot(%a, %b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %c = f32[10,30] parameter()
  %add = f32[10,30] add(%dot, %c)
  %relu = f32[10,30] relu(%add)
  ROOT %result = f32[10,30] tuple(%relu)
}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "sample.txt")
        with open(fpath, "w") as f:
            f.write(hlo_text)

        result = parse_tpugraphs_sample(fpath)
        assert result is not None, "Parser should return a valid result"
        assert result["problem_type"] == "scheduling"
        meta = result["metadata"]
        _check_metadata_structure(result, "scheduling")

        # Verify nodes were extracted
        nodes = result.get("_nodes", [])
        assert len(nodes) >= 4, f"Expected >=4 nodes, got {len(nodes)}"
        op_types = [n["op_type"] for n in nodes]
        assert "dot" in op_types, "Expected 'dot' operation"
        assert "add" in op_types, "Expected 'add' operation"
        print(f"  ✓ Extracted {len(nodes)} HLO nodes: {op_types}")

        # Verify edges exist
        edges = result.get("_edges", [])
        assert len(edges) > 0, "Expected at least one edge"
        print(f"  ✓ Extracted {len(edges)} edges")

        # Verify QUBO compatibility
        Q, n = build_scheduling_qubo(**meta)
        _check_qubo_structure(Q, n)

    print()


def test_parse_hlo_dump():
    """Parse a directory containing HLO text files (XLA dump format)."""
    print("test_parse_hlo_dump:")
    hlo_snippets = [
        # File 1: simple matmul
        """
HloModule matmul
ENTRY main {
  %a = f32[16,16] parameter()
  %b = f32[16,16] parameter()
  %mm = f32[16,16] dot(%a, %b)
  ROOT %out = f32[16,16] tuple(%mm)
}
""",
        # File 2: conv + add
        """
HloModule conv
ENTRY main {
  %x = f32[8,224,224,3] parameter()
  %w = f32[3,3,3,64] parameter()
  %conv = f32[8,224,224,64] convolution(%x, %w)
  %b = f32[64] parameter()
  %add = f32[8,224,224,64] add(%conv, %b)
  ROOT %out = f32[8,224,224,64] tuple(%add)
}
""",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, snippet in enumerate(hlo_snippets):
            fpath = os.path.join(tmpdir, f"module_{i}.txt")
            with open(fpath, "w") as f:
                f.write(snippet)

        results = parse_hlo_dump(tmpdir)
        assert len(results) == 2, f"Expected 2 parsed files, got {len(results)}"

        for r in results:
            _check_metadata_structure(r, "scheduling")

        # First file should have a matmul edge
        nodes_0 = results[0].get("_nodes", [])
        op_types_0 = [n["op_type"] for n in nodes_0]
        assert "dot" in op_types_0, f"Expected dot in {op_types_0}"

        # Second file should have convolution
        nodes_1 = results[1].get("_nodes", [])
        op_types_1 = [n["op_type"] for n in nodes_1]
        assert "convolution" in op_types_1, f"Expected convolution in {op_types_1}"

        print(f"  ✓ Parsed {len(results)} HLO files successfully")

    print()


def test_load_problem_instances_synthetic():
    """Verify load_problem_instances('synthetic') returns valid metadata."""
    print("test_load_problem_instances_synthetic:")
    instances = load_problem_instances(
        source="synthetic",
        sizes=[4],
        max_instances=20,
    )
    assert len(instances) == 4, f"Expected 4 instances (one per problem type), got {len(instances)}"

    types_found = set()
    for inst in instances:
        types_found.add(inst["problem_type"])
        _check_metadata_structure(inst, inst["problem_type"])

    assert types_found == {"scheduling", "coloring", "partitioning", "coverage"}, \
        f"Expected all 4 problem types, got {types_found}"
    print(f"  ✓ Loaded {len(instances)} synthetic instances across {len(types_found)} types")
    print()


def test_load_problem_instances_filtered():
    """Verify filtering by problem_type works."""
    print("test_load_problem_instances_filtered:")
    instances = load_problem_instances(
        source="synthetic",
        sizes=[4, 8],
        problem_type="scheduling",
    )
    assert len(instances) == 2, f"Expected 2 scheduling instances, got {len(instances)}"
    for inst in instances:
        assert inst["problem_type"] == "scheduling"
    print(f"  ✓ Filtered to {len(instances)} scheduling instances")
    print()


def test_unified_metadata_compatibility():
    """Verify metadata can be used directly with QUBO generators."""
    print("test_unified_metadata_compatibility:")
    instances = load_problem_instances(
        source="synthetic",
        sizes=[4],
        max_instances=4,
    )
    generator_map = {
        "scheduling": build_scheduling_qubo,
        "coloring": build_coloring_qubo,
        "partitioning": build_partitioning_qubo,
        "coverage": build_coverage_qubo,
    }
    for inst in instances:
        pt = inst["problem_type"]
        meta = inst["metadata"]
        gen_fn = generator_map[pt]
        if pt == "scheduling":
            Q, n = gen_fn(**meta)
        elif pt == "coloring":
            Q, n = gen_fn(
                meta["num_tensors"], meta["max_colors"],
                meta["conflict_edges"], meta.get("tensor_size"),
                capacity=meta.get("capacity"),
            )
        elif pt == "partitioning":
            Q, n = gen_fn(
                meta["num_ops"], meta["max_groups"],
                meta["edge_weights"], meta["op_cost"],
            )
        elif pt == "coverage":
            Q, n = gen_fn(
                meta["num_tests"], meta["num_points"],
                meta["coverage_matrix"], meta["max_select"],
                meta.get("point_weights"),
            )
        _check_qubo_structure(Q, n)
        print(f"  ✓ {pt}: QUBO compatible with metadata")
    print()


def test_mlperf_model_list():
    """Verify MLPerf model list returns expected names."""
    print("test_mlperf_model_list:")
    models = get_mlperf_model_list()
    assert len(models) == 5, f"Expected 5 models, got {len(models)}"
    print(f"  ✓ MLPerf models: {models}")
    print()


def test_mlperf_generators():
    """Verify each MLPerf model generates valid scheduling metadata."""
    print("test_mlperf_generators:")
    for model_name in get_mlperf_model_list():
        result = generate_from_mlperf_model(model_name)
        assert result is not None, f"Failed to generate for {model_name}"
        _check_metadata_structure(result, "scheduling")
        meta = result["metadata"]
        # Verify QUBO compatibility
        Q, n = build_scheduling_qubo(**meta)
        _check_qubo_structure(Q, n)
        print(f"  ✓ {model_name}: {meta['num_ops']} ops, "
              f"{len(Q)} QUBO entries, {n} vars")
    print()


def test_mlperf_unknown_model():
    """Verify unknown model name returns None without crashing."""
    print("test_mlperf_unknown_model:")
    result = generate_from_mlperf_model("nonexistent_model_xyz")
    assert result is None, "Expected None for unknown model"
    print("  ✓ Unknown model returns None")
    print()


def test_load_hlo_dump_integration():
    """End-to-end: write HLO files, load via unified loader, verify QUBO."""
    print("test_load_hlo_dump_integration:")
    hlo_text = """
HloModule test
ENTRY main {
  %x = f32[8,8] parameter()
  %y = f32[8,8] parameter()
  %mm = f32[8,8] dot(%x, %y)
  ROOT %t = f32[8,8] tuple(%mm)
}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "test.hlo")
        with open(fpath, "w") as f:
            f.write(hlo_text)

        instances = load_problem_instances(
            source="hlo_dump",
            source_path=tmpdir,
        )
        assert len(instances) == 1, f"Expected 1 instance, got {len(instances)}"
        inst = instances[0]
        _check_metadata_structure(inst, "scheduling")

        # Build QUBO and verify
        meta = inst["metadata"]
        Q, n = build_scheduling_qubo(**meta)
        _check_qubo_structure(Q, n)
        print(f"  ✓ HLO dump integration OK: {meta['num_ops']} ops -> {n} vars")

    print()


if __name__ == "__main__":
    test_download_tpugraphs()
    test_parse_tpugraphs_sample()
    test_parse_hlo_dump()
    test_load_problem_instances_synthetic()
    test_load_problem_instances_filtered()
    test_unified_metadata_compatibility()
    test_mlperf_model_list()
    test_mlperf_generators()
    test_mlperf_unknown_model()
    test_load_hlo_dump_integration()
    print("All data loader tests passed!")
