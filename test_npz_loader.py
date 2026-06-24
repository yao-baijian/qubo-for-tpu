"""Quick integration test for TpuGraphs .npz data pipeline."""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
from src.tpu.data_loader import (
    TPU_V3_CONFIG, estimate_exec_time, infer_lifetimes,
    compute_comm_cost, load_tpugraphs_npz, load_problem_instances,
)

# Test 1: estimate_exec_time
print("Test 1: estimate_exec_time")
feat = np.zeros(140)
feat[28] = 1000.0
feat[5] = 1.0
for oc, name in [(45, "DOT"), (57, "ADD"), (5, "CONV")]:
    t = estimate_exec_time(oc, feat)
    print(f"  {name}({oc}) = {t:.2f} ns")
    assert t > 0
print("  PASS")

# Test 2: infer_lifetimes
print("Test 2: infer_lifetimes")
ei = np.array([[0, 1], [1, 2], [0, 2]])
et = np.array([10.0, 20.0, 5.0])
st, ft, intervals = infer_lifetimes(ei, et, 3)
print(f"  start={st}, finish={ft}")
assert st[0] == 0.0
assert st[2] == 30.0
print("  PASS")

# Test 3: compute_comm_cost
print("Test 3: compute_comm_cost")
cost = compute_comm_cost(1024)
print(f"  1024 bytes: {cost:.6f} ns")
assert cost > 0
print("  PASS")

# Test 4: load a real npz file
print("Test 4: load_tpugraphs_npz")
sample = "benchmarks/v0/npz/layout/xla/default/test/937ee0eb0d5d6151b7b8252933b5c1c9.npz"
full_path = os.path.join(os.getcwd(), sample)
if os.path.exists(full_path):
    result = load_tpugraphs_npz(full_path)
    assert result is not None
    meta = result["metadata"]
    n_comm = sum(1 for row in meta["comm_cost"] for v in row if v > 0)
    print(f"  num_ops={meta['num_ops']}, time_horizon={meta['time_horizon']}")
    print(f"  exec_time[0]={meta['exec_time'][0]:.2f} ns")
    print(f"  nonzero comm_cost entries: {n_comm}")
    print("  PASS")
else:
    print(f"  SKIP (file not found)")

# Test 5: load_problem_instances with tpugraphs source
print("Test 5: load_problem_instances(tpugraphs)")
npz_dir = "benchmarks/v0/npz"
if os.path.isdir(npz_dir):
    insts = load_problem_instances(source="tpugraphs", source_path=npz_dir, max_instances=3)
    print(f"  Loaded {len(insts)} instances")
    assert len(insts) > 0
    assert insts[0]["problem_type"] == "scheduling"
    print("  PASS")
else:
    print(f"  SKIP (npz dir not found)")

# Test 6: verify QUBO compatibility
print("Test 6: QUBO compatibility")
from src.tpu.generators import build_scheduling_qubo
if os.path.exists(full_path):
    result = load_tpugraphs_npz(full_path)
    meta = result["metadata"]
    Q, n = build_scheduling_qubo(**meta)
    print(f"  QUBO: {len(Q)} entries, {n} vars")
    assert len(Q) > 0
    assert n == meta["num_ops"] * meta["num_processors"] * meta["time_horizon"]
    print("  PASS")

print("\nAll tests passed!")
