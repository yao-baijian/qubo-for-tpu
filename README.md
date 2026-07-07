# qubo-for-tpu

A Python library for solving **QUBO** (Quadratic Unconstrained Binary Optimization) problems using physics-inspired and quantum-inspired solvers. Focus on **TPU Full-Stack Optimization** problems across the entire lifecycle: **architecture design**, **compiler mapping**, and **runtime management**.

## Overview

This project provides a **unified QUBO framework** for four fundamental classes of combinatorial optimization problems that arise in TPU systems:

| QUBO Class | TPU Problems | Lifecycle Stage |
|------------|--------------|-----------------|
Ising/QUBO solvers (FEM, SBM) are provided by the external
**[qubo-solver](https://github.com/yao-baijian/qubo-solver)** submodule
at ``lib/qubo-solver/``.  Import helpers in ``src/__init__.py``
automatically add the submodule to ``sys.path``.

| **Assignment** | Instruction scheduling, task-to-core mapping, DVFS level selection | Compile-time, Runtime |
| **Coloring** | Tensor lifetime memory allocation, SRAM bank mapping | Compile-time, Design-time |
| **Partitioning** | Operator fusion, NoC topology/routing partition | Compile-time, Design-time |
| **Set Coverage** | Test case selection for functional verification | Design-time |

The framework enables:
- **Cross-stage co-optimization** using a single QUBO solver backend
- **Fair comparison** between QUBO-based methods and classical heuristics
- **Integration with real-world TPU data** (TpuGraphs dataset, XLA HLO dumps)

## Project Status

| Component | Status | Description |
|-----------|--------|-------------|
| QUBO generators (4 problem types) | ✅ Complete | Build sparse QUBO matrices for scheduling, coloring, partitioning, coverage |
| Baseline heuristics (4 types) | ✅ Complete | List scheduling, greedy coloring, KL partitioning, greedy coverage |
| Benchmark orchestrator | ✅ Complete | Unified benchmarking pipeline with CSV output |
| Test suite | ✅ Complete | Unit tests for generators, baselines, benchmark |
| TpuGraphs data loader | ✅ Complete | Load `.npz` files, extract HLO graphs, convert to QUBO input |
| Time-window pruning | ✅ Complete | EST/LST variable reduction for scheduling QUBO |
| Graph compression | ✅ Complete | Degree-1 chain folding for DAG reduction |
| Constraint violation reporting | ✅ Complete | Per-constraint violation breakdown in CSV output |
| **Node exec_time estimator** | 🚧 In Progress | FLOPs-based execution time estimation from HLO features |
| **Lifetime inference** | 🚧 In Progress | ASAP scheduling to infer tensor lifetimes from HLO DAG |
| **Hardware config module** | 🚧 In Progress | TPU v3 parameter definitions and validation |
| **End-to-end data pipeline** | 🚧 In Progress | `.npz` → metadata → QUBO → solver → evaluation |

## Data Sources

The project supports multiple data sources for TPU optimization problems:

| Source | Format | Description | Status |
|--------|--------|-------------|--------|
| **Synthetic** | Generated on-the-fly | Random instances for testing and scaling studies | ✅ Available |
| **TpuGraphs** | `.npz` files | Google's public dataset of HLO graphs with TPU execution times | 🚧 Loader in progress |
| **XLA HLO dump** | `.txt` / `.hlo` files | Custom HLO graphs from JAX/TensorFlow via `XLA_FLAGS="--xla_dump_to=..."` | 🚧 Loader in progress |
| **MLPerf** | Model definitions | Standard benchmark models (ResNet-50, BERT, 3D-Unet, etc.) | 🚧 Synthetic generator in progress |

### TpuGraphs Dataset

The TpuGraphs dataset is located at `http://download.tensorflow.org/data/tpu_graphs/v0`. It contains:

- **Tile collection**: Graph-level configurations (kernel-level optimization)
- **Layout collection**: Node-level configurations (full program optimization)

Each `.npz` file contains:
- `node_opcode`: HLO opcode for each node
- `node_feat`: 140-dim feature vector (shapes, types, convolution parameters, etc.)
- `edge_index`: Directed edges representing data dependencies
- `config_feat`: Configuration features
- `config_runtime`: Actual execution time on TPU v3 (nanoseconds)

The data pipeline converts TpuGraphs `.npz` files to QUBO generator input metadata.

## Solvers

All solvers expose a standard `solve(Q, num_vars)` interface where `Q` is a sparse upper-triangular QUBO matrix as `List[Tuple[int, int, float]]`.

| Solver | Import | Description |
|--------|--------|-------------|
| **FEM** | `from src.fem import FemSolver` | Mean-field entropy minimization with simulated annealing |
| **SBM** | `from src.sbm import SbmSolver` | Simulated Bifurcation — physics-inspired Ising machine solver |
| **QIS3** | `from src.qis3 import Qis3Solver` | Quantum-Inspired Solver v3 (SB + branch & bound + adaptive perturbation) |

### Standard Solver Interface

```python
from src.fem import FemSolver

solver = FemSolver(num_trials=10, num_steps=1000)
solution = solver.solve(Q, num_vars)  # returns List[int] of 0/1
```

## TPU Optimization Module

Located in `src/tpu/`:

| File | Description | Status |
|------|-------------|--------|
| `generators.py` | QUBO builders for 4 problem types | ✅ Complete |
| `baselines.py` | Classical heuristic baselines | ✅ Complete |
| `benchmark.py` | Unified benchmark orchestrator | ✅ Complete |
| `data_loader.py` | TpuGraphs/HLO data loading and conversion | 🚧 In Progress |

### Problem Formulations

| Problem | Variables | Constraints | Objective |
|---------|-----------|-------------|-----------|
| **Scheduling** | `num_ops × num_processors × time_horizon` | Unique assignment, dependencies, resource capacity | Minimize makespan |
| **Coloring** | `num_tensors × K + K` (aux y_c) | Unique color, conflict edges, link x ≤ y, capacity | Minimize colors used |
| **Partitioning** | `num_ops × G` | Unique group, load balancing | Minimize cut weight |
| **Coverage** | `num_tests + num_points` | Implication (x → y), no false positives, exact-K | Maximize coverage |

### Hardware Configuration

The framework uses TPU v3 specifications as the default hardware model:

```python
TPU_V3_CONFIG = {
    "num_processors": 4,                # 4-core topology
    "peak_tops": 92.0,                  # BF16/INT8 peak (TOPS)
    "effective_tops": 70.0,             # Effective after utilization
    "sram_capacity_mb": 32,             # On-chip SRAM per core (MiB)
    "hbm_bandwidth_gbps": 900,          # HBM bandwidth (GB/s)
    "ici_bandwidth_gbps": 100,          # Inter-chip interconnect (GB/s)
    "num_banks": 16,                    # SRAM banks for coloring
}
```

### Data Pipeline (TpuGraphs → QUBO)

The pipeline converts raw TpuGraphs `.npz` data to QUBO input metadata:

```
┌─────────────────┐
│   TpuGraphs     │
│   .npz file     │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Extract HLO     │  ← node_opcode, node_feat (140-dim), edge_index
│ Graph Structure │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Compute exec_time│  ← FLOPs estimation from opcode + shape_features
│ per node        │      exec_time = FLOPs / effective_tops
└────────┬────────┘
         ▼
┌─────────────────┐
│ Infer lifetimes │  ← ASAP scheduling on topological order
│ (ASAP schedule) │      interval = [finish(pred), start(node)]
└────────┬────────┘
         ▼
┌─────────────────┐
│ Compute comm_cost│  ← size / ici_bandwidth × hops
│ per edge        │      hops = 0 (same core) or 1 (cross-core)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Build QUBO      │  ← generators.build_scheduling_qubo(**metadata)
│ Matrix (Q)      │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Solver          │  ← FEM / SBM / QIS3
│ (QUBO solve)    │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Evaluate        │  ← Compare QUBO vs baselines (makespan, colors, cut, coverage)
│ & Compare       │
└─────────────────┘
```

## Acceleration

- **`torch.compile`** support (opt-in) for FEM `Solver.iterate()` and SBM `bsb_torch_batch` step function.

```python
compile_fem = True    # compile FEM Solver.iterate()
compile_sbm = True    # compile SBM bsb_torch_batch step function
```

## Project Structure

```
src/
├── solver_base.py       # Solver base classes (FemSolver, SbmSolver, Qis3Solver)
├── method_registry.py   # Method registry + JSON config loading
├── fem/                 # Flexible Entropy Minimization solver
│   ├── __init__.py      #   FemSolver (standard solve interface)
│   ├── interface.py     #   FEM class
│   ├── problem.py       #   OptimizationProblem, QUBO wrapper
│   ├── solver_fem.py    #   Mean-field iteration solver
│   └── utils.py         #   Utilities
├── sbm/                 # Simulated Bifurcation Machines
│   ├── __init__.py      #   SbmSolver (standard solve interface)
│   └── sbm.py           #   BSB/DSB solver
├── qis3/                # Quantum-Inspired Solver v3
│   ├── __init__.py      #   Qis3Solver (standard solve interface)
│   └── qis3.py          #   SB + branch & bound
├── tpu/                 # TPU Full-Stack Optimization
│   ├── __init__.py      #   Public API exports
│   ├── generators.py    #   QUBO builders for 4 problem types (✅ Complete)
│   ├── baselines.py     #   Classical heuristic baselines (✅ Complete)
│   ├── benchmark.py     #   Benchmark orchestrator (✅ Complete)
│   └── data_loader.py   #   TpuGraphs/HLO data loader (🚧 In Progress)
├── digcim/              # Digital Co-Ising Machine experiments
└── configs/             # Default solver JSON configs (fem, sbm)

tests/                   # Test suite
├── test_generators.py   # QUBO generator tests (✅ Complete)
├── test_baselines.py    # Baseline heuristic tests (✅ Complete)
├── test_benchmark.py    # Benchmark integration tests (✅ Complete)
├── test_data_loader.py  # Data loader tests (🚧 In Progress)
├── config/              # Working config copies (gitignored)
└── build/               # Benchmark CSV outputs

benchmarks/
├── v0/npz_all/          # TpuGraphs v0 dataset (download manually)
│   └── npz/
│       ├── tile/xla/
│       │   ├── train/   # Training split
│       │   ├── valid/   # Validation split
│       │   └── test/    # Test split
│       └── layout/      # Layout collections (xla/nlp, random/default)
│
doc/                    # Module documentation
config/                 # Working solver configs (copied from src/configs/)
build/                  # Benchmark result CSVs
```

## Installation

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate fem

# 2. Install PyTorch (see https://pytorch.org/)
pip3 install torch torchvision torchaudio
```

## Configuration

Each solver has a default JSON config under `src/configs/`. At runtime these are copied to `config/` (gitignored) where you can override them:

```
config/
├── fem.json
└── sbm.json
```

Use `method_registry.ensure_configs()` to populate the working directory, or manually edit the JSON files in `config/`.

## Running Tests

Run from project root:

```bash
python -u tests/test_generators.py     # QUBO generator unit tests
python -u tests/test_baselines.py      # Baseline heuristic tests
python -u tests/test_benchmark.py      # Benchmark integration tests
```

## Running the TPU Benchmark

```bash
# Quick test (smallest sizes)
python -m src.tpu.benchmark --quick

# Full benchmark
python -m src.tpu.benchmark --sizes 10,50,100 --trials 5

# Use TpuGraphs data (when data_loader is complete)
python -m src.tpu.benchmark --data-source tpugraphs --data-path benchmarks/v0/npz_all/npz/tile/xla/train/
```

Results are written to `build/tpu_benchmark_results.csv` with columns:
`problem, size, trial, solver, runtime_seconds, solution_quality, metric_name`

## Documentation

See `doc/` for detailed module documentation:
- `doc/fem.md` — FEM solver details
- `doc/sbm.md` — Simulated Bifurcation details
- `doc/qis3.md` — Quantum-Inspired Solver v3 details

## References

- **TPU Architecture**: Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit" (ISCA 2017)
- **TpuGraphs Dataset**: Phothilimthana et al., "TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs" (NeurIPS 2023)
- **FEM framework**: Shen et al., "Free-energy machine for combinatorial optimization" (Nature Computational Science 2025)
- **Simulated Bifurcation**: Goto et al., "Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems" (Science Advances 2019)
- **QIS3**: Tatsumura et al., "Scaling out Ising machines using a multi-chip architecture for simulated bifurcation" (Nature Electronics 2021)

## License

This is not an officially supported product. See LICENSE for details.