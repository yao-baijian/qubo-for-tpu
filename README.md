# fem-partition

A Python library for solving **QUBO** (Quadratic Unconstrained Binary Optimization) problems using physics-inspired and quantum-inspired solvers. Originally focused on graph/hypergraph partitioning, now extended to **TPU Full-Stack Optimization** problems including instruction scheduling, memory allocation, operator fusion, and test coverage selection.

## Problem Types

| Category | Type | Description |
|----------|------|-------------|
| **Classic** | Balanced minimum cut | Partition graph/hypergraph vertices into `k` equal-weight blocks minimizing cut edges |
| **Classic** | Max-cut | Partition vertices into two blocks maximizing cut edges |
| **Classic** | Max-SAT | Approximate maximum satisfiability via QUBO encoding |
| **TPU** | Instruction Scheduling | Assign operations to (processor, time) slots under dependency and capacity constraints |
| **TPU** | Memory Allocation | Assign tensors to memory blocks with conflict avoidance and capacity limits |
| **TPU** | Operator Fusion | Cluster operations into groups minimizing inter-cluster data traffic |
| **TPU** | Test Coverage | Select tests to maximize functional coverage under cardinality constraints |

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

| File | Description |
|------|-------------|
| `generators.py` | QUBO builders: `build_scheduling_qubo`, `build_coloring_qubo`, `build_partitioning_qubo`, `build_coverage_qubo` |
| `baselines.py` | Classical heuristics: `list_scheduling`, `greedy_coloring`, `kl_partitioning`, `greedy_coverage` |
| `benchmark.py` | Orchestrator: generates instances, solves with all 3 solvers + baselines, outputs CSV |

### Problem Formulations

| Problem | Variables | Constraints | Objective |
|---------|-----------|-------------|-----------|
| **Scheduling** | `num_ops × num_processors × time_horizon` | Unique assignment, dependencies, resource capacity | Minimize makespan |
| **Coloring** | `num_tensors × K + K` (aux y_c) | Unique color, conflict edges, link x ≤ y, capacity | Minimize colors used |
| **Partitioning** | `num_ops × G` | Unique group, load balancing | Minimize cut weight |
| **Coverage** | `num_tests + num_points` | Implication (x → y), no false positives, exact-K | Maximize coverage |

## Acceleration

- **`torch.compile`** support (opt-in) for FEM `Solver.iterate()` and SBM `bsb_torch_batch` step function.

```python
compile_fem = True    # compile FEM Solver.iterate()
compile_sbm = True    # compile SBM bsb_torch_batch step function
```

## Project Structure

```
src/
├── solver_base.py       # Solver base classes (FemSolver, SbmSolver)
├── method_registry.py   # Method registry + JSON config loading
├── fem/                 # Flexible Entropy Minimization solver + QUBO wrapper
│   ├── __init__.py      #   FemSolver (standard solve interface)
│   ├── interface.py     #   FEM class
│   ├── problem.py       #   OptimizationProblem, expected_qubo, manual_grad_qubo
│   ├── solver_fem.py    #   Solver (mean-field iteration)
│   └── utils.py         #   Graph I/O utilities
├── sbm/                 # Simulated Bifurcation Machines
│   ├── __init__.py      #   SbmSolver (standard solve interface)
│   └── sbm.py           #   BSB/DSB solver
├── qis3/                # Quantum-Inspired Solver v3
│   ├── __init__.py      #   Qis3Solver (standard solve interface)
│   └── qis3.py          #   QIS3 (SB + branch & bound)
├── tpu/                 # TPU Full-Stack Optimization (NEW)
│   ├── __init__.py      #   Public API exports
│   ├── generators.py    #   QUBO matrix builders for 4 problem types
│   ├── baselines.py     #   Classical heuristic baselines
│   └── benchmark.py     #   Benchmark orchestrator
├── digcim/              # Digital Co-Ising Machine experiments
└── configs/             # Default solver JSON configs (fem, sbm)
tests/                   # Test suite
├── test_generators.py   # QUBO generator tests
├── test_baselines.py    # Baseline heuristic tests
├── test_benchmark.py    # Benchmark integration tests
├── config/              # Working config copies (gitignored)
└── build/               # Benchmark CSV outputs
benchmarks/
├── bmincut/             # Balanced min-cut benchmarks
├── maxcut/              # Max-cut benchmarks (Gset, WK2000)
├── maxsat/              # Max-SAT benchmarks
└── tpu/                 # (NEW) TPU instance data
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

```powershell
python -u tests/test_generators.py     # QUBO generator unit tests
python -u tests/test_baselines.py      # Baseline heuristic tests
python -u tests/test_benchmark.py      # Benchmark integration tests
```

### Running the TPU Benchmark

```powershell
# Quick test (smallest sizes)
python -m src.tpu.benchmark --quick

# Full benchmark
python -m src.tpu.benchmark --sizes 10,50,100 --trials 5
```

Results are written to `build/tpu_benchmark_results.csv` with columns:
`problem, size, trial, solver, runtime_seconds, solution_quality, metric_name`

## Documentation

See `doc/` for detailed module documentation:
- `doc/fem.md` — FEM solver details
- `doc/sbm.md` — Simulated Bifurcation details
- `doc/qis3.md` — Quantum-Inspired Solver v3 details

## References

- FEM framework: mean-field entropy minimization with annealing
- Simulated Bifurcation: Goto et al., Science Advances (2019)
- QIS3: SB + Branch & Bound hybrid solver
