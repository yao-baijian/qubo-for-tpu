"""Tests for the unified solver architecture (strategy + mixins)."""

import sys
import importlib.util
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUBO_SOLVER_SRC = ROOT / "lib" / "qubo-solver" / "src"

# Import sbm/sbm.py directly via file path to avoid namespace clash with TPU's src/
_spec = importlib.util.spec_from_file_location(
    "qubo_solver_sbm",
    QUBO_SOLVER_SRC / "sbm" / "sbm.py",
)
_qss = importlib.util.module_from_spec(_spec)
sys.modules["qubo_solver_sbm"] = _qss
_spec.loader.exec_module(_qss)

Solver = _qss.Solver
BaseSolver = _qss.BaseSolver
BSBStrategy = _qss.BSBStrategy
DSBStrategy = _qss.DSBStrategy
AdiabaticStrategy = _qss.AdiabaticStrategy
DigCIMStrategy = _qss.DigCIMStrategy
GSBMixin = _qss.GSBMixin
GGSBMixin = _qss.GGSBMixin
QuantizationMixin = _qss.QuantizationMixin


def _make_ising(N=10):
    J = torch.randn(N, N)
    J = (J + J.T) / 2
    J.fill_diagonal_(0)
    return J


def _make_qubo(J):
    N = J.shape[0]
    return [(i, j, J[i, j].item()) for i in range(N) for j in range(i, N) if J[i, j] != 0]


def test_solver_string_strategy():
    print("test_solver_string_strategy:")
    J = _make_ising(6)
    Q = _make_qubo(J)
    s = Solver(strategy="bsb", num_iters=100, num_trials=4, dt=0.1)
    sol = s.solve(Q, 6)
    assert len(sol) == 6
    assert all(v in (0, 1) for v in sol)
    print(f"  \u2713 Solution: {sol}")
    print()


def test_bsb_gsb_mixin():
    print("test_bsb_gsb_mixin:")
    J = _make_ising(8)
    J_ising = -J / 2.0
    base = BaseSolver(
        strategy=BSBStrategy(dt=0.1),
        enhancements=[GSBMixin(A=0.5)],
        num_iters=200, num_trials=4,
    )
    sols, engs = base.solve(J_ising)
    assert sols.shape == (4, 8)
    assert engs.shape == (4,)
    print(f"  \u2713 Energies: {engs.tolist()}")
    print()


def test_dsb_strategy():
    print("test_dsb_strategy:")
    J_ising = -_make_ising(8) / 2.0
    base = BaseSolver(strategy=DSBStrategy(dt=0.1), num_iters=200, num_trials=4)
    sols, engs = base.solve(J_ising)
    assert sols.shape == (4, 8)
    print(f"  \u2713 Energies: {engs.tolist()}")
    print()


def test_digcim_strategy():
    print("test_digcim_strategy:")
    J_ising = -_make_ising(8) / 2.0
    base = BaseSolver(strategy=DigCIMStrategy(dt=0.1), num_iters=200, num_trials=4)
    sols, engs = base.solve(J_ising)
    assert sols.shape == (4, 8)
    print(f"  \u2713 Energies: {engs.tolist()}")
    print()


def test_all_enhancements():
    print("test_all_enhancements:")
    J_ising = -_make_ising(8) / 2.0
    base = BaseSolver(
        strategy=BSBStrategy(dt=0.1),
        enhancements=[
            GSBMixin(A=1.0),
            GGSBMixin(k=20, strength=0.05),
            QuantizationMixin(num_bits=8),
        ],
        num_iters=200, num_trials=4,
    )
    sols, engs = base.solve(J_ising)
    assert sols.shape == (4, 8)
    print(f"  \u2713 Energies: {engs.tolist()}")
    print()


def test_gsb_zero_A_fallback():
    """GSB with A=0 should match plain BSB."""
    print("test_gsb_zero_A_fallback:")
    J_ising = -_make_ising(10) / 2.0

    base_bsb = BaseSolver(strategy=BSBStrategy(dt=0.1), num_iters=200, num_trials=4)
    base_gsb = BaseSolver(
        strategy=BSBStrategy(dt=0.1),
        enhancements=[GSBMixin(A=0.0)],
        num_iters=200, num_trials=4,
    )
    sols_bsb, engs_bsb = base_bsb.solve(J_ising.clone())
    sols_gsb, engs_gsb = base_gsb.solve(J_ising.clone())
    assert torch.allclose(engs_bsb, engs_gsb, atol=1e-4), \
        f"A=0 GSB should match BSB: {engs_bsb.tolist()} vs {engs_gsb.tolist()}"
    print(f"  \u2713 Engines match: {engs_bsb.tolist()} == {engs_gsb.tolist()}")
    print()


def test_adiabatic_strategy():
    print("test_adiabatic_strategy:")
    J_ising = -_make_ising(8) / 2.0
    base = BaseSolver(strategy=AdiabaticStrategy(dt=0.1), num_iters=200, num_trials=4)
    sols, engs = base.solve(J_ising)
    assert sols.shape == (4, 8)
    print(f"  \u2713 Energies: {engs.tolist()}")
    print()


if __name__ == "__main__":
    test_solver_string_strategy()
    test_bsb_gsb_mixin()
    test_dsb_strategy()
    test_digcim_strategy()
    test_adiabatic_strategy()
    test_all_enhancements()
    test_gsb_zero_A_fallback()
    print("All unified solver tests passed!")
