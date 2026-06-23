"""
Solver base classes — each solver type becomes a class that loads its own
config once on initialisation and exposes phase methods (coarsen,
initial_partition, refine).

Usage::
    from src.solver_base import FemSolver, SbmSolver

    fem = FemSolver()              # loads config/fem.json
    fem.get_param("num_trials")    # read any config key
    fem.set_param("num_trials", 20)  # override at runtime
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch

from src.method_registry import load_config


def _default_config_dir() -> Path:
    return Path.cwd() / "config"


class SolverBase:
    type: str = "base"

    def __init__(self, config_dir: Optional[Path] = None):
        self._config_dir = Path(config_dir) if config_dir else _default_config_dir()
        self._config: Dict[str, Any] = load_config(self.type, self._config_dir)

    def get_param(self, key: str, default=None):
        return self._config.get(key, default)

    def set_param(self, key: str, value):
        self._config[key] = value

    def update_params(self, **kwargs):
        self._config.update(kwargs)

    def get_all_params(self) -> Dict[str, Any]:
        return dict(self._config)

    def initial_partition(self, coarse_J, coarse_weights, coarse_groups, q, **overrides):
        raise NotImplementedError

    def refine(self, J, initial_partition, q, **overrides):
        raise NotImplementedError

    def solve_direct(self, J, q, **overrides):
        raise NotImplementedError


class FemSolver(SolverBase):
    type = "fem"

    def initial_partition(self, coarse_J, coarse_weights, coarse_groups, q, **overrides):
        import torch
        from src.fem import FEM
        p = {**self._config, **overrides}
        nc = coarse_J.shape[0]
        nnz = coarse_J._nnz() if coarse_J.is_sparse else torch.count_nonzero(coarse_J).item()
        case = FEM.from_couplings("bmincut", nc, int(nnz // 2),
                                   coarse_J, node_weights=coarse_weights)
        case.set_up_solver(p.get("num_trials", 10), p.get("num_steps", 1000),
                            anneal=p.get("anneal", "lin"), dev=p.get("dev", "cpu"),
                            q=q, manual_grad=p.get("manual_grad", False),
                            use_compile=p.get("use_compile", False))
        config, result = case.solve()
        best = config[torch.argwhere(result == result.min()).reshape(-1)[0]]
        return best.argmax(dim=1).cpu().numpy()

    def solve_direct(self, case_type, instance, index_start, q, **overrides):
        import time
        from src.fem import FEM
        p = {**self._config, **overrides}
        case = FEM.from_file(case_type, instance, index_start)
        case.set_up_solver(p.get("num_trials", 10), p.get("num_steps", 1000),
                            anneal=p.get("anneal", "lin"), dev=p.get("dev", "cpu"),
                            q=q, manual_grad=p.get("manual_grad", False),
                            use_compile=p.get("use_compile", False))
        t0 = time.perf_counter()
        config, result = case.solve()
        dt = time.perf_counter() - t0
        out = config[torch.argwhere(result == result.min()).reshape(-1)[0]]
        return out, result.min().item(), dt


class SbmSolver(SolverBase):
    type = "sbm"

    def initial_partition(self, coarse_J, coarse_weights, coarse_groups, q, **overrides):
        from src.sbm.sbm import bsb_bmincut_batch
        p = {**self._config, **overrides}
        nc = coarse_J.shape[0]
        nt = p.get("num_trials", 10)
        ns = p.get("num_steps", 1000)
        dt = p.get("dt", 0.1)
        lb = p.get("lambda_balance", 1.0)
        uc = p.get("use_compile", False)
        dev = p.get("dev", "cpu")

        if q == 2:
            ix = 2 * torch.rand(nt, nc, device=dev) - 1
            iy = 2 * torch.rand(nt, nc, device=dev) - 1
            _, sol, cuts, _ = bsb_bmincut_batch(coarse_J, ix, iy, ns, dt,
                                                  lambda_balance=lb, use_compile=uc)
            return np.where(sol[torch.argmin(cuts)].cpu().numpy() == 1, 0, 1)
        else:
            part = torch.zeros(nc, dtype=torch.long, device=dev)
            n_p, nxt = 1, 1
            while n_p < q:
                sizes = torch.bincount(part, minlength=n_p)
                largest = int(torch.argmax(sizes))
                mask = part == largest
                sub = torch.where(mask)[0]
                if len(sub) <= 1:
                    break
                Jsub = coarse_J[sub][:, sub]
                bs = max(nt, 5)
                ix = 2 * torch.rand(bs, len(sub), device=dev) - 1
                iy = 2 * torch.rand(bs, len(sub), device=dev) - 1
                _, sol, cuts, _ = bsb_bmincut_batch(Jsub, ix, iy, max(ns // n_p, 50),
                                                      dt, lambda_balance=lb, use_compile=uc)
                part[sub[sol[torch.argmin(cuts)] == -1]] = nxt
                nxt += 1
                n_p += 1
            return part.cpu().numpy()

    def solve_direct(self, case_type, instance, index_start, q, **overrides):
        import time
        from src.sbm.sbm import bsb_bmincut_batch
        from src.fem import FEM
        from src.fem.problem import infer_bmincut
        p = {**self._config, **overrides}
        case = FEM.from_file(case_type, instance, index_start)
        J = case.problem.coupling_matrix.to(p.get("dev", "cpu"))
        n = J.shape[0]
        t0 = time.perf_counter()
        dt_sbm = p.get("dt", 0.1)
        nt = p.get("num_trials", 10)
        ns = p.get("num_steps", 1000)
        uc = p.get("use_compile", False)
        dev = p.get("dev", "cpu")

        if q == 2:
            ix = 2 * torch.rand(nt, n, device=dev) - 1
            iy = 2 * torch.rand(nt, n, device=dev) - 1
            _, sol, cuts, _ = bsb_bmincut_batch(J, ix, iy, ns, dt_sbm,
                                                  lambda_balance=1.0, use_compile=uc)
            spins = sol[torch.argmin(cuts)]
            pmat = torch.zeros(n, q, device=dev)
            pmat[spins == 1, 0] = 1.0
            pmat[spins == -1, 1] = 1.0
        else:
            part = torch.zeros(n, dtype=torch.long, device=dev)
            n_p, nxt = 1, 1
            while n_p < q:
                sizes = torch.bincount(part, minlength=n_p)
                largest = int(torch.argmax(sizes))
                mask = part == largest
                sub = torch.where(mask)[0]
                if len(sub) <= 1:
                    break
                Jsub = J[sub][:, sub]
                bs = max(nt, 5)
                ix = 2 * torch.rand(bs, len(sub), device=dev) - 1
                iy = 2 * torch.rand(bs, len(sub), device=dev) - 1
                _, sol, cuts, _ = bsb_bmincut_batch(Jsub, ix, iy, max(ns // n_p, 50),
                                                      dt_sbm, lambda_balance=1.0, use_compile=uc)
                part[sub[sol[torch.argmin(cuts)] == -1]] = nxt
                nxt += 1
                n_p += 1
            pmat = torch.zeros(n, q, device=dev)
            for i in range(n):
                pmat[i, min(int(part[i].item()), q - 1)] = 1.0

        dt = time.perf_counter() - t0
        _, cut = infer_bmincut(J, pmat.unsqueeze(0))
        return pmat, cut.item(), dt
