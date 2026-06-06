#!/usr/bin/env python
"""Numerical-vs-analytic Jacobian audit for splatreg's residuals + Lie ops.

The discipline every serious geometric-optimisation library enforces (GTSAM's
``numericalDerivative`` / ``EXPECT_CORRECT_FACTOR_JACOBIANS``; SymForce's 10k-sample
manifold-native check; Theseus's ``autograd.functional.jacobian``-vs-analytic): no
hand-derived analytic Jacobian ships without being checked against a numerical one.

splatreg's ICP and SDF residual Jacobians are hand-derived (with full docstring
derivations) but had never been numerically audited. This does it: a tangent-space
central difference (right-perturbation ``T @ se3_exp(xi)``) with the correspondences
/ SDF field FROZEN at the linearisation point — exactly the Gauss-Newton assumption
the analytic Jacobian makes — compared column-by-column to the analytic Jacobian.

Run standalone:  PYTHONPATH=. python tests/test_jacobians.py
Or via pytest:   pytest tests/test_jacobians.py
"""

from __future__ import annotations

import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from splatreg.core.types import Gaussians  # noqa: E402
from splatreg.core.lie import se3_exp, se3_log, sim3_exp, sim3_log  # noqa: E402
from splatreg.residuals import ICP, SDF  # noqa: E402

DTYPE = torch.float64  # audit in double precision for a tight check
DEV = "cpu"


def _make_splat(n: int, seed: int) -> Gaussians:
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n, 3, generator=g, dtype=DTYPE) * 0.1
    q = torch.randn(n, 4, generator=g, dtype=DTYPE)
    q = q / q.norm(dim=1, keepdim=True)
    base = torch.tensor([0.010, 0.012, 0.003], dtype=DTYPE)  # anisotropic -> real normals
    scales = base.repeat(n, 1) * (1.0 + 0.2 * torch.rand(n, 1, generator=g, dtype=DTYPE))
    return Gaussians(
        means=means.to(DEV),
        quats=q.to(DEV),
        scales=scales.to(DEV),
        opacities=torch.ones(n, dtype=DTYPE, device=DEV),
        log_scales=False,
    )


def _num_jac(r_frozen, T, eps=1e-6, dof=6):
    """Central-difference Jacobian of ``r_frozen`` wrt the right-perturbation tangent."""
    cols = []
    for i in range(dof):
        e = torch.zeros(dof, dtype=T.dtype, device=T.device)
        e[i] = eps
        cols.append((r_frozen(T @ se3_exp(e)) - r_frozen(T @ se3_exp(-e))) / (2 * eps))
    return torch.stack(cols, dim=1)


def _rand_T(scale: float, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return se3_exp(torch.randn(6, generator=g, dtype=DTYPE) * scale)


def _check(name, J_an, J_num, atol):
    err = (J_an - J_num).abs().max().item()
    ok = err < atol
    print(
        f"  [{'OK  ' if ok else 'FAIL'}] {name:24s} max|Δ|={err:.2e}  (atol {atol:.0e}, {tuple(J_an.shape)})"
    )
    return ok


def audit_residual_jacobians():
    target = _make_splat(300, 1)
    source = _make_splat(200, 2)
    results = []
    for tname, T in [("near-identity", _rand_T(0.05)), ("large-rotation", _rand_T(0.9, 11))]:
        print(f"\n== pose: {tname} ==")
        icp_pp = ICP(point_to_plane=False)
        _, q, _, src, _ = icp_pp._correspondences(T, target, source)
        results.append(
            _check(
                "ICP point-to-point",
                icp_pp.jacobian(T, target, source),
                _num_jac(lambda Tp, s=src, qq=q: (s @ Tp[:3, :3].T + Tp[:3, 3] - qq).norm(dim=-1), T),
                1e-5,
            )
        )
        icp_pl = ICP(point_to_plane=True)
        _, q2, n2, src2, _ = icp_pl._correspondences(T, target, source)
        results.append(
            _check(
                "ICP point-to-plane",
                icp_pl.jacobian(T, target, source),
                _num_jac(
                    lambda Tp, s=src2, qq=q2, nn=n2: ((s @ Tp[:3, :3].T + Tp[:3, 3] - qq) * nn).sum(-1), T
                ),
                1e-5,
            )
        )
        sdf = SDF(sigma=0.02, n_points=0)
        src_s = sdf._source_points(source)
        results.append(
            _check(
                "SDF",
                sdf.jacobian(T, target, source),
                _num_jac(
                    lambda Tp, s=src_s, sd=sdf, tg=target: sd._sdf(tg, s @ Tp[:3, :3].T + Tp[:3, 3])[0]
                    * sd.weight,
                    T,
                ),
                1e-3,
            )
        )
    return results


def audit_lie_roundtrips():
    print("\n== Lie roundtrips (1000 random tangents) ==")
    g = torch.Generator().manual_seed(3)
    se3_err = sim3_err = 0.0
    for _ in range(1000):
        xi = torch.randn(6, generator=g, dtype=DTYPE)
        xi[3:] *= 2.0
        T = se3_exp(xi)
        se3_err = max(se3_err, (se3_exp(se3_log(T)) - T).abs().max().item())
        xis = torch.randn(7, generator=g, dtype=DTYPE)
        Ts = sim3_exp(xis)
        sim3_err = max(sim3_err, (sim3_exp(sim3_log(Ts)) - Ts).abs().max().item())
    ok_se3 = se3_err < 1e-8  # near-pi se3_log precision floor (float64) — not a bug
    ok_sim3 = sim3_err < 1e-8
    print(f"  [{'OK  ' if ok_se3 else 'FAIL'}] se3  exp(log(T))==T   max|Δ|={se3_err:.2e}")
    print(f"  [{'OK  ' if ok_sim3 else 'FAIL'}] sim3 exp(log(T))==T   max|Δ|={sim3_err:.2e}")
    return [ok_se3, ok_sim3]


def test_residual_jacobians():
    assert all(audit_residual_jacobians()), "an analytic residual Jacobian disagrees with numerical"


def test_lie_roundtrips():
    assert all(audit_lie_roundtrips()), "a Lie exp/log roundtrip is not identity"


if __name__ == "__main__":
    print("=" * 72)
    print("splatreg Jacobian audit (numerical vs analytic; float64, tangent-space)")
    print("=" * 72)
    res = audit_residual_jacobians() + audit_lie_roundtrips()
    print("\n" + "=" * 72)
    ok = all(res)
    print(f"RESULT: {'ALL PASS' if ok else '*** FAILURES DETECTED ***'}  ({sum(res)}/{len(res)} checks)")
    print("=" * 72)
    sys.exit(0 if ok else 1)
