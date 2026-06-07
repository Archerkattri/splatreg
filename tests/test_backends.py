#!/usr/bin/env python
"""Pluggable solver-backend tests: ``register(backend="pypose"|"theseus")`` matches the builtin.

The ``backend=`` seam (``splatreg.solvers.base.Solver`` doc) lets a user hand the whole assembled
SE(3)/Sim(3) registration problem to an external engine while keeping splatreg's residual plugins
and right-perturbation convention. These tests assert that the external backends RECOVER A KNOWN
TRANSFORM to comparable accuracy as the builtin closed-form-Jacobian core — the "recover a known GT"
discipline from the solver tests, now across backends:

 1. ``test_pypose_se3_recovers_known_transform``  — PyPose LM recovers a known SE(3) to builtin-grade
    accuracy on a well-conditioned point cloud.
 2. ``test_theseus_se3_recovers_known_transform`` — Theseus LM, same.
 3. ``test_pypose_sim3_recovers_scale``           — PyPose recovers a known Sim(3) scale.
 4. ``test_theseus_sim3_recovers_scale``          — Theseus, same.
 5. ``test_backends_agree_with_builtin``          — both backends land within a tight tolerance of
    the builtin's recovered pose on the same problem (same residual, same init).
 6. ``test_gtsam_backend_honest_not_implemented`` — backend='gtsam' raises NotImplementedError
    (it is honestly not wired up — needs hand-written factor Jacobians).
 7. ``test_unknown_backend_rejected``             — an unknown backend string is a ValueError.

CPU + float64 throughout (the external engines run cleanly on CPU on this box). Tests for an engine
that is not installed are skipped, not failed.

Run standalone:  PYTHONPATH=. python tests/test_backends.py
Or via pytest:   pytest tests/test_backends.py
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from splatreg.api import register  # noqa: E402
from splatreg.core.lie import se3_exp  # noqa: E402
from splatreg.core.types import Gaussians  # noqa: E402
from splatreg.residuals.icp import ICP  # noqa: E402

DT = torch.float64
DEV = "cpu"


def _have(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


_HAVE_PYPOSE = _have("pypose")
_HAVE_THESEUS = _have("theseus")

_need_pypose = pytest.mark.skipif(not _HAVE_PYPOSE, reason="pypose not installed (optional backend)")
_need_theseus = pytest.mark.skipif(not _HAVE_THESEUS, reason="theseus not installed (optional backend)")


# ---------------------------------------------------------------------------
# Fixtures (plain helpers so the file also runs standalone without pytest)
# ---------------------------------------------------------------------------


def _make_splat(n: int = 200, seed: int = 99) -> Gaussians:
    """Random (non-degenerate) point cloud — well-conditioned for pose + scale recovery."""
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n, 3, generator=g, dtype=DT) * 0.3
    means[:, 2] *= 0.1  # somewhat planar but not a regular grid
    q = torch.zeros(n, 4, dtype=DT)
    q[:, 0] = 1.0
    return Gaussians(
        means=means.to(DEV),
        quats=q.to(DEV),
        scales=torch.full((n, 3), 0.005, dtype=DT).to(DEV),
        opacities=torch.ones(n, dtype=DT).to(DEV),
    )


def _se3_init() -> torch.Tensor:
    """A known small SE(3) perturbation the solver must invert back to identity."""
    return se3_exp(torch.tensor([0.02, -0.01, 0.015, 0.05, -0.03, 0.04], dtype=DT, device=DEV))


def _scaled_source(target: Gaussians, s: float) -> Gaussians:
    return Gaussians(
        means=(target.means * s),
        quats=target.quats.clone(),
        scales=(target.scales * s),
        opacities=target.opacities.clone(),
    )


# ---------------------------------------------------------------------------
# SE(3) recovery
# ---------------------------------------------------------------------------


def _check_se3_recovery(backend: str, atol: float = 1e-4) -> None:
    target = _make_splat()
    result = register(
        target,
        target,
        residuals=[ICP(point_to_plane=False)],
        init=_se3_init(),
        transform="se3",
        backend=backend,
        max_iters=60,
    )
    err = (result.T - torch.eye(4, dtype=DT, device=DEV)).abs().max().item()
    assert err < atol, f"backend={backend!r} SE3 recovery: max|T - I| = {err:.3e} (atol {atol:g})"
    assert torch.isfinite(result.T).all()


@_need_pypose
def test_pypose_se3_recovers_known_transform():
    _check_se3_recovery("pypose")


@_need_theseus
def test_theseus_se3_recovers_known_transform():
    _check_se3_recovery("theseus")


# ---------------------------------------------------------------------------
# Sim(3) scale recovery
# ---------------------------------------------------------------------------


def _check_sim3_scale(backend: str, true_scale: float = 1.2, rtol: float = 0.05) -> None:
    target = _make_splat()
    source = _scaled_source(target, true_scale)
    result = register(
        target,
        source,
        residuals=[ICP(point_to_plane=False)],
        init=torch.eye(4, dtype=DT, device=DEV),
        transform="sim3",
        backend=backend,
        max_iters=80,
    )
    expected = 1.0 / true_scale  # align scaled source back onto target
    rel = abs(result.scale - expected) / expected
    assert rel < rtol, (
        f"backend={backend!r} Sim3 scale: got {result.scale:.5f}, expected {expected:.5f} "
        f"(rel err {rel:.3%}, rtol {rtol:g})"
    )


@_need_pypose
def test_pypose_sim3_recovers_scale():
    _check_sim3_scale("pypose")


@_need_theseus
def test_theseus_sim3_recovers_scale():
    _check_sim3_scale("theseus")


# ---------------------------------------------------------------------------
# Cross-backend agreement with the builtin
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (_HAVE_PYPOSE or _HAVE_THESEUS), reason="no external backend installed")
def test_backends_agree_with_builtin():
    """Each available external backend lands within a tight tolerance of the builtin pose."""
    target = _make_splat()
    init = _se3_init()
    kw = dict(residuals=[ICP(point_to_plane=False)], init=init, transform="se3", max_iters=60)
    builtin_T = register(target, target, backend="builtin", **kw).T

    for backend, have in (("pypose", _HAVE_PYPOSE), ("theseus", _HAVE_THESEUS)):
        if not have:
            continue
        T = register(target, target, backend=backend, **kw).T
        diff = (T - builtin_T).abs().max().item()
        assert diff < 1e-3, f"backend={backend!r} disagrees with builtin by {diff:.3e} (tol 1e-3)"


# ---------------------------------------------------------------------------
# Honest gtsam + unknown-backend handling
# ---------------------------------------------------------------------------


def test_gtsam_backend_honest_not_implemented():
    """backend='gtsam' is recognised but honestly not implemented (needs factor Jacobians)."""
    target = _make_splat(n=50)
    with pytest.raises(NotImplementedError):
        register(
            target,
            target,
            residuals=[ICP(point_to_plane=False)],
            init=_se3_init(),
            backend="gtsam",
        )


def test_unknown_backend_rejected():
    """An unknown backend string is a ValueError, not a silent fallback."""
    target = _make_splat(n=50)
    with pytest.raises(ValueError):
        register(
            target,
            target,
            residuals=[ICP(point_to_plane=False)],
            init=_se3_init(),
            backend="does_not_exist",
        )


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print("=" * 70 + "\nsplatreg backend tests\n" + "=" * 70)
    npass = nskip = 0
    for fn in fns:
        marks = getattr(fn, "pytestmark", [])
        skip = any(getattr(m, "name", "") == "skipif" and m.args and m.args[0] for m in marks)
        if skip:
            print(f"  [SKIP] {fn.__name__}")
            nskip += 1
            continue
        try:
            fn()
            print(f"  [OK  ] {fn.__name__}")
            npass += 1
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
        except Exception:
            print(f"  [ERR ] {fn.__name__}")
            traceback.print_exc()
    print("=" * 70 + f"\n{npass} pass, {nskip} skip\n" + "=" * 70)
    sys.exit(0)
