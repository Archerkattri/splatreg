"""Pose information / covariance exposure (`run_lm` -> `RegisterResult.info`).

The builtin LM already forms JᵀWJ every iteration; these tests lock the contract of the
exposed pair:

* ``info["information"]`` — the UNDAMPED Gauss-Newton information matrix at the final accepted
  linearisation (6x6 for SE(3), 7x7 for Sim(3), tangent order ``[t, r, (log_s)]``);
* ``info["covariance"]`` — ``σ̂² (JᵀWJ)⁻¹`` with ``σ̂² = ||Wr||²/(R − dof)``, so it responds to
  the data's actual noise level: 2x residual noise => ~4x looser covariance.

Checked: shape per transform, symmetry, positive-definiteness on a well-constrained synthetic
registration, the noise response, the singular (under-constrained) fallback to ``None``, and
that the keys survive the high-level ``register`` API.
"""

from __future__ import annotations

import torch

from splatreg import register
from splatreg.core.lie import se3_exp
from splatreg.core.types import Gaussians
from splatreg.residuals.icp import ICP
from splatreg.solvers.lm import run_lm

DT = torch.float64


def _noisy_pair(n: int = 300, noise: float = 0.0, seed: int = 0) -> tuple:
    """A well-constrained (non-planar, anisotropic) cloud and a noise-perturbed copy."""
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n, 3, generator=g, dtype=DT) * torch.tensor([0.3, 0.2, 0.1], dtype=DT)
    quats = torch.zeros(n, 4, dtype=DT)
    quats[:, 0] = 1.0
    scales = torch.full((n, 3), 0.005, dtype=DT)
    opac = torch.ones(n, dtype=DT)
    src = Gaussians(means=means, quats=quats, scales=scales, opacities=opac)
    tgt_means = means + noise * torch.randn(n, 3, generator=g, dtype=DT)
    tgt = Gaussians(means=tgt_means, quats=quats.clone(), scales=scales.clone(), opacities=opac.clone())
    return tgt, src


def _solve(tgt, src, transform="se3"):
    T0 = se3_exp(torch.tensor([0.004, -0.002, 0.003, 0.01, -0.02, 0.015], dtype=DT))
    return run_lm(T0, [ICP(point_to_plane=False)], tgt, src, transform=transform, n_iters=15)


def test_information_and_covariance_shape_symmetry_spd():
    for transform, dof in (("se3", 6), ("sim3", 7)):
        out = _solve(*_noisy_pair(noise=1e-3), transform=transform)
        H = out.info["information"]
        C = out.info["covariance"]
        assert H is not None and H.shape == (dof, dof)
        assert C is not None and C.shape == (dof, dof)
        assert torch.allclose(H, H.T, atol=1e-9)
        assert torch.allclose(C, C.T, atol=1e-9)
        # SPD on a well-constrained problem
        assert float(torch.linalg.eigvalsh(H).min()) > 0.0
        assert float(torch.linalg.eigvalsh(C).min()) > 0.0


def test_covariance_loosens_with_noise():
    """2x the residual noise must report a looser covariance (σ̂² scaling, ~4x in trace)."""
    out1 = _solve(*_noisy_pair(noise=1e-3, seed=3))
    out2 = _solve(*_noisy_pair(noise=2e-3, seed=3))
    tr1 = float(torch.diagonal(out1.info["covariance"]).sum())
    tr2 = float(torch.diagonal(out2.info["covariance"]).sum())
    assert tr2 > 2.0 * tr1  # σ̂² quadruples in theory; demand at least 2x to stay seed-robust
    # information stays the same order (geometry unchanged): the loosening is the noise, not J
    h1 = float(torch.diagonal(out1.info["information"]).sum())
    h2 = float(torch.diagonal(out2.info["information"]).sum())
    assert 0.2 < h2 / h1 < 5.0


def test_covariance_consistent_with_information():
    """covariance == σ̂² · information⁻¹ exactly (same final linearisation)."""
    out = _solve(*_noisy_pair(noise=1e-3, seed=5))
    H = out.info["information"]
    C = out.info["covariance"]
    prod = C @ H  # = σ̂² I
    sigma2 = float(torch.diagonal(prod).mean())
    assert sigma2 > 0.0
    assert torch.allclose(prod, sigma2 * torch.eye(6, dtype=prod.dtype), atol=1e-8 * max(1.0, sigma2))


def test_singular_problem_reports_none_covariance():
    """An under-constrained pose (all points identical -> rotation unobservable) must not fake
    a covariance: information is returned, covariance falls back to None."""
    n = 50
    means = torch.zeros(n, 3, dtype=DT)  # a single repeated point
    quats = torch.zeros(n, 4, dtype=DT)
    quats[:, 0] = 1.0
    g = Gaussians(means=means, quats=quats, scales=torch.full((n, 3), 0.01, dtype=DT),
                  opacities=torch.ones(n, dtype=DT))
    out = run_lm(torch.eye(4, dtype=DT), [ICP(point_to_plane=False)], g, g, transform="se3", n_iters=3)
    H = out.info["information"]
    assert H is not None and H.shape == (6, 6)
    assert float(torch.linalg.eigvalsh(H).min()) < 1e-9  # genuinely rank-deficient
    assert out.info["covariance"] is None


def test_register_surfaces_information():
    """The high-level register() result carries the same keys (builtin backend)."""
    tgt, src = _noisy_pair(noise=1e-3, seed=7)
    out = register(tgt.to(torch.float32), src.to(torch.float32), init=torch.eye(4), transform="se3",
                   max_iters=10, quality="low")
    assert out.info["information"].shape == (6, 6)
    assert out.info["covariance"] is None or out.info["covariance"].shape == (6, 6)
