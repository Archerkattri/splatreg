"""Splat-to-splat photometric refinement (PhotoReg-style) — `refine="photometric"` tests.

The load-bearing scenario: a splat whose GEOMETRY is rotationally symmetric (points on a sphere)
but whose COLORS are not (azimuth-painted). Geometric residuals (ICP/SDF) plateau — they cannot
see rotation about the symmetry axis — while the photometric stage, which compares renders of the
two splats from a shared synthetic camera ring, recovers it. That is exactly the seam error
PhotoReg (arXiv 2410.05044) targets, adapted here to splat-vs-splat (no real images).

Two render backends are exercised:

* a pure-torch differentiable MOCK renderer (Gaussian point-splatting) — always runs, on CPU; it
  also validates the ``jac_mode="autodiff"`` path (run_lm's row-chunked ``jacrev`` fallback)
  against the default finite-difference Jacobian;
* the real gsplat rasterizer — skip-marked unless gsplat + CUDA are available (gsplat
  rasterization is a CUDA op), tiny scenes/resolutions only.
"""

from __future__ import annotations

import math

import pytest
import torch

from splatreg import register, merge
from splatreg.core.lie import se3_exp, sim3_exp
from splatreg.core.types import Gaussians
from splatreg.quality import resolve_quality, QualityConfig
from splatreg.residuals.photometric import (
    SplatPhotometric,
    camera_ring,
    refine_photometric,
)
from splatreg.solvers.lm import _autodiff_jacobian

# ---------------------------------------------------------------------------- helpers


def make_color_sphere(n: int, seed: int, radius: float = 0.1, device: str = "cpu") -> Gaussians:
    """Geometry-symmetric / color-asymmetric splat: random sphere points, azimuth-painted RGB.

    Two different ``seed``\\ s give two *resamplings of the same surface + paint job*, so the true
    aligning transform between them is the identity — while no point of one has an exact twin in
    the other (which is what makes the geometric residuals genuinely plateau instead of snapping
    onto exact correspondences).
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, 3, generator=g)
    v = (v / v.norm(dim=1, keepdim=True) * radius).to(device)
    az = torch.atan2(v[:, 1], v[:, 0])
    colors = torch.stack(
        [
            0.5 + 0.5 * torch.cos(az),
            0.5 + 0.5 * torch.cos(az + 2 * math.pi / 3),
            0.5 + 0.5 * torch.cos(az + 4 * math.pi / 3),
        ],
        dim=1,
    )
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    return Gaussians(
        means=v,
        quats=quats,
        scales=torch.full((n, 3), 0.08 * radius, device=device),
        opacities=torch.full((n,), 0.9, device=device),
        colors=colors,
    )


def mock_render(splat: Gaussians, T_CW, K, width, height, sh_degree, sigma_px: float = 1.3):
    """Pure-torch differentiable point-splat renderer -> (V, H, W, 3).

    Projects means through the pinhole and accumulates isotropic Gaussian blobs of the per-anchor
    color, normalised by total blob weight (+1 so empty pixels fade to black). Smooth in the
    means, so both autodiff (``torch.func``-compatible) and finite differences see real gradients.
    Ignores quats/scales/sh_degree — pose observability through means + colors is all these tests
    need.
    """
    imgs = []
    xs = torch.arange(width, dtype=splat.means.dtype, device=splat.means.device)
    ys = torch.arange(height, dtype=splat.means.dtype, device=splat.means.device)
    for v_i in range(T_CW.shape[0]):
        R = T_CW[v_i, :3, :3].to(splat.means.dtype)
        t = T_CW[v_i, :3, 3].to(splat.means.dtype)
        Xc = splat.means @ R.T + t
        z = Xc[:, 2].clamp_min(1e-6)
        u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
        vv = K[1, 1] * Xc[:, 1] / z + K[1, 2]
        du = xs.view(1, 1, -1) - u.view(-1, 1, 1)  # (N, 1, W)
        dv = ys.view(1, -1, 1) - vv.view(-1, 1, 1)  # (N, H, 1)
        w = torch.exp(-(du * du + dv * dv) / (2 * sigma_px**2)) * splat.opacities.reshape(-1, 1, 1)
        num = torch.einsum("nhw,nc->hwc", w, splat.colors)
        den = w.sum(0).unsqueeze(-1) + 1.0
        imgs.append(num / den)
    return torch.stack(imgs, 0)


def rot_err_deg(T: torch.Tensor) -> float:
    """Geodesic rotation angle (deg) of ``T``'s rotation block vs identity (scale divided out)."""
    s = T[:3, :3].det().abs().clamp_min(1e-18) ** (1.0 / 3.0)
    R = T[:3, :3] / s
    c = (float(torch.trace(R)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def trans_err(T: torch.Tensor) -> float:
    return float(T[:3, 3].norm())


def scale_of(T: torch.Tensor) -> float:
    return float(T[:3, :3].det().abs().clamp_min(1e-18) ** (1.0 / 3.0))


MOCK = dict(render_fn=mock_render, n_views=6, width=32, height=32)


# ---------------------------------------------------------------------------- camera ring


def test_camera_ring_looks_at_center():
    target = make_color_sphere(300, seed=1)
    T_WC, K = camera_ring(target, 8, width=48, height=48)
    assert T_WC.shape == (8, 4, 4) and K.shape == (3, 3)
    center = target.means.mean(dim=0)
    brad = float((target.means - center).norm(dim=1).max())
    for v in range(8):
        cam = T_WC[v, :3, 3]
        fwd = T_WC[v, :3, 2]  # +z column = optical axis in world
        to_center = center - cam
        dist = float(to_center.norm())
        assert abs(dist - 2.5 * brad) < 1e-4  # default radius_mult stand-off
        cosang = float(torch.dot(fwd, to_center / to_center.norm()))
        assert cosang > 0.999  # looking at the center
        # rotation block orthonormal
        R = T_WC[v, :3, :3]
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
    # intrinsics: principal point at the image center
    assert float(K[0, 2]) == pytest.approx(24.0) and float(K[1, 2]) == pytest.approx(24.0)


def test_camera_ring_validations():
    target = make_color_sphere(50, seed=1)
    with pytest.raises(ValueError):
        camera_ring(target, 0)
    with pytest.raises(ValueError):
        camera_ring(Gaussians(torch.zeros(0, 3), torch.zeros(0, 4), torch.zeros(0, 3), torch.zeros(0)), 4)


# ---------------------------------------------------------------------------- residual basics


def test_residual_near_zero_when_aligned():
    target = make_color_sphere(300, seed=1)
    cams, K = camera_ring(target, 4, width=24, height=24)
    res = SplatPhotometric(cams, K, width=24, height=24, render_fn=mock_render)
    r0 = res.residual(torch.eye(4), target, target)  # same splat, identity -> identical renders
    assert res.dim() == r0.shape[0] == 4 * 24 * 24 * 3
    assert float(r0.abs().max()) < 1e-6
    # a perturbed pose must light the residual up
    T = se3_exp(torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, math.radians(5.0)]))
    r1 = res.residual(T, target, target)
    assert float(r1.abs().max()) > 1e-3


def test_residual_requires_colors():
    target = make_color_sphere(100, seed=1)
    bare = Gaussians(
        means=target.means, quats=target.quats, scales=target.scales, opacities=target.opacities, colors=None
    )
    cams, K = camera_ring(target, 4, width=16, height=16)
    res = SplatPhotometric(cams, K, width=16, height=16, render_fn=mock_render)
    with pytest.raises(ValueError, match="colors"):
        res.residual(torch.eye(4), target, bare)
    with pytest.raises(ValueError, match="colors"):
        res.residual(torch.eye(4), bare, target)


def test_dssim_rows_appended():
    target = make_color_sphere(200, seed=1)
    source = make_color_sphere(200, seed=2)
    cams, K = camera_ring(target, 4, width=16, height=16)
    plain = SplatPhotometric(cams, K, width=16, height=16, render_fn=mock_render)
    with_ds = SplatPhotometric(cams, K, width=16, height=16, render_fn=mock_render, dssim_weight=0.2)
    T = torch.eye(4)
    n_plain = plain.residual(T, target, source).shape[0]
    r_ds = with_ds.residual(T, target, source)
    assert r_ds.shape[0] == 2 * n_plain  # one D-SSIM row per RGB row
    assert torch.isfinite(r_ds).all()


# ---------------------------------------------------------------------------- jacobians


def test_fd_jacobian_matches_autodiff():
    """The default FD Jacobian must agree with run_lm's jacrev fallback on a smooth renderer."""
    target = make_color_sphere(200, seed=1)
    source = make_color_sphere(200, seed=2)
    cams, K = camera_ring(target, 4, width=20, height=20)
    T = se3_exp(torch.tensor([0.004, -0.002, 0.001, 0.01, -0.02, math.radians(3.0)]))

    fd = SplatPhotometric(cams, K, width=20, height=20, render_fn=mock_render, jac_mode="fd")
    ad = SplatPhotometric(cams, K, width=20, height=20, render_fn=mock_render, jac_mode="autodiff")
    assert ad.jacobian(T, target, source, dof=6) is None  # autodiff mode defers to the solver

    for dof, exp_fn in ((6, se3_exp), (7, sim3_exp)):
        J_fd = fd.jacobian(T, target, source, dof=dof)
        J_ad = _autodiff_jacobian(ad, T, target, source, dof, exp_fn, jac_row_chunk=256).reshape(-1, dof)
        assert J_fd.shape == J_ad.shape == (4 * 20 * 20 * 3, dof)
        rel = float((J_fd - J_ad).abs().max() / J_ad.abs().max())
        assert rel < 5e-3, f"dof={dof}: FD vs autodiff Jacobian rel err {rel}"


def test_jacobian_dof_validation():
    target = make_color_sphere(50, seed=1)
    cams, K = camera_ring(target, 2, width=8, height=8)
    res = SplatPhotometric(cams, K, width=8, height=8, render_fn=mock_render)
    with pytest.raises(ValueError, match="dof"):
        res.jacobian(torch.eye(4), target, target, dof=5)


# ---------------------------------------------------------------------------- refinement (mock)


def test_refine_reduces_pose_error_se3():
    """The headline: a pose offset the geometry cannot see, fixed photometrically."""
    target = make_color_sphere(500, seed=1)
    source = make_color_sphere(500, seed=2)
    T0 = se3_exp(torch.tensor([0.008, -0.004, 0.0, 0.0, 0.0, math.radians(6.0)]))
    assert rot_err_deg(T0) == pytest.approx(6.0, abs=0.1)

    out = refine_photometric(target, source, T0, transform="se3", max_iters=12, **MOCK)
    assert out.info["stage"] == "photometric" and out.info["n_views"] == 6
    assert rot_err_deg(out.T) < 2.5  # 6 deg -> ~1.7 deg (mock-render floor)
    assert trans_err(out.T) < 0.003  # ~9 mm -> ~1 mm
    assert out.scale == pytest.approx(1.0)


def test_refine_sim3_recovers_scale():
    target = make_color_sphere(500, seed=1)
    source = make_color_sphere(500, seed=2)
    T0 = sim3_exp(torch.tensor([0.005, 0.0, 0.0, 0.0, 0.0, math.radians(4.0), math.log(1.06)]))
    out = refine_photometric(target, source, T0, transform="sim3", max_iters=14, **MOCK)
    assert abs(scale_of(out.T) - 1.0) < 0.01  # 6% scale offset pulled back via silhouette size
    assert rot_err_deg(out.T) < 2.5
    assert out.scale == pytest.approx(scale_of(out.T), abs=1e-4)


def test_refine_autodiff_mode_runs_jacrev_fallback():
    """jac_mode='autodiff' returns J=None so run_lm's row-chunked jacrev does the work.

    This is an INTEGRATION check of the autodiff path inside run_lm (the jacrev fallback with
    ``jac_row_chunk``), kept deliberately tiny for CPU runtime: at 16 px / 3 views the rotation
    about the sphere axis is too aliased to converge, so the assertions are the ones this config
    reliably earns — the LM cost drops and the translation error shrinks. Jacobian CORRECTNESS of
    the autodiff path (vs FD) is covered by ``test_fd_jacobian_matches_autodiff``; convergence
    quality by the FD-mode tests above.
    """
    target = make_color_sphere(150, seed=1)
    source = make_color_sphere(150, seed=2)
    T0 = se3_exp(torch.tensor([0.004, 0.0, 0.0, 0.0, 0.0, math.radians(3.0)]))
    out = refine_photometric(
        target, source, T0, transform="se3", render_fn=mock_render,
        n_views=3, width=16, height=16, jac_mode="autodiff", max_iters=4, jac_row_chunk=128,
    )
    assert out.info["n_iters"] >= 1
    assert out.info["cost_history"][-1] < out.info["cost_history"][0]  # measured 3.65 -> 3.31
    assert trans_err(out.T) < 0.7 * trans_err(T0)  # measured 4.0 mm -> 2.0 mm


def test_refine_with_dssim_still_converges():
    target = make_color_sphere(300, seed=1)
    source = make_color_sphere(300, seed=2)
    T0 = se3_exp(torch.tensor([0.005, 0.0, 0.0, 0.0, 0.0, math.radians(5.0)]))
    out = refine_photometric(
        target, source, T0, transform="se3", max_iters=8, dssim_weight=0.2, **MOCK
    )
    assert rot_err_deg(out.T) < rot_err_deg(T0) * 0.6


def test_refine_validations():
    target = make_color_sphere(100, seed=1)
    source = make_color_sphere(100, seed=2)
    bare = Gaussians(
        means=target.means, quats=target.quats, scales=target.scales, opacities=target.opacities, colors=None
    )
    with pytest.raises(ValueError, match="colors"):
        refine_photometric(bare, source, torch.eye(4), render_fn=mock_render)
    with pytest.raises(ValueError, match="colors"):
        refine_photometric(target, bare, torch.eye(4), render_fn=mock_render)
    cams, _K = camera_ring(target, 4)
    with pytest.raises(ValueError, match="explicit K"):
        refine_photometric(target, source, torch.eye(4), cameras=cams, render_fn=mock_render)


# ---------------------------------------------------------------------------- register / merge API


def test_register_refine_beats_geometric_on_symmetric_splat():
    """register(refine='photometric') must beat geometric-only register on the symmetric sphere."""
    target = make_color_sphere(500, seed=1)
    source = make_color_sphere(500, seed=2)
    T0 = se3_exp(torch.tensor([0.008, -0.004, 0.0, 0.0, 0.0, math.radians(6.0)]))

    geo = register(target, source, init=T0, transform="se3", quality="low", max_iters=15)
    both = register(
        target, source, init=T0, transform="se3", quality="low", max_iters=15,
        refine="photometric", refine_kwargs=dict(max_iters=12, **MOCK),
    )
    # The sphere's geometry is rotation-invariant: the geometric stage cannot fix (and may worsen)
    # the rotation. The photometric stage sees the azimuth paint and fixes it.
    assert rot_err_deg(geo.T) > 4.0
    assert rot_err_deg(both.T) < 3.0
    assert rot_err_deg(both.T) < rot_err_deg(geo.T)
    # diagnostics: both stages visible, honestly
    assert both.info["refine"]["stage"] == "photometric"
    assert both.info["refine"]["n_iters"] >= 1
    assert "quality" in both.info and "cost" in both.info


def test_register_refine_validation_fails_fast():
    target = make_color_sphere(50, seed=1)
    source = make_color_sphere(50, seed=2)
    with pytest.raises(ValueError, match="refine"):
        register(target, source, refine="bogus")


def test_register_refine_iters_defaults_from_quality():
    """The quality policy sizes the refine stage unless refine_kwargs overrides it."""
    target = make_color_sphere(250, seed=1)
    source = make_color_sphere(250, seed=2)
    T0 = se3_exp(torch.tensor([0.004, 0.0, 0.0, 0.0, 0.0, math.radians(4.0)]))
    out = register(
        target, source, init=T0, transform="se3", quality="low", max_iters=5,
        refine="photometric", refine_kwargs=dict(render_fn=mock_render, n_views=4, width=20, height=20),
    )
    # LOW quality => refine_iters=5; convergence may stop earlier but never exceed it.
    assert 1 <= out.info["refine"]["n_iters"] <= 5


def test_merge_refine_passthrough():
    """merge(..., refine='photometric') runs the photometric stage on each pairwise registration."""
    a = make_color_sphere(250, seed=1)
    b = make_color_sphere(250, seed=2)
    fused = merge(
        [a, b], transform="se3", init=torch.eye(4), quality="low", max_iters=8,
        refine="photometric", refine_kwargs=dict(max_iters=4, **MOCK),
    )
    assert isinstance(fused, Gaussians)
    assert 0 < len(fused) <= len(a) + len(b)
    assert fused.colors is not None


# ---------------------------------------------------------------------------- quality policy


def test_quality_policy_sets_refine_iters():
    assert QualityConfig().refine_iters == 10
    full = resolve_quality("full", torch.device("cpu"))
    bal = resolve_quality("balanced", torch.device("cpu"))
    low = resolve_quality("low", torch.device("cpu"))
    assert full.refine_iters == 10 and bal.refine_iters == 8 and low.refine_iters == 5
    assert full.refine_iters > bal.refine_iters > low.refine_iters
    scaled = resolve_quality(0.5, torch.device("cpu"))
    assert low.refine_iters <= scaled.refine_iters <= full.refine_iters


# ---------------------------------------------------------------------------- gsplat degradation


def test_import_error_when_gsplat_missing(monkeypatch):
    """Without gsplat and without a render_fn the residual must raise a clear hint AT CALL TIME."""
    import splatreg.residuals.photometric as pm

    monkeypatch.setattr(pm, "_GSPLAT_AVAILABLE", False)
    target = make_color_sphere(50, seed=1)
    cams, K = camera_ring(target, 4)
    with pytest.raises(ImportError, match=r"splatreg\[render\]"):
        SplatPhotometric(cams, K)
    with pytest.raises(ImportError, match=r"splatreg\[render\]"):
        refine_photometric(target, target, torch.eye(4))
    # an explicit render_fn keeps everything working without gsplat
    res = SplatPhotometric(cams, K, width=16, height=16, render_fn=mock_render)
    assert torch.isfinite(res.residual(torch.eye(4), target, target)).all()


def test_invalid_jac_mode():
    target = make_color_sphere(50, seed=1)
    cams, K = camera_ring(target, 2)
    with pytest.raises(ValueError, match="jac_mode"):
        SplatPhotometric(cams, K, render_fn=mock_render, jac_mode="exact")


# ---------------------------------------------------------------------------- gsplat (real) path


gsplat_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="gsplat rasterization needs CUDA"
)


@gsplat_required
def test_gsplat_refine_reduces_pose_error():
    pytest.importorskip("gsplat")
    target = make_color_sphere(2000, seed=1, device="cuda")
    source = make_color_sphere(2000, seed=2, device="cuda")
    T0 = se3_exp(torch.tensor([0.006, -0.003, 0.002, 0.0, 0.0, math.radians(5.0)], device="cuda"))
    out = refine_photometric(
        target, source, T0, transform="se3", n_views=6, width=64, height=64, max_iters=10
    )
    assert rot_err_deg(out.T) < 1.0  # measured ~0.36 deg from 5 deg
    assert trans_err(out.T) < 0.002  # measured ~0.5 mm from ~7 mm


@gsplat_required
def test_gsplat_refine_sim3():
    pytest.importorskip("gsplat")
    target = make_color_sphere(2000, seed=1, device="cuda")
    source = make_color_sphere(2000, seed=2, device="cuda")
    T0 = sim3_exp(torch.tensor([0.003, 0.0, 0.0, 0.0, 0.0, math.radians(4.0), math.log(1.05)], device="cuda"))
    out = refine_photometric(
        target, source, T0, transform="sim3", n_views=6, width=64, height=64, max_iters=12
    )
    assert abs(scale_of(out.T) - 1.0) < 0.01  # measured 0.9997 from 1.05
    assert rot_err_deg(out.T) < 1.0
