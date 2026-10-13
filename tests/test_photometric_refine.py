"""Splat-to-splat photometric refinement (PhotoReg-style) — `refine="photometric"` tests.

The load-bearing scenario: a splat whose GEOMETRY is rotationally symmetric (points on a sphere)
but whose COLORS are not (azimuth-painted). Geometric residuals (ICP/SDF) plateau — they cannot
see rotation about the symmetry axis — while the photometric stage, which compares renders of the
two splats from a shared synthetic camera ring, recovers it. That is exactly the seam error
PhotoReg (arXiv 2410.05044) targets, adapted here to splat-vs-splat (no real images).

Two render backends are exercised:

* a deterministic pure-torch point renderer runs CPU end-to-end integrations for pose, exposure,
  ladder, and the ``jac_mode="autodiff"`` solver path; it is not gsplat;
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
from splatreg.residuals.photometric import SplatPhotometric, camera_ring, refine_photometric

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
    reliably earns — the LM cost drops and the translation error shrinks.
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


def test_register_refine_iters_defaults_from_quality():
    """The quality policy sizes the refine stage unless refine_kwargs overrides it."""
    target = make_color_sphere(250, seed=1)
    source = make_color_sphere(250, seed=2)
    T0 = se3_exp(torch.tensor([0.004, 0.0, 0.0, 0.0, 0.0, math.radians(4.0)]))
    out = register(
        target, source, init=T0, transform="se3", quality="low", max_iters=5,
        refine="photometric",
        refine_kwargs=dict(render_fn=mock_render, n_views=4, width=20, height=20, exposure=False),
    )
    # LOW quality => refine_iters=5 (exposure off: a single LM stage); convergence may stop
    # earlier but never exceed it.
    assert 1 <= out.info["refine"]["n_iters"] <= 5

    out_exp = register(
        target, source, init=T0, transform="se3", quality="low", max_iters=5,
        refine="photometric", refine_kwargs=dict(render_fn=mock_render, n_views=4, width=20, height=20),
    )
    # Default exposure=True adds the alternation polish round (max(2, refine_iters//3) iters),
    # so the total is bounded by refine_iters + that round — still policy-sized.
    assert 1 <= out_exp.info["refine"]["n_iters"] <= 5 + max(2, 5 // 3)
    assert "exposure" in out_exp.info["refine"]


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


# ---------------------------------------------------------------------------- exposure compensation


def tint(g: Gaussians, gain: float = 1.3, bias: float = 0.05) -> Gaussians:
    """A globally mis-exposed copy of ``g`` (different capture session, same content)."""
    return Gaussians(
        means=g.means, quats=g.quats, scales=g.scales, opacities=g.opacities,
        colors=g.colors * gain + bias,
    )


def test_exposure_compensation_recovers_tinted_pair():
    """Item-2 headline: a x1.3 + 0.05 source tint absorbs into the Sim(3) pose without
    compensation (a brighter render is 'explained' by a bigger object — the scale DoF), and the
    bounded per-pair gain/bias model recovers the untinted accuracy. Both directions asserted.

    Measured (this exact config): clean 0.10% scale err · tinted/no-comp 3.99% · tinted/comp
    0.47% — and the fitted gain lands at ~1/1.3 per channel.
    """
    target = make_color_sphere(500, seed=1)
    source = make_color_sphere(500, seed=2)
    tinted = tint(source)
    T0 = sim3_exp(torch.tensor([0.005, 0.0, 0.0, 0.0, 0.0, math.radians(4.0), math.log(1.06)]))
    kw = dict(render_fn=mock_render, n_views=6, width=32, height=32, transform="sim3", max_iters=14)

    clean = refine_photometric(target, source, T0, exposure=False, **kw)
    off = refine_photometric(target, tinted, T0, exposure=False, **kw)
    on = refine_photometric(target, tinted, T0, exposure=True, **kw)

    e_clean = abs(scale_of(clean.T) - 1.0)
    e_off = abs(scale_of(off.T) - 1.0)
    e_on = abs(scale_of(on.T) - 1.0)
    # direction 1: without compensation the tint degrades the pose vs the untinted run
    assert e_off > 5.0 * max(e_clean, 1e-4) and e_off > 0.02
    # direction 2: with compensation the tinted pair matches the untinted accuracy
    assert e_on < 0.01 and e_on < 0.25 * e_off
    # the fitted appearance model is the tint's inverse (gain ~ 1/1.3), within the bounds
    g = on.info["exposure"]["gain"]
    assert all(abs(gi - 1.0 / 1.3) < 0.08 for gi in g)


def test_exposure_on_clean_pair_is_harmless():
    """Default-ON must not degrade an already-consistent pair (bounds + alternation keep the
    model near identity). Measured: 0.10% (off) vs 0.01% (on) scale err on the clean pair."""
    target = make_color_sphere(500, seed=1)
    source = make_color_sphere(500, seed=2)
    T0 = sim3_exp(torch.tensor([0.005, 0.0, 0.0, 0.0, 0.0, math.radians(4.0), math.log(1.06)]))
    kw = dict(render_fn=mock_render, n_views=6, width=32, height=32, transform="sim3", max_iters=14)
    on = refine_photometric(target, source, T0, exposure=True, **kw)
    assert abs(scale_of(on.T) - 1.0) < 0.005
    assert rot_err_deg(on.T) < 2.5
    g, b = on.info["exposure"]["gain"], on.info["exposure"]["bias"]
    assert all(abs(gi - 1.0) < 0.1 for gi in g) and all(abs(bi) < 0.05 for bi in b)


# ---------------------------------------------------------------------------- coarse-to-fine ladder


def test_ladder_beats_single_rung_96px():
    """Item-3 headline: the 32->64->96 ladder lands a strictly smaller rotation error than a
    single 96 px rung with the same per-stage budget (the fine rung's basin is too narrow to
    pull in a 6 deg offset cold; the coarse rung supplies the basin, the fine rung the floor).

    Measured (this exact config): single-96 5.61 deg vs ladder 2.55 deg from a 6 deg offset.
    """
    target = make_color_sphere(300, seed=1)
    source = make_color_sphere(300, seed=2)
    T0 = se3_exp(torch.tensor([0.008, -0.004, 0.0, 0.0, 0.0, math.radians(6.0)]))
    kw = dict(render_fn=mock_render, n_views=4, transform="se3", exposure=False, max_iters=8)

    single = refine_photometric(target, source, T0, width=96, height=96, **kw)
    ladder = refine_photometric(target, source, T0, ladder=(32, 64, 96), **kw)

    assert rot_err_deg(ladder.T) < rot_err_deg(single.T)  # the spec'd strict win
    assert rot_err_deg(single.T) > 4.0  # the cold fine rung genuinely stalls...
    assert rot_err_deg(ladder.T) < 3.5  # ...and the ladder genuinely converges
    # per-rung diagnostics recorded, coarse -> fine
    sizes = [r["size"] for r in ladder.info["ladder"]]
    assert sizes == [(32, 32), (64, 64), (96, 96)]


def test_register_refine_ladder_passthrough():
    """The public register API carries ladder settings into the refinement integration."""
    target = make_color_sphere(100, seed=1)
    source = make_color_sphere(100, seed=2)
    result = register(
        target,
        source,
        init=torch.eye(4),
        transform="se3",
        quality="low",
        max_iters=3,
        refine="photometric",
        refine_kwargs=dict(
            render_fn=mock_render,
            n_views=3,
            ladder=(12, 16),
            max_iters=2,
            exposure=False,
        ),
    )
    assert [r["size"] for r in result.info["refine"]["ladder"]] == [(12, 12), (16, 16)]


# ---------------------------------------------------------------------------- optional-renderer failure boundary


def test_missing_gsplat_has_actionable_install_error(monkeypatch):
    """Absence of the optional real backend fails clearly while injected CPU rendering works."""
    import splatreg.residuals.photometric as photometric

    monkeypatch.setattr(photometric, "_GSPLAT_AVAILABLE", False)
    target = make_color_sphere(50, seed=1)
    cameras, intrinsics = camera_ring(target, 4)
    with pytest.raises(ImportError, match=r"splatreg\[render\]"):
        SplatPhotometric(cameras, intrinsics)
    with pytest.raises(ImportError, match=r"splatreg\[render\]"):
        refine_photometric(target, target, torch.eye(4))

    residual = SplatPhotometric(
        cameras, intrinsics, width=16, height=16, render_fn=mock_render
    )
    assert torch.isfinite(residual.residual(torch.eye(4), target, target)).all()


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
