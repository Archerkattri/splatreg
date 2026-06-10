"""MAC maximal-clique seed (``splatreg.mac``, Zhang et al. CVPR 2023) — synthetic validation.

CPU-only, synthetic correspondence sets (no 3DMatch data on this box; the 3DLoMatch recall
claim is therefore *pending* — see ``docs_site/init-modes.md``). What IS asserted here:

(a) outlier-contamination sweep 30/60/90 %: MAC recovers the pose, never worse than the
    existing clique-prefilter+RANSAC hypothesis engine (``_robust_register_batched``, the one
    behind ``init="fast"``), and decisively better on a *structured-decoy* 90 % set (a
    reflection-consistent outlier cluster that defeats the greedy max-degree prefilter:
    measured RANSAC ~78 deg vs MAC <1 deg);
(b) the clique machinery on known graphs (compatibility edges + expected maximal cliques);
(c) all-outlier degenerate set -> honest ``success=False`` identity, no crash;
(d) integration: ``register(init="mac")`` end-to-end on synthetic splats, and the MAC
    registrar beats the fast-init hypothesis engine on the high-outlier correspondence set;
(e) runtime budget: 500 correspondences < 5 s on CPU.
"""

from __future__ import annotations

import math
import time

import pytest
import torch

from splatreg import register
from splatreg.core.types import Gaussians
from splatreg.mac import (
    compatibility_graph,
    enumerate_maximal_cliques,
    mac_feature_align,
    mac_pose,
)
from splatreg.align_features import _robust_register_batched

DT = torch.float32


# ---------------------------------------------------------------------------
# Synthetic correspondence sets
# ---------------------------------------------------------------------------


def _make_T(deg: float, axis, t) -> torch.Tensor:
    ax = torch.tensor(axis, dtype=torch.float64)
    ax = ax / ax.norm()
    a = math.radians(deg)
    K = torch.tensor([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], dtype=torch.float64)
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = torch.eye(3, dtype=torch.float64) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)
    T[:3, 3] = torch.tensor(t, dtype=torch.float64)
    return T


_T_TRUE = _make_T(40.0, [0.3, 1.0, 0.2], [0.1, -0.05, 0.2])


def _rot_err_deg(Ta: torch.Tensor, Tb: torch.Tensor) -> float:
    """Geodesic rotation error (deg) between two (possibly scaled) 4x4 transforms."""
    Ra, Rb = Ta[:3, :3].double(), Tb[:3, :3].double()
    Ra = Ra / Ra.det().abs().clamp_min(1e-12) ** (1.0 / 3.0)
    Rb = Rb / Rb.det().abs().clamp_min(1e-12) ** (1.0 / 3.0)
    c = float(((Ra @ Rb.T).trace() - 1.0) * 0.5)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _make_corr(
    n: int, frac_out: float, *, noise: float = 0.003, decoy: bool = False, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contaminated correspondence set under ``_T_TRUE``.

    ``decoy=True`` makes 60 % of the outliers *structured*: a reflection-consistent cluster
    (pairwise distances preserved -> it forms a large first-order-compatible component and
    out-degrees the true inliers, defeating a greedy max-degree prefilter) that admits NO
    proper rigid pose (so any hypothesis fit on it scores few inliers). This is exactly the
    multi-consensus regime MAC's exhaustive maximal-clique enumeration is built for.
    """
    g = torch.Generator().manual_seed(seed)
    R, t = _T_TRUE[:3, :3], _T_TRUE[:3, 3]
    n_in = int(round(n * (1.0 - frac_out)))
    n_out = n - n_in
    src_in = torch.randn(n_in, 3, generator=g, dtype=torch.float64) * 0.3
    tgt_in = src_in @ R.T + t + torch.randn(n_in, 3, generator=g, dtype=torch.float64) * noise
    if decoy:
        n_d = int(n_out * 0.6)
        src_d = torch.randn(n_d, 3, generator=g, dtype=torch.float64) * 0.3
        M = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64))  # reflection
        tgt_d = (
            src_d @ M.T
            + torch.tensor([0.3, 0.2, -0.1], dtype=torch.float64)
            + torch.randn(n_d, 3, generator=g, dtype=torch.float64) * noise
        )
        src_r = torch.randn(n_out - n_d, 3, generator=g, dtype=torch.float64) * 0.3
        tgt_r = torch.rand(n_out - n_d, 3, generator=g, dtype=torch.float64) * 1.2 - 0.6
        src_o = torch.cat([src_d, src_r])
        tgt_o = torch.cat([tgt_d, tgt_r])
    else:
        src_o = torch.randn(n_out, 3, generator=g, dtype=torch.float64) * 0.3
        tgt_o = torch.rand(n_out, 3, generator=g, dtype=torch.float64) * 1.2 - 0.6
    cs = torch.cat([src_in, src_o])
    ct = torch.cat([tgt_in, tgt_o])
    perm = torch.randperm(n, generator=g)
    return cs[perm].to(DT), ct[perm].to(DT)


def _ransac_baseline(cs: torch.Tensor, ct: torch.Tensor) -> dict:
    """The existing fast-init hypothesis engine on the same correspondence set."""
    idx = torch.arange(cs.shape[0])
    return _robust_register_batched(cs, ct, idx, idx, n_iters=2000, inlier_tol=0.02, with_scale=False)


# ---------------------------------------------------------------------------
# (a) outlier-contamination sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frac_out", [0.3, 0.6, 0.9])
def test_mac_recovers_under_random_outliers(frac_out):
    """MAC recovers the pose at 30/60/90 % random outliers, never worse than RANSAC."""
    cs, ct = _make_corr(200, frac_out)
    rr = mac_pose(cs, ct, inlier_tol=0.02)
    assert rr["success"]
    err_mac = _rot_err_deg(rr["T"], _T_TRUE)
    assert err_mac < 2.0, f"MAC rot err {err_mac:.2f} deg at {frac_out:.0%} outliers"
    # Never worse than the existing hypothesis engine on the same set.
    err_ransac = _rot_err_deg(_ransac_baseline(cs, ct)["T"], _T_TRUE)
    assert err_mac <= err_ransac + 0.5


@pytest.mark.parametrize("frac_out", [0.6, 0.9])
def test_mac_beats_ransac_on_structured_decoy(frac_out):
    """Structured (reflection-consistent) decoy outliers: MAC recovers; at 90 % the greedy
    max-degree prefilter latches onto the decoy cluster and RANSAC returns a wrong pose."""
    cs, ct = _make_corr(200, frac_out, decoy=True)
    rr = mac_pose(cs, ct, inlier_tol=0.02)
    assert rr["success"]
    err_mac = _rot_err_deg(rr["T"], _T_TRUE)
    assert err_mac < 2.0
    err_ransac = _rot_err_deg(_ransac_baseline(cs, ct)["T"], _T_TRUE)
    assert err_mac <= err_ransac + 0.5
    if frac_out >= 0.9:
        # The decoy cluster out-degrees the 20 true inliers -> the greedy prefilter keeps the
        # wrong consensus and RANSAC fails hard (measured ~78 deg); MAC enumerates BOTH
        # consensus sets and the true clique wins on inlier count.
        assert err_ransac > 10.0, (
            f"expected the RANSAC baseline to fail on the 90% decoy set, got {err_ransac:.2f} deg "
            "(if this starts passing, the baseline improved — update the claim, keep MAC's <2 deg)"
        )


def test_mac_sim3_scale_first():
    """Sim(3): median-ratio scale seed + SE(3) MAC + consensus scale refit recovers s and R."""
    g = torch.Generator().manual_seed(3)
    R, t = _T_TRUE[:3, :3], _T_TRUE[:3, 3]
    s_true = 1.7
    src_in = torch.randn(100, 3, generator=g, dtype=torch.float64) * 0.3
    tgt_in = s_true * (src_in @ R.T) + t + torch.randn(100, 3, generator=g, dtype=torch.float64) * 0.003
    src_o = torch.randn(100, 3, generator=g, dtype=torch.float64) * 0.3
    tgt_o = torch.rand(100, 3, generator=g, dtype=torch.float64) * 1.2 - 0.6
    cs = torch.cat([src_in, src_o]).to(DT)
    ct = torch.cat([tgt_in, tgt_o]).to(DT)
    rr = mac_pose(cs, ct, with_scale=True, inlier_tol=0.03)
    assert rr["success"]
    assert abs(rr["scale"] - s_true) / s_true < 0.02
    T_s = _T_TRUE.clone()
    T_s[:3, :3] *= s_true
    assert _rot_err_deg(rr["T"], T_s) < 2.0


# ---------------------------------------------------------------------------
# (b) the clique machinery on known graphs
# ---------------------------------------------------------------------------


def test_compatibility_graph_known_edges():
    """Rigidity-consistent pairs are edges; a far-off outlier correspondence is isolated."""
    R, t = _T_TRUE[:3, :3], _T_TRUE[:3, 3]
    src = torch.tensor([[0.0, 0, 0], [0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]], dtype=torch.float64)
    tgt = src @ R.T + t  # 0-3 exact inliers
    src = torch.cat([src, torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float64)])
    tgt = torch.cat([tgt, torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float64)])  # gross outlier
    adj, w2 = compatibility_graph(src.to(DT), tgt.to(DT), gamma=0.04)
    assert bool(adj[:4, :4].sum() == 12)  # the 4 inliers are fully mutually compatible
    assert not bool(adj[4].any()) and not bool(adj[:, 4].any())  # the outlier is isolated
    assert not bool(adj.diagonal().any())
    assert torch.equal(adj, adj.T)
    assert (w2[:4, :4][adj[:4, :4]] > 0).all()  # SC^2 weights positive on the inlier clique


def test_enumerate_maximal_cliques_known_graph():
    """Two triangles sharing an edge -> exactly those two maximal 3-cliques."""
    adj = torch.zeros(5, 5, dtype=torch.bool)
    for i, j in [(0, 1), (1, 2), (0, 2), (1, 3), (2, 3)]:  # triangles {0,1,2} and {1,2,3}; 4 isolated
        adj[i, j] = adj[j, i] = True
    cliques, truncated = enumerate_maximal_cliques(adj, min_size=3)
    assert not truncated
    assert sorted(tuple(sorted(c)) for c in cliques) == [(0, 1, 2), (1, 2, 3)]


def test_enumerate_maximal_cliques_count_cap():
    """The clique-count cap cuts the lazy enumeration and reports truncation."""
    n = 24
    adj = torch.ones(n, n, dtype=torch.bool) ^ torch.eye(n, dtype=torch.bool)
    adj[0, 1] = adj[1, 0] = False  # not one big clique -> several maximal ones
    cliques, truncated = enumerate_maximal_cliques(adj, max_cliques=1, min_size=3)
    assert truncated and len(cliques) == 1


# ---------------------------------------------------------------------------
# (c) degenerate inputs — honest failure, no crash
# ---------------------------------------------------------------------------


def test_mac_all_outliers_honest_failure():
    cs, ct = _make_corr(200, 1.0)
    rr = mac_pose(cs, ct, inlier_tol=0.02)
    assert rr["success"] is False
    assert torch.allclose(rr["T"], torch.eye(4))
    assert rr["n_inliers"] < 6  # only chance-compatible cliques exist


@pytest.mark.parametrize("m", [0, 2])
def test_mac_too_few_correspondences(m):
    cs = torch.randn(m, 3)
    ct = torch.randn(m, 3)
    rr = mac_pose(cs, ct, inlier_tol=0.02)
    assert rr["success"] is False and torch.allclose(rr["T"], torch.eye(4))


def test_mac_shape_mismatch_raises():
    with pytest.raises(ValueError, match="matched"):
        mac_pose(torch.randn(10, 3), torch.randn(8, 3))


def test_mac_feature_align_all_outliers_flags_ambiguous():
    """Splat-level entry: an all-outlier injected correspondence set -> identity + ambiguous."""
    g = torch.Generator().manual_seed(7)
    means = torch.randn(150, 3, generator=g) * 0.3
    splat = _splat_from(means)
    cs, ct = _make_corr(150, 1.0)
    T, info = mac_feature_align(splat, splat, correspondences=(cs, ct))
    assert info["success"] is False and info["ambiguous"] is True
    assert info["confidence"] == 0.0
    assert torch.allclose(T, torch.eye(4))


# ---------------------------------------------------------------------------
# (d) integration with register()
# ---------------------------------------------------------------------------


def _splat_from(means: torch.Tensor) -> Gaussians:
    n = means.shape[0]
    q = torch.zeros(n, 4, dtype=DT)
    q[:, 0] = 1.0
    return Gaussians(
        means=means.to(DT),
        quats=q,
        scales=torch.full((n, 3), 0.005, dtype=DT),
        opacities=torch.ones(n, dtype=DT),
    )


def test_register_init_mac_end_to_end():
    """register(init='mac') runs end-to-end on synthetic splats and recovers a 40 deg offset."""
    g = torch.Generator().manual_seed(11)
    means = torch.randn(400, 3, generator=g, dtype=torch.float64) * 0.3
    means[:, 2] *= 0.4  # non-degenerate, slightly anisotropic
    target = _splat_from(means.to(DT))
    src_means = ((means - _T_TRUE[:3, 3]) @ _T_TRUE[:3, :3]).to(DT)  # T_true maps source->target
    source = _splat_from(src_means)
    result = register(target, source, init="mac", transform="se3")
    assert _rot_err_deg(result.T, _T_TRUE) < 2.0
    assert (result.T[:3, 3] - _T_TRUE[:3, 3].to(DT)).norm() < 0.02
    # MAC diagnostics surface on the result like the other registrars'.
    assert result.info["feature"]["success"] is True
    assert result.info["confidence"] > 0.5 and not result.info["ambiguous"]


def test_mac_align_beats_fast_engine_on_high_outlier_corr():
    """On the 90 % structured-decoy correspondence set the MAC registrar lands < 2 deg while the
    fast-init hypothesis engine (greedy prefilter + RANSAC) returns a wrong pose (> 10 deg)."""
    cs, ct = _make_corr(200, 0.9, decoy=True)
    # Splat pair consistent with the true pose so the overlap polish refines (not fights) the seed.
    g = torch.Generator().manual_seed(13)
    tgt_means = torch.randn(400, 3, generator=g, dtype=torch.float64) * 0.3
    target = _splat_from(tgt_means.to(DT))
    source = _splat_from(((tgt_means - _T_TRUE[:3, 3]) @ _T_TRUE[:3, :3]).to(DT))
    T_mac, info = mac_feature_align(target, source, correspondences=(cs, ct))
    assert info["success"]
    err_mac = _rot_err_deg(T_mac, _T_TRUE)
    err_fast = _rot_err_deg(_ransac_baseline(cs, ct)["T"], _T_TRUE)
    assert err_mac < 2.0
    assert err_fast > 10.0
    assert err_mac < err_fast


# ---------------------------------------------------------------------------
# (e) runtime budget
# ---------------------------------------------------------------------------


def test_mac_runtime_budget_500_corr_cpu():
    """500 correspondences (60 % outliers) under 5 s on CPU (the documented budget)."""
    cs, ct = _make_corr(500, 0.6)
    t0 = time.monotonic()
    rr = mac_pose(cs, ct, inlier_tol=0.02)
    elapsed = time.monotonic() - t0
    assert rr["success"]
    assert _rot_err_deg(rr["T"], _T_TRUE) < 2.0
    assert elapsed < 5.0, f"MAC took {elapsed:.2f} s for 500 correspondences (budget 5 s)"
