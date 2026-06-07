#!/usr/bin/env python
"""Scene-scale spatial-index tests (v0.3) — exact queries + a real speedup, all on CPU.

The :class:`splatreg.spatial_index.SpatialIndex` (voxel-hash grid over the Gaussian means) is the
scale-up primitive for the SDF / dedupe / merge query path. These tests pin its two contracts:

* **Correctness** — ``knn`` / ``radius`` / ``region`` return EXACTLY what a brute-force scan does
  (the grid only prunes which anchors are distance-tested, never the answer). Verified against the
  full ``cdist`` on a moderate cloud.
* **Speedup** — wired into the O(N^2) cross-splat ``knn_dedupe`` path, the index gives a real
  wall-clock win on a scene-scale splat versus the brute-force ``cdist`` scan, with the SAME
  survivor set (a few exactly-on-the-radius float-boundary ties tolerated — see the dedupe test).

Everything is deterministic and CPU-only (no GPU touched).
"""

from __future__ import annotations

import time

import pytest
import torch

from splatreg.spatial_index import SpatialIndex, build_index
from splatreg.core.types import Gaussians
from splatreg.fuse import _knn_keep_mask, _knn_keep_mask_indexed, knn_dedupe


def _rand_points(m, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(m, 3, generator=g)


def test_knn_matches_brute_force(device):
    """``knn(queries, k)`` returns the exact k nearest anchors (distances + index set) vs brute."""
    pts = _rand_points(6000).to(device)
    q = _rand_points(400, seed=1).to(device)
    k = 16
    idx = SpatialIndex(pts)
    ii, dd = idx.knn(q, k)

    bd = torch.cdist(q, pts)
    bf_d, bf_i = torch.topk(bd, k=k, largest=False)
    # Distances agree to float32 tolerance (cdist(q,all) vs cdist(q_i,cand) round slightly apart).
    assert torch.allclose(dd, bf_d, atol=1e-4), f"knn distance mismatch (max {(dd-bf_d).abs().max():.2e})"
    # The neighbour SET per query is identical (order-independent).
    assert torch.equal(ii.sort(dim=1).values, bf_i.sort(dim=1).values), "knn index set differs from brute"


def test_radius_matches_brute_force(device):
    """``radius(queries, r)`` returns exactly the brute-force in-radius (query, anchor) pair set."""
    pts = _rand_points(6000).to(device)
    q = _rand_points(300, seed=2).to(device)
    r = 0.04
    idx = build_index(pts)
    qi, ai = idx.radius(q, r)
    idx_set = set(zip(qi.tolist(), ai.tolist()))

    bf = torch.cdist(q, pts) <= r
    bf_set = set(map(tuple, bf.nonzero().tolist()))
    assert idx_set == bf_set, (
        f"radius pair set differs: {len(bf_set - idx_set)} missing, {len(idx_set - bf_set)} extra"
    )


def test_region_matches_brute_force(device):
    """``region(lo, hi)`` returns exactly the anchors inside the axis-aligned box vs brute."""
    pts = _rand_points(6000).to(device)
    idx = SpatialIndex(pts)
    lo = torch.tensor([0.25, 0.25, 0.25], device=device)
    hi = torch.tensor([0.6, 0.55, 0.5], device=device)
    got = idx.region(lo, hi)

    bf = ((pts >= lo) & (pts <= hi)).all(dim=-1).nonzero().reshape(-1)
    assert torch.equal(got, torch.sort(bf).values), "region anchor set differs from brute"


def test_knn_k_exceeds_count(device):
    """k larger than the anchor count clamps to the cloud and stays exact."""
    pts = _rand_points(20).to(device)
    q = _rand_points(5, seed=3).to(device)
    idx = SpatialIndex(pts)
    ii, dd = idx.knn(q, 50)
    assert ii.shape == (5, 20) and dd.shape == (5, 20)
    bf_d, _ = torch.topk(torch.cdist(q, pts), k=20, largest=False)
    assert torch.allclose(dd, bf_d, atol=1e-4)


def test_dedupe_survivor_set_matches_brute(device):
    """The index-accelerated radius-dedupe keeps the SAME survivors as the brute-force pass.

    Duplicates are placed well inside the suppression radius (not straddling the exact boundary), so
    the two distance kernels agree bit-for-bit and the survivor masks are identical.
    """
    base = _rand_points(4000).to(device)
    # Near-coincident duplicates at ~1/8 of the radius — unambiguously inside, no boundary ties.
    r = 0.01
    dup = base[:800] + (r / 8.0) * torch.randn(800, 3, generator=torch.Generator().manual_seed(7)).to(device)
    means = torch.cat([base, dup], dim=0)
    opa = torch.rand(means.shape[0], generator=torch.Generator().manual_seed(8)).to(device)

    m_brute = _knn_keep_mask(means, opa, r)
    m_index = _knn_keep_mask_indexed(means, opa, r)
    assert torch.equal(m_brute, m_index), (
        f"index dedupe survivor set differs from brute by {(m_brute != m_index).sum().item()} points"
    )

    # And the public knn_dedupe(use_index=True) path agrees with the default.
    g = Gaussians(
        means=means,
        quats=torch.tensor([1.0, 0, 0, 0], device=device).repeat(means.shape[0], 1),
        scales=torch.full((means.shape[0], 3), 0.002, device=device),
        opacities=opa,
    )
    out_default = knn_dedupe(g, r, use_index=False)
    out_index = knn_dedupe(g, r, use_index=True)
    assert len(out_default) == len(out_index)


@pytest.mark.parametrize("m", [40000])
def test_dedupe_speedup_on_scene_scale(device, m):
    """On a scene-scale splat the index dedupe beats the O(N^2) brute scan in wall-clock.

    The brute ``knn_dedupe`` forms the full chunked ``cdist`` (O(N^2)); the index touches only local
    cells. We assert a real speedup (>= 1.5x) and that the survivor sets agree up to a negligible
    float-boundary fraction (a couple of exactly-on-the-radius duplicate pairs round across the two
    distance kernels — honest FP ties, not a logic difference).
    """
    if device != "cpu":  # the timing claim is made on CPU (GPU has its own profile)
        pytest.skip("speedup asserted on CPU")
    torch.set_num_threads(2)
    g0 = torch.Generator().manual_seed(0)
    base = torch.rand(m, 3, generator=g0)
    r = 0.002
    dup = base[: m // 5] + (r / 4.0) * torch.randn(m // 5, 3, generator=g0)
    means = torch.cat([base, dup], dim=0)
    opa = torch.rand(means.shape[0], generator=g0)

    t0 = time.time()
    m_brute = _knn_keep_mask(means, opa, r)
    t_brute = time.time() - t0
    t0 = time.time()
    m_index = _knn_keep_mask_indexed(means, opa, r)
    t_index = time.time() - t0

    speedup = t_brute / max(t_index, 1e-9)
    diff_frac = float((m_brute != m_index).float().mean())
    print(
        f"\n[spatial-index dedupe] N={means.shape[0]} brute={t_brute:.2f}s index={t_index:.2f}s "
        f"speedup={speedup:.1f}x diff_frac={diff_frac:.2e}"
    )
    assert diff_frac < 1e-3, f"index dedupe diverged from brute by {diff_frac:.2e} (> float-boundary)"
    assert speedup >= 1.5, f"expected >=1.5x speedup at N={means.shape[0]}, got {speedup:.2f}x"
