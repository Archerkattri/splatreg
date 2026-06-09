#!/usr/bin/env python
"""BENCH — PhotoReg-style photometric refinement on a REAL 103k-Gaussian capture.

Protocol (the merge-seam scenario, controlled):

1. Load a real exported 3DGS splat (``--ply``; defaults to the GaussianFeels 103k capture used by
   ``examples/merge_demo.py``).
2. Random-split it into two DISJOINT halves A (target) and B (source) — same surface, *different
   Gaussians*, like two captures of the same object — and move B by a KNOWN small SE(3)
   (default 2 deg + 0.5%-of-extent translation: the residual pose error a geometric registration
   typically leaves on a seam).
3. Register B onto A geometrically (``register`` with the default ICP+SDF residual set), then run
   the photometric stage on top (``refine="photometric"`` machinery via
   :func:`splatreg.residuals.photometric.refine_photometric`).
4. Score each stage against the KNOWN ground truth: rotation error (deg), translation error (mm),
   plus a SEAM PROXY — the mean |L1| between renders of the moved source and the target from the
   shared camera ring (what your eye sees at the seam).

Light by design: a handful of small renders (default 6 views x 96 px) — runs on a busy GPU.

    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 python benchmarks/photometric_refine_bench.py
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from splatreg import register  # noqa: E402
from splatreg.core.lie import se3_exp  # noqa: E402
from splatreg.core.types import Gaussians  # noqa: E402
from splatreg.io import load_ply  # noqa: E402
from splatreg.residuals.photometric import (  # noqa: E402
    SplatPhotometric,
    camera_ring,
    refine_photometric,
)

_DEFAULT_PLY = "/home/krishi/workspace/gaussianfeels/outputs/_opt_fscore_sd05_b10_h4/final.ply"


def rot_err_deg(T: torch.Tensor, T_gt: torch.Tensor) -> float:
    s = lambda M: M[:3, :3].det().abs().clamp_min(1e-18) ** (1.0 / 3.0)
    R = (T[:3, :3] / s(T)) @ (T_gt[:3, :3] / s(T_gt)).T
    c = (float(torch.trace(R)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def take(g: Gaussians, idx: torch.Tensor) -> Gaussians:
    return Gaussians(
        means=g.means[idx],
        quats=g.quats[idx],
        scales=g.scales[idx],
        opacities=g.opacities.reshape(-1)[idx],
        colors=g.colors[idx] if g.colors is not None else None,
        log_scales=g.log_scales,
    )


def split_halves(g: Gaussians, seed: int = 0) -> tuple[Gaussians, Gaussians]:
    n = len(g)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return take(g, perm[: n // 2]), take(g, perm[n // 2 :])


def subsample(g: Gaussians, n: int, seed: int = 0) -> Gaussians:
    if len(g) <= n:
        return g
    perm = torch.randperm(len(g), generator=torch.Generator().manual_seed(seed))
    return take(g, perm[:n])


def apply_T(g: Gaussians, T: torch.Tensor) -> Gaussians:
    from splatreg.residuals.photometric import _transform_gaussians_diff

    out = _transform_gaussians_diff(g, T)
    return Gaussians(
        means=out.means.detach(),
        quats=out.quats.detach(),
        scales=out.scales.detach(),
        opacities=g.opacities.reshape(-1).clone(),
        colors=g.colors.clone() if g.colors is not None else None,
        log_scales=g.log_scales,
    )


def seam_l1(target: Gaussians, source: Gaussians, T: torch.Tensor, cams, K, px: int, sh_degree) -> float:
    """Mean |render(source under T) - render(target)| over the ring — the visual seam proxy."""
    res = SplatPhotometric(cams, K, width=px, height=px, sh_degree=sh_degree, huber_k=0.0)
    r = res.residual(T, target, source)
    return float(r.abs().mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply", default=_DEFAULT_PLY)
    ap.add_argument("--rot-deg", type=float, default=2.0, help="known seam rotation error")
    ap.add_argument("--trans-frac", type=float, default=0.005, help="translation as fraction of extent")
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--px", type=int, default=96)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--geo-anchors", type=int, default=8000, help="anchor subsample for the geometric stage")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    full = load_ply(args.ply).to(device)
    sh_degree = None
    if full.colors is not None and full.colors.dim() == 3:
        sh_degree = int(round(math.sqrt(full.colors.shape[1]))) - 1
    extent = float((full.means.amax(0) - full.means.amin(0)).norm())
    print(f"splat: {len(full)} Gaussians, extent {extent*1000:.0f} mm, sh_degree={sh_degree}, {device}")

    A, B = split_halves(full, seed=args.seed)
    # KNOWN seam error: rotate about a skew axis + translate.
    axis = torch.tensor([0.25, 0.9, 0.35], device=device)
    axis = axis / axis.norm()
    d = torch.cat([torch.tensor([1.0, -0.6, 0.4], device=device) * args.trans_frac * extent,
                   axis * math.radians(args.rot_deg)])
    M_gt = se3_exp(d)  # B_moved = M_gt * B; the true aligning transform is M_gt^-1
    B_moved = apply_T(B, M_gt)
    T_true = torch.linalg.inv(M_gt)

    cams, K = camera_ring(A, args.views, width=args.px, height=args.px)
    seam0 = seam_l1(A, B_moved, torch.eye(4, device=device), cams, K, args.px, sh_degree)

    # Geometric stage on an anchor SUBSAMPLE (the default ICP residual builds a dense source x
    # target cdist — 51k x 51k would be ~10 GiB; 8k x 8k is ~0.25 GiB and registration pipelines
    # subsample anyway). The photometric refine then runs on the FULL halves.
    t0 = time.time()
    geo = register(
        subsample(A, args.geo_anchors, seed=1),
        subsample(B_moved, args.geo_anchors, seed=2),
        init=torch.eye(4, device=device),
        transform="se3",
        quality="balanced",
    )
    t_geo = time.time() - t0
    seam_geo = seam_l1(A, B_moved, geo.T, cams, K, args.px, sh_degree)

    t0 = time.time()
    pho = refine_photometric(
        A, B_moved, geo.T, transform="se3", n_views=args.views, width=args.px, height=args.px,
        sh_degree=sh_degree, max_iters=args.iters,
    )
    t_pho = time.time() - t0
    seam_pho = seam_l1(A, B_moved, pho.T, cams, K, args.px, sh_degree)

    # Photometric stage ALONE, straight from the injected seam error (no geometric stage): its own
    # pull/basin on real data.
    t0 = time.time()
    solo = refine_photometric(
        A, B_moved, torch.eye(4, device=device), transform="se3", n_views=args.views,
        width=args.px, height=args.px, sh_degree=sh_degree, max_iters=args.iters,
    )
    t_solo = time.time() - t0
    seam_solo = seam_l1(A, B_moved, solo.T, cams, K, args.px, sh_degree)

    fmt = lambda T: (rot_err_deg(T, T_true), float((T[:3, 3] - T_true[:3, 3]).norm()) * 1000)
    r0, t0mm = args.rot_deg, float(d[:3].norm()) * 1000
    rg, tg = fmt(geo.T)
    rp, tp = fmt(pho.T)
    rs, ts = fmt(solo.T)
    print(f"{'stage':<22}{'rot err (deg)':>14}{'trans err (mm)':>16}{'seam L1':>10}{'time (s)':>10}")
    print(f"{'injected seam error':<22}{r0:>14.3f}{t0mm:>16.2f}{seam0:>10.4f}{'-':>10}")
    print(f"{'geometric register':<22}{rg:>14.3f}{tg:>16.2f}{seam_geo:>10.4f}{t_geo:>10.1f}")
    print(f"{'+ photometric refine':<22}{rp:>14.3f}{tp:>16.2f}{seam_pho:>10.4f}{t_pho:>10.1f}")
    print(f"{'photometric ALONE':<22}{rs:>14.3f}{ts:>16.2f}{seam_solo:>10.4f}{t_solo:>10.1f}")


if __name__ == "__main__":
    main()
