#!/usr/bin/env python
"""From-scratch registration SPEED benchmark — median + p90 ms/registration.

Times the FULL ``register(target, source, init=..., transform=...)`` call (coarse init + LM
refine) over ``N`` random SE(3) + Sim(3) pairs, reporting median and p90 wall-clock ms. This is
the head-to-head speed number for splatreg's from-scratch path (vs GeoTransformer ~50 ms and the
classical ICP/FGR baselines).

Two init policies are compared:

  * ``fast``   — FPFH descriptors + GPU-batched 3-point RANSAC seed, then the closed-form-Jacobian
                 LM refine (the new default; target median < 50 ms).
  * ``global`` — the robust blind super-Fibonacci SO(3) sweep + trimmed ICP seed, then LM (the
                 correspondence-free fallback; robust to ANY rotation but slow).

Run (GPU1 only, the box shares GPU0):

    CUDA_VISIBLE_DEVICES=1 SPLATREG_DEVICE=cuda \
        PYTHONPATH=. python benchmarks/speed_bench.py --n 50

Self-contained: torch + numpy + the examples' splat generator. Deterministic per pair index.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_EX = os.path.join(_ROOT, "examples")
for _p in (_ROOT, _EX):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from splatreg import register  # noqa: E402
from splatreg.residuals import ICP  # noqa: E402

from _example_utils import (  # noqa: E402
    axis_angle_R,
    make_object_splat,
    rot_angle_deg,
    sim3_matrix,
)

DEVICE = os.environ.get("SPLATREG_DEVICE", "cpu")
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    print("SPLATREG_DEVICE=cuda requested but CUDA is unavailable; falling back to CPU.")
    DEVICE = "cpu"
DTYPE = torch.float32


def _sync():
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


def _make_pair(idx: int, transform: str):
    """Deterministic random (A, B=M_gt.A) pair plus the GT (R, s) for the accuracy spot-check."""
    g = torch.Generator(device="cpu").manual_seed(idx)
    axis = torch.randn(3, generator=g).tolist()
    rot_deg = float(torch.empty(1).uniform_(10.0, 175.0, generator=g).item())
    s_gt = float(torch.empty(1).uniform_(0.8, 1.3, generator=g).item()) if transform == "sim3" else 1.0
    A = make_object_splat(N_POINTS, seed=idx, device=DEVICE, dtype=DTYPE)
    R = axis_angle_R(axis, rot_deg, device=DEVICE, dtype=DTYPE)
    t = (torch.randn(3, generator=g) * 0.05).to(device=DEVICE, dtype=DTYPE)
    B = make_object_splat.apply_to(A, sim3_matrix(s_gt, R, t))
    return A, B, R, s_gt


def _acc(T: torch.Tensor, R_gt: torch.Tensor, s_gt: float, transform: str):
    block = T[:3, :3]
    s_est = float(torch.linalg.det(block).abs().clamp_min(1e-18) ** (1.0 / 3.0))
    R_est = block / s_est
    rot = rot_angle_deg(R_est, R_gt)
    scl = 100.0 * abs(s_est - s_gt) / s_gt
    ok = rot < 5.0 and (transform == "se3" or scl < 5.0)
    return rot, scl, ok


def run(init: str, transform: str, n: int, max_iters: int, quality: str = "full", light: bool = False):
    """Time ``register`` over ``n`` pairs. ``light`` uses an ICP-only few-iter LM (the headline fast
    config: the feature seed is near-exact, so a heavy SDF+autodiff full-quality polish is wasted)."""
    residuals = [ICP(point_to_plane=False, weight=1.0)] if light else None
    times, rots, oks = [], [], 0
    # warmup (CUDA kernels / autograd graph).
    A, B, R, s = _make_pair(0, transform)
    register(B, A, residuals=residuals, init=init, transform=transform, max_iters=max_iters, quality=quality)
    _sync()
    for idx in range(n):
        A, B, R, s = _make_pair(idx, transform)
        _sync()
        t0 = time.perf_counter()
        res = register(
            B, A, residuals=residuals, init=init, transform=transform, max_iters=max_iters, quality=quality
        )
        _sync()
        times.append((time.perf_counter() - t0) * 1e3)
        rot, scl, ok = _acc(res.T, R, s, transform)
        rots.append(rot)
        oks += int(ok)
    times = np.array(times)
    rots = np.array(rots)
    cfg = f"{init}+{'ICP-only' if light else 'default'}/{max_iters}it"
    print(
        f"  {cfg:24s} {transform}: "
        f"median {np.median(times):7.1f} ms   p90 {np.percentile(times, 90):7.1f} ms   "
        f"| rot med {np.median(rots):6.2f}° max {rots.max():6.2f}°  recov {oks}/{n}"
    )
    return {"median": float(np.median(times)), "p90": float(np.percentile(times, 90)), "recov": oks, "n": n}


N_POINTS = 1400


def main():
    global N_POINTS
    ap = argparse.ArgumentParser(description="splatreg from-scratch registration speed benchmark.")
    ap.add_argument("--n", type=int, default=50, help="pairs per (init, transform) block (default 50)")
    ap.add_argument("--n-points", type=int, default=1400, help="object anchor count (default 1400)")
    ap.add_argument(
        "--mode",
        default="all",
        choices=["all", "fast", "robust"],
        help="'fast'=headline ICP-only few-iter config; 'robust'=full-quality 60-iter; 'all'=both + "
        "the global-sweep baseline (default all)",
    )
    args = ap.parse_args()
    N_POINTS = args.n_points

    print(
        f"\nsplatreg from-scratch speed bench — N={args.n} pairs/block, "
        f"N_points={N_POINTS}, device={DEVICE.upper()}\n"
        "  FAST   = init='fast' (FPFH+batched-RANSAC) + ICP-only LM, 6 iters  (target median < 50 ms)\n"
        "  ROBUST = init='fast' + the default full-quality SDF+ICP LM, 60 iters\n"
        "  GLOBAL = init='global' (blind super-Fibonacci sweep) + default LM, 60 iters (slow fallback)\n"
    )
    for transform in ("se3", "sim3"):
        if args.mode in ("all", "fast"):
            run("fast", transform, args.n, max_iters=6, quality="full", light=True)
        if args.mode in ("all", "robust"):
            run("fast", transform, args.n, max_iters=60, quality="full", light=False)
        if args.mode == "all":
            run("global", transform, args.n, max_iters=60, quality="full", light=False)
        print()


if __name__ == "__main__":
    main()
