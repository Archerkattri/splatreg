#!/usr/bin/env python
"""REAL-GEOMETRY multi-capture bundle benchmark (v0.3) — loop-consistency win on real splats.

``bundle_register(splats, pairs="auto")`` registers ``N`` overlapping splats JOINTLY (a pose-graph
Gauss-Newton over all absolute poses) instead of the sequential merge-to-ref chain. The unit test
proves the loop-consistency win on a synthetic ring; this file builds the SAME ring out of a **real
GaussianFeels object splat** so the win is measured on real geometry.

Protocol (multi-capture loop, real geometry)
--------------------------------------------
For each real splat:

  1. Build an ``N``-capture ring: place ``N`` cameras around the object and crop the splat to each
     camera's visible side (overlapping neighbours), then express each crop at a KNOWN absolute pose
     ``T_i`` (a ring of small rotations + translations). Each capture is a real partial view of the
     real object — overlapping its neighbours, as a real multi-capture scan does.
  2. Run ``bundle_register`` to get joint absolute poses; compare against the **sequential chain**
     (``_sequential_poses`` — exactly what ``merge`` does) on the SAME pairwise measurements.
  3. Report the **max pairwise inconsistency** (the loop-closure metric) for sequential vs joint, and
     the reduction factor — the headline loop-consistency win on REAL data.

WHAT IS REAL: the splat geometry, the partial overlapping crops, the pairwise ``register`` solves,
and the consistency numbers. WHAT REMAINS for an official number: the captures here are crops of one
splat under a known ring (controlled overlap + GT), NOT ``N`` independently-reconstructed real scans
with their own noise and an external GT trajectory (e.g. a ScanNet-GSReg / multi-scan-loop dataset).

Run:
    CUDA_VISIBLE_DEVICES=0 SPLATREG_DEVICE=cuda \\
        PYTHONPATH=. \\
        python benchmarks/bundle_real_bench.py --device cuda
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "examples"))

from _example_utils import _apply_to, axis_angle_R, sim3_matrix  # noqa: E402

from splatreg import bundle_register  # noqa: E402
from splatreg.bundle import _sequential_poses, pairwise_consistency, register  # noqa: E402
from splatreg.core.types import Gaussians  # noqa: E402
from splatreg.io import load_ply  # noqa: E402

REAL_SPLATS = {
    "sim_potted_meat_can": "outputs/_slam_fps_p_cap_600f/final.ply",
    "sim_pear": "outputs/_fix_a_baseline_016_pear/final.ply",
    "real_bell_pepper": "outputs/_fix_d_bmax60k_slam/final.ply",
    "real_peach": "outputs/_fps_real_op2_peach_pose_wtac/final.ply",
}


def _index(g: Gaussians, idx: torch.Tensor) -> Gaussians:
    return Gaussians(
        means=g.means[idx],
        quats=g.quats[idx],
        scales=g.scales[idx],
        opacities=g.opacities[idx],
        colors=None if g.colors is None else g.colors[idx],
        log_scales=g.log_scales,
    )


def _crop_side(g: Gaussians, direction: torch.Tensor, keep_frac: float) -> Gaussians:
    """Keep the ``keep_frac`` of anchors furthest along ``direction`` (a one-sided partial view)."""
    proj = g.means @ direction
    k = max(64, int(round(keep_frac * len(g))))
    idx = torch.topk(proj, k).indices
    return _index(g, idx)


def build_ring(g: Gaussians, n: int, keep_frac: float, ring_deg: float, ring_mm: float, device):
    """Return ``(captures, T_abs_gt)``: N overlapping one-sided crops at known absolute ring poses.

    Capture ``i`` is a crop of the object's ``i``-th side (overlapping its neighbours) expressed in a
    DIFFERENT frame: a small rotation ``i·ring_deg`` about an oblique axis + a translation around a
    ring. The known absolute pose ``T_i`` maps the capture back to the canonical frame; the bundle
    must recover the relative loop and close it consistently.
    """
    dtype = g.means.dtype
    center = 0.5 * (g.means.max(0).values + g.means.min(0).values)
    captures, T_abs = [], []
    for i in range(n):
        ang = 2 * math.pi * i / n
        direction = torch.tensor([math.cos(ang), math.sin(ang), 0.3], device=device, dtype=dtype)
        direction = direction / direction.norm()
        crop = _crop_side(g, direction, keep_frac)
        # Known absolute pose for this capture: incremental ring rotation + translation.
        R = axis_angle_R([0.2, 0.85, 0.3], ring_deg * i, device=device, dtype=dtype)
        t = torch.tensor(
            [ring_mm / 1000.0 * math.cos(ang), ring_mm / 1000.0 * math.sin(ang), 0.0],
            device=device, dtype=dtype,
        )
        T_i = sim3_matrix(1.0, R, t)
        T_i[:3, 3] = T_i[:3, 3] + (center - R @ center)  # rotate about the object centre
        # Express the capture in its own frame: apply inverse of T_i so register recovers T_i.
        T_inv = torch.linalg.inv(T_i)
        captures.append(_apply_to(crop, T_inv))
        T_abs.append(T_i)
    return captures, T_abs


def run_object(name, path, device, n, keep_frac, ring_deg, ring_mm, max_anchors):
    g = load_ply(path, device=device)
    if max_anchors > 0 and len(g) > max_anchors:
        gen = torch.Generator(device="cpu").manual_seed(7)
        idx = torch.randperm(len(g), generator=gen)[:max_anchors].to(g.means.device)
        g = _index(g, idx)
    captures, _T_gt = build_ring(g, n, keep_frac, ring_deg, ring_mm, device)
    sizes = [len(c) for c in captures]

    # Pairwise measurements (the ring edges) — shared by both sequential and joint.
    from splatreg.bundle import _auto_pairs
    edges = _auto_pairs(n)
    rel = {}
    t0 = time.time()
    for (i, j) in edges:
        res = register(captures[i], captures[j], init="global", transform="se3", quality="balanced")
        rel[(i, j)] = res.T.detach()
    t_pair = time.time() - t0

    # Sequential (merge-style) baseline vs joint bundle on the SAME measurements.
    seq = _sequential_poses(captures, rel, 0, n, torch.device(device), g.means.dtype)
    seq_max, seq_mean = pairwise_consistency(seq, rel, "se3")

    t1 = time.time()
    poses, info = bundle_register(
        captures, ref=0, pairs="auto", transform="se3",
        register_kwargs=dict(quality="balanced"), return_info=True,
    )
    t_joint = time.time() - t1
    # Recompute joint consistency on the same rel for an apples-to-apples comparison.
    joint_max, joint_mean = pairwise_consistency(poses, rel, "se3")

    factor = seq_max / max(joint_max, 1e-12)
    print(f"  {name:22s} N={n} caps={sizes} keep={keep_frac:.2f} ring={ring_deg:.0f}deg/{ring_mm:.0f}mm")
    print(f"    pairwise {t_pair:.1f}s, joint {t_joint:.1f}s")
    print(f"    SEQUENTIAL chain : max-edge inconsistency {seq_max:.4e}   mean {seq_mean:.4e}")
    print(f"    JOINT bundle     : max-edge inconsistency {joint_max:.4e}   mean {joint_mean:.4e}")
    print(f"    --> loop-consistency win: {factor:.2f}x lower max inconsistency")
    return {"obj": name, "seq_max": seq_max, "joint_max": joint_max, "factor": factor,
            "seq_mean": seq_mean, "joint_mean": joint_mean}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=os.environ.get("SPLATREG_DEVICE", "cpu"))
    ap.add_argument("--n", type=int, default=5, help="captures in the ring")
    ap.add_argument("--keep-frac", type=float, default=0.55, help="one-sided crop keep fraction")
    ap.add_argument("--ring-deg", type=float, default=8.0, help="per-capture ring rotation (deg)")
    ap.add_argument("--ring-mm", type=float, default=12.0, help="ring translation radius (mm)")
    ap.add_argument("--max-anchors", type=int, default=15000)
    ap.add_argument("--objects", default="")
    args = ap.parse_args()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    objects = REAL_SPLATS
    if args.objects:
        wanted = [o.strip() for o in args.objects.split(",") if o.strip()]
        objects = {k: REAL_SPLATS[k] for k in wanted if k in REAL_SPLATS}
    objects = {k: v for k, v in objects.items() if os.path.exists(v)}
    if not objects:
        print("ERROR: no real splat files present; cannot run real-geometry bundle benchmark.")
        sys.exit(2)

    torch.manual_seed(0)
    np.random.seed(0)
    print(f"REAL-GEOMETRY multi-capture bundle benchmark on {device} — {len(objects)} real GaussianFeels splats.")
    print("NOTE: N overlapping crops of the REAL splat at a KNOWN ring of poses. Real-geometry loop,")
    print("      NOT N independently-reconstructed real scans with an external GT trajectory.\n")

    rows = []
    for name, path in objects.items():
        rows.append(run_object(name, path, device, args.n, args.keep_frac,
                               args.ring_deg, args.ring_mm, args.max_anchors))
        print()

    factors = np.array([r["factor"] for r in rows])
    print("=" * 80)
    print(f"SUMMARY (n={len(rows)} real objects, N={args.n}-capture ring):")
    print(f"  median loop-consistency win: {np.median(factors):.2f}x  (range {factors.min():.2f}-{factors.max():.2f}x)")
    print(f"  median seq max  = {np.median([r['seq_max'] for r in rows]):.4e}")
    print(f"  median joint max= {np.median([r['joint_max'] for r in rows]):.4e}")


if __name__ == "__main__":
    main()
