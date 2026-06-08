#!/usr/bin/env python
"""REAL-GEOMETRY 6-DoF object-pose benchmark (v0.2) — ADD / ADD-S / AUC on real splats.

The companion ``object_pose_bench.py`` scores the SAME estimate→ADD/ADD-S/AUC pipeline on a
*procedural* object splat. This file runs it on **real GaussianFeels object splats**
(``outputs/*/final.ply`` — the real captures used by the merge demo), so the ADD-S AUC is on real,
noisy, anisotropic-scale splat geometry rather than a clean ellipsoid.

Protocol (FoundationPose/YCB-Video-style, real geometry)
--------------------------------------------------------
For each real object splat ``model`` and each of a grid of KNOWN 6-DoF poses ``T_gt``:

  1. ``observation = T_gt · model``  — the object placed at a known pose in the scene frame.
  2. **Corrupt** the observation to mimic an independent capture: random subsample (partial view /
     different sampling) + additive Gaussian position noise ∝ the median Gaussian footprint, and an
     optional occlusion keep-fraction sweep (drop a contiguous-ish random fraction of points).
  3. ``op = estimate_object_pose(model, observation, init="fast", transform="se3")``.
  4. Score **ADD** (Hinterstoisser), **ADD-S** (symmetric), and the **ADD-S AUC** (0–10 cm) — the
     exact ``splatreg.add_metric`` / ``adds_metric`` / ``add_auc`` functions a real-dataset run calls.

WHAT IS REAL: the object geometry (real exported splats), the full estimate→score pipeline, and the
ADD-S AUC numbers. WHAT REMAINS for an official number: this is a **real-geometry** benchmark with a
known applied SE(3), NOT the official **YCB-Video / FoundationPose protocol** (their RGB-D frames,
their GT object poses, their per-object symmetry labels, the BOP toolkit). The plug point for that
is ``iter_dataset()`` — yield ``(model, observation, T_gt)`` from a real-dataset loader and the rest
is unchanged.

Run:
    CUDA_VISIBLE_DEVICES=0 SPLATREG_DEVICE=cuda \\
        PYTHONPATH=. \\
        python benchmarks/object_pose_real_bench.py --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "examples"))

from _example_utils import _apply_to, axis_angle_R, rot_angle_deg, sim3_matrix  # noqa: E402

from splatreg import add_auc, add_metric, adds_metric, estimate_object_pose  # noqa: E402
from splatreg.core.types import Gaussians  # noqa: E402
from splatreg.io import load_ply  # noqa: E402

# Real GaussianFeels object splats (a SIM + REAL mix spanning asymmetric / near-symmetric objects).
REAL_SPLATS = {
    "sim_potted_meat_can": "outputs/_slam_fps_p_cap_600f/final.ply",
    "sim_pear": "outputs/_fix_a_baseline_016_pear/final.ply",
    "sim_rubiks_cube": "outputs/_fix_b_lazy_baseline_077_rubiks_cube_v2/final.ply",
    "real_bell_pepper": "outputs/_fix_d_bmax60k_slam/final.ply",
    "real_peach": "outputs/_fps_real_op2_peach_pose_wtac/final.ply",
}

# Known 6-DoF pose grid (object units ≈ metres): an oblique axis at several angles + translation.
POSE_GRID = [
    ([0.3, 0.9, 0.25], 15.0, [0.03, -0.02, 0.02]),
    ([0.3, 0.9, 0.25], 45.0, [0.05, -0.03, 0.04]),
    ([1.0, 0.2, 0.4], 90.0, [-0.04, 0.05, -0.03]),
    ([0.1, 1.0, 0.6], 135.0, [0.06, 0.02, 0.05]),
]


def _median_footprint(g: Gaussians) -> float:
    sc = g.scales.exp() if g.log_scales else g.scales
    per = sc.mean(dim=-1)
    per = per[torch.isfinite(per) & (per > 0)]
    return float(per.median()) if per.numel() else 1e-3


def _cap(g: Gaussians, max_anchors: int, gen: torch.Generator) -> Gaussians:
    if max_anchors <= 0 or len(g) <= max_anchors:
        return g
    idx = torch.randperm(len(g), generator=gen, device="cpu")[:max_anchors].to(g.means.device)
    return _index(g, idx)


def _index(g: Gaussians, idx: torch.Tensor) -> Gaussians:
    return Gaussians(
        means=g.means[idx],
        quats=g.quats[idx],
        scales=g.scales[idx],
        opacities=g.opacities[idx],
        colors=None if g.colors is None else g.colors[idx],
        log_scales=g.log_scales,
    )


def _corrupt(g: Gaussians, keep_frac: float, noise_std: float, gen: torch.Generator) -> Gaussians:
    """Independent-capture corruption: subsample to keep_frac + additive position noise."""
    n = len(g)
    k = max(50, int(round(keep_frac * n)))
    idx = torch.randperm(n, generator=gen, device="cpu")[:k].to(g.means.device)
    out = _index(g, idx)
    means = out.means.clone()
    if noise_std > 0:
        means = means + noise_std * torch.randn(means.shape, generator=gen, device="cpu").to(means)
    return Gaussians(
        means=means,
        quats=out.quats.clone(),
        scales=out.scales.clone(),
        opacities=out.opacities.clone(),
        colors=None if out.colors is None else out.colors.clone(),
        log_scales=out.log_scales,
    )


def iter_dataset(objects, device, keep_frac, noise_mult, max_anchors):
    """Yield ``(name, model, observation, T_gt, meta)`` over real splats × the pose grid.

    REPLACE with a YCB-Video / FoundationPose loader (CAD/splat model + RGB-D observation crop + GT
    pose) for the official protocol; the estimate + ADD/ADD-S/AUC scoring below is dataset-agnostic.
    """
    cap_gen = torch.Generator(device="cpu").manual_seed(7)
    for name, path in objects.items():
        model = load_ply(path, device=device)
        model = _cap(model, max_anchors, cap_gen)
        noise_std = noise_mult * _median_footprint(model)
        for gi, (axis, rot_deg, trans) in enumerate(POSE_GRID):
            R_gt = axis_angle_R(axis, rot_deg, device=device)
            t_gt = torch.tensor(trans, device=device, dtype=model.means.dtype)
            T_gt = sim3_matrix(1.0, R_gt, t_gt)
            obs = _apply_to(model, T_gt)
            gen = torch.Generator(device="cpu").manual_seed(1000 * gi + 17)
            obs = _corrupt(obs, keep_frac, noise_std, gen)
            yield name, model, obs, T_gt, {"obj": name, "rot_deg": rot_deg, "keep": keep_frac}


def run(objects, device, keep_frac, noise_mult, max_anchors, transform, args):
    add_errs, adds_errs, rot_errs = [], [], []
    per_obj: dict = {}
    n = 0
    t0 = time.time()
    for name, model, obs, T_gt, meta in iter_dataset(objects, device, keep_frac, noise_mult, max_anchors):
        op = estimate_object_pose(model, obs, init=args.init, transform=transform, quality="balanced")
        add = add_metric(model, op.T_SO, T_gt)
        adds = adds_metric(model, op.T_SO, T_gt)
        rot = rot_angle_deg(op.T_SO[:3, :3] / op.scale, T_gt[:3, :3] / op.scale)
        add_errs.append(add)
        adds_errs.append(adds)
        rot_errs.append(rot)
        per_obj.setdefault(name, []).append((add, adds, rot))
        n += 1
        flag = " AMBIGUOUS" if op.info.get("ambiguous") else ""
        print(
            f"  {meta['obj']:20s} rot{int(meta['rot_deg']):>3}deg keep{meta['keep']:.2f}: "
            f"ADD {add*1000:8.2f}mm  ADD-S {adds*1000:8.2f}mm  rot_err {rot:7.2f}deg{flag}"
        )
    dt = time.time() - t0
    add_t = torch.tensor(add_errs)
    print(f"\n  === keep={keep_frac:.2f}  noise={noise_mult:.1f}xfootprint  transform={transform}  (n={n}, {dt:.1f}s) ===")
    print(f"  ADD   AUC (0-10cm): {add_auc(add_errs):.4f}   median {1000*float(add_t.median()):.2f}mm")
    print(f"  ADD-S AUC (0-10cm): {add_auc(adds_errs):.4f}   median {1000*float(torch.tensor(adds_errs).median()):.2f}mm")
    succ = float((torch.tensor(adds_errs) < 0.02).float().mean())
    print(f"  ADD-S < 2cm success:{succ*100:.1f}%   median rot_err {float(torch.tensor(rot_errs).median()):.2f}deg")
    print("  per-object ADD-S median:")
    for name, vals in per_obj.items():
        a = torch.tensor([v[1] for v in vals])
        r = torch.tensor([v[2] for v in vals])
        print(f"    {name:22s} ADD-S {1000*float(a.median()):8.2f}mm  rot {float(r.median()):6.2f}deg")
    return {
        "keep": keep_frac,
        "add_auc": add_auc(add_errs),
        "adds_auc": add_auc(adds_errs),
        "adds_median_mm": 1000 * float(torch.tensor(adds_errs).median()),
        "success_2cm": succ,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=os.environ.get("SPLATREG_DEVICE", "cpu"))
    ap.add_argument("--transform", default="se3", choices=["se3", "sim3"])
    ap.add_argument("--keep", type=float, nargs="+", default=[1.0, 0.6, 0.4],
                    help="observation keep-fractions (occlusion sweep)")
    ap.add_argument("--noise-mult", type=float, default=0.5, help="pos-noise std / footprint")
    ap.add_argument("--max-anchors", type=int, default=12000, help="cap loaded splat (speed/mem)")
    ap.add_argument("--init", default="global", help="coarse-init mode (global|fast); global is the timed default")
    ap.add_argument("--objects", default="", help="comma list to subset")
    args = ap.parse_args()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    objects = REAL_SPLATS
    if args.objects:
        wanted = [o.strip() for o in args.objects.split(",") if o.strip()]
        objects = {k: REAL_SPLATS[k] for k in wanted if k in REAL_SPLATS}
    objects = {k: v for k, v in objects.items() if os.path.exists(v)}
    if not objects:
        print("ERROR: no real splat files present; cannot run real-geometry object-pose benchmark.")
        sys.exit(2)

    torch.manual_seed(0)
    np.random.seed(0)
    print(f"REAL-GEOMETRY 6-DoF object-pose benchmark on {device} — {len(objects)} real GaussianFeels splats.")
    print("NOTE: real splat geometry + KNOWN applied SE(3). This is a real-geometry benchmark, NOT the")
    print("      official YCB-Video / FoundationPose protocol (their RGB-D frames + GT poses + BOP).\n")
    results = []
    for kf in args.keep:
        results.append(run(objects, device, kf, args.noise_mult, args.max_anchors, args.transform, args))
        print()
    print("=" * 88)
    print("SUMMARY (real-geometry ADD-S AUC vs occlusion keep-fraction):")
    for r in results:
        print(f"  keep={r['keep']:.2f}: ADD-S AUC {r['adds_auc']:.4f}  median {r['adds_median_mm']:.2f}mm  "
              f"<2cm {r['success_2cm']*100:.0f}%")


if __name__ == "__main__":
    main()
