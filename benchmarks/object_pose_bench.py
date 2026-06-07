#!/usr/bin/env python
"""6-DoF object-pose benchmark (v0.2) — FoundationPose / YCB-Video-style ADD / ADD-S AUC.

This is the evaluation **harness** the FoundationPose / YCB-Video / BOP literature reports on, run
here on a SYNTHETIC PROXY because the real datasets are not fetched in this environment:

  * **Metrics** — ADD (Hinterstoisser), ADD-S (symmetric closest-point), and the YCB-Video-style
    area-under-curve recall (`add_auc`, 0–0.1 m threshold). The metric code is `splatreg.add_metric`
    / `adds_metric` / `add_auc` (unit-tested in `tests/test_object_pose.py`), i.e. the exact same
    functions a real-dataset run would call.
  * **Protocol** — for each "object" (a procedurally-generated object splat) and each of a grid of
    KNOWN 6-DoF poses, render/observe the object at that pose, estimate it with
    `splatreg.estimate_object_pose`, and score ADD/ADD-S vs the known pose. Optional occlusion
    (random point dropout) sweeps the partial-view regime FoundationPose stresses.

WHAT IS REAL HERE: the full estimate→score pipeline, the metrics, and the recovery numbers on the
synthetic proxy (printed below). WHAT REMAINS for a published number: point the same harness at real
YCB-Video / FoundationPose objects (a CAD/splat model + RGB-D observations + GT poses). The plug
point is `iter_dataset()` — replace the synthetic generator with a real-dataset loader yielding
`(model_gaussians, observation, T_gt)` and the rest (estimate + ADD/ADD-S/AUC) is unchanged.

Run:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python benchmarks/object_pose_bench.py --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "examples"))

from _example_utils import axis_angle_R, make_object_splat, rot_angle_deg, sim3_matrix  # noqa: E402

from splatreg import add_auc, add_metric, adds_metric, estimate_object_pose  # noqa: E402
from splatreg.core.types import Gaussians  # noqa: E402

# A grid of poses spanning the SO(3) basin + a translation offset (object units ≈ metres).
POSE_GRID = [
    # (axis, rot_deg, trans)
    ([0.3, 0.9, 0.25], 15.0, [0.03, -0.02, 0.02]),
    ([0.3, 0.9, 0.25], 45.0, [0.05, -0.03, 0.04]),
    ([1.0, 0.2, 0.4], 90.0, [-0.04, 0.05, -0.03]),
    ([0.1, 1.0, 0.6], 135.0, [0.06, 0.02, 0.05]),
]
N_OBJECTS = 3  # distinct procedural object splats (seeds)
N_POINTS = 1200


def _occlude(g: Gaussians, keep_frac: float, seed: int) -> Gaussians:
    """Drop a random (1 - keep_frac) of the observed points to mimic partial views / occlusion."""
    if keep_frac >= 1.0:
        return g
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = len(g)
    k = max(50, int(round(keep_frac * n)))
    idx = torch.randperm(n, generator=gen)[:k].to(g.means.device)
    return Gaussians(
        means=g.means[idx],
        quats=g.quats[idx],
        scales=g.scales[idx],
        opacities=g.opacities[idx],
        colors=None if g.colors is None else g.colors[idx],
        log_scales=g.log_scales,
    )


def iter_dataset(device: str, keep_frac: float):
    """Yield ``(model, observation, T_gt, meta)`` tuples — the synthetic-proxy 'dataset'.

    REPLACE THIS with a real YCB-Video / FoundationPose loader (CAD/splat model + observation crop +
    GT pose) to produce a published number; the scoring loop below is dataset-agnostic.
    """
    for obj_seed in range(N_OBJECTS):
        model = make_object_splat(N_POINTS, seed=obj_seed, device=device)
        for gi, (axis, rot_deg, trans) in enumerate(POSE_GRID):
            R_gt = axis_angle_R(axis, rot_deg, device=device)
            t_gt = torch.tensor(trans, device=device)
            T_gt = sim3_matrix(1.0, R_gt, t_gt)
            obs = make_object_splat.apply_to(model, T_gt)
            obs = _occlude(obs, keep_frac, seed=1000 * obj_seed + gi)
            yield model, obs, T_gt, {"obj": obj_seed, "rot_deg": rot_deg, "keep": keep_frac}


def run(device: str, keep_frac: float, transform: str):
    add_errs, adds_errs, rot_errs = [], [], []
    n = 0
    t0 = time.time()
    for model, obs, T_gt, meta in iter_dataset(device, keep_frac):
        op = estimate_object_pose(model, obs, init="global", transform=transform, quality="balanced")
        add = add_metric(model, op.T_SO, T_gt)
        adds = adds_metric(model, op.T_SO, T_gt)
        rot = rot_angle_deg(op.T_SO[:3, :3] / op.scale, T_gt[:3, :3] / op.scale)
        add_errs.append(add)
        adds_errs.append(adds)
        rot_errs.append(rot)
        n += 1
        flag = " AMBIGUOUS" if op.info.get("ambiguous") else ""
        print(
            f"  obj{meta['obj']} rot{int(meta['rot_deg']):>3}deg keep{meta['keep']:.2f}: "
            f"ADD {add*1000:7.2f}mm  ADD-S {adds*1000:7.2f}mm  rot_err {rot:6.2f}deg{flag}"
        )
    dt = time.time() - t0
    add_t = torch.tensor(add_errs)
    print(f"\n  === keep={keep_frac:.2f}  transform={transform}  (n={n}, {dt:.1f}s) ===")
    print(f"  ADD   AUC (0-10cm): {add_auc(add_errs):.4f}   median {1000*float(add_t.median()):.2f}mm")
    print(f"  ADD-S AUC (0-10cm): {add_auc(adds_errs):.4f}   median {1000*float(torch.tensor(adds_errs).median()):.2f}mm")
    # ADD < 2 cm = the common YCB 'correct pose' threshold for ~10-cm-class objects.
    succ = float((add_t < 0.02).float().mean())
    print(f"  ADD < 2cm success:  {succ*100:.1f}%   median rot_err {float(torch.tensor(rot_errs).median()):.2f}deg")
    return {"add_auc": add_auc(add_errs), "adds_auc": add_auc(adds_errs), "success": succ}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=os.environ.get("SPLATREG_DEVICE", "cpu"))
    ap.add_argument("--transform", default="se3", choices=["se3", "sim3"])
    ap.add_argument("--keep", type=float, nargs="+", default=[1.0, 0.6, 0.4],
                    help="observation keep-fractions (occlusion sweep)")
    args = ap.parse_args()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    print(f"6-DoF object-pose benchmark (SYNTHETIC PROXY) on {device}\n")
    print("NOTE: numbers are on a procedural object splat, not YCB-Video/FoundationPose data.")
    print("      Swap iter_dataset() for a real loader to produce a published ADD-S AUC.\n")
    for kf in args.keep:
        run(device, kf, args.transform)
        print()


if __name__ == "__main__":
    main()
