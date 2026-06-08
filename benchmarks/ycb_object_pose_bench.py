#!/usr/bin/env python
"""YCB-CAD 6-DoF object-pose benchmark — ADD / ADD-S / AUC on the canonical YCB ``google_16k`` models.

Clean-geometry protocol on the **canonical BOP YCB CAD models** (the exact meshes YCB-Video / BOP use
for ADD/ADD-S). Unlike the FeelSight RGB-D back-projection — which is unusable here because it drags in
hand + background clutter — this uses the clean object geometry directly:

  * **model**       = the YCB ``google_16k/nontextured.ply`` vertices (object frame), points-only Gaussians.
  * **observation** = the SAME model placed at a KNOWN 6-DoF pose ``T_gt`` then corrupted to mimic an
    independent partial/noisy capture (subsample to a keep-fraction + additive position noise).
  * ``estimate_object_pose(model, observation)`` recovers ``T_gt``; score ADD (Hinterstoisser),
    ADD-S (symmetric), and the 0-10 cm ADD-S AUC — the exact ``splatreg`` metric functions.

Symmetric objects (cans / bottles / round fruit / ball) are scored by ADD-S; the rest by ADD, matching
the YCB-Video symmetry convention. This is a clean-CAD real-geometry benchmark; it is NOT the full
YCB-Video RGB-D + per-frame-GT pipeline (those frames here are low-visibility in-hand and unusable).

Run:
    CUDA_VISIBLE_DEVICES=0 SPLATREG_DEVICE=cuda \\
        PYTHONPATH=. \\
        python -u benchmarks/ycb_object_pose_bench.py --device cuda
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
sys.path.insert(0, os.path.join(_REPO, "benchmarks"))
sys.path.insert(0, os.path.join(_REPO, "examples"))

from _example_utils import axis_angle_R, rot_angle_deg, sim3_matrix  # noqa: E402

from splatreg import add_auc, add_metric, adds_metric, estimate_object_pose  # noqa: E402
from splatreg.core.types import Gaussians  # noqa: E402

YCB_ROOT = "data/assets/gt_models/ycb"


def load_cad_vertices(obj: str) -> np.ndarray:
    """Load YCB ``google_16k/nontextured.ply`` CAD vertices (the BOP ADD/ADD-S model), in metres."""
    import trimesh

    path = os.path.join(YCB_ROOT, obj, "google_16k", "nontextured.ply")
    m = trimesh.load(path, process=False)
    return np.asarray(m.vertices, dtype=np.float64)

# YCB google_16k objects + symmetry flag (True -> ADD-S is the reported metric, per YCB-Video).
YCB = {
    "002_master_chef_can": True,
    "003_cracker_box": False,
    "004_sugar_box": False,
    "005_tomato_soup_can": True,
    "006_mustard_bottle": False,
    "008_pudding_box": False,
    "009_gelatin_box": False,
    "010_potted_meat_can": False,
    "011_banana": False,
    "013_apple": True,
    "014_lemon": True,
    "016_pear": True,
    "035_power_drill": False,
    "055_baseball": True,
}

# Known 6-DoF pose grid (object units = metres): oblique axis at several angles + translation.
POSE_GRID = [
    ([0.3, 0.9, 0.25], 15.0, [0.03, -0.02, 0.02]),
    ([0.3, 0.9, 0.25], 45.0, [0.05, -0.03, 0.04]),
    ([1.0, 0.2, 0.4], 90.0, [-0.04, 0.05, -0.03]),
    ([0.1, 1.0, 0.6], 135.0, [0.06, 0.02, 0.05]),
]


def _to_g(pts: np.ndarray, device, dtype, scale: float) -> Gaussians:
    m = torch.as_tensor(pts, device=device, dtype=dtype)
    n = m.shape[0]
    return Gaussians(
        means=m,
        quats=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).repeat(n, 1),
        scales=torch.full((n, 3), scale, device=device, dtype=dtype),
        opacities=torch.ones(n, device=device, dtype=dtype),
        log_scales=False,
    )


def _corrupt(g: Gaussians, keep: float, noise: float, gen: torch.Generator) -> Gaussians:
    n = len(g)
    k = max(50, int(round(keep * n)))
    idx = torch.randperm(n, generator=gen, device="cpu")[:k].to(g.means.device)
    means = g.means[idx].clone()
    if noise > 0:
        means = means + noise * torch.randn(means.shape, generator=gen, device="cpu").to(means)
    return Gaussians(
        means=means,
        quats=g.quats[idx].clone(),
        scales=g.scales[idx].clone(),
        opacities=g.opacities[idx].clone(),
        log_scales=g.log_scales,
    )


def run(args):
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    dtype = torch.float32
    objs = [o.strip() for o in args.objects.split(",")] if args.objects else list(YCB)
    objs = [o for o in objs if o in YCB]

    print(f"YCB-CAD 6-DoF object-pose benchmark on {device} — {len(objs)} google_16k models (clean geometry)")
    print("  model = YCB google_16k CAD vertices; observation = model @ KNOWN T_gt + subsample + noise\n")

    summary = []
    for kf in args.keep:
        add_e, adds_e, rot_e = [], [], []
        per: dict = {}
        t0 = time.time()
        for obj in objs:
            cad = load_cad_vertices(obj)
            diag = float(np.linalg.norm(cad.max(0) - cad.min(0)))
            cap = np.linspace(0, len(cad) - 1, min(len(cad), args.max_anchors)).round().astype(int)
            model = _to_g(cad[cap], device, dtype, 0.005 * diag)
            noise = args.noise_mult * 0.005 * diag
            for gi, (axis, rd, tr) in enumerate(POSE_GRID):
                R = axis_angle_R(axis, rd, device=device)
                t = torch.tensor(tr, device=device, dtype=dtype)
                T = sim3_matrix(1.0, R, t)
                obs_means = model.means @ T[:3, :3].T + T[:3, 3]
                obs = Gaussians(means=obs_means, quats=model.quats, scales=model.scales,
                                opacities=model.opacities, log_scales=False)
                obs = _corrupt(obs, kf, noise, torch.Generator(device="cpu").manual_seed(1000 * gi + 17))
                try:
                    op = estimate_object_pose(model, obs, init=args.init, transform="se3", quality="balanced")
                except Exception as e:
                    print(f"  {obj:22s} rot{int(rd):>3} keep{kf:.2f}: FAILED ({e})")
                    continue
                add = add_metric(model, op.T_SO, T)
                adds = adds_metric(model, op.T_SO, T)
                rot = rot_angle_deg(op.T_SO[:3, :3], T[:3, :3])
                add_e.append(add); adds_e.append(adds); rot_e.append(rot)
                per.setdefault(obj, []).append((add, adds, rot))
                print(f"  {obj:22s} rot{int(rd):>3} keep{kf:.2f}: "
                      f"ADD {add*1000:7.2f}mm  ADD-S {adds*1000:7.2f}mm  rot {rot:6.2f}deg")
        dt = time.time() - t0
        if not adds_e:
            print(f"  keep={kf:.2f}: no frames scored.\n")
            continue
        adds_t = torch.tensor(adds_e)
        print(f"\n  === keep={kf:.2f}  noise={args.noise_mult:.1f}xfootprint  (n={len(adds_e)}, {dt:.1f}s) ===")
        print(f"  ADD   AUC(0-10cm): {add_auc(add_e):.4f}   median {1000*float(torch.tensor(add_e).median()):.2f}mm")
        print(f"  ADD-S AUC(0-10cm): {add_auc(adds_e):.4f}   median {1000*float(adds_t.median()):.2f}mm")
        print(f"  ADD-S <2cm: {100*float((adds_t<0.02).float().mean()):.1f}%   "
              f"median rot {float(torch.tensor(rot_e).median()):.2f}deg")
        print("  per-object (ADD-S for symmetric, ADD for asymmetric):")
        for obj, vals in per.items():
            sym = YCB[obj]
            metric = torch.tensor([v[1] if sym else v[0] for v in vals])
            rr = torch.tensor([v[2] for v in vals])
            print(f"    {obj:22s} {'sym ' if sym else 'asym'} "
                  f"{'ADD-S' if sym else 'ADD  '} {1000*float(metric.median()):7.2f}mm  rot {float(rr.median()):6.2f}deg")
        print()
        summary.append((kf, add_auc(add_e), add_auc(adds_e), 1000 * float(adds_t.median()),
                        100 * float((adds_t < 0.02).float().mean())))

    print("=" * 92)
    print("SUMMARY (YCB-CAD ADD-S AUC vs occlusion keep-fraction):")
    for kf, aauc, sauc, smed, s2 in summary:
        print(f"  keep={kf:.2f}: ADD AUC {aauc:.4f} | ADD-S AUC {sauc:.4f}  median {smed:.2f}mm  <2cm {s2:.0f}%")
    print("=" * 92)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=os.environ.get("SPLATREG_DEVICE", "cuda"))
    ap.add_argument("--objects", default="", help="comma list to subset")
    ap.add_argument("--keep", type=float, nargs="+", default=[1.0, 0.6], help="observation keep-fractions")
    ap.add_argument("--noise-mult", type=float, default=0.5, help="pos-noise std / (0.005*diag)")
    ap.add_argument("--max-anchors", type=int, default=6000)
    ap.add_argument("--init", default="fast", help="coarse-init mode (fast|global)")
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    run(args)


if __name__ == "__main__":
    main()
