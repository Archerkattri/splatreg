#!/usr/bin/env python
"""REAL-GEOMETRY camera-localization benchmark (v0.2) — recover a perturbed camera in a real splat.

``localize_camera(splat, frame, init_T_WC)`` refines a camera pose by optimizing the right-
perturbation tangent **through gsplat's differentiable rasteriser**. The unit test validates it on a
synthetic textured scene; this file runs it on **real GaussianFeels object splats**
(``outputs/*/final.ply``), so the rotation/translation recovery is measured against real, exported
3DGS appearance + geometry.

Protocol (relocalisation / pose-refinement, real geometry)
----------------------------------------------------------
For each real splat and each of a grid of KNOWN camera perturbations:

  1. Place a GT camera ``T_WC_gt`` looking at the object centre from a fixed standoff.
  2. **Render the real splat** from ``T_WC_gt`` (gsplat) → this is the query image (a real-geometry
     observation: the appearance/depth is the real exported splat, not a synthetic texture).
  3. Perturb the GT pose by a known rotation (deg) + translation (mm) → ``init_T_WC`` (a cold-ish
     but in-basin prior, as iNeRF / photometric SLAM front-ends assume).
  4. ``res = localize_camera(splat, frame(query_rgb, K), init_T_WC)``.
  5. Report rotation error (deg) and translation error (mm) of ``res.T`` vs ``T_WC_gt``, alongside
     the starting error, so the *reduction* is explicit.

WHAT IS REAL: the splat appearance + geometry, the differentiable-render pose refinement, the
recovered rot/trans errors. WHAT REMAINS for an official number: the query image here is a render of
the same splat (no exposure/sensor gap), and this refines a prior within the direct-alignment basin
— it is NOT a real-photo cross-modal relocaliser nor a priorless global one. The honest gap is a
held-out real RGB query (a different sensor/illumination) + a coarse global seed for wide baselines.

Run:
    CUDA_VISIBLE_DEVICES=0 SPLATREG_DEVICE=cuda \\
        PYTHONPATH=. \\
        python benchmarks/camera_loc_real_bench.py --device cuda
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

from _example_utils import axis_angle_R, rot_angle_deg  # noqa: E402

from splatreg import localize_camera  # noqa: E402
from splatreg.core.types import Frame, Gaussians  # noqa: E402
from splatreg.io import load_ply  # noqa: E402

REAL_SPLATS = {
    "sim_potted_meat_can": "outputs/_slam_fps_p_cap_600f/final.ply",
    "sim_pear": "outputs/_fix_a_baseline_016_pear/final.ply",
    "real_bell_pepper": "outputs/_fix_d_bmax60k_slam/final.ply",
    "real_peach": "outputs/_fps_real_op2_peach_pose_wtac/final.ply",
}

# Known camera perturbations: (rot_deg, trans_mm) added to the GT look-at pose.
PERTURB_GRID = [
    (3.0, 5.0),
    (5.0, 10.0),
    (8.0, 15.0),
]

H, W = 160, 160


def _look_at(eye: torch.Tensor, target: torch.Tensor, device, dtype) -> torch.Tensor:
    """Camera→world pose looking from ``eye`` at ``target`` (OpenCV +z forward convention)."""
    fwd = (target - eye)
    fwd = fwd / fwd.norm().clamp_min(1e-9)
    up0 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    if abs(float(fwd @ up0)) > 0.95:
        up0 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    right = torch.cross(up0, fwd, dim=0)
    right = right / right.norm().clamp_min(1e-9)
    down = torch.cross(fwd, right, dim=0)  # +y is down in OpenCV image frame
    R = torch.stack([right, down, fwd], dim=1)  # columns = cam axes in world
    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T


def _perturb_pose(T_WC: torch.Tensor, rot_deg: float, trans_mm: float, seed: int) -> torch.Tensor:
    """Apply a known rotation (deg) + translation (mm) perturbation to a 4x4 camera→world pose."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    device, dtype = T_WC.device, T_WC.dtype
    axis = torch.randn(3, generator=g).tolist()
    dR = axis_angle_R(axis, rot_deg, device=device, dtype=dtype)
    dt = torch.randn(3, generator=g).to(device=device, dtype=dtype)
    dt = dt / dt.norm().clamp_min(1e-9) * (trans_mm / 1000.0)
    out = T_WC.clone()
    out[:3, :3] = dR @ T_WC[:3, :3]
    out[:3, 3] = T_WC[:3, 3] + dt
    return out


def _render(splat: Gaussians, T_WC, K, sh_degree):
    from splatreg.camera_loc import _render_rgb_depth
    with torch.no_grad():
        rgb, depth = _render_rgb_depth(splat, T_WC, K, W, H, sh_degree)
    return rgb, depth


def run_object(name, path, device, iters, lr):
    splat = load_ply(path, device=device)
    # SH degree from colors shape: (N, (deg+1)^2, 3).
    sh_degree = None
    if splat.colors is not None and splat.colors.dim() == 3:
        sh_degree = int(round(math.sqrt(splat.colors.shape[1]))) - 1
    dtype = splat.means.dtype
    center = 0.5 * (splat.means.max(0).values + splat.means.min(0).values)
    diag = float((splat.means.max(0).values - splat.means.min(0).values).norm())
    standoff = 2.2 * diag
    eye = center + torch.tensor([0.6, 0.4, 1.0], device=device, dtype=dtype) / math.sqrt(0.6**2 + 0.4**2 + 1.0**2) * standoff
    T_gt = _look_at(eye, center, device, dtype)

    f = 1.2 * W
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=device, dtype=dtype)
    rgb_gt, depth_gt = _render(splat, T_gt, K, sh_degree)
    cover = float((depth_gt.abs() > 1e-6).float().mean())

    print(f"  {name:22s} N={len(splat)} diag={1000*diag:.0f}mm sh={sh_degree} coverage={cover*100:.0f}%")
    rows = []
    for gi, (rot_deg, trans_mm) in enumerate(PERTURB_GRID):
        T_init = _perturb_pose(T_gt, rot_deg, trans_mm, seed=100 * gi + 3)
        start_rot = rot_angle_deg(T_init[:3, :3], T_gt[:3, :3])
        start_trans_mm = 1000.0 * float((T_init[:3, 3] - T_gt[:3, 3]).norm())
        frame = Frame(rgb=rgb_gt, K=K)
        t0 = time.time()
        res = localize_camera(splat, frame, T_init, iters=iters, lr=lr, sh_degree=sh_degree)
        dt = time.time() - t0
        end_rot = rot_angle_deg(res.T[:3, :3], T_gt[:3, :3])
        end_trans_mm = 1000.0 * float((res.T[:3, 3] - T_gt[:3, 3]).norm())
        rows.append({"obj": name, "rot": end_rot, "trans": end_trans_mm,
                     "start_rot": start_rot, "start_trans": start_trans_mm})
        print(f"    perturb rot {start_rot:5.2f}deg trans {start_trans_mm:6.2f}mm  ->  "
              f"rot {end_rot:5.2f}deg trans {end_trans_mm:6.2f}mm  ({dt:.1f}s, loss {res.info.get('loss', float('nan')):.4f})")
    del splat
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=os.environ.get("SPLATREG_DEVICE", "cpu"))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--objects", default="")
    args = ap.parse_args()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    objects = REAL_SPLATS
    if args.objects:
        wanted = [o.strip() for o in args.objects.split(",") if o.strip()]
        objects = {k: REAL_SPLATS[k] for k in wanted if k in REAL_SPLATS}
    objects = {k: v for k, v in objects.items() if os.path.exists(v)}
    if not objects:
        print("ERROR: no real splat files present; cannot run real-geometry camera-loc benchmark.")
        sys.exit(2)

    torch.manual_seed(0)
    np.random.seed(0)
    print(f"REAL-GEOMETRY camera-localization benchmark on {device} — {len(objects)} real GaussianFeels splats.")
    print("NOTE: query image = gsplat render of the REAL splat from a GT camera; recovery refines a")
    print("      known perturbation. Real-geometry benchmark, NOT a real-photo cross-modal relocaliser.\n")

    all_rows = []
    for name, path in objects.items():
        all_rows += run_object(name, path, device, args.iters, args.lr)

    rot = np.array([r["rot"] for r in all_rows])
    trans = np.array([r["trans"] for r in all_rows])
    srot = np.array([r["start_rot"] for r in all_rows])
    strans = np.array([r["start_trans"] for r in all_rows])
    print("\n" + "=" * 80)
    print(f"SUMMARY (n={len(all_rows)} real-geometry localizations):")
    print(f"  start : median rot {np.median(srot):.2f}deg  trans {np.median(strans):.2f}mm")
    print(f"  final : median rot {np.median(rot):.2f}deg  trans {np.median(trans):.2f}mm")
    print(f"  final : worst  rot {rot.max():.2f}deg  trans {trans.max():.2f}mm")
    conv = float(((rot < srot) & (trans < strans)).mean())
    print(f"  converged (both errors reduced): {conv*100:.0f}%")


if __name__ == "__main__":
    main()
