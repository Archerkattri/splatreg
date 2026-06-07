#!/usr/bin/env python
"""Head-to-head: splatreg vs the named ICP-only splat-registration tools on REAL splat pairs.

Competitors (the README-named ICP-only splat tools):
  * **splatalign**  (terminusfilms/splatalign) — pure multi-scale point-to-point ICP from identity,
    no global init.  Cloned to ``--splatalign-dir``; run via its ``--cli`` headless path.
  * **GaussianSplattingRegistration** (DarkTemplar91) — its registration core is Open3D
    RANSAC(FPFH)+ICP (``do_ransac_registration`` + ``do_icp_registration``); we call that engine
    directly (the GUI is irrelevant to the algorithm).  Represented here as ``o3d_ransac_icp``.

Protocol (controlled GT, identical to the merge demo): take a REAL GaussianFeels ``final.ply`` splat,
split it into two overlapping crops A / B along its principal axis, apply a **KNOWN Sim(3)** (rot,
trans, scale!=1) to B, then ask each tool to recover the transform that maps B_moved -> A.  GT is
``M_gt^{-1}``.  We report rotation (deg), translation (mm) and scale error per tool — honest, since
the GT is exact.

Run::

    CUDA_VISIBLE_DEVICES=1 SPLATREG_DEVICE=cuda python benchmarks/splat_competitors_bench.py \
        --splatalign-dir /tmp/splatreg_official/splatalign
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import splatreg  # noqa: E402
from splatreg.core.types import Gaussians  # noqa: E402
from splatreg.io import load_ply, save_ply  # noqa: E402

DEVICE = os.environ.get("SPLATREG_DEVICE", "cpu")
DTYPE = torch.float32
_DEFAULT_PLY = "/home/krishi/workspace/gaussianfeels/outputs/_opt_fscore_sd05_b10_h4/final.ply"


# --------------------------------------------------------------------- Sim(3) helpers
def axis_angle_R(axis, deg, device, dtype):
    a = torch.tensor(axis, device=device, dtype=dtype)
    a = a / a.norm()
    th = math.radians(deg)
    K = torch.tensor(
        [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], device=device, dtype=dtype
    )
    return torch.eye(3, device=device, dtype=dtype) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def sim3_matrix(s, R, t):
    T = torch.eye(4, device=R.device, dtype=R.dtype)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def decompose_sim3(T):
    block = T[:3, :3]
    s = float(torch.linalg.det(block).abs().clamp_min(1e-18) ** (1.0 / 3.0))
    return s, block / s, T[:3, 3]


def rot_angle_deg(Ra, Rb):
    c = (torch.trace(Ra.transpose(0, 1) @ Rb) - 1.0) * 0.5
    return math.degrees(math.acos(float(c.clamp(-1.0, 1.0))))


def _index(g: Gaussians, mask) -> Gaussians:
    return Gaussians(
        means=g.means[mask], quats=g.quats[mask], scales=g.scales[mask],
        opacities=g.opacities[mask], log_scales=False,
    )


def apply_sim3(g: Gaussians, M) -> Gaussians:
    m = (g.means @ M[:3, :3].T) + M[:3, 3]
    return Gaussians(means=m, quats=g.quats, scales=g.scales, opacities=g.opacities, log_scales=False)


def principal_axis(means):
    X = means - means.mean(0, keepdim=True)
    cov = (X.transpose(0, 1) @ X) / max(X.shape[0], 1)
    _, evecs = torch.linalg.eigh(cov.double())
    return evecs[:, -1].to(means.dtype)


# --------------------------------------------------------------------- competitor wrappers
def run_splatalign(A: Gaussians, B_moved: Gaussians, splatalign_dir: str,
                   use_all_points: bool = False) -> np.ndarray | None:
    """Write A and B_moved as plain PLYs, call splatalign --cli, parse its row-major transform.
    splatalign aligns secondary->primary; we pass primary=A, secondary=B_moved (recover B_moved->A).

    ``use_all_points``: splatalign's CLI hardcodes a bottom-40%-of-Z 'ground filter' (built for
    outdoor temporal captures).  On an object splat that discards 60% of points and keeps DIFFERENT
    points for the rotated B, crippling ICP.  When True we monkeypatch the percentile to 100 (use all
    points) to give splatalign its FAIREST shot — isolating its ICP-from-identity quality."""
    script = os.path.join(splatalign_dir, "splat_align.py")
    if not os.path.isfile(script):
        return None
    with tempfile.TemporaryDirectory() as td:
        pa, pb = os.path.join(td, "A.ply"), os.path.join(td, "B.ply")
        save_ply(A, pa)
        save_ply(B_moved, pb)
        env = dict(os.environ)
        if use_all_points:
            # Run via a tiny shim that patches the default Z-percentile to 100 before main().
            shim = os.path.join(td, "shim.py")
            with open(shim, "w") as f:
                f.write(
                    "import sys, runpy\n"
                    f"sys.path.insert(0, {splatalign_dir!r})\n"
                    "import splat_align\n"
                    "_orig = splat_align.load_ply_fast\n"
                    "splat_align.load_ply_fast = lambda fp, **k: _orig(fp, "
                    "**{**k, 'z_percentile_max': 100})\n"
                    f"sys.argv = ['splat_align.py', '--cli', {pa!r}, {pb!r}, {td!r}]\n"
                    "splat_align.main()\n"
                )
            cmd = [sys.executable, shim]
        else:
            cmd = [sys.executable, script, "--cli", pa, pb, td]
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, cwd=splatalign_dir, env=env,
            )
        except Exception as e:
            print(f"    splatalign run failed: {e}")
            return None
        # Parse its saved JSON transform.
        js = [f for f in os.listdir(td) if f.startswith("alignment_") and f.endswith(".json")]
        if not js:
            print("    splatalign produced no transform json. stdout tail:\n" + out.stdout[-400:])
            return None
        with open(os.path.join(td, sorted(js)[-1])) as f:
            data = json.load(f)
        for key in ("matrix_row_major", "matrix", "transform", "transformation"):
            if key in data:
                return np.asarray(data[key], dtype=np.float64).reshape(4, 4)
        # Some versions nest it; search any 16-len / 4x4 list.
        for v in data.values():
            arr = np.asarray(v, dtype=np.float64) if isinstance(v, list) else None
            if arr is not None and arr.size == 16:
                return arr.reshape(4, 4)
        print(f"    splatalign json keys unrecognised: {list(data.keys())}")
        return None


def run_o3d_ransac_icp(A: Gaussians, B_moved: Gaussians, voxel: float) -> np.ndarray | None:
    """GaussianSplattingRegistration's engine: Open3D FPFH+RANSAC global + ICP local (B_moved->A)."""
    try:
        import open3d as o3d
    except Exception:
        return None
    src = B_moved.means.detach().cpu().numpy().astype(np.float64)
    tgt = A.means.detach().cpu().numpy().astype(np.float64)

    def prep(p):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(p)
        pc = pc.voxel_down_sample(voxel)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
        f = o3d.pipelines.registration.compute_fpfh_feature(
            pc, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
        return pc, f

    sp, sf = prep(src)
    tp, tf = prep(tgt)
    d = voxel * 1.5
    r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        sp, tp, sf, tf, True, d,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(d)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    r = o3d.pipelines.registration.registration_icp(
        sp, tp, d, r.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return np.asarray(r.transformation, dtype=np.float64)


def report(name, T_est, M_gt_inv, dt):
    if T_est is None:
        print(f"  {name:28s} : UNAVAILABLE")
        return
    T = torch.as_tensor(T_est, device=DEVICE, dtype=DTYPE)
    s_g, R_g, t_g = decompose_sim3(M_gt_inv)
    s_h, R_h, t_h = decompose_sim3(T)
    rot = rot_angle_deg(R_h, R_g)
    tr = 1000.0 * float((t_h - t_g).norm())
    sc = abs(s_h - s_g)
    print(f"  {name:28s} : rot {rot:7.3f} deg | trans {tr:8.2f} mm | scale_err {sc:.4f} "
          f"(s_hat={s_h:.3f}) | {dt:.2f}s")
    return dict(rot=rot, trans_mm=tr, scale_err=sc, s_hat=s_h, sec=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default=_DEFAULT_PLY)
    ap.add_argument("--splatalign-dir", default="/tmp/splatreg_official/splatalign")
    ap.add_argument("--overlap-lo", type=float, default=0.40)
    ap.add_argument("--overlap-hi", type=float, default=0.60)
    ap.add_argument("--rot-deg", type=float, default=12.0)
    ap.add_argument("--scale", type=float, default=1.08)
    args = ap.parse_args()

    full = load_ply(args.ply).to(DEVICE)
    diag = float((full.means.amax(0) - full.means.amin(0)).norm())
    print(f"REAL splat: {len(full)} Gaussians, bbox diag {diag:.4f} m, device={DEVICE}\n")

    pa = principal_axis(full.means)
    proj = full.means @ pa
    lo_q = float(torch.quantile(proj, args.overlap_lo))
    hi_q = float(torch.quantile(proj, args.overlap_hi))
    A = _index(full, proj <= hi_q)
    B = _index(full, proj >= lo_q)
    print(f"split: A={len(A)} pts, B={len(B)} pts, overlap band "
          f"~{100*(args.overlap_hi-args.overlap_lo):.0f}%\n")

    gen = torch.Generator().manual_seed(7)
    axis = torch.randn(3, generator=gen).tolist()
    R_gt = axis_angle_R(axis, args.rot_deg, DEVICE, DTYPE)
    t_gt = (0.03 * torch.randn(3, generator=gen)).to(DEVICE, DTYPE)
    M_gt = sim3_matrix(args.scale, R_gt, t_gt)
    B_moved = apply_sim3(B, M_gt)
    M_gt_inv = torch.linalg.inv(M_gt)
    print(f"KNOWN Sim(3) on B: rot {args.rot_deg} deg, trans {1000*float(t_gt.norm()):.1f} mm, "
          f"scale {args.scale}\n")
    print("Recovering B_moved -> A  (GT = M_gt^-1; rot/trans/scale error vs GT):\n")

    voxel = max(diag / 50.0, 1e-3)

    # splatreg sim3 (its flagship — full scale DoF).
    t0 = time.perf_counter()
    r = splatreg.register(A, B_moved, transform="sim3", init="robust")
    report("splatreg (sim3, robust)", r.T.detach().cpu().numpy(), M_gt_inv, time.perf_counter() - t0)

    # splatreg se3 (fair vs SE(3)-only competitors — no scale DoF, scale stays 1).
    t0 = time.perf_counter()
    r2 = splatreg.register(A, B_moved, transform="se3", init="robust")
    report("splatreg (se3, robust)", r2.T.detach().cpu().numpy(), M_gt_inv, time.perf_counter() - t0)

    # GaussianSplattingRegistration engine (Open3D RANSAC+ICP, SE(3) only).
    t0 = time.perf_counter()
    To = run_o3d_ransac_icp(A, B_moved, voxel)
    report("GSReg (o3d RANSAC+ICP, se3)", To, M_gt_inv, time.perf_counter() - t0)

    # splatalign (pure ICP from identity, SE(3) only) — as-shipped (bottom-40%-Z ground filter).
    t0 = time.perf_counter()
    Ts = run_splatalign(A, B_moved, args.splatalign_dir, use_all_points=False)
    report("splatalign (as-shipped, se3)", Ts, M_gt_inv, time.perf_counter() - t0)

    # splatalign with all points (fairest: removes the outdoor ground-filter that hurts objects).
    t0 = time.perf_counter()
    Ts2 = run_splatalign(A, B_moved, args.splatalign_dir, use_all_points=True)
    report("splatalign (all-pts, se3)", Ts2, M_gt_inv, time.perf_counter() - t0)

    print("\nNote: SE(3)-only tools (splatalign, GSReg) cannot model the scale!=1 in this GT, so "
          "their scale_err >= |1 - s_gt_inv|; splatreg's sim3 path is the only one that recovers scale.")


if __name__ == "__main__":
    main()
