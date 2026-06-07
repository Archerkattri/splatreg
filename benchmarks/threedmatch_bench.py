#!/usr/bin/env python
"""Real external benchmark: splatreg vs Open3D FPFH+RANSAC on the standard 3DMatch test split.

3DMatch (Zeng et al., CVPR 2017) is the canonical real-scan pairwise point-cloud registration
benchmark.  The test split ships 8 indoor scenes of overlapping RGB-D fragments (``cloud_bin_*.ply``)
plus per-fragment **camera-to-world** poses (``poses/cloud_bin_*.txt``).  The ground-truth relative
transform that maps fragment ``i``'s points into fragment ``j``'s frame is::

    T_ji = inv(pose_j) @ pose_i

We sample ~``--n-pairs`` overlapping fragment pairs (overlap measured on a voxel-downsampled cloud),
register each with splatreg ``register(target, source, init="fast")`` AND with the classical Open3D
FPFH + RANSAC pipeline on the **same** pair, and report the standard 3DMatch metrics:

* **RRE** — geodesic rotation error (deg) of the estimated vs GT rotation.
* **RTE** — translation error (m).
* **RR (Registration Recall)** — the 3DMatch PRIMARY metric: fraction of pairs whose RMSE over the
  GT correspondences (points within the GT inlier band, transformed by the ESTIMATED pose) is below
  ``--rmse-thresh`` (default 0.2 m).
* median ms/pair.

HONEST framing: 3DMatch fragments are partial-overlap real scans.  splatreg is a classical-style
geometric registrar (FPFH/RANSAC seed + LM/ICP polish), so its RR is expected to sit well below the
80-92 % that *learned* descriptors (FCGF, Predator, GeoTransformer) reach.  We report whatever the
real numbers are — this is splatreg's FIRST external-benchmark datapoint, not a tuned win.

Run (GPU 1 only)::

    CUDA_VISIBLE_DEVICES=1 SPLATREG_DEVICE=cuda \
        python benchmarks/threedmatch_bench.py --n-pairs 60
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from splatreg import register  # noqa: E402
from splatreg.core.types import Gaussians  # noqa: E402
from splatreg.io import _read_ply_vertex  # noqa: E402

DEFAULT_DATA = os.path.join(_REPO_ROOT, "data", "3dmatch", "3dmatch", "test")


# ----------------------------------------------------------------------- IO + geometry
def read_points(path: str) -> np.ndarray:
    """Load a plain ``x y z`` 3DMatch fragment PLY -> ``(N, 3)`` float64 points."""
    cols = _read_ply_vertex(path)
    return np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float64)


def read_pose(path: str) -> np.ndarray:
    """Read a 3DMatch pose ``.txt`` (1 header line + 4x4 matrix) -> ``(4, 4)`` camera-to-world."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    mat = [[float(x) for x in ln.split()] for ln in lines[1:5]]
    return np.asarray(mat, dtype=np.float64)


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Deterministic voxel-grid downsample: one (first) point per occupied voxel."""
    if pts.shape[0] == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def points_to_gaussians(pts: np.ndarray, device, dtype=torch.float32) -> Gaussians:
    """Wrap a plain point cloud as a :class:`Gaussians` (means = points; neutral other fields)."""
    m = torch.as_tensor(pts, device=device, dtype=dtype)
    n = m.shape[0]
    return Gaussians(
        means=m,
        quats=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).repeat(n, 1),
        scales=torch.full((n, 3), 0.02, device=device, dtype=dtype),
        opacities=torch.ones(n, device=device, dtype=dtype),
        log_scales=False,
    )


def rot_err_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) * 0.5
    return math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))


def overlap_ratio(src: np.ndarray, tgt: np.ndarray, T_gt: np.ndarray, thresh: float) -> float:
    """Fraction of source points landing within ``thresh`` of a target point under the GT transform."""
    s = (src @ T_gt[:3, :3].T) + T_gt[:3, 3]
    st = torch.as_tensor(s, dtype=torch.float32)
    tt = torch.as_tensor(tgt, dtype=torch.float32)
    d = torch.cdist(st, tt).min(dim=1).values
    return float((d < thresh).float().mean().item())


def correspondence_rmse(
    src: np.ndarray, tgt: np.ndarray, T_gt: np.ndarray, T_est: np.ndarray, band: float
) -> float:
    """3DMatch RR RMSE: over the GT-overlap correspondences, RMSE after the ESTIMATED transform.

    Correspondences are source points whose GT-transformed position is within ``band`` of a target
    point (the true overlap set, with their matched target point).  RMSE is then measured with the
    SAME correspondences but the source transformed by the *estimated* pose — so a pose close to GT
    keeps the overlap aligned (low RMSE) and a wrong pose scatters it (high RMSE).
    """
    s_gt = (src @ T_gt[:3, :3].T) + T_gt[:3, 3]
    st = torch.as_tensor(s_gt, dtype=torch.float32)
    tt = torch.as_tensor(tgt, dtype=torch.float32)
    d, nn = torch.cdist(st, tt).min(dim=1)
    m = d < band
    if int(m.sum().item()) < 3:
        return float("inf")
    src_corr = src[m.cpu().numpy()]
    tgt_corr = tgt[nn[m].cpu().numpy()]
    s_est = (src_corr @ T_est[:3, :3].T) + T_est[:3, 3]
    return float(np.sqrt(np.mean(np.sum((s_est - tgt_corr) ** 2, axis=1))))


# ----------------------------------------------------------------------- Open3D baseline
def open3d_fpfh_ransac(src: np.ndarray, tgt: np.ndarray, voxel: float) -> np.ndarray:
    """Classical FPFH + RANSAC global registration via Open3D -> estimated 4x4 (source->target)."""
    import open3d as o3d

    def make(p):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(p)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pc, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100)
        )
        return pc, fpfh

    sp, sf = make(src)
    tp, tf = make(tgt)
    dist = voxel * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        sp,
        tp,
        sf,
        tf,
        True,
        dist,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return np.asarray(result.transformation, dtype=np.float64)


# ----------------------------------------------------------------------- pair selection
def gather_pairs(
    data_root: str, voxel: float, overlap_thresh: float, min_overlap: float, n_pairs: int, seed: int
):
    """Sample ~``n_pairs`` overlapping fragment pairs across all test scenes (deterministic)."""
    rng = random.Random(seed)
    scenes = sorted(p for p in Path(data_root).iterdir() if (p / "fragments").is_dir())
    candidates = []  # (scene, i, j)
    for sc in scenes:
        frags = sorted((sc / "fragments").glob("cloud_bin_*.ply"), key=lambda p: int(p.stem.split("_")[-1]))
        n = len(frags)
        # Adjacent + near-adjacent pairs are the overlapping ones in a sequential scan.
        for i in range(n):
            for j in range(i + 1, min(i + 4, n)):
                candidates.append((sc, i, j, frags))
    rng.shuffle(candidates)
    pairs = []
    for sc, i, j, frags in candidates:
        if len(pairs) >= n_pairs:
            break
        try:
            pi = read_points(str(frags[i]))
            pj = read_points(str(frags[j]))
            posei = read_pose(str(sc / "poses" / f"{frags[i].stem}.txt"))
            posej = read_pose(str(sc / "poses" / f"{frags[j].stem}.txt"))
        except Exception:
            continue
        T_gt = np.linalg.inv(posej) @ posei  # maps i's points into j's frame
        di = voxel_downsample(pi, voxel)
        dj = voxel_downsample(pj, voxel)
        ov = overlap_ratio(di, dj, T_gt, voxel * 2.0)
        if ov < min_overlap:
            continue
        pairs.append({"scene": sc.name, "i": i, "j": j, "src": di, "tgt": dj, "T_gt": T_gt, "overlap": ov})
    return pairs


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="splatreg vs Open3D on the 3DMatch test split.")
    ap.add_argument("--data-root", default=DEFAULT_DATA)
    ap.add_argument("--n-pairs", type=int, default=60)
    ap.add_argument("--voxel", type=float, default=0.05, help="downsample voxel (m)")
    ap.add_argument("--min-overlap", type=float, default=0.3, help="min GT overlap to keep a pair")
    ap.add_argument("--rmse-thresh", type=float, default=0.2, help="RR RMSE threshold (m)")
    ap.add_argument("--corr-band", type=float, default=0.1, help="GT-correspondence band (m)")
    ap.add_argument("--device", default=os.environ.get("SPLATREG_DEVICE", "cpu"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-open3d", action="store_true")
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("cuda requested but unavailable; CPU.")
        device = "cpu"

    print(f"Sampling overlapping pairs (voxel={args.voxel} m, min_overlap={args.min_overlap}) ...")
    pairs = gather_pairs(
        args.data_root, args.voxel, args.voxel * 2.0, args.min_overlap, args.n_pairs, args.seed
    )
    print(
        f"  -> {len(pairs)} pairs across "
        f"{len(set(p['scene'] for p in pairs))} scenes (median overlap "
        f"{np.median([p['overlap'] for p in pairs]):.2f})\n"
    )

    has_o3d = not args.no_open3d
    if has_o3d:
        try:
            import open3d  # noqa: F401
        except Exception:
            print("open3d unavailable -> skipping baseline.")
            has_o3d = False

    sr = {"rre": [], "rte": [], "rmse": [], "ms": []}
    o3 = {"rre": [], "rte": [], "rmse": [], "ms": []}

    for k, p in enumerate(pairs):
        src, tgt, T_gt = p["src"], p["tgt"], p["T_gt"]

        # splatreg (init="fast"): registers source onto target -> T maps src into tgt frame.
        gs_src = points_to_gaussians(src, device)
        gs_tgt = points_to_gaussians(tgt, device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = register(gs_tgt, gs_src, init="fast", transform="se3")
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        sr["ms"].append((time.perf_counter() - t0) * 1000.0)
        T_est = res.T.detach().cpu().numpy().astype(np.float64)
        sr["rre"].append(rot_err_deg(T_gt[:3, :3], T_est[:3, :3]))
        sr["rte"].append(float(np.linalg.norm(T_gt[:3, 3] - T_est[:3, 3])))
        sr["rmse"].append(correspondence_rmse(src, tgt, T_gt, T_est, args.corr_band))

        if has_o3d:
            t0 = time.perf_counter()
            T_o = open3d_fpfh_ransac(src, tgt, args.voxel)
            o3["ms"].append((time.perf_counter() - t0) * 1000.0)
            o3["rre"].append(rot_err_deg(T_gt[:3, :3], T_o[:3, :3]))
            o3["rte"].append(float(np.linalg.norm(T_gt[:3, 3] - T_o[:3, 3])))
            o3["rmse"].append(correspondence_rmse(src, tgt, T_gt, T_o, args.corr_band))

        if (k + 1) % 10 == 0:
            print(f"  ...{k + 1}/{len(pairs)}")

    def report(name, d):
        if not d["rre"]:
            print(f"{name}: no results")
            return
        rmse = np.asarray(d["rmse"])
        rr = float(np.mean(rmse < args.rmse_thresh))
        succ = rmse < args.rmse_thresh  # report RRE/RTE on the recall-success subset too
        print(f"\n=== {name} (n={len(d['rre'])}) ===")
        print(f"  RR (RMSE<{args.rmse_thresh} m)   : {rr * 100:.1f}%")
        print(f"  median RRE (all)        : {np.median(d['rre']):.2f} deg")
        print(f"  median RTE (all)        : {np.median(d['rte']):.3f} m")
        if succ.any():
            print(f"  median RRE (RR-success) : {np.median(np.asarray(d['rre'])[succ]):.2f} deg")
            print(f"  median RTE (RR-success) : {np.median(np.asarray(d['rte'])[succ]):.3f} m")
        print(f"  median ms/pair          : {np.median(d['ms']):.1f}")

    print("\n" + "=" * 60)
    report("splatreg  (init='fast')", sr)
    if has_o3d:
        report("Open3D  FPFH+RANSAC", o3)
    print("=" * 60)


if __name__ == "__main__":
    main()
