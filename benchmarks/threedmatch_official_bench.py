#!/usr/bin/env python
"""OFFICIAL 3DMatch / 3DLoMatch geometric-registration protocol for splatreg.

Unlike ``threedmatch_bench.py`` (which used splatreg's OWN overlapping-pair sampler and an ICP-RMSE
recall), this runner uses the *canonical* Choi-et-al. protocol that every published learned method
(FCGF / Predator / GeoTransformer) reports on:

* **Pairs** — the fixed list in the official ``gt.log`` for each of the 8 test scenes, restricted to
  the NON-ADJACENT pairs (``frag_id1 > frag_id0 + 1``) exactly as the official evaluator does.  This
  is the canonical 1623-pair 3DMatch set (and the harder ~1781-pair 3DLoMatch set).
* **Metric** — the covariance-weighted transform error from ``gt.info`` (Choi 2015 / Zeng 2017):
  ``e = er^T C er / C[0,0]`` with ``er = [t; q_{1:3}]`` of ``inv(T_gt) @ T_est``.  A pair is a
  success iff ``e <= 0.2**2``.  **RR = successes / num_gt_pairs.**  RRE/RTE are the mean over the
  successful subset (again exactly the official definition).

Ground-truth / info files ship with GeoTransformer under
``third_party_models/GeoTransformer/data/3DMatch/metadata/benchmarks/{3DMatch,3DLoMatch}/``.
Point-cloud fragments are the standard ``cloud_bin_*.ply`` under ``data/3dmatch/3dmatch/test/``.

Run (GPU 1)::

    CUDA_VISIBLE_DEVICES=1 SPLATREG_DEVICE=cuda \
        python benchmarks/threedmatch_official_bench.py --split 3DMatch --init learned
"""

from __future__ import annotations

import argparse
import os
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

# Official GeoTransformer metric helpers (covariance error + quaternion).
_GEOT = os.path.join(_REPO_ROOT, "third_party_models", "GeoTransformer")
if _GEOT not in sys.path:
    sys.path.insert(0, _GEOT)

DEFAULT_FRAG = os.path.join(_REPO_ROOT, "data", "3dmatch", "3dmatch", "test")
DEFAULT_GT = os.path.join(_GEOT, "data", "3DMatch", "metadata", "benchmarks")


# --------------------------------------------------------------------- official metric (self-contained)
def _mat2quat(M: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion [w, x, y, z] — verbatim transforms3d.quaternions.mat2quat
    (the exact convention GeoTransformer's official evaluator uses)."""
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    K = (
        np.array(
            [
                [Qxx - Qyy - Qzz, 0, 0, 0],
                [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0],
                [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
                [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz],
            ]
        )
        / 3.0
    )
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if q[0] < 0:
        q = -q
    return q


def compute_transform_error(transform: np.ndarray, covariance: np.ndarray, est: np.ndarray) -> float:
    """Official Choi/Zeng covariance-weighted transform error (== GeoTransformer's)."""
    rel = np.linalg.inv(transform) @ est
    R, t = rel[:3, :3], rel[:3, 3]
    q = _mat2quat(R)
    er = np.concatenate([t, q[1:]], axis=0)
    return float((er.reshape(1, 6) @ covariance @ er.reshape(6, 1) / covariance[0, 0]).item())


def rre_rte(transform: np.ndarray, est: np.ndarray) -> tuple[float, float]:
    """Isotropic RRE (deg) + RTE (m) — official definition."""
    Rg, Re = transform[:3, :3], est[:3, :3]
    c = (np.trace(Rg.T @ Re) - 1.0) * 0.5
    rre = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    rte = float(np.linalg.norm(transform[:3, 3] - est[:3, 3]))
    return float(rre), rte


def read_log_file(path: str):
    with open(path) as f:
        lines = [ln.strip() for ln in f.readlines()]
    out = []
    for i in range(len(lines) // 5):
        s = lines[i * 5].split()
        T = np.array([lines[i * 5 + j].split() for j in range(1, 5)], dtype=np.float64)
        out.append({"pair": [int(s[0]), int(s[1])], "num_fragments": int(s[2]), "transform": T})
    return out


def read_info_file(path: str):
    with open(path) as f:
        lines = [ln.strip() for ln in f.readlines()]
    out = []
    for i in range(len(lines) // 7):
        s = lines[i * 7].split()
        info = np.array([lines[i * 7 + j].split() for j in range(1, 7)], dtype=np.float64)
        out.append({"pair": [int(s[0]), int(s[1])], "covariance": info})
    return out


# --------------------------------------------------------------------- IO + register helpers
def read_points(path: str) -> np.ndarray:
    cols = _read_ply_vertex(path)
    return np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float64)


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def to_gaussians(pts: np.ndarray, device, dtype=torch.float32) -> Gaussians:
    m = torch.as_tensor(pts, device=device, dtype=dtype)
    n = m.shape[0]
    return Gaussians(
        means=m,
        quats=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).repeat(n, 1),
        scales=torch.full((n, 3), 0.02, device=device, dtype=dtype),
        opacities=torch.ones(n, device=device, dtype=dtype),
        log_scales=False,
    )


def open3d_fpfh_ransac(src: np.ndarray, tgt: np.ndarray, voxel: float) -> np.ndarray:
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
        sp, tp, sf, tf, True, dist,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    # ICP polish (standard in the Open3D RANSAC recipe).
    result = o3d.pipelines.registration.registration_icp(
        sp, tp, dist, result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return np.asarray(result.transformation, dtype=np.float64)


SCENES = [
    "7-scenes-redkitchen",
    "sun3d-home_at-home_at_scan1_2013_jan_1",
    "sun3d-home_md-home_md_scan9_2012_sep_30",
    "sun3d-hotel_uc-scan3",
    "sun3d-hotel_umd-maryland_hotel1",
    "sun3d-hotel_umd-maryland_hotel3",
    "sun3d-mit_76_studyroom-76-1studyroom2",
    "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika",
]


def main():
    ap = argparse.ArgumentParser(description="OFFICIAL 3DMatch/3DLoMatch protocol for splatreg.")
    ap.add_argument("--split", default="3DMatch", choices=["3DMatch", "3DLoMatch"])
    ap.add_argument("--frag-root", default=DEFAULT_FRAG)
    ap.add_argument("--gt-root", default=DEFAULT_GT)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument(
        "--learned-voxel",
        type=float,
        default=0.025,
        help=(
            "Voxel for init='learned' ONLY. GeoTransformer's official 3DMatch protocol runs at "
            "init_voxel_size=0.025; the legacy 0.05 downsample halved its input resolution (~5k vs "
            "~19k pts/fragment) and was the dominant cause of the gap vs. the published ~92/74 RR. "
            "Set =0 to use the same --voxel as the other inits (legacy behaviour)."
        ),
    )
    ap.add_argument("--init", default="robust",
                    choices=["fast", "robust", "learned", "features", "global", "open3d"])
    ap.add_argument("--device", default=os.environ.get("SPLATREG_DEVICE", "cpu"))
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all official pairs")
    ap.add_argument("--scenes", default="", help="comma-sep scene subset (default all 8)")
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    scenes = args.scenes.split(",") if args.scenes else SCENES
    gt_split = os.path.join(args.gt_root, args.split)

    # Per-scene accumulation, then official aggregation (mean of per-scene recalls).
    scene_recalls, scene_rre, scene_rte = [], [], []
    total_gt = total_pos = total_pred = 0
    all_errors = []
    ms_list = []
    t_start = time.perf_counter()

    for scene in scenes:
        logs = read_log_file(os.path.join(gt_split, scene, "gt.log"))
        infos = read_info_file(os.path.join(gt_split, scene, "gt.info"))
        # Build non-adjacent gt index set.
        gt_pairs = []
        for li, lg in enumerate(logs):
            a, b = lg["pair"]
            if b > a + 1:
                gt_pairs.append((li, a, b))
        num_gt = len(gt_pairs)
        total_gt += num_gt

        frag_dir = Path(args.frag_root) / scene / "fragments"
        pos = 0
        rres, rtes = [], []
        for (li, a, b) in gt_pairs:
            if args.max_pairs and total_pred >= args.max_pairs:
                break
            fa = frag_dir / f"cloud_bin_{a}.ply"
            fb = frag_dir / f"cloud_bin_{b}.ply"
            if not (fa.exists() and fb.exists()):
                continue
            # init='learned' feeds GeoTransformer, whose released 3DMatch model expects its native
            # init_voxel_size=0.025 input resolution.  The legacy 0.05 downsample discarded ~75 % of
            # the points the learned matcher was trained on and crippled low-overlap (3DLoMatch) in
            # particular.  Run it at GeoTransformer-native voxel; every other init keeps --voxel so
            # their numbers (and the 44 tests) are byte-for-byte unchanged.
            eff_voxel = (
                args.learned_voxel
                if (args.init == "learned" and args.learned_voxel > 0.0)
                else args.voxel
            )
            tgt = voxel_downsample(read_points(str(fa)), eff_voxel)   # frag a = target
            src = voxel_downsample(read_points(str(fb)), eff_voxel)   # frag b = source
            T_gt = logs[li]["transform"]      # maps b (src) -> a (tgt)
            cov = infos[li]["covariance"]
            total_pred += 1

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if args.init == "open3d":
                T_est = open3d_fpfh_ransac(src, tgt, args.voxel)
            else:
                res = register(to_gaussians(tgt, device), to_gaussians(src, device),
                               init=args.init, transform="se3")
                T_est = res.T.detach().cpu().numpy().astype(np.float64)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            ms_list.append((time.perf_counter() - t0) * 1000.0)

            err = compute_transform_error(T_gt, cov, T_est)
            all_errors.append(err)
            if err <= 0.2 ** 2:
                pos += 1
                r, t = rre_rte(T_gt, T_est)
                rres.append(r)
                rtes.append(t)
        total_pos += pos
        rec = pos / num_gt if num_gt else 0.0
        scene_recalls.append(rec)
        if rres:
            scene_rre.append(np.mean(rres))
            scene_rte.append(np.mean(rtes))
        print(f"  {scene:52s} RR={rec*100:5.1f}%  ({pos}/{num_gt})")

    dt = time.perf_counter() - t_start
    print("\n" + "=" * 68)
    print(f"OFFICIAL {args.split}   init={args.init!r}   voxel={args.voxel}   device={device}")
    print("=" * 68)
    print(f"  pairs evaluated      : {total_pred} / {total_gt} official non-adjacent")
    # Official RR = mean of per-scene recalls (Choi/GeoTransformer report this).
    print(f"  RR (mean-of-scenes)  : {np.mean(scene_recalls)*100:.1f}%")
    print(f"  RR (pooled)          : {total_pos/total_gt*100:.1f}%  ({total_pos}/{total_gt})")
    if scene_rre:
        print(f"  RRE (mean, success)  : {np.mean(scene_rre):.2f} deg")
        print(f"  RTE (mean, success)  : {np.mean(scene_rte):.3f} m")
    print(f"  median ms/pair       : {np.median(ms_list):.1f}")
    print(f"  total wall           : {dt:.1f} s")
    print("=" * 68)


if __name__ == "__main__":
    main()
