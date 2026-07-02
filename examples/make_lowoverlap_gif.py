"""Animate real low-overlap 3DGS/point-cloud registration: the classical seed
failing, the BUFFER-X seed locking on.

Renders ``assets/registration_lowoverlap.gif`` -- a three-phase animation of a
single **real 3DMatch** low-overlap fragment pair (7-scenes-redkitchen 35->46,
GT overlap 0.10):

  1. **Unaligned**        -- the source fragment sits in its raw local frame,
                             clearly misplaced against the gray target.
  2. **Classical seed**   -- the source slews to the classical FPFH+RANSAC
     (FPFH+RANSAC)           result, which on this low-overlap pair lands ~151.5
                             deg off (catastrophic).
  3. **BUFFER-X seed +    -- the source interpolates (slerp rotation + lerp
     splatreg refine**       translation) onto the BUFFER-X result and locks
                             onto the target at ~2.0 deg.

The two transforms are **not faked**: the script calls the shipped library
functions :func:`splatreg.align_features.robust_feature_align` (classical) and
:func:`splatreg.align_features.bufferx_feature_align` (BUFFER-X zero-shot seed +
splatreg refine) on the real fragment pair, then animates the source cloud from
one estimate to the next. RRE numbers overlaid in each phase are computed
against the fragments' ground-truth relative pose.

Deterministic (fixed subsample RNG + fixed view). Palette: dataviz categorical
slot 1 (blue, BUFFER-X) and slot 8 (orange, classical) -- the same pairing as
``assets/bufferx_recall.*`` so the two figures read as one system.

Requires the BUFFER-X backend + weights to be built (see
``third_party_models/README-BUFFERX.md``); run under the GPU venv. Example::

    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \\
      python examples/make_lowoverlap_gif.py
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from splatreg.align_features import bufferx_feature_align, robust_feature_align  # noqa: E402

# ── palette (dataviz light surface) ─────────────────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
TARGET_GRAY = "#9c9a93"
C_BUFFERX = "#2a78d6"  # slot 1, blue  -> BUFFER-X track
C_ROBUST = "#eb6834"  # slot 8, orange -> classical track
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"


# ── data IO (3DMatch original geometric-registration layout) ────────────────────
def _read_pth_points(path: Path) -> np.ndarray:
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    if hasattr(d, "numpy"):
        d = d.numpy()
    return np.ascontiguousarray(np.asarray(d, dtype=np.float64))


def _read_info_pose(path: Path) -> np.ndarray:
    lines = path.read_text().strip().splitlines()
    return np.asarray([[float(x) for x in ln.split()] for ln in lines[1:5]], dtype=np.float64)


def _rre_rte(T_est: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    R_err = T_est[:3, :3].T @ T_gt[:3, :3]
    rre = float(np.degrees(np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))))
    rte = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
    return rre, rte


def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def _interp_transforms(T_a: np.ndarray, T_b: np.ndarray, n: int) -> list[np.ndarray]:
    """slerp rotation + lerp translation between two 4x4 poses -> n frames (a..b)."""
    from scipy.spatial.transform import Rotation, Slerp

    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([T_a[:3, :3], T_b[:3, :3]])))
    out = []
    for f in np.linspace(0.0, 1.0, n):
        T = np.eye(4)
        T[:3, :3] = slerp(f).as_matrix()
        T[:3, 3] = (1.0 - f) * T_a[:3, 3] + f * T_b[:3, 3]
        out.append(T)
    return out


def main() -> str:
    ap = argparse.ArgumentParser(description=__doc__)
    default_data = "/home/krishi/workspace/brain/workspace/projects/3dgs-registration/data"
    ap.add_argument("--data", default=os.environ.get("SPLATREG_3DGS_DATA", default_data))
    ap.add_argument("--scene", default="7-scenes-redkitchen")
    ap.add_argument("--i", type=int, default=35, help="target fragment index")
    ap.add_argument("--j", type=int, default=46, help="source fragment index")
    ap.add_argument("--voxel", type=float, default=0.025)
    ap.add_argument("--n-target", type=int, default=2600, help="max target points drawn")
    ap.add_argument("--n-source", type=int, default=2200, help="max source points drawn")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default=str(REPO / "assets" / "registration_lowoverlap.gif"),
    )
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load the real pair + GT relative pose (source j -> target i) ────────────
    sd = Path(a.data) / "3dmatch" / "extracted" / "data" / "indoor" / "test" / a.scene
    spec = importlib.util.spec_from_file_location(
        "tdm", str(REPO / "benchmarks" / "threedmatch_official_bench.py")
    )
    tdm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tdm)

    pts_i = tdm.voxel_downsample(_read_pth_points(sd / f"cloud_bin_{a.i}.pth"), a.voxel)
    pts_j = tdm.voxel_downsample(_read_pth_points(sd / f"cloud_bin_{a.j}.pth"), a.voxel)
    pose_i = _read_info_pose(sd / f"cloud_bin_{a.i}.info.txt")
    pose_j = _read_info_pose(sd / f"cloud_bin_{a.j}.info.txt")
    T_gt = np.linalg.inv(pose_i) @ pose_j

    # ── run the actual library calls to get the two estimates (NOT faked) ───────
    tgt_g = tdm.to_gaussians(pts_i, dev)
    src_g = tdm.to_gaussians(pts_j, dev)
    T_robust, info_r = robust_feature_align(tgt_g, src_g, transform="se3", voxel=a.voxel)
    T_bufferx, info_b = bufferx_feature_align(tgt_g, src_g, transform="se3", voxel=a.voxel)
    T_robust = T_robust.detach().cpu().numpy().astype(np.float64)
    T_bufferx = T_bufferx.detach().cpu().numpy().astype(np.float64)
    if not info_b.get("used_bufferx", False):
        raise RuntimeError(
            "BUFFER-X backend unavailable (fell back to classical seed); build it first "
            "(third_party_models/README-BUFFERX.md) and run under the GPU venv."
        )

    rre_r, rte_r = _rre_rte(T_robust, T_gt)
    rre_b, rte_b = _rre_rte(T_bufferx, T_gt)
    ov = tdm.__dict__.get("overlap_ratio", None)
    print(f"classical robust : RRE {rre_r:.2f} deg  RTE {rte_r:.3f} m")
    print(f"BUFFER-X + refine: RRE {rre_b:.2f} deg  RTE {rte_b:.3f} m  (seed={info_b.get('seed')})")

    # ── subsample the drawn clouds (seeded) ─────────────────────────────────────
    def _sub(p, k):
        return p if len(p) <= k else p[rng.choice(len(p), k, replace=False)]

    tgt = _sub(pts_i, a.n_target)
    src = _sub(pts_j, a.n_source)

    # ── build the frame schedule ────────────────────────────────────────────────
    I = np.eye(4)
    frames: list[dict] = []

    def hold(T, n, **kw):
        for _ in range(n):
            frames.append({"T": T, **kw})

    def slew(Ta, Tb, n, **kw):
        for T in _interp_transforms(Ta, Tb, n):
            frames.append({"T": T, **kw})

    p1 = {"color": C_ROBUST, "phase": "1 / 3   Unaligned",
          "metric": f"real 3DMatch pair  •  GT overlap 0.10", "mcolor": INK_2}
    p2 = {"color": C_ROBUST, "phase": "2 / 3   Classical seed  (FPFH + RANSAC)",
          "metric": f"RRE {rre_r:.1f}°   ✗  wrong basin", "mcolor": CRITICAL}
    p3 = {"color": C_BUFFERX, "phase": "3 / 3   BUFFER-X seed + splatreg refine",
          "metric": f"RRE {rre_b:.1f}°   ✓  locked on", "mcolor": GOOD}

    hold(I, 8, **p1)                     # phase 1: unaligned
    slew(I, T_robust, 14, **p2)          # phase 2: slew to classical result
    hold(T_robust, 10, **p2)             #          hold on the wrong basin
    slew(T_robust, T_bufferx, 20, **p3)  # phase 3: interpolate onto BUFFER-X
    hold(T_bufferx, 16, **p3)            #          locked

    # ── fixed cube limits over every pose the source visits ─────────────────────
    allpts = [tgt]
    for T in (I, T_robust, T_bufferx):
        allpts.append(_apply(T, src))
    allpts = np.concatenate(allpts, 0)
    ctr = 0.5 * (allpts.min(0) + allpts.max(0))
    half = 0.55 * float((allpts.max(0) - allpts.min(0)).max())
    lims = np.stack([ctr - half, ctr + half], 1)

    # ── figure ──────────────────────────────────────────────────────────────────
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    plt.rcParams.update(
        {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"]}
    )

    fig = plt.figure(figsize=(7.0, 7.0), dpi=a.dpi)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(SURFACE)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=18, azim=-58)

    for setlim, lo, hi in ((ax.set_xlim, *lims[0]), (ax.set_ylim, *lims[1]), (ax.set_zlim, *lims[2])):
        setlim(lo, hi)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1.0, 1.0, 1.0, 0.0))
        pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)

    # static target cloud (gray environment)
    ax.scatter(tgt[:, 0], tgt[:, 1], tgt[:, 2], s=3, c=TARGET_GRAY, alpha=0.32,
               linewidths=0, depthshade=False, zorder=1)

    # persistent text overlays
    fig.text(0.045, 0.955, "splatreg  —  low-overlap registration on real 3DMatch",
             fontsize=15, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.045, 0.918,
             "gray = target fragment 35   •   colored = source fragment 46 being registered",
             fontsize=10.5, color=INK_2, ha="left", va="top")
    t_phase = fig.text(0.045, 0.085, "", fontsize=13.5, color=INK, fontweight="bold",
                       ha="left", va="bottom")
    t_metric = fig.text(0.045, 0.048, "", fontsize=12, color=INK_2, ha="left", va="bottom")

    src_artist = {"a": None}

    def update(k):
        fr = frames[k]
        P = _apply(fr["T"], src)
        if src_artist["a"] is not None:
            src_artist["a"].remove()
        src_artist["a"] = ax.scatter(
            P[:, 0], P[:, 1], P[:, 2], s=6, c=fr["color"], alpha=0.95,
            linewidths=0, depthshade=False, zorder=3,
        )
        t_phase.set_text(fr["phase"])
        t_phase.set_color(fr["color"])
        t_metric.set_text(fr["metric"])
        t_metric.set_color(fr["mcolor"])
        return (src_artist["a"], t_phase, t_metric)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / a.fps, blit=False)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=a.fps), dpi=a.dpi,
              savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)

    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({size_mb:.2f} MB, {len(frames)} frames, ~{len(frames)/a.fps:.1f}s)")
    return str(out)


if __name__ == "__main__":
    main()
