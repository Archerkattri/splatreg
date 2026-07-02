#!/usr/bin/env python
"""MERGE FUSION GIF — splatreg's headline merge, animated on two REAL 3DMatch scans.

Loads two genuinely overlapping real indoor fragments from 3DMatch (default:
``7-scenes-redkitchen`` cloud_bin_0 + cloud_bin_1, ~19k points each, ~0.67 overlap), each in its
OWN sensor frame. The 3DMatch fragment-to-world poses give the ground-truth relative transform, so
the recovered pose can be scored honestly. Then it runs the REAL library calls —
``splatreg.register`` (recover the rigid transform) and ``splatreg.merge`` (align + concatenate +
voxel-dedupe the overlap) — and animates the three stages as a slow turntable:

    1. TWO OVERLAPPING CAPTURES  — A (teal) + B in its own wrong sensor frame (orange), misaligned
    2. REGISTERED  (SE(3))       — B snapped onto A by the recovered transform
    3. MERGED + OVERLAP DEDUP    — one fused splat, the double-density seam collapsed

with the MEASURED rotation/translation error vs GT, overlap fraction, seam gap, and point counts
printed in frame. Every number is this run's actual measurement.

Run (GPU)::

    CUDA_VISIBLE_DEVICES=0 SPLATREG_DEVICE=cuda \
        .gpu_venv/bin/python examples/make_merge_fusion_gif.py

Writes ``assets/merge_fusion.gif`` (+ ``examples/_out/merge_fusion_stats.json``).
Deterministic. Palette: splatreg dark-instrument (teal / orange).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EXAMPLES_DIR)
for _p in (_REPO_ROOT, _EXAMPLES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import splatreg  # noqa: E402
from splatreg.core.types import Gaussians  # noqa: E402
from splatreg.api import apply_transform  # noqa: E402
from _example_utils import rot_angle_deg  # noqa: E402

DEVICE = os.environ.get("SPLATREG_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
DATA = os.environ.get(
    "SPLATREG_3DMATCH_SCENE",
    "/home/krishi/workspace/brain/workspace/projects/3dgs-registration/data/3dmatch/extracted/"
    "data/indoor/test/7-scenes-redkitchen",
)
FRAG_A, FRAG_B = 0, 1
EPS = 0.05  # 5 cm — the 3DMatch inlier standard

# --- dark-instrument palette (matches docs_site/stylesheets/extra.css) -------------------------
BG = "#0b1417"
TEAL = "#2fd6e0"
ORANGE = "#ff6b3d"
INK = "#eafcff"
MUTED = "#7fa6ad"
GOOD = "#37e0a0"


def _load_fragment(i: int) -> torch.Tensor:
    p = torch.load(f"{DATA}/cloud_bin_{i}.pth", weights_only=False)
    return torch.tensor(np.asarray(p), dtype=DTYPE, device=DEVICE)


def _info_pose(i: int) -> torch.Tensor:
    L = open(f"{DATA}/cloud_bin_{i}.info.txt").read().split("\n")
    return torch.tensor([[float(x) for x in L[r].split()] for r in range(1, 5)], dtype=DTYPE, device=DEVICE)


def _to_splat(P: torch.Tensor, opacity: float) -> Gaussians:
    n = P.shape[0]
    return Gaussians(
        means=P,
        quats=torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE).repeat(n, 1),
        scales=torch.full((n, 3), 0.02, device=DEVICE),  # 2 cm anchor footprint
        opacities=torch.full((n,), float(opacity), device=DEVICE),
        colors=None,
        log_scales=False,
    )


def _frac_within(b: torch.Tensor, a: torch.Tensor, eps: float) -> float:
    return float((torch.cdist(b, a).min(1).values <= eps).float().mean())


def _sub(x: torch.Tensor, n: int) -> np.ndarray:
    if x.shape[0] > n:
        sel = torch.linspace(0, x.shape[0] - 1, n, device=x.device).round().long()
        x = x[sel]
    return x.detach().cpu().numpy()


def main() -> str:
    scene = Path(DATA).name
    print(f"loading real 3DMatch fragments: {scene} cloud_bin_{FRAG_A} + cloud_bin_{FRAG_B}")
    Pa, Pb = _load_fragment(FRAG_A), _load_fragment(FRAG_B)
    Ma, Mb = _info_pose(FRAG_A), _info_pose(FRAG_B)
    T_gt = torch.linalg.inv(Ma) @ Mb  # maps B-local -> A-local
    A = _to_splat(Pa, 0.90)
    B = _to_splat(Pb, 0.40)
    print(f"  A {len(A)} pts | B {len(B)} pts | scene diag {float((Pa.amax(0) - Pa.amin(0)).norm()):.2f} m")

    # --- REAL register ------------------------------------------------------------------------
    print("register(A, B, transform='se3', init='robust') ...")
    res = splatreg.register(A, B, transform="se3", init="robust")
    rre = rot_angle_deg(res.T[:3, :3], T_gt[:3, :3])
    rte_mm = 1000.0 * float((res.T[:3, 3] - T_gt[:3, 3]).norm())
    print(f"  RRE {rre:.2f} deg | RTE {rte_mm:.1f} mm | confidence {res.info.get('confidence', 0):.3f}")

    B_reg = apply_transform(B, res.T, res.scale)
    B_gt = apply_transform(B, T_gt, 1.0)  # GT-aligned B, used to define the true overlap set

    # overlap fraction (frac of B within 5 cm of an A surface point): own frame vs registered
    ov_naive = _frac_within(B.means, A.means, EPS)
    ov_reg = _frac_within(B_reg.means, A.means, EPS)

    # seam gap: for the true-overlap subset of B (NN(A) < 10 cm under GT), mean NN distance to A
    nn_gt = torch.cdist(B_gt.means, A.means).min(1).values
    ov_mask = nn_gt < 0.10
    seam_naive_mm = 1000.0 * float(torch.cdist(B.means[ov_mask], A.means).min(1).values.mean())
    seam_reg_mm = 1000.0 * float(torch.cdist(B_reg.means[ov_mask], A.means).min(1).values.mean())

    # --- REAL merge ---------------------------------------------------------------------------
    # voxel sized to the fragments' native ~2.5 cm resolution: it preserves each scan (99% of A
    # survives dedupe alone) so the removed points are the double-density OVERLAP, not decimation.
    VOXEL = 0.025
    print("merge([A, B], transform='se3', init='robust', voxel=0.025) ...")
    merged = splatreg.merge([A, B], ref=0, transform="se3", init="robust", max_iters=40, voxel=VOXEL)
    n_cat = len(A) + len(B)

    stats = {
        "scene": scene,
        "frag_a": FRAG_A,
        "frag_b": FRAG_B,
        "n_A": len(A),
        "n_B": len(B),
        "n_cat": n_cat,
        "n_merged": len(merged),
        "n_dedup_removed": n_cat - len(merged),
        "rre_deg": rre,
        "rte_mm": rte_mm,
        "confidence": float(res.info.get("confidence", 0.0)),
        "overlap_naive": ov_naive,
        "overlap_reg": ov_reg,
        "overlap_ratio": ov_reg / max(ov_naive, 1e-6),
        "seam_naive_mm": seam_naive_mm,
        "seam_reg_mm": seam_reg_mm,
        "seam_ratio": seam_naive_mm / max(seam_reg_mm, 1e-9),
    }
    stat_dir = Path(_EXAMPLES_DIR) / "_out"
    stat_dir.mkdir(exist_ok=True)
    (stat_dir / "merge_fusion_stats.json").write_text(json.dumps(stats, indent=2))
    print("MEASURED:", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}))

    # --- capture the stages for rendering -----------------------------------------------------
    NPTS = 6500
    A_np = _sub(A.means, NPTS)
    Bm_np = _sub(B.means, NPTS)
    Br_np = _sub(B_reg.means, NPTS)

    m_means = merged.means
    m_sub = m_means if m_means.shape[0] <= 2 * NPTS else m_means[
        torch.linspace(0, m_means.shape[0] - 1, 2 * NPTS, device=m_means.device).round().long()
    ]
    A_ref = A.means[torch.linspace(0, len(A) - 1, min(6000, len(A)), device=DEVICE).round().long()]
    B_ref = B_reg.means[torch.linspace(0, len(B_reg) - 1, min(6000, len(B_reg)), device=DEVICE).round().long()]
    from_A = (torch.cdist(m_sub, A_ref).min(1).values <= torch.cdist(m_sub, B_ref).min(1).values).cpu().numpy()
    m_np = m_sub.detach().cpu().numpy()

    out_path = Path(_REPO_ROOT) / "assets" / "merge_fusion.gif"
    _render_gif(stats, A_np, Bm_np, Br_np, m_np, from_A, out_path)
    return str(out_path)


def _render_gif(stats, A_np, Bm_np, Br_np, m_np, from_A, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    plt.rcParams.update({"font.family": "monospace", "font.monospace": ["DejaVu Sans Mono"]})

    allpts = np.concatenate([A_np, Bm_np, Br_np, m_np], axis=0)
    ctr = allpts.mean(0)
    rad = np.percentile(np.linalg.norm(allpts - ctr, axis=1), 99) * 1.02
    lims = [(ctr[i] - rad, ctr[i] + rad) for i in range(3)]

    stages = [
        {
            "key": "misaligned",
            "kicker": "STAGE 1 / 3",
            "title": "two overlapping captures",
            "sub": f"real 3DMatch {stats['scene']}: two ~19k-point indoor scans,\n"
            f"B in its own sensor frame  (~0.67 true overlap)",
            "metric": f"overlap  {stats['overlap_naive']:.2f}      seam gap  {stats['seam_naive_mm']:.0f} mm",
            "metric_color": ORANGE,
        },
        {
            "key": "registered",
            "kicker": "STAGE 2 / 3",
            "title": "registered  ·  SE(3)",
            "sub": "splatreg.register(A, B, init='robust')  vs 3DMatch ground truth:\n"
            f"rotation err {stats['rre_deg']:.2f} deg   translation err {stats['rte_mm']:.0f} mm",
            "metric": f"seam gap  {stats['seam_naive_mm']:.0f}  ->  {stats['seam_reg_mm']:.0f} mm"
            f"     overlap  {stats['overlap_naive']:.2f} -> {stats['overlap_reg']:.2f}",
            "metric_color": TEAL,
        },
        {
            "key": "merged",
            "kicker": "STAGE 3 / 3",
            "title": "merged  +  overlap dedup",
            "sub": f"splatreg.merge([A, B])  ->  {stats['n_merged']:,} Gaussians\n"
            f"registered concat {stats['n_cat']:,}; voxel-dedupe (2.5 cm) dropped {stats['n_dedup_removed']:,} overlap dupes",
            "metric": f"{stats['n_cat']:,} concat  ->  {stats['n_merged']:,} fused"
            f"   ({stats['n_dedup_removed']:,} dupes dropped)",
            "metric_color": GOOD,
        },
    ]
    frames_per_stage = [11, 11, 14]
    az0 = -60.0

    frames = []
    fnum = 0
    total = sum(frames_per_stage)
    for si, st in enumerate(stages):
        for _k in range(frames_per_stage[si]):
            az = az0 + 340.0 * fnum / total
            fig = plt.figure(figsize=(7.2, 4.55), dpi=100)
            fig.patch.set_facecolor(BG)
            ax = fig.add_axes([0.0, 0.0, 1.0, 0.80], projection="3d")
            ax.set_facecolor(BG)

            if st["key"] == "misaligned":
                ax.scatter(*A_np.T, s=1.5, c=TEAL, alpha=0.5, linewidths=0, depthshade=False)
                ax.scatter(*Bm_np.T, s=1.5, c=ORANGE, alpha=0.5, linewidths=0, depthshade=False)
            elif st["key"] == "registered":
                ax.scatter(*A_np.T, s=1.5, c=TEAL, alpha=0.5, linewidths=0, depthshade=False)
                ax.scatter(*Br_np.T, s=1.5, c=ORANGE, alpha=0.5, linewidths=0, depthshade=False)
            else:
                ax.scatter(*m_np[from_A].T, s=1.6, c=TEAL, alpha=0.68, linewidths=0, depthshade=False)
                ax.scatter(*m_np[~from_A].T, s=1.6, c=ORANGE, alpha=0.68, linewidths=0, depthshade=False)

            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])
            ax.set_zlim(*lims[2])
            try:
                ax.set_box_aspect((1, 1, 1), zoom=1.5)
            except TypeError:
                ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=18, azim=az)
            ax.set_axis_off()

            fig.text(0.035, 0.945, "splatreg", color=TEAL, fontsize=15, fontweight="bold")
            fig.text(0.20, 0.951, "merge()  ·  align + fuse + dedupe two 3DGS scans", color=MUTED, fontsize=9)
            fig.text(0.965, 0.945, st["kicker"], color=MUTED, fontsize=8.5, ha="right")
            fig.text(0.035, 0.888, st["title"], color=INK, fontsize=13.5, fontweight="bold")
            fig.text(0.035, 0.848, st["sub"], color=MUTED, fontsize=8.2, va="top", linespacing=1.4)
            fig.text(0.5, 0.043, st["metric"], color=st["metric_color"], fontsize=11.5, ha="center", fontweight="bold")

            if st["key"] != "merged":
                fig.text(0.685, 0.888, "● A (target)", color=TEAL, fontsize=8.5)
                fig.text(0.83, 0.888, "● B (source)", color=ORANGE, fontsize=8.5)

            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            plt.close(fig)
            fnum += 1

    frames += [frames[-1]] * 5
    # imageio >= 2.28 reads `duration` in MILLISECONDS; pass a per-frame list so the
    # turntable runs at ~110 ms/frame and the final merged frame holds for ~1.3 s.
    per_frame_ms = [110] * (len(frames) - 1) + [1300]
    imageio.mimsave(out_path, frames, format="GIF", duration=per_frame_ms, loop=0,
                    subrectangles=True)
    sz = os.path.getsize(out_path) / 1e6
    print(f"wrote {out_path}  ({sz:.2f} MB, {len(frames)} frames)")
    print(f"  STAGE-3 counts: {stats['n_cat']:,} concat -> {stats['n_merged']:,} fused "
          f"({stats['n_dedup_removed']:,} dupes dropped)")
    print(f"  seam {stats['seam_naive_mm']:.0f}->{stats['seam_reg_mm']:.0f} mm | "
          f"overlap {stats['overlap_naive']:.2f}->{stats['overlap_reg']:.2f} | "
          f"RRE {stats['rre_deg']:.2f} deg RTE {stats['rte_mm']:.0f} mm")


if __name__ == "__main__":
    main()
