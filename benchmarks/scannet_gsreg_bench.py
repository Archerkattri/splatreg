#!/usr/bin/env python
"""OFFICIAL ScanNet-GSReg (GaussReg, ECCV 2024) splat-to-splat registration protocol for splatreg.

The benchmark GaussReg introduced: 82 test scenes from ScanNet, each reconstructed twice (two
random continuous image subsequences A and B -> two independent 3DGS models with 0.2-0.8 overlap),
each side perturbed by a recorded random Sim(3); the task is to recover the relative Sim(3) that
maps the perturbed B (source) splat onto the perturbed A (reference) splat.

Protocol mirrored 1:1 from GaussReg's official ``experiments/geotransformer.gaussian_splatting.
indoor/test.py``:

* **Input** — ``test/<scene>/{A,B}/output/point_cloud/iteration_10000/point_cloud.ply`` with the
  per-scene ``ref``/``src`` Sim(3) from ``test_transformations.npz`` baked in first.
* **Metrics** — their exact ``compute_registration_error_w_scale``:
  - scale  ``s = sqrt(trace(M M^T) / 3)``, ``R = M/s``, ``t' = t/s`` (their decomposition),
  - RRE  = arccos((trace(R_est^T R_gt) - 1) / 2) in degrees,
  - RTE  = ||t'_gt - t'_est|| / ||t'_gt||   (relative),
  - RSE  = |s_gt - s_est| / s_gt,
  plus their printed threshold ratios (RRE<5/10 deg, RTE<0.1/0.2, RSE<0.1/0.2) and wall time.

* **Published reference numbers** (GaussReg paper, Table 1, same split):
  - GaussReg (coarse):        RRE 2.827  RTE 0.042  RSE 0.032  time 4.8 s
  - GaussReg (coarse-to-fine):RRE 1.851  RTE 0.029  RSE 0.030
  - HLoc (SP+SG):             RRE 2.725  RTE 0.099  RSE 0.098  time 212.3 s (75.6 % solved)
  - FGR:                      RRE 157.1  RTE 3.328  RSE 0.268  time 3.4 s
  - REGTR:                    RRE 80.10  RTE 2.768  RSE 0.408  time 3.5 s
  (PhotoReg, arXiv 2410.05044, does NOT report on this benchmark - image-quality metrics only on
  Playroom/Truck/etc. The photometric-refine comparison is therefore method-level, not number-level.)

GaussReg's loader additionally drops low-opacity floaters (sigmoid(opacity) <= 0.7) and the 5-95
percentile spatial outliers before sampling 30 k FPS points; that is THEIR method-side
preprocessing. splatreg consumes the raw splats; ``--prefilter`` applies the same opacity (and
optional percentile) gate as method-side preprocessing for a like-for-like floater regime.

Run (GPU 0)::

    CUDA_VISIBLE_DEVICES=0 SPLATREG_DEVICE=cuda python benchmarks/scannet_gsreg_bench.py \
        --data /home/krishi/workspace/data/scannet_gsreg/ScanNet-GSReg \
        --init learned --transform sim3 --refine photometric
"""

from __future__ import annotations

import argparse
import json
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
from splatreg.api import apply_transform  # noqa: E402
from splatreg.core.types import Gaussians  # noqa: E402
from splatreg.io import load_ply  # noqa: E402

PUBLISHED = {
    "GaussReg (coarse)": dict(rre=2.827, rte=0.042, rse=0.032, time=4.8),
    "GaussReg (c2f)": dict(rre=1.851, rte=0.029, rse=0.030, time=None),
    "HLoc (SP+SG, 75.6%)": dict(rre=2.725, rte=0.099, rse=0.098, time=212.3),
    "FGR": dict(rre=157.126, rte=3.328, rse=0.268, time=3.4),
    "REGTR": dict(rre=80.095, rte=2.768, rse=0.408, time=3.5),
}


# ------------------------------------------------------------------ official GaussReg metric
def _decompose_w_scale(T: np.ndarray):
    """GaussReg's get_rotation_translation_from_transform_w_scale (numpy variant), verbatim."""
    scale = ((T[:3, :3] @ T[:3, :3].T).trace() / 3) ** 0.5
    rotation = T[:3, :3] / scale
    translation = T[:3, 3] / scale
    return rotation, translation, scale


def registration_error_w_scale(gt: np.ndarray, est: np.ndarray):
    """GaussReg's compute_registration_error_w_scale, verbatim semantics."""
    gt_R, gt_t, gt_s = _decompose_w_scale(gt.astype(np.float64))
    est_R, est_t, est_s = _decompose_w_scale(est.astype(np.float64))
    x = 0.5 * (np.trace(est_R.T @ gt_R) - 1.0)
    rre = np.degrees(np.arccos(np.clip(x, -1.0, 1.0)))
    rte = np.linalg.norm(gt_t - est_t) / np.linalg.norm(gt_t)
    rse = np.linalg.norm(gt_s - est_s) / np.linalg.norm(gt_s)
    return float(rre), float(rte), float(rse)


# ------------------------------------------------------------------ data
def _sim3_parts(T: np.ndarray):
    s = float(((T[:3, :3] @ T[:3, :3].T).trace() / 3) ** 0.5)
    return torch.from_numpy(T.astype(np.float32)), s


def _prefilter(g: Gaussians, opacity_thresh: float = 0.7, percentile: bool = False) -> Gaussians:
    """GaussReg-style method-side floater gate: sigmoid(opacity) > 0.7 (+ optional 5-95 pct box)."""
    keep = torch.sigmoid(g.opacities.flatten()) > opacity_thresh
    if percentile:
        for d in range(3):
            x = g.means[:, d]
            lo, hi = torch.quantile(x, 0.05), torch.quantile(x, 0.95)
            keep &= (x > lo) & (x < hi)
    idx = torch.nonzero(keep).flatten()
    return Gaussians(
        means=g.means[idx],
        scales=g.scales[idx],
        quats=g.quats[idx],
        opacities=g.opacities[idx],
        colors=None if g.colors is None else g.colors[idx],
        log_scales=g.log_scales,
    )


def load_scene(scene_dir: Path, npz_tr: dict, scene: str, device: str,
               prefilter: bool, percentile: bool):
    ref_T, ref_s = _sim3_parts(npz_tr["ref_transformations_list"][scene])
    src_T, src_s = _sim3_parts(npz_tr["src_transformations_list"][scene])
    gt = npz_tr["gt_transformations_list"][scene]

    def _load(side: str, T: torch.Tensor, s: float) -> Gaussians:
        ply = scene_dir / side / "output" / "point_cloud" / "iteration_10000" / "point_cloud.ply"
        g = load_ply(ply, device=device)
        if prefilter:
            g = _prefilter(g, percentile=percentile)
        return apply_transform(g, T.to(device), s)

    return _load("A", ref_T, ref_s), _load("B", src_T, src_s), gt


# ------------------------------------------------------------------ runner
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="/home/krishi/workspace/data/scannet_gsreg/ScanNet-GSReg")
    ap.add_argument("--init", default="learned",
                    choices=["learned", "robust", "features", "mac", "global", "fast"])
    ap.add_argument("--transform", default="sim3", choices=["se3", "sim3"])
    ap.add_argument("--refine", default="photometric", choices=["none", "photometric"])
    ap.add_argument("--quality", default="auto")
    ap.add_argument("--no-prefilter", action="store_true",
                    help="feed raw splats (keep floaters) instead of the GaussReg-style opacity gate")
    ap.add_argument("--percentile-crop", action="store_true",
                    help="also apply GaussReg's 5-95 percentile spatial crop in the prefilter")
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--scenes", nargs="*", default=None, help="explicit scene names to run")
    ap.add_argument("--out", default=None, help="output json (default benchmarks/scannet_gsreg_<tag>.json)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data)
    npz = np.load(data / "test_transformations.npz", allow_pickle=True)["transformations"].item()
    scenes = sorted(npz["gt_transformations_list"].keys())
    have = [s for s in scenes if (data / "test" / s / "A" / "output" / "point_cloud"
                                  / "iteration_10000" / "point_cloud.ply").exists()
            and (data / "test" / s / "B" / "output" / "point_cloud"
                 / "iteration_10000" / "point_cloud.ply").exists()]
    missing = [s for s in scenes if s not in have]
    if args.scenes:
        have = [s for s in have if s in set(args.scenes)]
    if args.max_scenes:
        have = have[: args.max_scenes]
    print(f"[data] {len(scenes)} test scenes in GT; {len(have)} runnable, {len(missing)} missing on disk")
    if not have:
        sys.exit("no scenes available - is the download finished?")

    refine = None if args.refine == "none" else args.refine
    tag = f"{args.init}-{args.transform}" + ("" if refine is None else f"-{refine}")
    rows = []
    for i, scene in enumerate(have):
        ref_g, src_g, gt = load_scene(data / "test" / scene, npz, scene, device,
                                      prefilter=not args.no_prefilter,
                                      percentile=args.percentile_crop)
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.time()
        res = register(ref_g, src_g, init=args.init, transform=args.transform,
                       quality=args.quality, refine=refine)
        torch.cuda.synchronize() if device == "cuda" else None
        dt = time.time() - t0
        est = res.T.detach().cpu().numpy().astype(np.float64)
        rre, rte, rse = registration_error_w_scale(gt, est)
        rows.append(dict(scene=scene, rre=rre, rte=rte, rse=rse, time=dt,
                         n_ref=len(ref_g), n_src=len(src_g)))
        print(f"[{i+1:3d}/{len(have)}] {scene}: RRE {rre:7.3f}  RTE {rte:6.3f}  RSE {rse:6.3f}  "
              f"{dt:5.1f}s  ({len(ref_g):,} vs {len(src_g):,} gaussians)", flush=True)

    rre = np.array([r["rre"] for r in rows])
    rte = np.array([r["rte"] for r in rows])
    rse = np.array([r["rse"] for r in rows])
    ts = np.array([r["time"] for r in rows])
    summary = dict(
        n=len(rows), init=args.init, transform=args.transform, refine=args.refine,
        prefilter=not args.no_prefilter, percentile_crop=args.percentile_crop,
        rre_avg=float(rre.mean()), rte_avg=float(rte.mean()), rse_avg=float(rse.mean()),
        rre_median=float(np.median(rre)), rte_median=float(np.median(rte)),
        rse_median=float(np.median(rse)),
        rre_lt5=float((rre < 5).mean()), rre_lt10=float((rre < 10).mean()),
        rte_lt01=float((rte < 0.1).mean()), rte_lt02=float((rte < 0.2).mean()),
        rse_lt01=float((rse < 0.1).mean()), rse_lt02=float((rse < 0.2).mean()),
        time_avg=float(ts.mean()),
    )

    print("\n================ ScanNet-GSReg (official 82-scene protocol) ================")
    print(f"splatreg  init={args.init} transform={args.transform} refine={args.refine} "
          f"prefilter={not args.no_prefilter}  [{len(rows)} scenes]")
    print(f"  RRE avg {summary['rre_avg']:.3f} deg (median {summary['rre_median']:.3f})  "
          f"RTE avg {summary['rte_avg']:.3f} (median {summary['rte_median']:.3f})  "
          f"RSE avg {summary['rse_avg']:.3f} (median {summary['rse_median']:.3f})  "
          f"time {summary['time_avg']:.1f}s")
    print(f"  RRE<5 {summary['rre_lt5']*100:.1f}%  RRE<10 {summary['rre_lt10']*100:.1f}%  "
          f"RTE<0.1 {summary['rte_lt01']*100:.1f}%  RTE<0.2 {summary['rte_lt02']*100:.1f}%  "
          f"RSE<0.1 {summary['rse_lt01']*100:.1f}%  RSE<0.2 {summary['rse_lt02']*100:.1f}%")
    print("---- published (GaussReg paper, Table 1) ----")
    for name, m in PUBLISHED.items():
        t = "-" if m["time"] is None else f"{m['time']:.1f}s"
        print(f"  {name:24s} RRE {m['rre']:8.3f}  RTE {m['rte']:6.3f}  RSE {m['rse']:6.3f}  time {t}")

    out = args.out or os.path.join(_REPO_ROOT, "benchmarks", f"scannet_gsreg_{tag}.json")
    with open(out, "w") as f:
        json.dump(dict(summary=summary, scenes=rows, missing=missing), f, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
