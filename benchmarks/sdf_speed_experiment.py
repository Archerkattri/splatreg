#!/usr/bin/env python
"""SDF speed experiment — does truncation preserve accuracy at much higher speed?

vs-ICP showed splatreg's SE(3) is ~1000x slower than ICP, dominated by the Gaussian-SDF
field evaluation (every source point sums over every target anchor: N x M). The field
supports `trunc_sigmas` (only anchors within k.sigma of a query contribute, via a top-k
gather: N x k). This compares default (no trunc) vs trunc on a few recovery cells, measuring
rotation/scale error AND wall time — to see if trunc is a free order-of-magnitude speed-up.

Run:  CUDA_VISIBLE_DEVICES=0 SPLATREG_DEVICE=cuda PYTHONPATH=. python benchmarks/sdf_speed_experiment.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

_BENCH = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BENCH)
for p in (_REPO, os.path.join(_REPO, "examples"), _BENCH):
    if p not in sys.path:
        sys.path.insert(0, p)

from splatreg import register  # noqa: E402
from splatreg.residuals import ICP, SDF  # noqa: E402
from splatreg.api import _auto_sdf_sigma  # noqa: E402
from _example_utils import make_object_splat, axis_angle_R, rot_angle_deg, sim3_matrix  # noqa: E402

DEV = os.environ.get("SPLATREG_DEVICE", "cuda")
if DEV.startswith("cuda") and not torch.cuda.is_available():
    DEV = "cpu"
DT = torch.float32
AXIS, TRANS = [0.3, 0.9, 0.25], [0.03, -0.02, 0.025]


def cell(transform, rot, label, trunc):
    A = make_object_splat(1400, seed=0, device=DEV, dtype=DT)
    s_gt = 1.0 if transform == "se3" else 1.3
    R = axis_angle_R(AXIS, rot, device=DEV, dtype=DT)
    M = sim3_matrix(s_gt, R, torch.tensor(TRANS, device=DEV, dtype=DT))
    B = make_object_splat.apply_to(A, M)
    sigma = _auto_sdf_sigma(B)
    residuals = [
        ICP(point_to_plane=False, weight=1.0),
        SDF(sigma=sigma, weight=0.3, n_points=0, trunc_sigmas=trunc),
    ]
    if DEV.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = register(B, A, residuals=residuals, init="global", transform=transform, max_iters=60)
    if DEV.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    rot_err = rot_angle_deg(res.T[:3, :3] / res.scale, M[:3, :3] / s_gt)
    scl_err = 100.0 * abs(res.scale - s_gt) / s_gt
    return rot_err, scl_err, dt


def main():
    print(f"SDF truncation speed/accuracy on {DEV.upper()}\n")
    print(f"{'cell':>14} | {'config':>8} | {'rot_err':>9} | {'scale_err':>10} | {'sec':>7}")
    print("-" * 64)
    for transform in ("se3", "sim3"):
        for rot in (30.0, 90.0):
            base = None
            for label, trunc in [("default", None), ("trunc3", 3.0), ("trunc4", 4.0)]:
                # warm up once (kernel compile) so timings are fair
                if base is None:
                    cell(transform, rot, "warm", None)
                re, se, dt = cell(transform, rot, label, trunc)
                if label == "default":
                    base = dt
                sp = f"{base/dt:.1f}x" if base else ""
                print(
                    f"{transform+' '+str(int(rot))+'°':>14} | {label:>8} | {re:>8.3f}° | "
                    f"{se:>9.3f}% | {dt:>6.1f}s {sp}"
                )
        print("-" * 64)


if __name__ == "__main__":
    main()
