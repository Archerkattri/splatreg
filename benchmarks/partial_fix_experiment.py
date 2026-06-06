#!/usr/bin/env python
"""Validate the partial-overlap fix BEFORE making it the default.

The robustness sweep showed partial overlap fails 0/9 with the default (un-gated,
non-robust) ICP. Root cause (docs/03): the full reference's points over the target's
*missing* region match to wrong edges. Hypothesis: gating correspondences by distance
(``max_correspondence_dist``) drops those, so the fine LM fits only the overlap.

This runs the partial cells (keep 80/60/40%) under (a) the DEFAULT residual set
(baseline, expected ~0/9) and (b) a GATED point-to-point ICP at a few auto-derived
distance multiples — measuring rotation error. It changes NOTHING in the library
(custom ``residuals=``), so it is safe to run alongside other work.

Run:  CUDA_VISIBLE_DEVICES=0 SPLATREG_DEVICE=cuda PYTHONPATH=. python benchmarks/partial_fix_experiment.py
"""
from __future__ import annotations

import os
import sys

import torch

_BENCH = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BENCH)
for p in (_REPO, os.path.join(_REPO, "examples"), _BENCH):
    if p not in sys.path:
        sys.path.insert(0, p)

from splatreg import register                                   # noqa: E402
from splatreg.residuals import ICP                              # noqa: E402
from _example_utils import (make_object_splat, axis_angle_R,    # noqa: E402
                            rot_angle_deg, sim3_matrix)
from robustness_bench import perturb_partial, ROT_AXIS, FIXED_ROT, TRANS  # noqa: E402

DEV = os.environ.get("SPLATREG_DEVICE", "cuda")
if DEV.startswith("cuda") and not torch.cuda.is_available():
    DEV = "cpu"
DT = torch.float32
N = 1400
SEEDS = [0, 1, 2]
KEEPS = [0.8, 0.6, 0.4]


def median_nn_spacing(g):
    d = torch.cdist(g.means, g.means)
    d.fill_diagonal_(float("inf"))
    return float(d.min(dim=1).values.median())


def run(keep, seed, residuals, label):
    A = make_object_splat(N, seed=seed, device=DEV, dtype=DT)
    R_gt = axis_angle_R(ROT_AXIS, FIXED_ROT, device=DEV, dtype=DT)
    M = sim3_matrix(1.0, R_gt, torch.tensor(TRANS, device=DEV, dtype=DT))
    B = make_object_splat.apply_to(A, M)
    Bp = perturb_partial(B, keep, seed)
    res = register(Bp, A, residuals=residuals, init="global", transform="se3", max_iters=60)
    return rot_angle_deg(res.T[:3, :3] / res.scale, M[:3, :3]), median_nn_spacing(Bp)


def main():
    print(f"partial-overlap fix experiment on {DEV.upper()} (gate = k x median NN spacing)\n")
    configs = []  # (label, residual_factory(spacing))
    configs.append(("default (ungated)", lambda sp: None))
    for k in (3.0, 5.0, 10.0):
        configs.append((f"gated x{k:g}", lambda sp, k=k: [ICP(point_to_plane=False, max_correspondence_dist=k * sp)]))
    header = f"{'cell':>12} | " + " | ".join(f"{lbl:>16}" for lbl, _ in configs)
    print(header); print("-" * len(header))
    gate_ok = {lbl: 0 for lbl, _ in configs}
    n = 0
    for keep in KEEPS:
        for seed in SEEDS:
            n += 1
            row = f"keep{int(keep*100)}% s{seed:>1} | "
            for lbl, fac in configs:
                # derive spacing once from a quick build
                A = make_object_splat(N, seed=seed, device=DEV, dtype=DT)
                Rg = axis_angle_R(ROT_AXIS, FIXED_ROT, device=DEV, dtype=DT)
                Mg = sim3_matrix(1.0, Rg, torch.tensor(TRANS, device=DEV, dtype=DT))
                Bp = perturb_partial(make_object_splat.apply_to(A, Mg), keep, seed)
                sp = median_nn_spacing(Bp)
                res = register(Bp, A, residuals=fac(sp), init="global", transform="se3", max_iters=60)
                err = rot_angle_deg(res.T[:3, :3] / res.scale, Mg[:3, :3])
                gate_ok[lbl] += err < 2.0
                row += f"{err:>15.3f}° | "
            print(row)
    print("-" * len(header))
    print("success (<2deg): " + " | ".join(f"{lbl}={gate_ok[lbl]}/{n}" for lbl, _ in configs))


if __name__ == "__main__":
    main()
