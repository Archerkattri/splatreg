#!/usr/bin/env python
"""Partial-overlap fix v2 — overlap-aware (target->source) global alignment.

Experiment 1 showed gating the fine ICP is insufficient (2/9): the GLOBAL INIT is the
dominant failure. align.py matches SOURCE->TARGET (every full-reference point must find a
target match) and centroid/RMS-inits assuming FULL overlap, so a partial target's shifted
centroid corrupts the seed. Fix: match TARGET->SOURCE — every observed partial-B point
has a correct match in the full reference A (B_partial subset of T.A) — and fit the
transform on those inliers. Validated here on the partial cells (SE3) vs the default 0/9.
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

from splatreg.align import super_fibonacci_so3, _stride_subsample      # noqa: E402
from _example_utils import (make_object_splat, axis_angle_R,           # noqa: E402
                            rot_angle_deg, sim3_matrix)
from robustness_bench import perturb_partial, ROT_AXIS, FIXED_ROT, TRANS  # noqa: E402

DEV = os.environ.get("SPLATREG_DEVICE", "cuda")
if DEV.startswith("cuda") and not torch.cuda.is_available():
    DEV = "cpu"
DT = torch.float32


def _nn(X, Y):
    d = torch.cdist(X, Y)
    return d.min(dim=1)


def _kabsch(src, dst):
    sc = src - src.mean(0)
    dc = dst - dst.mean(0)
    U, _, Vh = torch.linalg.svd(sc.T @ dc)
    d = torch.sign(torch.linalg.det(Vh.transpose(-1, -2) @ U.transpose(-1, -2)))
    D = torch.diag(torch.tensor([1.0, 1.0, float(d)], device=src.device, dtype=src.dtype))
    R = Vh.transpose(-1, -2) @ D @ U.transpose(-1, -2)
    t = dst.mean(0) - R @ src.mean(0)
    return R, t


@torch.no_grad()
def overlap_aware_align(B_means, A_means, n_rot=256, icp_iters=25, trim=0.8):
    """Recover T: A->B by TARGET(B)->SOURCE(A) matching — robust to partial B."""
    Bs = _stride_subsample(B_means, 1024).float()
    As = _stride_subsample(A_means, 2048).float()
    Ac, Bc = As.mean(0), Bs.mean(0)
    keep_k = max(8, int(trim * Bs.shape[0]))
    best = (float("inf"), None, None)
    for R0 in super_fibonacci_so3(n_rot, device=B_means.device, dtype=torch.float32):
        R, t = R0.clone(), Bc - R0 @ Ac
        for _ in range(icp_iters):
            TA = As @ R.transpose(-1, -2) + t
            d, idx = _nn(Bs, TA)                       # each observed B -> nearest in T.A
            thr = torch.kthvalue(d, keep_k).values
            m = d <= thr                               # keep the inlier (overlapping) B points
            R, t = _kabsch(As[idx[m]], Bs[m])          # fit matched A -> B on inliers
        d, _ = _nn(Bs, As @ R.transpose(-1, -2) + t)
        score = d.topk(keep_k, largest=False).values.mean().item()
        if score < best[0]:
            best = (score, R.clone(), t.clone())
    T = torch.eye(4, device=B_means.device, dtype=B_means.dtype)
    T[:3, :3], T[:3, 3] = best[1], best[2]
    return T


def main():
    print(f"overlap-aware (B->A) partial align on {DEV.upper()}, rot={FIXED_ROT} deg\n")
    ok = n = 0
    for keep in [0.8, 0.6, 0.4]:
        for seed in [0, 1, 2]:
            n += 1
            A = make_object_splat(1400, seed=seed, device=DEV, dtype=DT)
            Rg = axis_angle_R(ROT_AXIS, FIXED_ROT, device=DEV, dtype=DT)
            M = sim3_matrix(1.0, Rg, torch.tensor(TRANS, device=DEV, dtype=DT))
            Bp = perturb_partial(make_object_splat.apply_to(A, M), keep, seed)
            T = overlap_aware_align(Bp.means, A.means)
            err = rot_angle_deg(T[:3, :3], M[:3, :3])
            ok += err < 2.0
            print(f"  keep{int(keep*100)}% s{seed}: overlap-aware init rot_err = {err:7.3f}°")
    print(f"\noverlap-aware init success (<2 deg): {ok}/{n}   (default global_align: 0/9)")


if __name__ == "__main__":
    main()
