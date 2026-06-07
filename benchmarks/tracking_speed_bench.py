#!/usr/bin/env python
"""Warm-start tracking speed + accuracy benchmark for :func:`splatreg.track`.

Simulates a realistic TRACKING SEQUENCE: a fixed reference object splat (the target) and a moving
source splat that drifts by a small incremental SE(3) delta each frame (~2 deg + 5 mm/frame). At
every frame the source is localised by warm-starting :func:`splatreg.track` at the *previous*
frame's estimate — exactly how a tracker runs (NO per-frame global init). This is the regime where
sub-frame (<40 ms) tracking is reachable; ``register``'s blind global-init sweep is skipped.

Reports, GPU-warm:
  * per-frame milliseconds (median / p90 / mean over the sequence, after warm-up frames),
  * tracking accuracy: rotation error (deg) and translation error (mm) of the estimated pose vs the
    KNOWN per-frame ground-truth pose. The tracker must actually follow the object, not just be fast.

Target: median < 40 ms/frame with rot_err < 0.5 deg (the GaussianFeels tracker it descends from
runs ~45 ms/frame).

Run:
    CUDA_VISIBLE_DEVICES=1 SPLATREG_DEVICE=cuda \\
        python benchmarks/tracking_speed_bench.py [--frames 30] [--n-gauss 6000] [--transform se3]
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time

import torch

# Make the repo importable when run directly from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.path.join(_REPO, "examples")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _example_utils import axis_angle_R, make_object_splat, rot_angle_deg  # noqa: E402
from splatreg.track import make_track_residuals, track  # noqa: E402


def _se3(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    T = torch.eye(4, device=R.device, dtype=R.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=os.environ.get("SPLATREG_DEVICE", "cpu"))
    ap.add_argument("--frames", type=int, default=30, help="tracking-sequence length")
    ap.add_argument("--warmup", type=int, default=5, help="frames dropped from the timing stats")
    ap.add_argument("--n-gauss", type=int, default=6000, help="target/source Gaussian count")
    ap.add_argument("--n-points", type=int, default=300, help="source anchors sampled per track()")
    ap.add_argument("--iters", type=int, default=4, help="LM iters per frame")
    ap.add_argument("--trunc-sigmas", type=float, default=3.0)
    ap.add_argument("--knn", type=int, default=16, help="truncated top-k anchors per query")
    ap.add_argument("--transform", default="se3", choices=["se3", "sim3"])
    ap.add_argument("--deg-per-frame", type=float, default=2.0)
    ap.add_argument("--mm-per-frame", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    dtype = torch.float32
    torch.manual_seed(args.seed)

    # Fixed reference object splat (the target the tracker localises against).
    target = make_object_splat(args.n_gauss, seed=args.seed, device=device, dtype=dtype)
    # The moving body: the SAME object, which we will displace by the GT pose each frame. The
    # tracker estimates the pose mapping this body's frame into the target frame.
    body = make_object_splat(args.n_gauss, seed=args.seed, device=device, dtype=dtype)

    # Per-frame incremental delta: deg_per_frame about a tilted axis + mm_per_frame translation.
    dR = axis_angle_R((0.3, 1.0, 0.5), args.deg_per_frame, device=device, dtype=dtype)
    dt = (args.mm_per_frame / 1000.0) * torch.tensor([1.0, -0.4, 0.7], device=device, dtype=dtype)
    dt = dt / dt.norm() * (args.mm_per_frame / 1000.0)
    delta = _se3(dR, dt)

    # The tracker aligns `source` onto `target`. `source` is the body transformed by the INVERSE of
    # the cumulative GT pose, so that the recovered T == cumulative GT pose maps source->target. We
    # build source per frame by applying the cumulative pose to the body's means.
    T_gt = torch.eye(4, device=device, dtype=dtype)
    prior_T = torch.eye(4, device=device, dtype=dtype)  # warm-start: identity at frame 0

    # Build the tracking residual stack ONCE (the realistic tracker pattern): the target is fixed,
    # so its per-anchor normals are constant and cached on the residual — recomputing them every
    # frame (a full cdist + SVD over the target) would dominate a warm-started track.
    residuals = make_track_residuals(
        target, n_points=args.n_points, knn=args.knn, trunc_sigmas=args.trunc_sigmas
    )

    times_ms = []
    rot_errs = []
    trans_errs_mm = []

    for f in range(args.frames):
        T_gt = T_gt @ delta  # cumulative GT pose this frame
        # Source = body placed at the INVERSE GT pose, so aligning it back onto the target recovers
        # T_gt. apply_to bakes a 4x4 into the splat geometry (same convention as the GT harness).
        T_inv = torch.linalg.inv(T_gt)
        source = make_object_splat.apply_to(body, T_inv)

        _sync(device)
        t0 = time.perf_counter()
        res = track(
            target,
            source,
            prior_T,
            transform=args.transform,
            iters=args.iters,
            residuals=residuals,
        )
        _sync(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        T_est = res.T
        prior_T = T_est.detach().clone()  # warm-start the next frame

        rot_err = rot_angle_deg(T_est[:3, :3], T_gt[:3, :3])
        trans_err_mm = 1000.0 * float((T_est[:3, 3] - T_gt[:3, 3]).norm().item())

        if f >= args.warmup:
            times_ms.append(dt_ms)
            rot_errs.append(rot_err)
            trans_errs_mm.append(trans_err_mm)

        print(
            f"frame {f:3d}  {dt_ms:7.2f} ms   rot_err {rot_err:6.3f} deg   "
            f"trans_err {trans_err_mm:7.3f} mm"
        )

    def _stats(x):
        return (statistics.median(x), statistics.mean(x), sorted(x)[int(0.9 * (len(x) - 1))], max(x))

    med_ms, mean_ms, p90_ms, max_ms = _stats(times_ms)
    med_rot, mean_rot, _, max_rot = _stats(rot_errs)
    med_tr, mean_tr, _, max_tr = _stats(trans_errs_mm)

    print("\n" + "=" * 72)
    print(
        f"splatreg.track  transform={args.transform}  device={device}  "
        f"n_gauss={args.n_gauss}  n_points={args.n_points}  iters={args.iters}  "
        f"trunc_sigmas={args.trunc_sigmas}  knn={args.knn}"
    )
    print(f"timed frames (warmup={args.warmup} dropped): {len(times_ms)}")
    print("-" * 72)
    print(
        f"PER-FRAME ms     median {med_ms:7.2f}   mean {mean_ms:7.2f}   "
        f"p90 {p90_ms:7.2f}   max {max_ms:7.2f}"
    )
    print(f"ROT err  (deg)   median {med_rot:7.3f}   mean {mean_rot:7.3f}   max {max_rot:7.3f}")
    print(f"TRANS err (mm)   median {med_tr:7.3f}   mean {mean_tr:7.3f}   max {max_tr:7.3f}")
    print("-" * 72)
    goal_ms, goal_rot = 40.0, 0.5
    ok = med_ms < goal_ms and med_rot < goal_rot
    print(
        f"GOAL median < {goal_ms:.0f} ms  AND  median rot_err < {goal_rot} deg  ->  "
        f"{'PASS' if ok else 'MISS'}  (median {med_ms:.2f} ms, {med_rot:.3f} deg)"
    )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
