#!/usr/bin/env python
"""REAL-DATA Sim(3) / SE(3) recovery benchmark for splatreg.

This is the *external anchor* for splatreg: the synthetic harness
(``examples/validate_recovery.py``) proves the solver on a clean, hand-built ellipsoid;
this benchmark proves it on **real, noisy, non-synthetic 3D-Gaussian-Splatting geometry** —
``.ply`` files exported by the GaussianFeels VT-3DGS SLAM pipeline (~12k of them on disk).

Why this is a credible test and not circular
---------------------------------------------
There is no readily-available public splat-registration GT-pose dataset. So we use the same
KNOWN-transform protocol the synthetic harness uses,
but feed it REAL splat geometry and (optionally) corrupt the moved copy to mimic a *second* real
capture of the same object:

  1. Load a real exported splat ``A`` (real anchor positions, real anisotropic scales).
  2. Apply a KNOWN Sim(3)/SE(3) ``M_gt`` to A -> ``B`` (this is the ground truth).
  3. Corrupt ``B`` to look like an independent capture: random SUBSAMPLE (partial density) +
     additive Gaussian position NOISE proportional to the median Gaussian footprint. (Default
     on; ``--clean`` disables it to isolate the geometry effect.)
  4. Recover with ``register(B, A, init="global", transform=..., residuals=None)`` — the SAME
     default ICP-dominant + auto-SDF residual set and coarse global init as the synthetic harness.
  5. Score: rotation error (deg), translation error (mm), scale error (%), post-alignment
     Chamfer (mm, A-pushed-through-recovered-T vs the *clean* B so noise doesn't flatter it),
     and wall-time. A cell SUCCEEDS when rot < gate and (Sim3) |scale err| < gate.

The honest question this answers: *does real splat geometry — noisy, partial, with real anchor
clutter and near-symmetric objects (dice, cubes) — degrade recovery versus the clean synthetic
ellipsoid?* It is run across several real SIM and REAL objects so the gap is measured, not assumed.

This file ONLY adds a benchmark; it imports splatreg unchanged (no source edits).

Run (GPU1 only — GPU0 is busy):

    CUDA_VISIBLE_DEVICES=1 SPLATREG_DEVICE=cuda \\
        PYTHONPATH=. \\
        python benchmarks/realdata_bench.py

Useful flags: ``--clean`` (no noise/subsample), ``--max-anchors N`` (cap loaded splat size for
speed/memory), ``--objects sim_pear,real_peach`` (subset), ``--noise-mult``, ``--keep-frac``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

# Make runnable as `python benchmarks/realdata_bench.py` and with PYTHONPATH set: add the repo
# root (parent of benchmarks/) so `import splatreg` resolves when not pip-installed, plus examples/
# so we can borrow the same SO(3)/Chamfer/transform helpers the synthetic harness uses (no dup).
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_BENCH_DIR)
_EXAMPLES_DIR = os.path.join(_REPO_ROOT, "examples")
for _p in (_REPO_ROOT, _EXAMPLES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from splatreg import register  # noqa: E402
from splatreg.core.types import Gaussians  # noqa: E402
from splatreg.io import load_ply  # noqa: E402

from _example_utils import (  # noqa: E402  (reuse the synthetic harness' vetted helpers)
    axis_angle_R,
    chamfer_mm,
    rot_angle_deg,
    sim3_matrix,
)

# ------------------------------------------------------------------ configuration
DEVICE = os.environ.get("SPLATREG_DEVICE", "cpu")
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    print("SPLATREG_DEVICE=cuda requested but CUDA is unavailable; falling back to CPU.")
    DEVICE = "cpu"
DTYPE = torch.float32

MAX_ITERS = 60
QUALITY: object = "full"

# Same known-offset grid as the synthetic harness (so the two are directly comparable):
# rotation about a fixed oblique axis at small/medium/large, three scales, fixed translation.
ROT_AXIS = [0.3, 0.9, 0.25]
ROT_DEGS = [5.0, 30.0, 90.0]
SCALES = [0.8, 1.0, 1.3]
TRANS = [0.03, -0.02, 0.025]  # object units (~30 mm), applied AFTER s*R

# Success gates (match the synthetic harness: field-limited 2 deg / 2% scale).
SUCC_ROT_DEG = 2.0
SUCC_SCALE_PCT = 2.0

# Default "looks like a second real capture" corruption.
KEEP_FRAC = 0.6  # subsample B to 60% of A's anchors (partial density / different sampling)
NOISE_MULT = 1.0  # additive position noise std = NOISE_MULT * median Gaussian footprint

# Real exported splats: one per distinct GaussianFeels object (SIM + REAL), confirmed to load.
REAL_SPLATS = {
    "sim_potted_meat_can": "outputs/_slam_fps_p_cap_600f/final.ply",
    "sim_pear": "outputs/_fix_a_baseline_016_pear/final.ply",
    "sim_rubiks_cube": "outputs/_fix_b_lazy_baseline_077_rubiks_cube_v2/final.ply",
    "sim_large_dice": "outputs/_opt_ctrl_dice_wtac_30f_smax04/final.ply",
    "real_bell_pepper": "outputs/_fix_d_bmax60k_slam/final.ply",
    "real_large_dice": "outputs/_real_verify_dice_pose_wtac/final.ply",
    "real_peach": "outputs/_fps_real_op2_peach_pose_wtac/final.ply",
    "real_rubiks_small": "outputs/_real_verify_rubiks_pose_wtac/final.ply",
}


# ------------------------------------------------------------------ helpers
def _median_footprint(g: Gaussians) -> float:
    """Median per-Gaussian linear scale (the anchor footprint), in splat units (m)."""
    sc = g.scales.exp() if g.log_scales else g.scales
    per = sc.mean(dim=-1)
    per = per[torch.isfinite(per) & (per > 0)]
    return float(per.median()) if per.numel() else 1e-3


def _apply_sim3(g: Gaussians, M: torch.Tensor) -> Gaussians:
    """Bake a 4x4 Sim(3)/SE(3) ``M=[[s*R,t],[0,1]]`` into a real splat (means + scales).

    Mirrors ``examples/_example_utils._apply_to``: means map ``s*(R@x)+t``; linear scales x s
    (so a Sim(3) GT genuinely rescales the real footprint). Quats/opacity/colour carry through.
    """
    M = M.to(device=g.means.device, dtype=g.means.dtype)
    block = M[:3, :3]
    t = M[:3, 3]
    s = float(torch.linalg.det(block).abs().clamp_min(1e-18) ** (1.0 / 3.0))
    means = g.means @ block.transpose(-1, -2) + t
    lin = g.scales.exp() if g.log_scales else g.scales
    lin = lin * s
    scales = lin.log() if g.log_scales else lin
    return Gaussians(
        means=means,
        quats=g.quats.clone(),
        scales=scales,
        opacities=g.opacities.clone(),
        colors=None if g.colors is None else g.colors.clone(),
        log_scales=g.log_scales,
    )


def _corrupt(g: Gaussians, keep_frac: float, noise_std: float, gen: torch.Generator) -> Gaussians:
    """Mimic an independent real capture: random subsample + additive position noise.

    Subsampling gives a different anchor density/sampling than A (partial overlap of *which*
    anchors); the Gaussian position noise simulates depth/track jitter. Scales/quats/colour of the
    kept anchors are untouched. Returns a new Gaussians on the same device.
    """
    n = len(g)
    k = max(16, int(round(keep_frac * n)))
    idx = torch.randperm(n, generator=gen, device="cpu")[:k].to(g.means.device)
    means = g.means[idx].clone()
    if noise_std > 0:
        means = means + noise_std * torch.randn(means.shape, generator=gen, device="cpu").to(means)
    return Gaussians(
        means=means,
        quats=g.quats[idx].clone(),
        scales=g.scales[idx].clone(),
        opacities=g.opacities[idx].clone(),
        colors=None if g.colors is None else g.colors[idx].clone(),
        log_scales=g.log_scales,
    )


def _maybe_cap(g: Gaussians, max_anchors: int, gen: torch.Generator) -> Gaussians:
    """Cap a splat to ``max_anchors`` (random, deterministic) for speed/memory; identity if small."""
    if max_anchors <= 0 or len(g) <= max_anchors:
        return g
    idx = torch.randperm(len(g), generator=gen, device="cpu")[:max_anchors].to(g.means.device)
    return Gaussians(
        means=g.means[idx].clone(),
        quats=g.quats[idx].clone(),
        scales=g.scales[idx].clone(),
        opacities=g.opacities[idx].clone(),
        colors=None if g.colors is None else g.colors[idx].clone(),
        log_scales=g.log_scales,
    )


def recover_once(A: Gaussians, B_clean: Gaussians, M_gt, s_gt, transform, corrupt, args, seed):
    """Apply M_gt -> (optionally) corrupt -> register(B, A) -> metrics dict.

    ``B_clean`` is the un-corrupted transformed splat used for an honest Chamfer reference (so
    added noise does not artificially inflate or deflate the post-alignment Chamfer).
    """
    gen = torch.Generator(device="cpu").manual_seed(1000 + seed)
    B = B_clean
    if corrupt:
        noise_std = args.noise_mult * _median_footprint(A)
        B = _corrupt(B_clean, args.keep_frac, noise_std, gen)

    t0 = time.perf_counter()
    res = register(B, A, init="global", transform=transform, max_iters=MAX_ITERS, quality=QUALITY)
    dt = time.perf_counter() - t0

    s_est = res.scale
    R_est = res.T[:3, :3] / s_est
    R_gt = M_gt[:3, :3] / s_gt
    rot_err = rot_angle_deg(R_est, R_gt)
    trans_err_mm = 1000.0 * float((res.T[:3, 3] - M_gt[:3, 3]).norm())
    scale_err_pct = 100.0 * abs(s_est - s_gt) / s_gt
    A_aligned = A.means @ res.T[:3, :3].transpose(-1, -2) + res.T[:3, 3]
    cham_mm = chamfer_mm(A_aligned, B_clean.means)

    success = rot_err < SUCC_ROT_DEG and (transform == "se3" or scale_err_pct < SUCC_SCALE_PCT)
    return {
        "rot_err": rot_err,
        "trans_err_mm": trans_err_mm,
        "scale_err_pct": scale_err_pct,
        "scale_est": s_est,
        "cham_mm": cham_mm,
        "iters": res.info.get("n_iters", -1),
        "rmse": res.info.get("rmse", float("nan")),
        "secs": dt,
        "success": success,
    }


def run_object(name, path, transform, scales, corrupt, args):
    """Run the full rot x scale grid for one real object/transform; print a table, return rows."""
    cap_gen = torch.Generator(device="cpu").manual_seed(7)
    A = load_ply(path, device=DEVICE, dtype=DTYPE)
    n_full = len(A)
    A = _maybe_cap(A, args.max_anchors, cap_gen)
    foot_mm = 1000.0 * _median_footprint(A)

    print("-" * 108)
    print(
        f"OBJECT {name!r}  ({transform}, N={len(A)}/{n_full} anchors, footprint~{foot_mm:.2f}mm, "
        f"corrupt={'noise+subsample' if corrupt else 'clean'})"
    )
    header = (
        f"{'rot_gt':>7} {'scl_gt':>7} {'|':>1} {'rot_err°':>9} {'trans_mm':>9} "
        f"{'scale_err%':>11} {'cham_mm':>9} {'iters':>6} {'sec':>7}  ok"
    )
    print(header)

    rows = []
    for rot_deg in ROT_DEGS:
        for s_gt in scales:
            R_gt = axis_angle_R(ROT_AXIS, rot_deg, device=DEVICE, dtype=DTYPE)
            t_gt = torch.tensor(TRANS, device=DEVICE, dtype=DTYPE)
            M_gt = sim3_matrix(s_gt, R_gt, t_gt)
            B_clean = _apply_sim3(A, M_gt)
            m = recover_once(A, B_clean, M_gt, s_gt, transform, corrupt, args, seed=len(rows))
            m.update(object=name, rot_gt=rot_deg, scale_gt=s_gt, transform=transform)
            rows.append(m)
            ok = "Y" if m["success"] else "."
            print(
                f"{rot_deg:>6.1f}° {s_gt:>7.2f} {'|':>1} {m['rot_err']:>9.4f} "
                f"{m['trans_err_mm']:>9.3f} {m['scale_err_pct']:>11.3f} {m['cham_mm']:>9.4f} "
                f"{m['iters']:>6d} {m['secs']:>7.2f}  {ok}"
            )
    del A
    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def summarize(rows, transform, label):
    """Aggregate success-rate + medians + worst-case for a set of rows."""
    n = len(rows)
    if n == 0:
        print(f"  {label}: no rows.")
        return {"n": 0, "n_ok": 0}
    n_ok = sum(r["success"] for r in rows)

    def med(k):
        return float(np.median([r[k] for r in rows]))

    def worst(k):
        return float(np.max([r[k] for r in rows]))

    gate = f"rot<{SUCC_ROT_DEG}°" if transform == "se3" else f"rot<{SUCC_ROT_DEG}° & scale<{SUCC_SCALE_PCT}%"
    print(f"  {label} SUMMARY: success {n_ok}/{n} = {100.0 * n_ok / n:.1f}%   (gate: {gate})")
    print(f"    median rot_err  = {med('rot_err'):.4f}°    worst = {worst('rot_err'):.4f}°")
    print(f"    median trans    = {med('trans_err_mm'):.3f} mm   worst = {worst('trans_err_mm'):.3f} mm")
    if transform == "sim3":
        print(f"    median scale_err= {med('scale_err_pct'):.3f}%    worst = {worst('scale_err_pct'):.3f}%")
    print(f"    median Chamfer  = {med('cham_mm'):.4f} mm   worst = {worst('cham_mm'):.4f} mm")
    print(f"    median iters    = {med('iters'):.0f}      median sec = {med('secs'):.2f}")
    return {"n": n, "n_ok": n_ok, "rows": rows, "transform": transform, "label": label}


def main():
    global DEVICE, QUALITY, MAX_ITERS
    ap = argparse.ArgumentParser(description="splatreg REAL-DATA Sim(3)/SE(3) recovery benchmark.")
    ap.add_argument("--quality", default="full", help="full|balanced|low|auto|<0..1> (default full)")
    ap.add_argument("--device", default=DEVICE, help="cpu|cuda (default $SPLATREG_DEVICE or cpu)")
    ap.add_argument("--iters", type=int, default=MAX_ITERS, help=f"LM iters (default {MAX_ITERS})")
    ap.add_argument(
        "--max-anchors",
        type=int,
        default=20000,
        help="cap loaded splat size for speed/memory (random subsample); 0 = use all (default 20000)",
    )
    ap.add_argument("--clean", action="store_true", help="disable noise+subsample (isolate geometry)")
    ap.add_argument(
        "--keep-frac", type=float, default=KEEP_FRAC, help=f"B subsample fraction (default {KEEP_FRAC})"
    )
    ap.add_argument(
        "--noise-mult",
        type=float,
        default=NOISE_MULT,
        help=f"pos-noise std / footprint (default {NOISE_MULT})",
    )
    ap.add_argument("--objects", default="", help="comma list to subset (default: all real splats)")
    ap.add_argument("--se3", action="store_true", help="also run the SE(3) rigid block")
    args = ap.parse_args()

    DEVICE = args.device
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        print("--device cuda requested but CUDA is unavailable; falling back to CPU.")
        DEVICE = "cpu"
    try:
        QUALITY = float(args.quality)
    except ValueError:
        QUALITY = args.quality
    MAX_ITERS = args.iters

    objects = REAL_SPLATS
    if args.objects:
        wanted = [o.strip() for o in args.objects.split(",") if o.strip()]
        objects = {k: REAL_SPLATS[k] for k in wanted if k in REAL_SPLATS}
        missing = [k for k in wanted if k not in REAL_SPLATS]
        if missing:
            print(f"WARNING: unknown objects ignored: {missing}")
    # Drop any whose file is missing, honestly reporting it.
    present = {k: v for k, v in objects.items() if os.path.exists(v)}
    absent = [k for k in objects if k not in present]
    if absent:
        print(f"WARNING: real splat files missing on disk (skipped): {absent}")
    objects = present
    if not objects:
        print("ERROR: no real splat files available. Cannot run the real-data benchmark.")
        sys.exit(2)

    torch.manual_seed(0)
    np.random.seed(0)
    if DEVICE.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    corrupt = not args.clean

    print(
        f"\nsplatreg REAL-DATA recovery benchmark — {len(objects)} real GaussianFeels splats, "
        f"{len(ROT_DEGS)}x{len(SCALES)} grid each, on {DEVICE.upper()}, quality={QUALITY!r}.\n"
        f"Corruption: {'subsample keep=%.2f + noise=%.1fxfootprint' % (args.keep_frac, args.noise_mult) if corrupt else 'NONE (clean)'}; "
        f"max_anchors={args.max_anchors}.\n"
    )

    # SIM(3) block over all objects.
    print("=" * 108)
    print("SIM(3) RECOVERY (rotation + translation + SCALE) on REAL splats")
    sim3_rows = []
    for name, path in objects.items():
        sim3_rows += run_object(name, path, "sim3", SCALES, corrupt, args)
    print("-" * 108)
    sim3 = summarize(sim3_rows, "sim3", "SIM(3) ALL-OBJECTS")
    # Per-object breakdown.
    print("  per-object success (sim3):")
    for name in objects:
        r = [x for x in sim3_rows if x["object"] == name]
        ok = sum(x["success"] for x in r)
        mr = float(np.median([x["rot_err"] for x in r]))
        ms = float(np.median([x["scale_err_pct"] for x in r]))
        mc = float(np.median([x["cham_mm"] for x in r]))
        print(
            f"     {name:22s} {ok}/{len(r)}  med rot {mr:7.3f}°  med scale {ms:6.3f}%  med cham {mc:8.3f}mm"
        )

    se3 = {"n": 0, "n_ok": 0}
    if args.se3:
        print("=" * 108)
        print("SE(3) RECOVERY (rigid: rotation + translation, scale==1) on REAL splats")
        se3_rows = []
        for name, path in objects.items():
            se3_rows += run_object(name, path, "se3", [1.0], corrupt, args)
        print("-" * 108)
        se3 = summarize(se3_rows, "se3", "SE(3) ALL-OBJECTS")

    print("=" * 108)
    total = sim3["n"] + se3["n"]
    total_ok = sim3["n_ok"] + se3["n_ok"]
    print(
        f"OVERALL REAL-DATA: {total_ok}/{total} cells within gate "
        f"({100.0 * total_ok / total:.1f}%)   wall {time.perf_counter() - t_start:.1f}s"
    )
    if DEVICE.startswith("cuda"):
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"PEAK GPU: {peak:.2f} GiB allocated")
    print("=" * 108)


if __name__ == "__main__":
    main()
