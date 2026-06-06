#!/usr/bin/env python
"""splatreg vs plain ICP baseline + residual ablation benchmark.

Part A — splatreg full LM vs plain ICP baseline:
  - Same synthetic recovery cells as examples/validate_recovery.py (same data gen, same metrics).
  - Method A: splatreg full registration (default residuals: ICP + SDF, global init, LM).
  - Method B: plain point-to-point ICP (standalone, centroid init — standard ICP, NOT splatreg's
    global init; that would conflate the initializer with the solver).
  - Method C: splatreg-init + plain ICP (tests whether the super-Fibonacci global init alone closes
    the gap or whether the LM multi-residual fine step contributes independently).
  - ICP CANNOT solve scale: for Sim(3) cells Method B reports scale_err% from identity scale (=1),
    which is honest — point-to-point ICP gives no scale estimate.
  - Report per regime: SE(3) rigid × {5°, 30°, 90°}; Sim(3) × {5°, 30°, 90°} × {scale 0.8, 1.3}.

Part B — residual ablation:
  - Enumerate splatreg's composable residuals (ICP, SDF).
  - Prior is NOT part of the default set and is excluded (it needs a warm-start reference pose and
    is only meaningful in tracking; ablating it here would be artificial).
  - Photometric is also NOT in the default set (requires rendered RGB / camera intrinsics).
  - Ablation variants: ICP-only, SDF-only, ICP+SDF (full default). Shows which residuals are
    load-bearing for success rate and accuracy.

Metrics (matching validate_recovery.py):
  rot_err (deg), trans_mm, scale_err%, Chamfer_mm, success (rot < 2° and |scale%| < 2%), wall-sec.

Run:
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 \\
        PYTHONPATH=/path/to/splatreg \\
        python benchmarks/icp_baseline_bench.py [--device cuda] [--seeds 3]

Notes:
  - N_POINTS, ROT_DEGS, SCALES, TRANS, SUCC_ROT_DEG, SUCC_SCALE_PCT, SEEDS, axis, MAX_ITERS all
    match validate_recovery.py exactly so results are comparable.
  - ICP baseline uses centroid alignment as init (standard ICP practice). For the "splatreg-init +
    ICP" row we run the global aligner first, then feed that init to the plain ICP solver.
  - Scale is NOT estimated by plain ICP; for Sim(3) cells we report scale_err% = 100*|1 - s_gt|/s_gt
    (honest: ICP recovered scale = 1 always).
  - Real numbers only. If a cell fails / is inconclusive, it is counted as failure (not excluded).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

# Make runnable from anywhere: add repo root + examples/ dir so sibling utils resolve.
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_BENCH_DIR)
_EXAMPLES_DIR = os.path.join(_REPO_ROOT, "examples")
for _p in (_REPO_ROOT, _EXAMPLES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from splatreg import register  # noqa: E402
from splatreg.core.types import Gaussians  # noqa: E402

from _example_utils import (  # noqa: E402
    SEEDS,
    axis_angle_R,
    chamfer_mm,
    make_object_splat,
    rot_angle_deg,
    sim3_matrix,
)

# ─── Exact match with validate_recovery.py config ─────────────────────────────
ROT_AXIS = [0.3, 0.9, 0.25]
ROT_DEGS = [5.0, 30.0, 90.0]
SCALES = [0.8, 1.0, 1.3]
TRANS = [0.03, -0.02, 0.025]
SUCC_ROT_DEG = 2.0
SUCC_SCALE_PCT = 2.0
N_POINTS = 1400
MAX_ITERS = 60  # same as validate_recovery.py default

# ─── Plain ICP parameters ─────────────────────────────────────────────────────
# Give ICP a FAIR budget: same iteration count as the LM (60 iters). Point-to-point Procrustes ICP.
ICP_ITERS = 60  # iteration budget — same as splatreg's max_iters
ICP_TRIM_KEEP = 1.0  # no trimming: use all correspondences (standard ICP; trimming would be a variant)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone plain ICP implementation (point-to-point Procrustes)
# ──────────────────────────────────────────────────────────────────────────────


def _kabsch_umeyama(src: torch.Tensor, dst: torch.Tensor, with_scale: bool = False):
    """Closed-form optimal SE(3)/Sim(3) that maps src -> dst (Kabsch / Umeyama).

    Args:
        src: (N, 3) source points.
        dst: (N, 3) corresponding target points.
        with_scale: if True, also solve for scale (Umeyama); if False (default), rigid only.

    Returns:
        (s, R, t): scale (1.0 for rigid), (3, 3) rotation, (3,) translation;
                   such that dst ≈ s * (R @ src.T).T + t.
    """
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    sc = src - src_mean
    dc = dst - dst_mean
    cov = dc.T @ sc / src.shape[0]
    U, D, Vh = torch.linalg.svd(cov)
    S = torch.eye(3, device=src.device, dtype=src.dtype)
    if torch.linalg.det(U) * torch.linalg.det(Vh) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vh
    if with_scale:
        var_src = sc.pow(2).sum() / src.shape[0]
        s = float((D * torch.diagonal(S)).sum() / var_src.clamp_min(1e-12))
    else:
        s = 1.0
    t = dst_mean - s * (R @ src_mean)
    return s, R, t


def plain_icp(
    source_pts: torch.Tensor,
    target_pts: torch.Tensor,
    *,
    max_iters: int = ICP_ITERS,
    init_T: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standalone point-to-point ICP, rigid SE(3) (no scale estimation).

    Standard init: centroid alignment when init_T is None.
    When init_T is given (for the "splatreg-init + ICP" variant), starts from that pose.

    Returns a 4x4 SE(3) matrix T such that T @ [src; 1] = [aligned; 1].
    """
    dev = source_pts.device
    dtype = source_pts.dtype
    tgt = target_pts.to(device=dev, dtype=dtype)

    if init_T is not None:
        T = init_T.to(device=dev, dtype=dtype).clone()
        R = T[:3, :3].clone()
        t = T[:3, 3].clone()
    else:
        # Standard ICP centroid init: translate source centroid to target centroid.
        src_c = source_pts.mean(0)
        tgt_c = tgt.mean(0)
        R = torch.eye(3, device=dev, dtype=dtype)
        t = tgt_c - src_c
        T = torch.eye(4, device=dev, dtype=dtype)
        T[:3, :3] = R
        T[:3, 3] = t

    src = source_pts.clone()

    for _ in range(max_iters):
        # Transform source.
        p = (R @ src.T).T + t  # (N, 3)
        # Nearest neighbour in target.
        dists = torch.cdist(p, tgt)  # (N, M)
        nn = dists.argmin(dim=1)  # (N,)
        corresp = tgt[nn]  # (N, 3)
        # Closed-form rigid (Kabsch) on correspondences.
        _, R_new, t_new = _kabsch_umeyama(p, corresp, with_scale=False)
        # Compose with current cumulative T.
        R = R_new @ R
        t = R_new @ t + t_new
        T_new = torch.eye(4, device=dev, dtype=dtype)
        T_new[:3, :3] = R
        T_new[:3, 3] = t
        # Convergence check: step norm.
        delta = (T_new - T).norm()
        T = T_new
        if delta < 1e-8:
            break

    return T


# ──────────────────────────────────────────────────────────────────────────────
# Core cell evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────


def _score_T(
    T_est: torch.Tensor,
    M_gt: torch.Tensor,
    s_gt: float,
    A: Gaussians,
    B: Gaussians,
    transform: str,
    wall_sec: float,
    scale_est: float = 1.0,
) -> dict:
    """Compute rot_err, trans_mm, scale_err%, chamfer_mm, success from a recovered T."""
    s_est = scale_est
    R_est = T_est[:3, :3] / s_est if abs(s_est - 1.0) > 1e-9 else T_est[:3, :3].clone()
    R_gt = M_gt[:3, :3] / s_gt
    rot_err = rot_angle_deg(R_est, R_gt)
    trans_err_mm = 1000.0 * float((T_est[:3, 3] - M_gt[:3, 3]).norm())
    scale_err_pct = 100.0 * abs(s_est - s_gt) / s_gt
    A_aligned = A.means @ T_est[:3, :3].transpose(-1, -2) + T_est[:3, 3]
    cham = chamfer_mm(A_aligned, B.means)
    success = rot_err < SUCC_ROT_DEG and (transform == "se3" or scale_err_pct < SUCC_SCALE_PCT)
    return dict(
        rot_err=rot_err,
        trans_err_mm=trans_err_mm,
        scale_err_pct=scale_err_pct,
        scale_est=s_est,
        cham_mm=cham,
        secs=wall_sec,
        success=success,
    )


def eval_splatreg(A: Gaussians, M_gt: torch.Tensor, s_gt: float, transform: str, device: str, dtype) -> dict:
    """splatreg full LM: global init + ICP+SDF residuals."""
    B = make_object_splat.apply_to(A, M_gt)
    t0 = time.perf_counter()
    res = register(B, A, init="global", transform=transform, max_iters=MAX_ITERS, quality="full")
    dt = time.perf_counter() - t0
    s_est = float(res.scale)
    return _score_T(res.T, M_gt, s_gt, A, B, transform, dt, scale_est=s_est)


def eval_plain_icp(
    A: Gaussians,
    M_gt: torch.Tensor,
    s_gt: float,
    transform: str,
    device: str,
    dtype,
    init_T: torch.Tensor | None = None,
    method_label: str = "ICP-centroid",
) -> dict:
    """Standalone plain ICP (point-to-point, centroid init by default).

    Scale is NOT estimated. For Sim(3) cells, scale_est = 1.0 always (honest reporting).
    """
    B = make_object_splat.apply_to(A, M_gt)
    src_pts = A.means.to(device=device, dtype=dtype)
    tgt_pts = B.means.to(device=device, dtype=dtype)
    t0 = time.perf_counter()
    T_est = plain_icp(src_pts, tgt_pts, max_iters=MAX_ITERS, init_T=init_T)
    dt = time.perf_counter() - t0
    # ICP gives no scale: scale_est = 1.0 always.
    return _score_T(T_est, M_gt, s_gt, A, B, transform, dt, scale_est=1.0)


def eval_splatreg_init_then_icp(
    A: Gaussians, M_gt: torch.Tensor, s_gt: float, transform: str, device: str, dtype
) -> dict:
    """splatreg global init -> plain ICP (tests init contribution vs fine LM contribution)."""
    from splatreg.align import global_align

    B = make_object_splat.apply_to(A, M_gt)
    t0 = time.perf_counter()
    # Step 1: coarse global init from splatreg's super-Fibonacci aligner.
    T_coarse = global_align(B, A, transform="se3")  # always SE(3) init for ICP (no scale solve)
    T_coarse = T_coarse.to(device=device, dtype=dtype)
    # Step 2: plain ICP from that init.
    src_pts = A.means.to(device=device, dtype=dtype)
    tgt_pts = B.means.to(device=device, dtype=dtype)
    T_est = plain_icp(src_pts, tgt_pts, max_iters=MAX_ITERS, init_T=T_coarse)
    dt = time.perf_counter() - t0
    return _score_T(T_est, M_gt, s_gt, A, B, transform, dt, scale_est=1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Residual ablation helpers
# ──────────────────────────────────────────────────────────────────────────────


def _auto_sdf_sigma(target: Gaussians) -> float:
    """Mirror of api.py's _auto_sdf_sigma (2x median Gaussian scale)."""
    scales = target.scales.exp() if target.log_scales else target.scales
    per_gauss = scales.mean(dim=-1)
    finite = per_gauss[torch.isfinite(per_gauss) & (per_gauss > 0.0)]
    if finite.numel() > 0:
        med = float(finite.median().item())
        if med > 0.0:
            return 2.0 * med
    extent = float((target.means.amax(0) - target.means.amin(0)).norm().item())
    return max(0.01 * extent, 1e-12)


def eval_ablation(
    A: Gaussians, M_gt: torch.Tensor, s_gt: float, transform: str, residual_names: list[str]
) -> dict:
    """splatreg LM with a specific residual subset (global init always on).

    residual_names: list of strings from {"ICP", "SDF"}.
    """
    from splatreg.residuals import ICP, SDF

    B = make_object_splat.apply_to(A, M_gt)
    sigma = _auto_sdf_sigma(B)  # B is the target in register(B, A, ...)
    residuals = []
    if "ICP" in residual_names:
        residuals.append(ICP(point_to_plane=False, weight=1.0))
    if "SDF" in residual_names:
        residuals.append(SDF(sigma=sigma, weight=0.3, n_points=0))

    if not residuals:
        raise ValueError("ablation: must specify at least one residual")

    t0 = time.perf_counter()
    res = register(
        B, A, residuals=residuals, init="global", transform=transform, max_iters=MAX_ITERS, quality="full"
    )
    dt = time.perf_counter() - t0
    s_est = float(res.scale)
    return _score_T(res.T, M_gt, s_gt, A, B, transform, dt, scale_est=s_est)


# ──────────────────────────────────────────────────────────────────────────────
# Table printing / summarizing
# ──────────────────────────────────────────────────────────────────────────────


def _med(rows, key):
    return float(np.median([r[key] for r in rows]))


def _worst(rows, key):
    return float(np.max([r[key] for r in rows]))


def _succ_rate(rows):
    return sum(r["success"] for r in rows), len(rows)


def print_cell_header():
    print(
        f"{'seed':>4} {'rot_gt':>7} {'scl_gt':>7} | "
        f"{'rot_err°':>9} {'trans_mm':>9} {'scale%':>8} {'cham_mm':>9} {'sec':>6}  ok"
    )


def print_cell(seed, rot_gt, s_gt, m):
    ok = "Y" if m["success"] else "."
    print(
        f"{seed:>4} {rot_gt:>6.1f}° {s_gt:>7.2f} | "
        f"{m['rot_err']:>9.4f} {m['trans_err_mm']:>9.3f} "
        f"{m['scale_err_pct']:>8.3f} {m['cham_mm']:>9.4f} {m['secs']:>6.2f}  {ok}"
    )


def summarize_block(rows, label, transform):
    n_ok, n = _succ_rate(rows)
    gate = f"rot<{SUCC_ROT_DEG}°" + ("" if transform == "se3" else f" & scale<{SUCC_SCALE_PCT}%")
    print(f"  [{label}] success {n_ok}/{n} = {100.0*n_ok/n:.1f}%  (gate: {gate})")
    print(f"    median rot_err  = {_med(rows,'rot_err'):.4f}°   worst = {_worst(rows,'rot_err'):.4f}°")
    print(
        f"    median trans_mm = {_med(rows,'trans_err_mm'):.3f}   worst = {_worst(rows,'trans_err_mm'):.3f}"
    )
    if transform == "sim3":
        print(
            f"    median scale%   = {_med(rows,'scale_err_pct'):.3f}   worst = {_worst(rows,'scale_err_pct'):.3f}"
        )
    print(f"    median cham_mm  = {_med(rows,'cham_mm'):.4f}   worst = {_worst(rows,'cham_mm'):.4f}")
    print(f"    median sec      = {_med(rows,'secs'):.2f}")
    return {"n_ok": n_ok, "n": n, "rows": rows, "label": label, "transform": transform}


# ──────────────────────────────────────────────────────────────────────────────
# Per-regime (rot bin) breakdown helpers
# ──────────────────────────────────────────────────────────────────────────────


def regime_breakdown(rows, label):
    """Print per-rotation-magnitude (5/30/90°) success rates for a method's rows."""
    print(f"  {label} — per-rotation-regime:")
    for rot_deg in ROT_DEGS:
        subset = [r for r in rows if abs(r.get("rot_gt", 0) - rot_deg) < 0.5]
        if not subset:
            continue
        n_ok, n = _succ_rate(subset)
        print(
            f"    rot={rot_deg:4.0f}°  success {n_ok}/{n} = {100.0*n_ok/n:.1f}%  "
            f"median_rot={_med(subset,'rot_err'):.4f}°  "
            f"median_trans={_med(subset,'trans_err_mm'):.3f}mm"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main benchmark driver
# ──────────────────────────────────────────────────────────────────────────────


def run_part_a(seeds, device, dtype, transform):
    """Run Part A for one transform mode. Returns dict of method -> summary."""
    scales = SCALES if transform == "sim3" else [1.0]
    label_suffix = transform.upper()

    method_rows: dict[str, list] = {
        "splatreg-full": [],
        "ICP-centroid": [],
        "sfib-init+ICP": [],
    }

    for seed in seeds:
        A = make_object_splat(N_POINTS, seed=seed, device=device, dtype=dtype)
        for rot_deg in ROT_DEGS:
            for s_gt in scales:
                R_gt = axis_angle_R(ROT_AXIS, rot_deg, device=device, dtype=dtype)
                t_gt = torch.tensor(TRANS, device=device, dtype=dtype)
                M_gt = sim3_matrix(s_gt, R_gt, t_gt)

                m_spl = eval_splatreg(A, M_gt, s_gt, transform, device, dtype)
                m_icp = eval_plain_icp(A, M_gt, s_gt, transform, device, dtype)
                m_si = eval_splatreg_init_then_icp(A, M_gt, s_gt, transform, device, dtype)

                for m, key in [(m_spl, "splatreg-full"), (m_icp, "ICP-centroid"), (m_si, "sfib-init+ICP")]:
                    m.update(seed=seed, rot_gt=rot_deg, scale_gt=s_gt)
                    method_rows[key].append(m)

    summaries = {}
    for method, rows in method_rows.items():
        print(f"\n{'='*90}")
        print(f"PART A [{label_suffix}] — Method: {method}")
        print_cell_header()
        for r in rows:
            print_cell(r["seed"], r["rot_gt"], r["scale_gt"], r)
        print("-" * 90)
        s = summarize_block(rows, f"{method}/{label_suffix}", transform)
        regime_breakdown(rows, method)
        summaries[method] = s
    return summaries


def run_part_b(seeds, device, dtype):
    """Part B: residual ablation on SE(3) (rigid) only — simpler, avoids scale interaction."""
    transform = "se3"
    scales = [1.0]

    ablation_variants = {
        "ICP-only": ["ICP"],
        "SDF-only": ["SDF"],
        "ICP+SDF (full default)": ["ICP", "SDF"],
    }

    all_summaries = {}
    for variant_label, res_names in ablation_variants.items():
        rows = []
        for seed in seeds:
            A = make_object_splat(N_POINTS, seed=seed, device=device, dtype=dtype)
            for rot_deg in ROT_DEGS:
                for s_gt in scales:
                    R_gt = axis_angle_R(ROT_AXIS, rot_deg, device=device, dtype=dtype)
                    t_gt = torch.tensor(TRANS, device=device, dtype=dtype)
                    M_gt = sim3_matrix(s_gt, R_gt, t_gt)
                    m = eval_ablation(A, M_gt, s_gt, transform, res_names)
                    m.update(seed=seed, rot_gt=rot_deg, scale_gt=s_gt)
                    rows.append(m)

        print(f"\n{'='*90}")
        print(f"PART B — Ablation variant: {variant_label!r}  (transform=SE(3))")
        print_cell_header()
        for r in rows:
            print_cell(r["seed"], r["rot_gt"], r["scale_gt"], r)
        print("-" * 90)
        s = summarize_block(rows, f"ablation/{variant_label}", transform)
        regime_breakdown(rows, variant_label)
        all_summaries[variant_label] = s

    return all_summaries


def print_comparison_table(a_se3, a_sim3):
    """Print a compact side-by-side comparison table for Part A results."""
    print("\n" + "=" * 100)
    print("PART A — SUMMARY COMPARISON TABLE")
    print("=" * 100)

    methods = ["splatreg-full", "ICP-centroid", "sfib-init+ICP"]
    header = (
        f"{'Method':<22} {'Mode':<6} | {'succ':>6} {'med_rot°':>9} "
        f"{'med_trans_mm':>14} {'med_scale%':>11} {'med_cham_mm':>12} {'med_sec':>8}"
    )
    print(header)
    print("-" * 100)

    for block_label, summaries in [("SE(3)", a_se3), ("Sim(3)", a_sim3)]:
        for method in methods:
            if method not in summaries:
                continue
            s = summaries[method]
            rows = s["rows"]
            transform = s["transform"]
            n_ok, n = s["n_ok"], s["n"]
            succ_str = f"{n_ok}/{n}"
            med_rot = _med(rows, "rot_err")
            med_trans = _med(rows, "trans_err_mm")
            med_scale = _med(rows, "scale_err_pct")
            med_cham = _med(rows, "cham_mm")
            med_sec = _med(rows, "secs")
            scale_note = f"{med_scale:11.3f}" if transform == "sim3" else "    n/a (fixed)"
            print(
                f"{method:<22} {block_label:<6} | {succ_str:>6} {med_rot:>9.4f} "
                f"{med_trans:>14.3f} {scale_note:>11} {med_cham:>12.4f} {med_sec:>8.2f}"
            )
    print("=" * 100)
    print("Note: ICP-centroid and sfib-init+ICP do NOT estimate scale.")
    print("      scale% for ICP rows = |1 - s_gt| / s_gt × 100 (honest: scale_est=1 always).")
    print("      For SE(3) cells s_gt=1.0 so scale% = 0 for all methods.")


def print_ablation_table(b_summaries):
    """Print compact ablation table."""
    print("\n" + "=" * 100)
    print("PART B — RESIDUAL ABLATION SUMMARY TABLE  (SE(3), global init always on)")
    print("=" * 100)
    header = (
        f"{'Residuals':<28} | {'succ':>6} {'med_rot°':>9} {'med_trans_mm':>14} "
        f"{'med_cham_mm':>12} {'med_sec':>8}"
    )
    print(header)
    print("-" * 100)
    for variant_label, s in b_summaries.items():
        rows = s["rows"]
        n_ok, n = s["n_ok"], s["n"]
        succ_str = f"{n_ok}/{n}"
        print(
            f"{variant_label:<28} | {succ_str:>6} {_med(rows,'rot_err'):>9.4f} "
            f"{_med(rows,'trans_err_mm'):>14.3f} {_med(rows,'cham_mm'):>12.4f} "
            f"{_med(rows,'secs'):>8.2f}"
        )
    print("=" * 100)
    print("All ablation variants use splatreg's global (super-Fibonacci) init + LM solver.")
    print("Photometric residual excluded: requires RGB camera frames (not applicable here).")
    print("Prior residual excluded: requires a warm-start reference pose (tracking-only context).")


def main():
    global SEEDS
    ap = argparse.ArgumentParser(description="splatreg vs ICP baseline + residual ablation.")
    ap.add_argument(
        "--device",
        default=os.environ.get("SPLATREG_DEVICE", "cpu"),
        help="cpu|cuda (default: $SPLATREG_DEVICE or cpu)",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        default=len(SEEDS),
        help=f"number of seeds to run (default: {len(SEEDS)}, max: {len(SEEDS)})",
    )
    ap.add_argument(
        "--skip-sim3", action="store_true", help="skip Sim(3) block (saves time; SE(3) + ablation only)"
    )
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU.")
        device = "cpu"
    dtype = torch.float32

    # Truncate seeds if user asked for fewer.
    seeds = SEEDS[: args.seeds]
    SEEDS = seeds

    torch.manual_seed(0)
    np.random.seed(0)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    print(f"\nsplatreg vs ICP baseline + residual ablation")
    print(
        f"Config: seeds={seeds}, N_POINTS={N_POINTS}, MAX_ITERS={MAX_ITERS}, "
        f"device={device.upper()}, dtype={dtype}"
    )
    print(f"Grid: rot={ROT_DEGS}°, scales(sim3)={SCALES}, transform: SE(3) + Sim(3)")
    print(f"Success gate: rot < {SUCC_ROT_DEG}°, |scale%| < {SUCC_SCALE_PCT}% (Sim3 only)")
    print(f"ICP init: centroid (Method B standard) OR splatreg global init (Method C)")
    print(f"ICP does NOT estimate scale: scale_est=1.0 always for ICP methods.\n")

    t_start = time.perf_counter()

    # Part A: SE(3)
    print("\n" + "#" * 100)
    print("# PART A — SE(3) RIGID BLOCK")
    print("#" * 100)
    a_se3 = run_part_a(seeds, device, dtype, "se3")

    # Part A: Sim(3)
    if not args.skip_sim3:
        print("\n" + "#" * 100)
        print("# PART A — Sim(3) BLOCK (includes scale)")
        print("#" * 100)
        a_sim3 = run_part_a(seeds, device, dtype, "sim3")
    else:
        a_sim3 = {}
        print("\n[Sim(3) block skipped by --skip-sim3]")

    # Part B: residual ablation (SE(3) only)
    print("\n" + "#" * 100)
    print("# PART B — RESIDUAL ABLATION (SE(3))")
    print("#" * 100)
    b_summaries = run_part_b(seeds, device, dtype)

    # Summary tables
    print_comparison_table(a_se3, a_sim3)
    print_ablation_table(b_summaries)

    wall = time.perf_counter() - t_start
    print(f"\nTotal wall time: {wall:.1f}s")
    if device.startswith("cuda"):
        peak = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        print(f"Peak GPU memory: allocated {peak:.2f} GiB, reserved {reserved:.2f} GiB")

    print("Done.")


if __name__ == "__main__":
    main()
