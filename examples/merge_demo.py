#!/usr/bin/env python
"""MERGE DEMO — splatreg's headline: fuse two overlapping captures into ONE clean ``.ply``.

This is splatreg's headline demo: the thing SuperSplat / graphdeco-#990 / Cesium
users ask for and no tool provides — *merge two overlapping splats, aligned, into a single
deduped ``.ply``* — quantified against the status-quo baseline (naive ``torch.cat``).

What it does
------------
1. Take a realistic object splat ``full`` (synthetic by default; a real exported 3DGS ``.ply``
   if you point ``--ply`` at one). Split it by ``x`` into a LEFT half and a RIGHT half that
   SHARE a central overlap band — so the band is double-covered, exactly like two real captures
   that see the same surface from different viewpoints.
2. Apply a KNOWN relative Sim(3) (small rotation + translation + a scale != 1) to the RIGHT half
   — this is the inter-capture misalignment a real second scan would have. Give the two halves
   distinct opacities so the dedupe's "highest-opacity survivor wins" is visible.
3. Build the NAIVE baseline: ``torch.cat`` the two halves with NO registration — what SuperSplat
   users do today. It is both misaligned (the right half is off by the known Sim(3)) AND
   double-dense in the overlap band.
4. ``splatreg.merge([left, right_moved])`` — registers the right half back onto the left
   (``transform="sim3"``, ``init="global"``) and voxel-dedupes the overlap to single density.
5. Report the seam improvement:
     * overlap-band Gaussian COUNT: naive-cat (double) vs splatreg (deduped -> ~single);
     * Chamfer-to-ORIGINAL (mm): how close each merged cloud sits to the untouched ``full`` splat
       — the naive cat is far (right half misaligned), splatreg is tight (registered);
   then ``save_ply`` the merged result and round-trip it through ``load_ply`` as a sanity check.

A real-capture demo needs nothing more than YOUR two overlapping ``.ply`` files:

    merged = splatreg.merge([splatreg.io.load_ply("scanA.ply"),
                             splatreg.io.load_ply("scanB.ply")])
    splatreg.io.save_ply(merged, "merged.ply")

— the synthetic split here just gives a *known* ground truth (the original ``full``) so the
improvement is measurable. Pass ``--ply /path/to/export.ply`` to run the same split-and-merge on
a real exported splat instead of the synthetic object.

Run (CPU — the DiT owns the GPUs):

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 \
        PYTHONPATH=/path/to/splatreg python examples/merge_demo.py
    # or on a real export:  python examples/merge_demo.py --ply outputs/.../final.ply
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# Make this runnable BOTH ways: `python examples/merge_demo.py` from anywhere, and
# `PYTHONPATH=.../splatreg python ...`. Add the repo ROOT (parent of examples/) so `import splatreg`
# resolves even when splatreg is not pip-installed, AND the examples/ dir for `_example_utils`.
_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EXAMPLES_DIR)
for _p in (_REPO_ROOT, _EXAMPLES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import splatreg  # noqa: E402
from splatreg.core.types import Gaussians  # noqa: E402
from splatreg.io import load_ply, save_ply  # noqa: E402

from _example_utils import (  # noqa: E402
    axis_angle_R,
    chamfer_mm,
    make_object_splat,
    sim3_matrix,
)

# Device: defaults to CPU; set ``SPLATREG_DEVICE=cuda`` to run the merge on GPU (the Sim(3) autodiff
# Jacobian is row-chunked and the default SDF point sample is capped, so the GPU path is bounded).
DEVICE = os.environ.get("SPLATREG_DEVICE", "cpu")
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    print("SPLATREG_DEVICE=cuda requested but CUDA is unavailable; falling back to CPU.")
    DEVICE = "cpu"
DTYPE = torch.float32
# Quality / machine-adaptivity policy passed to `merge` (CLI `--quality`, default "full").
QUALITY: object = "full"
# Dedupe method passed to `merge` (CLI `--dedupe`): "voxel" (default) or "knn" (cross-splat,
# removes the ~16% residual overlap a voxel grid leaves at cell boundaries).
DEDUPE_METHOD = "voxel"


# --------------------------------------------------------------------- splat helpers
def _index(g: Gaussians, mask: torch.Tensor) -> Gaussians:
    """Row-subset a splat by a boolean mask, carrying every field."""
    opac = g.opacities.reshape(-1)[mask]
    return Gaussians(
        means=g.means[mask],
        quats=g.quats[mask],
        scales=g.scales[mask],
        opacities=opac,
        colors=None if g.colors is None else g.colors[mask],
        log_scales=g.log_scales,
    )


def _set_opacity(g: Gaussians, value: float) -> Gaussians:
    """Return a copy with every opacity set to a constant (so the dedupe winner is observable)."""
    n = len(g)
    return Gaussians(
        means=g.means,
        quats=g.quats,
        scales=g.scales,
        opacities=torch.full((n,), float(value), device=g.means.device, dtype=g.means.dtype),
        colors=g.colors,
        log_scales=g.log_scales,
    )


def naive_cat(a: Gaussians, b: Gaussians) -> Gaussians:
    """The SuperSplat status quo: concatenate two splats with NO registration, NO dedupe."""
    return Gaussians(
        means=torch.cat([a.means, b.means], 0),
        quats=torch.cat([a.quats, b.quats], 0),
        scales=torch.cat([a.scales, b.scales], 0),
        opacities=torch.cat([a.opacities.reshape(-1), b.opacities.reshape(-1)], 0),
        colors=(None if a.colors is None or b.colors is None else torch.cat([a.colors, b.colors], 0)),
        log_scales=a.log_scales,
    )


def band_count(means: torch.Tensor, lo: float, hi: float) -> int:
    """Number of Gaussian centres whose x lies in the overlap band ``[lo, hi]``."""
    return int(((means[:, 0] >= lo) & (means[:, 0] <= hi)).sum().item())


# --------------------------------------------------------------------- the demo
def run_demo(full: Gaussians, source_label: str) -> None:
    print("=" * 96)
    print(f"MERGE DEMO  —  source: {source_label}")
    print(
        f"  full splat: {len(full)} Gaussians, log_scales={full.log_scales}, "
        f"bbox diag {float((full.means.amax(0) - full.means.amin(0)).norm()):.4f} m, "
        f"device={DEVICE}"
    )

    # --- split into two overlapping halves (share a central band in x) ---------------
    x = full.means[:, 0]
    xmid = float(x.median())
    extent_x = float(x.max() - x.min())
    band = 0.08 * extent_x  # overlap band half-width ~ 8% of the x-extent
    lo, hi = xmid - band, xmid + band
    left = _index(full, x <= xmid + band)  # left half  + band
    right = _index(full, x >= xmid - band)  # right half + band
    # Distinct opacities so the "highest-opacity wins" dedupe survivor is observable in the band.
    left = _set_opacity(left, 0.90)
    right = _set_opacity(right, 0.40)
    print(f"  split at x_med={xmid:.4f}; overlap band x in [{lo:.4f}, {hi:.4f}] " f"(width {2 * band:.4f} m)")
    print(f"  left half {len(left)} (opacity 0.90) | right half {len(right)} (opacity 0.40)")

    # --- apply a KNOWN relative Sim(3) to the right half (the inter-capture offset) ---
    R_off = axis_angle_R([0.1, 0.25, 1.0], 6.0, device=DEVICE, dtype=DTYPE)
    t_off = torch.tensor([0.02, 0.012, -0.015], device=DEVICE, dtype=DTYPE)
    s_off = 1.10
    M_off = sim3_matrix(s_off, R_off, t_off)
    right_moved = make_object_splat.apply_to(right, M_off)
    print(f"  applied known offset to RIGHT half: rot 6deg, trans ~28 mm, scale {s_off:.2f}")

    # --- baseline: naive cat (no registration, no dedupe) ----------------------------
    cat = naive_cat(left, right_moved)
    # The band counts (using the ALIGNED, un-moved halves) define the double-density reference and
    # the single-density target; the naive cat leaves the right band misplaced AND duplicated.
    band_left = band_count(left.means, lo, hi)
    band_right = band_count(right.means, lo, hi)
    double_ref = band_left + band_right
    single_ref = max(band_left, band_right)

    # --- splatreg merge: register right->left (sim3, global init) + dedupe ------------
    t0 = time.perf_counter()
    merged = splatreg.merge(
        [left, right_moved],
        ref=0,
        transform="sim3",
        init="global",
        max_iters=40,
        quality=QUALITY,
        dedupe_method=DEDUPE_METHOD,
    )
    dt = time.perf_counter() - t0
    # No-dedupe variant: registered concatenation only (isolates dedupe vs registration effect).
    merged_nodedupe = splatreg.merge(
        [left, right_moved],
        ref=0,
        transform="sim3",
        init="global",
        dedupe=False,
        max_iters=40,
        quality=QUALITY,
    )

    band_cat = band_count(cat.means, lo, hi)
    band_merged = band_count(merged.means, lo, hi)
    # Chamfer of each merged cloud back to the ORIGINAL untouched full splat (lower = truer merge).
    cham_cat = chamfer_mm(cat.means, full.means)
    cham_merged = chamfer_mm(merged.means, full.means)
    cham_nodedupe = chamfer_mm(merged_nodedupe.means, full.means)

    # --- dedupe-winner check: who survives in the band? (left=0.90 should dominate) ---
    bmask = (merged.means[:, 0] >= lo) & (merged.means[:, 0] <= hi)
    band_opac = merged.opacities.reshape(-1)[bmask]
    frac_high = float((band_opac > 0.6).float().mean()) if band_opac.numel() else float("nan")

    print("-" * 96)
    print(f"  merge ran in {dt:.2f}s  ({len(merged)} Gaussians out)")
    print("  OVERLAP-BAND DENSITY (Gaussians in the band):")
    print(f"    aligned single-half reference (target) : ~{single_ref}")
    print(f"    aligned both-halves (double) reference : ~{double_ref}")
    print(f"    naive torch.cat (baseline)             :  {band_cat}")
    print(
        f"    splatreg merge (deduped)               :  {band_merged}    "
        f"({band_cat - band_merged} removed; "
        f"{100.0 * band_merged / max(band_cat, 1):.0f}% of cat density)"
    )
    print(
        f"    band survivors with high opacity (>0.6): {frac_high:.2f}  "
        f"(left half = 0.90 should win the overlap)"
    )
    print("  TOTAL COUNT:")
    print(
        f"    naive cat {len(cat)}  ->  splatreg merge {len(merged)}  "
        f"(removed {len(cat) - len(merged)} overlap duplicates)"
    )
    print(
        f"    registered-only (dedupe=False): {len(merged_nodedupe)}  "
        f"(== cat size {len(cat)}: {len(merged_nodedupe) == len(cat)})"
    )
    print("  CHAMFER-TO-ORIGINAL (mm; lower = closer to the untouched full splat):")
    print(f"    naive torch.cat            : {cham_cat:.4f} mm   (right half left misaligned)")
    print(
        f"    splatreg merge (registered): {cham_merged:.4f} mm   "
        f"({cham_cat / max(cham_merged, 1e-9):.1f}x closer than cat)"
    )
    print(f"    splatreg (registered, no dedupe): {cham_nodedupe:.4f} mm")

    # --- save_ply + round-trip -------------------------------------------------------
    out_dir = Path(__file__).resolve().parent / "_out"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"merged_{source_label.replace('/', '_').replace(' ', '_')}.ply"
    save_ply(merged, out_path)
    reloaded = load_ply(out_path, device=DEVICE)
    dmax = float((reloaded.means.to(DTYPE) - merged.means).abs().max())
    print(f"  saved merged splat -> {out_path}")
    print(
        f"  save_ply -> load_ply round-trip: {len(reloaded)} Gaussians "
        f"(== {len(merged)}: {len(reloaded) == len(merged)}), "
        f"max |mean| diff {dmax:.2e}"
    )
    print("=" * 96)


def _peak_report():
    """Print peak GPU (torch) + process-RSS memory (the 'is full quality healthy?' check)."""
    rss = None
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)  # KiB->GiB (Linux)
    except Exception:
        pass
    if DEVICE.startswith("cuda"):
        peak = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        print(
            f"  PEAK MEMORY: GPU allocated {peak:.2f} GiB, reserved {reserved:.2f} GiB"
            + (f"; python RSS {rss:.2f} GiB" if rss is not None else "")
        )
    elif rss is not None:
        print(f"  PEAK MEMORY: python RSS {rss:.2f} GiB (CPU run)")


def main():
    global DEVICE, QUALITY, DEDUPE_METHOD
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--ply",
        type=str,
        default=None,
        help="optional path to a REAL exported 3DGS .ply to split-and-merge "
        "instead of the synthetic object",
    )
    ap.add_argument("--n", type=int, default=5000, help="synthetic object anchor count (ignored with --ply)")
    ap.add_argument(
        "--quality",
        default="full",
        help="quality policy: full|balanced|low|auto|<0..1 float> (default: full)",
    )
    ap.add_argument("--device", default=DEVICE, help="cpu|cuda (default: $SPLATREG_DEVICE or cpu)")
    ap.add_argument(
        "--dedupe",
        default="voxel",
        choices=("voxel", "knn"),
        help="overlap dedupe pass: voxel (default) | knn (cross-splat, catches the "
        "~16%% residual overlap a voxel grid leaves at cell boundaries)",
    )
    args = ap.parse_args()

    DEVICE = args.device
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        print("--device cuda requested but CUDA is unavailable; falling back to CPU.")
        DEVICE = "cpu"
    try:
        QUALITY = float(args.quality)
    except ValueError:
        QUALITY = args.quality
    DEDUPE_METHOD = args.dedupe

    torch.manual_seed(0)
    if DEVICE.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    if args.ply:
        full = load_ply(args.ply, device=DEVICE, dtype=DTYPE)
        run_demo(full, source_label=f"real PLY {Path(args.ply).name}")
    else:
        full = make_object_splat(args.n, seed=0, device=DEVICE, dtype=DTYPE)
        run_demo(full, source_label="synthetic object (ellipsoid shell + blob)")
    _peak_report()


if __name__ == "__main__":
    main()
