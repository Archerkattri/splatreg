"""splatreg command-line interface.

Three subcommands wrap the library's public API for the common PLY-in / PLY-out
workflows (SuperSplat, gsplat, Nerfstudio, INRIA exports all speak this format):

* ``splatreg align target.ply source.ply -o aligned.ply``, register ``source`` onto
  ``target`` (:func:`splatreg.api.register`), bake the recovered SE(3)/Sim(3) into the
  source splat, and write it. Prints the 4x4 transform, scale, and solver diagnostics.
* ``splatreg merge a.ply b.ply [c.ply ...] -o fused.ply``, register every splat onto the
  reference (``--ref``, default the first), fuse, dedupe the overlap
  (:func:`splatreg.api.merge`), and write one splat.
* ``splatreg info x.ply``, print what is inside a 3DGS PLY (count, bounds, SH degree,
  opacity/scale statistics).

Everything heavy is delegated to :mod:`splatreg.api` / :mod:`splatreg.io`; this module is
argument parsing, helpful errors, and printing only.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Union

__all__ = ["main", "build_parser"]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _version() -> str:
    from splatreg import __version__

    return __version__


def _parse_quality(value: str) -> Union[str, float]:
    """``--quality`` accepts the named policies or a 0..1 float (per ``resolve_quality``)."""
    names = ("full", "balanced", "low", "auto")
    if value in names:
        return value
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"quality must be one of {names} or a float in [0, 1], got {value!r}"
        )
    if not (0.0 <= f <= 1.0):
        raise argparse.ArgumentTypeError(f"numeric quality must be in [0, 1], got {f}")
    return f


def _resolve_device(device: Optional[str]) -> str:
    """``--device`` default: CUDA when available, else CPU."""
    if device is not None:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load(path: str, device: str):
    """Load a 3DGS PLY with CLI-friendly errors."""
    from splatreg.io import load_ply

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: file not found: {p}")
    if p.suffix.lower() != ".ply":
        print(f"warning: {p} does not end in .ply, trying anyway", file=sys.stderr)
    try:
        return load_ply(p, device=device)
    except ValueError as e:
        raise SystemExit(f"error: could not read {p} as a 3DGS PLY: {e}")


def _format_matrix(T) -> str:
    rows = []
    for r in range(4):
        rows.append("  [" + "  ".join(f"{float(T[r, c]): .6f}" for c in range(4)) + "]")
    return "\n".join(rows)


def _print_result(result, elapsed: float) -> None:
    """Print a RegisterResult: T, scale, rmse + convergence diagnostics."""
    print("T (4x4, maps source -> target):")
    print(_format_matrix(result.T))
    print(f"scale     : {result.scale:.6f}")
    info = result.info or {}
    rmse = info.get("rmse")
    if rmse is not None:
        print(f"rmse      : {float(rmse):.6g}")
    if info.get("n_iters") is not None:
        print(f"iterations: {info['n_iters']}")
    print(f"converged : {result.converged}")
    if info.get("ambiguous"):
        conf = info.get("confidence", 0.0)
        print(
            f"WARNING   : pose flagged AMBIGUOUS (confidence {conf:.2f}); the overlap does "
            "not constrain the pose; treat the result as unreliable.",
            file=sys.stderr,
        )
    print(f"time      : {elapsed:.2f} s")


# --------------------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------------------
def _cmd_align(args: argparse.Namespace) -> int:
    from splatreg.api import apply_transform, register
    from splatreg.io import save_ply

    device = _resolve_device(args.device)
    target = _load(args.target, device)
    source = _load(args.source, device)
    print(f"target: {args.target} ({len(target)} Gaussians)")
    print(f"source: {args.source} ({len(source)} Gaussians)")
    print(f"registering on {device} (transform={args.transform}, init={args.init}, quality={args.quality})")

    t0 = time.perf_counter()
    result = register(
        target,
        source,
        transform=args.transform,
        init=args.init,
        quality=args.quality,
        max_iters=args.max_iters,
    )
    elapsed = time.perf_counter() - t0
    _print_result(result, elapsed)

    aligned = apply_transform(source, result.T, result.scale)
    save_ply(aligned, args.output)
    print(f"wrote {args.output} ({len(aligned)} Gaussians, source aligned into the target frame)")
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    from splatreg.api import merge
    from splatreg.io import save_ply

    if len(args.plys) < 2:
        raise SystemExit("error: merge needs at least two .ply files")
    if not (0 <= args.ref < len(args.plys)):
        raise SystemExit(f"error: --ref {args.ref} out of range for {len(args.plys)} inputs")

    device = _resolve_device(args.device)
    splats = []
    for p in args.plys:
        g = _load(p, device)
        print(f"loaded {p} ({len(g)} Gaussians)")
        splats.append(g)
    n_before = sum(len(g) for g in splats)
    print(
        f"merging {len(splats)} splats on {device} (ref={args.ref} -> {args.plys[args.ref]}, "
        f"transform={args.transform}, init={args.init}, dedupe={not args.no_dedupe})"
    )

    t0 = time.perf_counter()
    fused = merge(
        splats,
        ref=args.ref,
        transform=args.transform,
        init=args.init,
        quality=args.quality,
        dedupe=not args.no_dedupe,
        dedupe_method=args.dedupe_method,
        voxel=args.voxel,
    )
    elapsed = time.perf_counter() - t0

    save_ply(fused, args.output)
    print(f"fused {n_before} -> {len(fused)} Gaussians in {elapsed:.2f} s")
    print(f"wrote {args.output}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    import math

    import torch

    g = _load(args.ply, "cpu")
    n = len(g)
    print(f"file      : {args.ply}")
    print(f"gaussians : {n}")
    if n == 0:
        return 0

    lo = g.means.min(dim=0).values
    hi = g.means.max(dim=0).values
    ext = hi - lo
    print(f"bounds min: [{lo[0]:.4f}, {lo[1]:.4f}, {lo[2]:.4f}]")
    print(f"bounds max: [{hi[0]:.4f}, {hi[1]:.4f}, {hi[2]:.4f}]")
    print(f"extent    : [{ext[0]:.4f}, {ext[1]:.4f}, {ext[2]:.4f}]")

    if g.colors is None:
        print("colors    : none")
    elif g.colors.dim() == 2:
        print("colors    : SH degree 0 (DC only)")
    else:
        k = g.colors.shape[1]
        deg = int(round(math.sqrt(k))) - 1
        print(f"colors    : SH degree {deg} ({k} coefficients per channel)")

    # PLY opacities are pre-sigmoid logits; report the activated range too.
    opac = g.opacities.reshape(-1)
    act = torch.sigmoid(opac)
    print(f"opacity   : raw [{opac.min():.3f}, {opac.max():.3f}]  sigmoid mean {act.mean():.3f}")

    scales = torch.exp(g.scales) if g.log_scales else g.scales
    print(
        f"scales    : {'log-stored, ' if g.log_scales else ''}"
        f"linear median {scales.median():.5f}  max {scales.max():.5f}"
    )
    return 0


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="splatreg",
        description=(
            "Register Gaussian splats: align and merge 3DGS .ply scans into one "
            "SE(3)/Sim(3) frame. Docs: https://archerkattri.github.io/splatreg/"
        ),
    )
    parser.add_argument("--version", action="version", version=f"splatreg {_version()}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--quality",
        type=_parse_quality,
        default="full",
        metavar="Q",
        help="quality policy: full (default), balanced, low, auto, or a float in [0,1]",
    )
    common.add_argument(
        "--device",
        default=None,
        metavar="DEV",
        help="torch device (default: cuda if available, else cpu)",
    )
    common.add_argument(
        "--max-iters",
        type=int,
        default=None,
        metavar="N",
        help="LM iteration cap (default: from --quality)",
    )

    p_align = sub.add_parser(
        "align",
        parents=[common],
        help="register source.ply onto target.ply and write the aligned source",
        description=(
            "Register SOURCE onto TARGET, print the recovered transform, and write the "
            "source splat with the transform baked in (same frame as the target: open both "
            "in SuperSplat and they line up)."
        ),
    )
    p_align.add_argument("target", help="reference splat .ply (stays fixed)")
    p_align.add_argument("source", help="splat .ply to align onto the target")
    p_align.add_argument("-o", "--output", required=True, help="output .ply for the aligned source")
    p_align.add_argument(
        "--transform",
        choices=("se3", "sim3"),
        default="se3",
        help="rigid (se3, default) or similarity with scale (sim3)",
    )
    p_align.add_argument(
        "--init",
        choices=("fast", "robust", "learned", "mac", "global", "features"),
        default="fast",
        help=(
            "initializer: fast (default, FPFH+RANSAC seed), robust (Open3D FPFH+RANSAC, real "
            "scans), learned (GeoTransformer, best accuracy), mac (maximal-clique consensus, "
            "outlier-heavy correspondences), global (blind SO(3) sweep), features (partial overlap)"
        ),
    )
    p_align.set_defaults(func=_cmd_align)

    p_merge = sub.add_parser(
        "merge",
        parents=[common],
        help="register N splats into one frame, fuse, dedupe, write one .ply",
        description=(
            "Register every splat onto the reference (--ref), fuse them, dedupe the "
            "double-density overlap, and write a single splat."
        ),
    )
    p_merge.add_argument("plys", nargs="+", metavar="x.ply", help="two or more 3DGS .ply files")
    p_merge.add_argument("-o", "--output", required=True, help="output .ply for the fused splat")
    p_merge.add_argument("--ref", type=int, default=0, help="index of the reference splat (default 0)")
    p_merge.add_argument(
        "--transform",
        choices=("se3", "sim3"),
        default="sim3",
        help="per-pair transform: sim3 (default, absorbs scale differences) or se3",
    )
    p_merge.add_argument(
        "--init",
        choices=("fast", "robust", "learned", "mac", "global", "features"),
        default="global",
        help="per-pair initializer (default: global, robust to large inter-capture offsets)",
    )
    p_merge.add_argument(
        "--no-dedupe", action="store_true", help="skip the overlap dedupe (plain registered concat)"
    )
    p_merge.add_argument(
        "--dedupe-method",
        choices=("voxel", "knn"),
        default="voxel",
        help="overlap dedupe: voxel grid (default) or knn radius suppression",
    )
    p_merge.add_argument(
        "--voxel",
        type=float,
        default=None,
        metavar="EDGE",
        help="voxel edge in splat units (default: auto from anchor spacing)",
    )
    p_merge.set_defaults(func=_cmd_merge)

    p_info = sub.add_parser(
        "info",
        help="print what is inside a 3DGS .ply",
        description="Print count, bounds, SH degree, and opacity/scale statistics of a 3DGS PLY.",
    )
    p_info.add_argument("ply", help="a 3DGS .ply file")
    p_info.set_defaults(func=_cmd_info)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
