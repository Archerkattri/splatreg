#!/usr/bin/env python
"""PIPELINE DIAGRAM — splatreg's architecture, in the docs-site instrument identity.

One clean left-to-right diagram of the pipeline:

    two 3DGS splats  ->  6 coarse-init seeds (pick one)  ->  multi-residual Levenberg-Marquardt
    (ICP + Gaussian-SDF, SE(3)/Sim(3))  ->  T* + recovered scale + pose covariance  ->  the
    merge / align / track / pose-graph consumers.

Dark "instrument" palette + IBM Plex Mono wordmark, matching docs_site/stylesheets/extra.css.
Emits both ``assets/pipeline.svg`` (vector, text preserved, IBM Plex Mono for the live docs) and
``assets/pipeline.png`` (rasterized fallback). Pure matplotlib — no external SVG rasterizer needed.

Run::  .gpu_venv/bin/python examples/make_pipeline_diagram.py
"""

from __future__ import annotations

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# --- docs-site instrument palette --------------------------------------------------------------
HERO_A = "#0a2229"
HERO_B = "#071519"
INK = "#f2feff"
DIM = "#a9c9ce"
MUTED = "#6f959b"
TEAL = "#2fd6e0"
TEAL_D = "#0f8b96"
ORANGE = "#ff6b3d"
VIOLET = "#b08cf0"
PANEL = "#0c2027"
CHIP = "#0a1b21"
GRID = "#12313a"


def _box(ax, x, y, w, h, *, fc=CHIP, ec=TEAL_D, lw=1.2, r=0.06, z=2):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        fc=fc, ec=ec, lw=lw, zorder=z, mutation_aspect=1.0,
    )
    ax.add_patch(p)
    return p


def _arrow(ax, x0, y0, x1, y1, *, color=TEAL, lw=1.6, z=1, alpha=0.9, rad=0.0):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, lw=lw,
        color=color, zorder=z, alpha=alpha, shrinkA=1, shrinkB=1,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)


def build():
    plt.rcParams.update({"font.family": "monospace", "font.monospace": ["DejaVu Sans Mono"], "svg.fonttype": "none"})
    W, H = 200.0, 96.0
    fig = plt.figure(figsize=(13.9, 6.67), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor(HERO_B)

    # background: hero-style panel + faint point-cloud dot grid, fading downward
    ax.add_patch(FancyBboxPatch((0, 0), W, H, boxstyle="square,pad=0", fc=HERO_A, ec="none", zorder=0))
    for gy in range(6, int(H), 6):
        for gx in range(6, int(W), 6):
            ax.plot([gx], [gy], marker="o", ms=0.7, color="#ffffff",
                    alpha=0.05 * (gy / H), zorder=0)

    # header
    ax.text(5, 89.5, "splatreg", color=TEAL, fontsize=20, fontweight="bold", family="monospace")
    ax.text(35, 90.3, "the inverse of gsplat  ·  SE(3) / Sim(3) registration pipeline",
            color=DIM, fontsize=10.5, va="center")
    ax.plot([5, 195], [85.5, 85.5], color=GRID, lw=1.0, zorder=1)

    # stage kicker labels
    def kicker(x, s):
        ax.text(x, 80.5, s, color=TEAL, fontsize=8.2, family="monospace", letterspacing=None)

    # ---- STAGE 1: input ----------------------------------------------------------------------
    ax.text(4.5, 80.5, "INPUT", color=MUTED, fontsize=8.0)
    _box(ax, 3, 62, 26, 8.5, ec=TEAL, lw=1.4)
    ax.text(16, 66.2, "splat A  ·  target", color=TEAL, fontsize=9.3, ha="center", va="center", fontweight="bold")
    _box(ax, 3, 50, 26, 8.5, ec=ORANGE, lw=1.4)
    ax.text(16, 54.2, "splat B  ·  source", color=ORANGE, fontsize=9.3, ha="center", va="center", fontweight="bold")
    ax.text(16, 44.5, "gsplat / Nerfstudio / INRIA\nPLY  or  means+cov tensors",
            color=MUTED, fontsize=7.2, ha="center", va="top", linespacing=1.4)

    # ---- STAGE 2: init seeds -----------------------------------------------------------------
    ax.text(37.5, 80.5, "COARSE INIT  ·  pick one", color=MUTED, fontsize=8.0)
    _box(ax, 36, 26, 44, 51, fc=PANEL, ec=TEAL_D, lw=1.2, r=0.05)
    seeds = [
        ("fast", "FPFH + GPU RANSAC  ·  ~17 ms  (default)", TEAL),
        ("robust", "Open3D FPFH + RANSAC  ·  scale-safe", TEAL),
        ("learned", "GeoTransformer  ·  CVPR'22", TEAL),
        ("bufferx", "zero-shot BUFFER-X  ·  ICCV'25", ORANGE),
        ("mac", "maximal-clique consensus  ·  CVPR'23", TEAL),
        ("global", "super-Fibonacci SO(3) sweep", TEAL),
    ]
    sy = 70.5
    for name, desc, col in seeds:
        h = 6.6
        hl = col == ORANGE
        _box(ax, 38.5, sy - h + 1.3, 39, h, fc="#14262d" if hl else CHIP,
             ec=col, lw=1.5 if hl else 1.0, r=0.09)
        ax.text(41, sy - 1.0, f"init='{name}'", color=col, fontsize=8.6, va="center", fontweight="bold")
        ax.text(41, sy - 3.9, desc, color=DIM if hl else MUTED, fontsize=6.7, va="center")
        sy -= 7.7

    # ---- STAGE 3: LM core --------------------------------------------------------------------
    ax.text(89, 80.5, "MULTI-RESIDUAL SOLVE", color=MUTED, fontsize=8.0)
    _box(ax, 88, 34, 44, 40, fc=PANEL, ec=TEAL, lw=1.6, r=0.05)
    ax.text(110, 70.2, "Levenberg-Marquardt", color=INK, fontsize=11.2, ha="center", fontweight="bold")
    ax.text(110, 66.4, "from-scratch  ·  closed-form Jacobian", color=MUTED, fontsize=7.0, ha="center")
    _box(ax, 91, 55.5, 38, 7.2, fc=CHIP, ec=TEAL_D, lw=1.0, r=0.1)
    ax.text(110, 59.1, "ICP   point-to-point / point-to-plane", color=TEAL, fontsize=8.0, ha="center", va="center")
    _box(ax, 91, 46.5, 38, 7.2, fc="#14262d", ec=ORANGE, lw=1.3, r=0.1)
    ax.text(110, 51.1, "Gaussian-SDF   flagship residual", color=ORANGE, fontsize=8.0, ha="center", va="center", fontweight="bold")
    ax.text(110, 42.0, "smooth field from the target Gaussians", color=MUTED, fontsize=6.6, ha="center")
    _box(ax, 91, 36.0, 38, 4.6, fc=CHIP, ec=TEAL_D, lw=1.0, r=0.14)
    ax.text(110, 38.3, "solves the full  SE(3) · Sim(3)  tangent", color=DIM, fontsize=7.4, ha="center", va="center")

    # ---- STAGE 4: outputs --------------------------------------------------------------------
    ax.text(141, 80.5, "OUTPUT", color=MUTED, fontsize=8.0)
    _box(ax, 140, 63, 30, 9.5, ec=TEAL, lw=1.4)
    ax.text(155, 68.9, "T*   4x4", color=TEAL, fontsize=9.6, ha="center", va="center", fontweight="bold")
    ax.text(155, 65.4, "SE(3) / Sim(3)", color=DIM, fontsize=7.2, ha="center", va="center")
    _box(ax, 140, 52, 30, 8.0, ec=TEAL, lw=1.2)
    ax.text(155, 56.0, "recovered scale  s", color=TEAL, fontsize=8.3, ha="center", va="center")
    _box(ax, 140, 41, 30, 8.0, ec=VIOLET, lw=1.3)
    ax.text(155, 45.6, "pose covariance  Σ", color=VIOLET, fontsize=8.3, ha="center", va="center", fontweight="bold")
    ax.text(155, 42.3, "None when singular", color=MUTED, fontsize=6.4, ha="center", va="center")

    # ---- STAGE 5: consumers ------------------------------------------------------------------
    ax.text(177, 80.5, "CONSUMERS", color=MUTED, fontsize=8.0)
    cons = [
        ("merge()  + dedupe", TEAL, True),
        ("apply_transform / align", TEAL, False),
        ("Tracker  warm-start", TEAL, False),
        ("pose-graph weighting", VIOLET, False),
    ]
    cy = 72.0
    for label, col, strong in cons:
        _box(ax, 175, cy - 6.6, 22, 6.0, fc="#14262d" if strong else CHIP, ec=col, lw=1.4 if strong else 1.0, r=0.12)
        ax.text(186, cy - 3.6, label, color=col, fontsize=7.2, ha="center", va="center",
                fontweight="bold" if strong else "normal")
        cy -= 8.2

    # ---- arrows ------------------------------------------------------------------------------
    # input -> init panel
    _arrow(ax, 29, 66, 36, 58, color=TEAL, lw=1.6)
    _arrow(ax, 29, 54, 36, 50, color=ORANGE, lw=1.6)
    # init panel -> LM
    _arrow(ax, 80, 51, 88, 54, color=TEAL, lw=2.0)
    # LM -> outputs
    _arrow(ax, 132, 60, 140, 66, color=TEAL, lw=1.8)
    _arrow(ax, 132, 54, 140, 56, color=TEAL, lw=1.4)
    _arrow(ax, 132, 48, 140, 46, color=VIOLET, lw=1.4)
    # outputs -> consumers
    _arrow(ax, 170, 67, 175, 68, color=TEAL, lw=1.5)
    _arrow(ax, 170, 56, 175, 60, color=TEAL, lw=1.3)
    _arrow(ax, 170, 45, 175, 50, color=VIOLET, lw=1.3, rad=-0.15)

    # footer
    ax.text(5, 6.0, "Pure PyTorch  ·  no meshing, no CUDA extension, no point-cloud detour  ·  "
            "every builtin-LM solve reports its pose covariance", color=MUTED, fontsize=7.6)
    ax.text(5, 2.4, "docs: Archerkattri.github.io/splatreg   ·   pip install splatreg",
            color=TEAL_D, fontsize=7.6)

    out_png = os.path.join(_REPO_ROOT, "assets", "pipeline.png")
    out_svg = os.path.join(_REPO_ROOT, "assets", "pipeline.svg")
    fig.savefig(out_png, facecolor=HERO_B)
    fig.savefig(out_svg, facecolor=HERO_B)
    plt.close(fig)

    # inject IBM Plex Mono ahead of the fallback so the live docs render the brand font
    with open(out_svg, "r", encoding="utf-8") as fh:
        svg = fh.read()
    svg = svg.replace("DejaVu Sans Mono", "IBM Plex Mono, DejaVu Sans Mono, ui-monospace, monospace")
    svg = re.sub(r"font:([^;\"]*?)'DejaVu Sans Mono'", r"font:\1'IBM Plex Mono', 'DejaVu Sans Mono'", svg)
    with open(out_svg, "w", encoding="utf-8") as fh:
        fh.write(svg)

    print(f"wrote {out_png}  ({os.path.getsize(out_png) / 1e6:.2f} MB)")
    print(f"wrote {out_svg}  ({os.path.getsize(out_svg) / 1e3:.0f} KB)")
    return out_png


if __name__ == "__main__":
    build()
