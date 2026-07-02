"""Render the BUFFER-X vs classical-seed recall bar chart used in the README / docs.

A grouped bar chart of registration recall for the zero-shot BUFFER-X seed against the
classical robust (Open3D FPFH+RANSAC) seed, both pushed through the *identical* splatreg
refine so the comparison isolates the seed. Two regimes:

  * 3DMatch     -- complete official gt.log pair set (8/8 scenes, n=1619): 0.962 vs 0.630
  * 3DLoMatch   -- complete official gt.log pair set (n=1781): 0.777 vs 0.122

Writes ``assets/bufferx_recall.png`` (+ ``.svg``). CPU, deterministic, matplotlib only.
Palette: dataviz categorical slots 1 (blue) and 8 (orange) -- a CVD-safe pair (ΔE ~96).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- data --------------------------------------------------------------------
GROUPS = ["3DMatch\n(gt.log, n=1619)", "3DLoMatch\n(gt.log, n=1781)"]
BUFFERX = [0.962, 0.777]
ROBUST = [0.630, 0.122]

# --- palette (dataviz light surface) -----------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
C_BUFFERX = "#2a78d6"  # slot 1, blue
C_ROBUST = "#eb6834"  # slot 8, orange


def main() -> str:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = [0.0, 1.15]
    w = 0.34
    gap = 0.03  # 2px surface gap between the two adjacent bars

    bars_b = ax.bar(
        [xi - w / 2 - gap / 2 for xi in x], BUFFERX, w, color=C_BUFFERX, zorder=3, linewidth=0
    )
    bars_r = ax.bar(
        [xi + w / 2 + gap / 2 for xi in x], ROBUST, w, color=C_ROBUST, zorder=3, linewidth=0
    )

    # value labels on the caps
    for bars in (bars_b, bars_r):
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 0.018,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                color=INK,
                fontweight="bold" if h > 0.9 else "normal",
            )

    # axes / chrome
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"], fontsize=10, color=MUTED)
    ax.set_ylabel("Registration recall  (higher is better)", fontsize=11, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, fontsize=11.5, color=INK)
    ax.set_xlim(-0.62, 1.77)

    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", length=0)

    # title / subtitle (top-left)
    ax.set_title(
        "Zero-shot BUFFER-X seed vs classical FPFH seed",
        fontsize=14,
        color=INK,
        fontweight="bold",
        loc="left",
        pad=26,
    )
    ax.text(
        0.0,
        1.045,
        "Recall on the complete official 3DMatch / 3DLoMatch gt.log sets, identical splatreg refine",
        transform=ax.transAxes,
        fontsize=10.5,
        color=INK_2,
        ha="left",
        va="bottom",
    )

    legend = ax.legend(
        handles=[
            Patch(facecolor=C_BUFFERX, label="BUFFER-X (zero-shot, ICCV 2025)"),
            Patch(facecolor=C_ROBUST, label="classical robust (FPFH+RANSAC)"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=10,
        handlelength=1.1,
        handleheight=1.1,
        borderpad=0.2,
        labelcolor=INK_2,
    )
    legend.set_zorder(5)

    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "bufferx_recall.png")
    svg = os.path.join(out_dir, "bufferx_recall.svg")
    fig.savefig(png, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(svg, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print("wrote", png)
    print("wrote", svg)
    return png


if __name__ == "__main__":
    main()
