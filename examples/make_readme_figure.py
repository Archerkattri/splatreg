#!/usr/bin/env python
"""Generate the README hero figure: two Gaussian splats before/after splatreg registration.

Builds a realistic object splat A, transforms it by a known SE(3) to make splat B (unknown
relative pose), recovers the transform with ``register``, and renders a before/after
3D scatter to ``docs/assets/registration_demo.png``. CPU, deterministic.

    PYTHONPATH=. python examples/make_readme_figure.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from _example_utils import (  # noqa: E402
    axis_angle_R,
    chamfer_mm,
    make_object_splat,
    rot_angle_deg,
    sim3_matrix,
)
from splatreg.api import register  # noqa: E402

torch.manual_seed(0)
N = 800
A = make_object_splat(N, seed=0)  # target splat
R = axis_angle_R([1.0, 0.7, 0.3], 62.0)
t_gt = torch.tensor([0.18, -0.12, 0.10])
M_gt = sim3_matrix(1.0, R, t_gt)  # known SE(3): rotation + translation
B = make_object_splat.apply_to(A, M_gt)  # source = A under unknown SE(3)

res = register(B, A, init="global", transform="se3", max_iters=50)
T = res.T
A_to_B = A.means @ T[:3, :3].transpose(-1, -2) + T[:3, 3]  # A mapped into B's frame -> overlaps B
rot_err = rot_angle_deg(M_gt[:3, :3], T[:3, :3])
trans_mm = 1000.0 * float((T[:3, 3] - t_gt).norm())
cham = chamfer_mm(A_to_B, B.means)
print(f"recovered: rot_err={rot_err:.3f} deg | trans_err={trans_mm:.3f} mm | chamfer={cham:.3f} mm")

Am, Bm, ABm = A.means.numpy(), B.means.numpy(), A_to_B.detach().numpy()
TEAL, CORAL = "#17becf", "#ff6b5b"
fig = plt.figure(figsize=(12.5, 5.6), facecolor="white")


def panel(ax, p1, c1, l1, p2, c2, l2, title):
    ax.scatter(p1[:, 0], p1[:, 1], p1[:, 2], s=5, c=c1, alpha=0.55, label=l1, edgecolors="none")
    ax.scatter(p2[:, 0], p2[:, 1], p2[:, 2], s=5, c=c2, alpha=0.55, label=l2, edgecolors="none")
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=8)
    ax.set_axis_off()
    ax.legend(loc="upper right", fontsize=9, framealpha=0.0, markerscale=2.0)
    allp = np.vstack([p1, p2])
    c = allp.mean(0)
    r = (allp.max(0) - allp.min(0)).max() / 2 * 1.05
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=18, azim=-62)


panel(
    fig.add_subplot(121, projection="3d"),
    Am,
    TEAL,
    "splat A (target)",
    Bm,
    CORAL,
    "splat B  (unknown SE(3))",
    "Before  —  two splats, unknown relative pose",
)
panel(
    fig.add_subplot(122, projection="3d"),
    Bm,
    CORAL,
    "splat B",
    ABm,
    TEAL,
    "A registered into B",
    f"After splatreg  —  rot {rot_err:.2f}° · trans {trans_mm:.1f} mm · Chamfer {cham:.2f} mm",
)
plt.tight_layout()

out_dir = os.path.join(_REPO, "docs", "assets")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "registration_demo.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("saved", out)
