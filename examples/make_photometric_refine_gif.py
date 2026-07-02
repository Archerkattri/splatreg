#!/usr/bin/env python
"""PHOTOMETRIC-REFINE GIF — splatreg's opt-in photometric stage, rendered through gsplat.

A colour splat is knocked ~9 degrees / ~12 cm out of alignment (the residual seam a geometric
solve can leave), then the REAL library photometric refiner
(:func:`splatreg.residuals.photometric.refine_photometric`) is stepped one LM iteration at a time.
Each iteration renders the SOURCE splat under the current pose against a small synthetic camera
ring of the TARGET (PhotoReg-style, no real images) and polishes the pose. The GIF shows, per real
iteration:

    * a fixed-camera blend of TARGET (fixed) + SOURCE-under-current-pose — a ghosted double image
      that sharpens to one as the pose locks on;
    * the MEASURED rotation / translation error vs ground truth, ticking down;
    * a convergence curve drawn point-by-point.

Every pose and error is a real per-iteration measurement (LM damping is raised so the solver takes
gradual, visible steps instead of its usual 2-3 Newton jumps).

Run (GPU)::

    CUDA_VISIBLE_DEVICES=0 .gpu_venv/bin/python examples/make_photometric_refine_gif.py

Writes ``assets/photometric_refine.gif``. Deterministic. Palette: splatreg dark-instrument.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EXAMPLES_DIR)
for _p in (_REPO_ROOT, _EXAMPLES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gsplat import rasterization  # noqa: E402
from splatreg.core.types import Gaussians  # noqa: E402
from splatreg.api import apply_transform  # noqa: E402
from splatreg.residuals.photometric import camera_ring, refine_photometric  # noqa: E402
from _example_utils import axis_angle_R, rot_angle_deg  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISP = 340  # display render size

BG = "#0b1417"
TEAL = "#2fd6e0"
ORANGE = "#ff6b3d"
INK = "#eafcff"
MUTED = "#7fa6ad"
GOOD = "#37e0a0"


def _build_object(n: int = 30000) -> Gaussians:
    torch.manual_seed(0)
    u = torch.rand(n, device=DEVICE)
    v = torch.rand(n, device=DEVICE)
    phi = 2 * math.pi * u
    ct = 2 * v - 1
    st = torch.sqrt((1 - ct**2).clamp_min(0))
    p = torch.stack([st * torch.cos(phi) * 1.3, st * torch.sin(phi) * 0.8, ct * 1.0], 1)
    p[:, 0] += 0.35 * torch.exp(-((p[:, 0] - 1.3) ** 2) / 0.3)  # +x lobe breaks symmetry
    col = torch.stack(
        [(p[:, 0] + 1.4) / 2.8, (p[:, 1] + 0.8) / 1.6, (p[:, 2] + 1.0) / 2.0], 1
    ).clamp(0, 1)
    return Gaussians(
        means=p, quats=torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE).repeat(n, 1),
        scales=torch.full((n, 3), 0.05, device=DEVICE), opacities=torch.full((n,), 0.9, device=DEVICE),
        colors=col, log_scales=False,
    )


def _look_at(cam, target):
    cam = torch.as_tensor(cam, dtype=torch.float32, device=DEVICE)
    target = torch.as_tensor(target, dtype=torch.float32, device=DEVICE)
    up = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
    f = target - cam
    f = f / f.norm()
    r = torch.linalg.cross(f, up)
    r = r / r.norm()
    d = torch.linalg.cross(f, r)
    R = torch.stack([r, d, f], dim=0)
    T = torch.eye(4, device=DEVICE)
    T[:3, :3] = R
    T[:3, 3] = -R @ cam
    return T


def _render(g: Gaussians, V: torch.Tensor, K: torch.Tensor) -> np.ndarray:
    out, _, _ = rasterization(
        g.means.float(), g.quats.float(), g.scales.float(), g.opacities.float(),
        g.colors.float(), V[None], K[None], DISP, DISP, sh_degree=None, render_mode="RGB",
    )
    return out[0].clamp(0, 1).cpu().numpy()


def main() -> str:
    G = _build_object()
    Tgt = torch.eye(4, device=DEVICE)  # target pose (source should converge here)

    # a decent 3/4 fixed display camera
    Vdisp = _look_at([2.6, 1.5, -3.4], [0.15, 0.0, 0.0])
    fdisp = 330.0
    Kdisp = torch.tensor([[fdisp, 0, DISP / 2], [0, fdisp, DISP / 2], [0, 0, 1]], device=DEVICE)
    target_img = _render(G, Vdisp, Kdisp)

    # the refiner's synthetic camera ring (target views), 96 px
    ring_cams, ring_K = camera_ring(G, 8, width=96, height=96)

    # initial misalignment ~9 deg / ~12 cm — the residual seam a geometric solve leaves
    R0 = axis_angle_R([0.3, 1.0, 0.2], 9.0, device=DEVICE, dtype=torch.float32)
    T = torch.eye(4, device=DEVICE)
    T[:3, :3] = R0
    T[:3, 3] = torch.tensor([0.12, -0.07, 0.06], device=DEVICE)

    def err(Tc):
        return rot_angle_deg(Tc[:3, :3], Tgt[:3, :3]), 1000.0 * float((Tc[:3, 3] - Tgt[:3, 3]).norm())

    poses = [T.clone()]
    errs = [err(T)]
    N_ITERS = 15
    for _ in range(N_ITERS):
        res = refine_photometric(
            G, G, T, transform="se3", cameras=ring_cams, K=ring_K, width=96, height=96,
            max_iters=1, exposure=True, jac_mode="fd", max_rot_step=0.03, max_trans_step=0.04, damping=1.5,
        )
        T = res.T
        poses.append(T.clone())
        errs.append(err(T))
    print("RRE trajectory:", " ".join(f"{r:.1f}" for r, t in errs))
    print(f"start {errs[0][0]:.1f} deg / {errs[0][1]:.0f} mm  ->  final {errs[-1][0]:.2f} deg / {errs[-1][1]:.2f} mm")

    # render source-under-pose for each captured pose
    src_imgs = [_render(apply_transform(G, Tp, 1.0), Vdisp, Kdisp) for Tp in poses]

    _render_gif(target_img, src_imgs, errs)
    return os.path.join(_REPO_ROOT, "assets", "photometric_refine.gif")


def _render_gif(target_img, src_imgs, errs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    plt.rcParams.update({"font.family": "monospace", "font.monospace": ["DejaVu Sans Mono"]})

    rre = [e[0] for e in errs]
    rte = [e[1] for e in errs]
    nframes = len(errs)
    # two-tint overlay: TARGET in teal, SOURCE-under-pose in orange. Misaligned -> the orange peels
    # off the teal as separate ghosts; locked-on -> they coincide into one warm-lit object.
    teal_rgb = np.array([0.18, 0.84, 0.88])
    orange_rgb = np.array([1.0, 0.42, 0.24])
    lum_t = target_img.mean(-1)
    frames = []
    for i in range(nframes):
        lum_s = src_imgs[i].mean(-1)
        blend = np.clip(teal_rgb * lum_t[..., None] + orange_rgb * lum_s[..., None], 0, 1)

        fig = plt.figure(figsize=(7.4, 4.3), dpi=100)
        fig.patch.set_facecolor(BG)

        # left: blended render
        axr = fig.add_axes([0.02, 0.02, 0.55, 0.80])
        axr.imshow(blend)
        axr.set_xticks([])
        axr.set_yticks([])
        for s in axr.spines.values():
            s.set_color("#12313a")
        axr.text(0.03, 0.05, "target", transform=axr.transAxes, color=TEAL, fontsize=8, va="bottom", fontweight="bold")
        axr.text(0.155, 0.05, "vs source-under-pose", transform=axr.transAxes, color=ORANGE, fontsize=8,
                 va="bottom", fontweight="bold")
        axr.text(0.97, 0.05, "gsplat render", transform=axr.transAxes, color=MUTED, fontsize=7, va="bottom", ha="right")

        # header
        fig.text(0.03, 0.925, "splatreg", color=TEAL, fontsize=14, fontweight="bold")
        fig.text(0.165, 0.930, "photometric refine  ·  the pose geometry can't see", color=INK, fontsize=9.2)
        fig.text(0.03, 0.865, "PhotoReg-style splat-vs-splat photometric LM (no real images) —"
                 " renders source vs target and", color=MUTED, fontsize=7.3)
        fig.text(0.03, 0.835, "polishes the residual seam.  Real per-iteration trajectory.", color=MUTED, fontsize=7.3)

        # right: readouts
        conv = i == nframes - 1 or rre[i] < 0.1
        col_rre = GOOD if conv else TEAL
        fig.text(0.62, 0.72, f"iteration {i:2d} / {nframes - 1}", color=MUTED, fontsize=9)
        fig.text(0.62, 0.63, "rotation err", color=MUTED, fontsize=8.5)
        fig.text(0.62, 0.55, f"{rre[i]:5.2f}°", color=col_rre, fontsize=21, fontweight="bold")
        fig.text(0.80, 0.63, "translation err", color=MUTED, fontsize=8.5)
        fig.text(0.80, 0.55, f"{rte[i]:4.0f} mm", color=col_rre, fontsize=21, fontweight="bold")

        # convergence curve (RRE vs iter), revealed up to i
        axc = fig.add_axes([0.63, 0.12, 0.34, 0.33])
        axc.set_facecolor(BG)
        axc.plot(range(nframes), rre, color="#1c4650", lw=1.0, zorder=1)  # faint full path
        axc.plot(range(i + 1), rre[: i + 1], color=col_rre, lw=1.8, zorder=2)
        axc.scatter([i], [rre[i]], s=22, color=col_rre, zorder=3)
        axc.set_xlim(-0.5, nframes - 0.5)
        axc.set_ylim(-0.6, max(rre) * 1.08)
        axc.set_xlabel("LM iteration", color=MUTED, fontsize=7.5)
        axc.set_ylabel("RRE (deg)", color=MUTED, fontsize=7.5)
        axc.tick_params(colors=MUTED, labelsize=6.5, length=0)
        for sp in ("top", "right"):
            axc.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            axc.spines[sp].set_color("#2c4c54")
        axc.grid(True, color="#12313a", lw=0.6)
        axc.set_axisbelow(True)

        if conv:
            fig.text(0.62, 0.47, "locked on ✓", color=GOOD, fontsize=10, fontweight="bold")

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    # hold start and end
    frames = [frames[0]] * 3 + frames + [frames[-1]] * 6
    out = os.path.join(_REPO_ROOT, "assets", "photometric_refine.gif")
    # imageio >= 2.28 reads `duration` in MILLISECONDS: ~160 ms/frame, ~1.3 s end hold.
    per_frame_ms = [160] * (len(frames) - 1) + [1300]
    imageio.mimsave(out, frames, format="GIF", duration=per_frame_ms, loop=0, subrectangles=True)
    print(f"wrote {out}  ({os.path.getsize(out) / 1e6:.2f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
