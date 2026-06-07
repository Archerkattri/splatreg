#!/usr/bin/env python
"""Camera localization in a splat (v0.2) — differentiable-render pose refinement.

Renders a textured synthetic splat from a known camera pose to make the query image, perturbs the
pose, and asserts :func:`splatreg.localize_camera` pulls it back toward ground truth. Needs gsplat +
CUDA (gsplat rasterization is a CUDA op), so the whole module is skipped otherwise — it does not gate
the CPU CI suite.
"""

from __future__ import annotations

import math

import pytest
import torch

gsplat = pytest.importorskip("gsplat")
if not torch.cuda.is_available():
    pytest.skip("camera localization needs CUDA (gsplat rasterization)", allow_module_level=True)

from splatreg import localize_camera  # noqa: E402
from splatreg.core.lie import se3_exp  # noqa: E402
from splatreg.core.types import Frame, Gaussians  # noqa: E402


def _textured_sphere(n=8000, device="cuda"):
    """A smooth, low-frequency textured sphere in front of the camera (rotation observable)."""
    g = torch.Generator(device="cpu").manual_seed(0)
    u = torch.rand(n, generator=g)
    v = torch.rand(n, generator=g)
    phi = 2 * math.pi * u
    costh = 2 * v - 1
    sinth = torch.sqrt((1 - costh**2).clamp_min(0))
    dirs = torch.stack([sinth * torch.cos(phi), sinth * torch.sin(phi), costh], dim=1).to(device)
    means = dirs * 0.25
    means[:, 2] += 1.3  # in front of an at-origin camera looking +z
    colors = (0.5 + 0.5 * torch.stack([dirs[:, 0], dirs[:, 1], torch.sin(3 * dirs[:, 2])], dim=1)).clamp(0, 1)
    return Gaussians(
        means=means,
        quats=torch.tensor([1.0, 0, 0, 0], device=device).repeat(n, 1),
        scales=torch.full((n, 3), 0.02, device=device),
        opacities=torch.ones(n, device=device),
        colors=colors,
    )


def _angle(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2))))


@pytest.mark.parametrize("mag_t,mag_r", [(0.02, 0.02), (0.05, 0.05), (0.10, 0.10)])
def test_localize_camera_reduces_pose_error(mag_t, mag_r):
    """From a perturbed prior, localize_camera reduces both rotation and translation error."""
    device = "cuda"
    splat = _textured_sphere(device=device)
    H = W = 200
    fx = fy = 220.0
    K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1.0]], device=device)
    T_gt = torch.eye(4, device=device)

    # Ground-truth query image = render at T_gt.
    from splatreg.camera_loc import _render_rgb_depth

    rgb_gt, _ = _render_rgb_depth(splat, T_gt, K, W, H, None)
    frame = Frame(rgb=rgb_gt.detach(), K=K)

    d0 = torch.tensor([mag_t, -mag_t * 0.7, mag_t * 0.5, mag_r, -mag_r * 0.6, mag_r * 0.4], device=device)
    T0 = T_gt @ se3_exp(d0)
    rot0 = _angle(T0[:3, :3].T @ T_gt[:3, :3])
    trans0 = float((T0[:3, 3] - T_gt[:3, 3]).norm())

    res = localize_camera(splat, frame, T0, iters=120, lr=1e-2)
    rot1 = _angle(res.T[:3, :3].T @ T_gt[:3, :3])
    trans1 = float((res.T[:3, 3] - T_gt[:3, 3]).norm())

    assert res.info["mode"] == "camera_loc"
    # Both errors must drop meaningfully (not a no-op / divergence).
    assert rot1 < rot0 * 0.6, f"rotation not reduced: {rot0:.2f} -> {rot1:.2f} deg"
    assert trans1 < trans0 * 0.8, f"translation not reduced: {trans0*1000:.1f} -> {trans1*1000:.1f} mm"
    # Loss must monotone-ish decrease overall.
    assert res.info["loss_history"][-1] <= res.info["loss_history"][0]
