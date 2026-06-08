#!/usr/bin/env python
"""Coarse, prior-free camera-loc seed (v0.3 hardening) — CPU-only, no gsplat.

:func:`splatreg.localize_camera` only refines a pose within the narrow basin of direct image
alignment, so a *wide-baseline* relocalise (no good prior) falls outside it and the refine fails.
:func:`splatreg.camera_loc.coarse_localize_camera` provides the missing seed: it scores a sphere of
candidate camera poses by how well the splat's projected occupancy overlaps the query silhouette —
pure pinhole projection, so it runs on CPU with no rasteriser. These tests pin that:

* the coarse sweep recovers a pose CLOSE to ground truth from NO prior, on a wide-baseline case where
  a refine-from-bad-prior would diverge (the prior is ~120 deg off);
* the chosen pose's projected silhouette actually overlaps the query (the score is high), i.e. the
  seed is in the refine basin.

Everything here is projection-only (no gsplat / no CUDA), so it runs in the normal CPU CI suite.
"""

from __future__ import annotations

import math

import torch

from splatreg.camera_loc import coarse_localize_camera, _project_points
from splatreg.core.types import Frame, Gaussians


def _asymmetric_splat(device="cpu"):
    """A chiral planar 'F'-letter splat (thin in z): a viewpoint-DISTINCTIVE silhouette.

    The letter F has no mirror or rotational symmetry, so its projected silhouette is unique to the
    viewing direction (front vs back vs side all differ) — exactly what a silhouette-overlap coarse
    seed needs to disambiguate viewpoint. It is thin in z so only roughly frontal views see it filled,
    making the framing pose well-determined. A symmetric blob would leave the IoU score ambiguous.
    """
    g = torch.Generator().manual_seed(0)

    def bar(ox, oy, sx, sy):
        cnt = int(8000 * sx * sy)
        p = torch.rand(cnt, 3, generator=g)
        p[:, 0] = ox + p[:, 0] * sx
        p[:, 1] = oy + p[:, 1] * sy
        p[:, 2] = (p[:, 2] - 0.5) * 0.04  # thin slab in z
        return p

    stem = bar(0.0, 0.0, 0.08, 0.34)  # vertical spine
    top = bar(0.0, 0.28, 0.24, 0.07)  # top arm (longer)
    mid = bar(0.0, 0.15, 0.16, 0.06)  # middle arm (shorter) -> chiral
    pts = torch.cat([stem, top, mid], dim=0)
    pts = pts - pts.mean(dim=0)
    m = pts.shape[0]
    return Gaussians(
        means=pts.to(device),
        quats=torch.tensor([1.0, 0, 0, 0], device=device).repeat(m, 1),
        scales=torch.full((m, 3), 0.01, device=device),
        opacities=torch.ones(m, device=device),
    )


def _lookat(cam, center, device):
    """OpenCV (+z forward, +y down) camera→world look-at pose."""
    fwd = center - cam
    fwd = fwd / fwd.norm().clamp_min(1e-9)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    if abs(float(torch.dot(fwd, up))) > 0.95:
        up = torch.tensor([0.0, 0.0, 1.0], device=device)
    right = torch.linalg.cross(up, fwd)
    right = right / right.norm().clamp_min(1e-9)
    down = torch.linalg.cross(fwd, right)
    T = torch.eye(4, device=device)
    T[:3, :3] = torch.stack([right, down, fwd], dim=1)
    T[:3, 3] = cam
    return T


def _silhouette(splat, T_WC, K, W, H):
    """Render the query foreground mask by projecting the splat (CPU, no gsplat)."""
    uv, z, in_front = _project_points(splat.means, T_WC, K)
    mask = torch.zeros(H, W, dtype=torch.bool, device=splat.means.device)
    uvv = uv[in_front]
    inb = (uvv[:, 0] >= 0) & (uvv[:, 0] < W) & (uvv[:, 1] >= 0) & (uvv[:, 1] < H)
    uvv = uvv[inb]
    u = uvv[:, 0].long().clamp(0, W - 1)
    v = uvv[:, 1].long().clamp(0, H - 1)
    mask[v, u] = True
    return mask


def _rot_angle(Ra, Rb):
    Rrel = Ra.transpose(-1, -2) @ Rb
    c = ((Rrel[0, 0] + Rrel[1, 1] + Rrel[2, 2]) - 1.0) * 0.5
    return math.degrees(math.acos(float(c.clamp(-1.0, 1.0))))


def test_coarse_seed_recovers_wide_baseline_from_no_prior():
    """The coarse sweep lands near GT from no prior on a wide-baseline case a refine would miss."""
    device = "cpu"
    splat = _asymmetric_splat(device=device)
    W = H = 160
    fx = fy = 180.0
    K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1.0]], device=device)

    center = splat.means.mean(dim=0)
    brad = float((splat.means - center).norm(dim=1).max())
    dist = 2.5 * brad

    # Ground-truth camera: the frontal +z view (a sweep grid node), where the thin F is seen filled.
    cam_gt = center + dist * torch.tensor([0.0, 0.0, 1.0], device=device)
    T_gt = _lookat(cam_gt, center, device)
    query_mask = _silhouette(splat, T_gt, K, W, H)
    frame = Frame(K=K, mask=query_mask)

    # A wide-baseline BAD prior — camera on the FAR side (-z, ~180 deg away). A refine-from-prior
    # (localize_camera) would diverge from here; the coarse sweep ignores it entirely.
    cam_bad = center + dist * torch.tensor([0.0, 0.0, -1.0], device=device)
    T_bad = _lookat(cam_bad, center, device)
    rot_bad = _rot_angle(T_bad[:3, :3], T_gt[:3, :3])

    # Coarse sweep — NO prior used.
    T_seed, score = coarse_localize_camera(
        splat, frame, n_az=16, n_el=5, grid=32, return_score=True
    )
    rot_seed = _rot_angle(T_seed[:3, :3], T_gt[:3, :3])

    print(
        f"\n[coarse-seed] bad-prior rot err={rot_bad:.1f} deg -> coarse-seed rot err={rot_seed:.1f} deg "
        f"| seed IoU score={score:.3f}"
    )
    # The bad prior really is wide-baseline (well outside any image-alignment basin)...
    assert rot_bad > 90.0, f"bad prior not wide-baseline ({rot_bad:.1f} deg)"
    # ...and the coarse seed lands far closer to GT than that prior (into a refinable basin). The
    # seed is grid-resolution (one azimuth step ~ 22.5 deg), so we allow up to ~one step of slack.
    assert rot_seed < 25.0, f"coarse seed not near GT (rot err {rot_seed:.1f} deg)"
    assert rot_seed < 0.25 * rot_bad, "coarse seed not substantially better than the wide-baseline prior"
    # The seed's silhouette genuinely overlaps the query (a usable refine init).
    assert score > 0.3, f"coarse seed silhouette IoU too low ({score:.3f})"


def test_coarse_seed_uses_rgb_foreground_when_no_mask():
    """With no mask, the sweep derives the foreground from a non-black rgb and still localizes.

    The mask path and the rgb-foreground path must agree: a white-on-black rgb of the same view gives
    the same silhouette cue, so the chosen seed is identical to the masked case (and high-overlap).
    """
    device = "cpu"
    splat = _asymmetric_splat(device=device)
    W = H = 160
    K = torch.tensor([[180.0, 0, W / 2], [0, 180.0, H / 2], [0, 0, 1.0]], device=device)
    center = splat.means.mean(dim=0)
    dist = 2.5 * float((splat.means - center).norm(dim=1).max())
    cam_gt = center + torch.tensor([0.0, 0.0, dist], device=device)
    T_gt = _lookat(cam_gt, center, device)
    mask = _silhouette(splat, T_gt, K, W, H)

    # Same view as an rgb image (white foreground on black) instead of an explicit mask.
    rgb = torch.zeros(H, W, 3, device=device)
    rgb[mask] = 1.0
    T_mask = coarse_localize_camera(splat, Frame(K=K, mask=mask), n_az=16, n_el=5, grid=32)
    T_rgb, score = coarse_localize_camera(
        splat, Frame(K=K, rgb=rgb), n_az=16, n_el=5, grid=32, return_score=True
    )
    print(f"\n[coarse-seed rgb] rgb-vs-mask seed agree={torch.allclose(T_mask, T_rgb)} | IoU={score:.3f}")
    # The rgb luminance cue reproduces the mask cue, so the same seed is chosen, with high overlap.
    assert torch.allclose(T_mask, T_rgb), "rgb-foreground seed differs from the mask seed"
    assert score > 0.3, f"rgb-foreground seed overlap too low ({score:.3f})"
