"""Shared helpers for the splatreg examples (recovery harness + merge demo).

Pure torch + numpy (the only splatreg runtime deps). Provides:

* :func:`make_object_splat` — a REALISTIC synthetic object splat: an anisotropic ellipsoid
  SHELL (surface points spaced ~ the Gaussian footprint) + a filled interior blob + a ``+x``
  lobe. The anisotropy and lobe make rotation OBSERVABLE (so the geometry-first SDF/ICP minimum
  sits at the true pose), and the shell point-spacing matches the Gaussian scale so the library's
  auto SDF-sigma and auto dedupe-voxel size are sensible. This is on purpose NOT a 1-D spiral
  (ambiguous ICP correspondences + spacing that breaks the dedupe sizing).
  It carries an ``.apply_to(g, M)`` attribute that bakes a 4x4 Sim(3)/SE(3) into a splat's
  geometry (means, scales) the same way the GT transform is defined in the harness.
* small SO(3) / metric utilities (axis-angle rotation, geodesic rotation error, Chamfer).

Everything is deterministic given a seed.
"""

from __future__ import annotations

import math

import torch

from splatreg.core.types import Gaussians

__all__ = [
    "SEEDS",
    "axis_angle_R",
    "rot_angle_deg",
    "chamfer_mm",
    "sim3_matrix",
    "make_object_splat",
]

# A small fixed seed set so the examples are reproducible but still average over geometry RNG.
SEEDS = [0, 1, 2]


# ----------------------------------------------------------------------- SO(3) + metrics
def axis_angle_R(axis, deg: float, *, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Rodrigues rotation matrix for ``deg`` degrees about ``axis`` (need not be unit)."""
    axis = torch.as_tensor(axis, dtype=dtype, device=device)
    axis = axis / axis.norm().clamp_min(1e-12)
    th = math.radians(deg)
    K = torch.tensor(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=dtype,
        device=device,
    )
    eye = torch.eye(3, dtype=dtype, device=device)
    return eye + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def rot_angle_deg(Ra: torch.Tensor, Rb: torch.Tensor) -> float:
    """Geodesic angle (deg) between two rotation matrices: ``arccos((tr(Ra^T Rb) - 1) / 2)``."""
    Rrel = Ra.transpose(-1, -2) @ Rb
    c = ((Rrel[0, 0] + Rrel[1, 1] + Rrel[2, 2]) - 1.0) * 0.5
    return math.degrees(math.acos(float(c.clamp(-1.0, 1.0))))


def chamfer_mm(a: torch.Tensor, b: torch.Tensor, *, max_pts: int = 4000) -> float:
    """Symmetric mean Chamfer distance (in mm, assuming inputs are in metres) between point sets.

    Subsamples each side to ``max_pts`` (deterministic stride) to keep the ``cdist`` bounded, then
    averages the mean nearest-neighbour distance in each direction. Returned x1000 (m -> mm).
    """

    def sub(x):
        if x.shape[0] <= max_pts:
            return x
        sel = torch.linspace(0, x.shape[0] - 1, max_pts, device=x.device).round().long()
        return x[sel]

    a, b = sub(a), sub(b)
    d = torch.cdist(a, b)  # (Na, Nb)
    d_ab = d.min(dim=1).values.mean()
    d_ba = d.min(dim=0).values.mean()
    return 1000.0 * float(0.5 * (d_ab + d_ba))


def sim3_matrix(s: float, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Assemble the 4x4 similarity ``[[s*R, t], [0, 1]]`` on R's device/dtype."""
    T = torch.eye(4, device=R.device, dtype=R.dtype)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


# ----------------------------------------------------------------------- realistic splat
def _make_object_splat(
    n: int,
    *,
    seed: int = 0,
    device="cpu",
    dtype=torch.float32,
    scale: float = 0.004,
    center=(0.4, -0.2, 0.3),
    shell_frac: float = 0.7,
) -> Gaussians:
    """A realistic single-object splat: anisotropic ellipsoid SHELL + filled blob + ``+x`` lobe.

    ``shell_frac`` of the points are a thin anisotropic shell (surface), the rest fill the
    interior; a Gaussian ``+x`` lobe breaks rotational symmetry so pose is observable. Point count
    is chosen so the surface spacing is comparable to ``scale`` (the per-Gaussian footprint), which
    is what the library's auto SDF-sigma / auto dedupe-voxel sizing expects. All anchors get the
    same isotropic linear ``scale`` and unit quaternion; opacities default to 1 (the merge demo
    overrides them to make the dedupe winner observable). Deterministic given ``seed``.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    n_shell = int(round(shell_frac * n))
    n_fill = n - n_shell
    radii = torch.tensor([0.14, 0.09, 0.06], dtype=dtype)  # anisotropic extents (m)

    def unit_sphere(m):
        u = torch.rand(m, generator=g)
        v = torch.rand(m, generator=g)
        phi = 2 * math.pi * u
        costh = 2 * v - 1
        sinth = torch.sqrt((1 - costh**2).clamp_min(0))
        return torch.stack([sinth * torch.cos(phi), sinth * torch.sin(phi), costh], dim=1)

    # Shell: unit sphere -> anisotropic surface, with thin-shell radial jitter (~ Gaussian scale).
    sph = unit_sphere(n_shell)
    shell = sph * radii
    shell = shell * (1.0 + (scale / 0.1) * torch.randn(n_shell, 1, generator=g))

    # Filled interior: cube-root radial fill for ~uniform volume density inside the same ellipsoid.
    if n_fill > 0:
        dir_f = unit_sphere(n_fill)
        rad_f = torch.rand(n_fill, 1, generator=g) ** (1.0 / 3.0)
        fill = dir_f * radii * rad_f * 0.92  # 0.92 keeps fill just inside the shell
        pts = torch.cat([shell, fill], dim=0)
    else:
        pts = shell

    # +x lobe: a bump that makes the +x end distinct (rotation observable).
    bump = 0.03 * torch.exp(-((pts[:, 0:1] / radii[0] - 1.0) ** 2) / 0.2)
    pts = pts + bump * torch.tensor([1.0, 0.0, 0.0], dtype=dtype)

    pts = pts.to(device=device, dtype=dtype) + torch.tensor(center, device=device, dtype=dtype)
    m = pts.shape[0]
    return Gaussians(
        means=pts,
        quats=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).repeat(m, 1),
        scales=torch.full((m, 3), scale, device=device, dtype=dtype),
        opacities=torch.ones(m, device=device, dtype=dtype),
        log_scales=False,
    )


def _apply_to(g: Gaussians, M: torch.Tensor) -> Gaussians:
    """Bake a 4x4 Sim(3)/SE(3) ``M = [[s*R, t],[0,1]]`` into a splat's geometry, returning a copy.

    Means map as ``s*(R @ x) + t``; linear scales are multiplied by the recovered ``s`` (so a
    Sim(3) GT genuinely changes the splat's footprint, exercising the scale DoF). Quats/opacities/
    colors carry through (this is GT construction, not a quaternion-exact compose — the SDF/ICP
    residuals read geometry, and the harness scores against these means). ``s`` is read as the cube
    root of ``det(s*R)`` so the helper needs only the matrix.
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


# Expose as a callable with an ``.apply_to`` attribute so call sites read naturally:
#   A = make_object_splat(N, seed=...);  B = make_object_splat.apply_to(A, M_gt)
make_object_splat = _make_object_splat
make_object_splat.apply_to = _apply_to
