"""ICP residual — align ``source`` points to ``target``'s nearest Gaussian surface.

Each ``source`` point ``p_s`` is mapped into ``target``'s frame by the pose under
optimization, ``p = R·p_s + t`` (right-perturbation ``T_new = T · exp(ξ)``). Its nearest
``target`` Gaussian supplies a surface point ``q`` (the Gaussian mean) and a unit surface
normal ``n`` (the Gaussian's thinnest axis — the eigenvector of the smallest scale, which
for a surface-fit splat points across the surface). The residual is then

    point-to-plane (default):  r_i = nᵀ (p_i − q_i)            scalar per correspondence
    point-to-point:            r_i = ‖p_i − q_i‖               scalar per correspondence

so ``dim()`` equals the number of correspondences ``n`` and the Jacobian is ``(n, 6)``.

The analytic Jacobian uses the right-perturbation chain rule (porting the SDF/centroid
derivation in GaussianFeels' frozen_object_map.py). With ``p = R·p_s + t`` and the surface
geometry ``(q, n)`` held fixed for the linearization (Gauss–Newton / standard ICP):

    ∂p/∂v = R                              (translation block)
    ∂p/∂ω = −R · [p_s]×                     (rotation block, right-perturbation)

point-to-plane:  ∂r/∂ξ = nᵀ · [ R | −R·[p_s]× ]
point-to-point:  ∂r/∂ξ = ûᵀ · [ R | −R·[p_s]× ],  û = (p − q)/‖p − q‖

The convention ``ξ = [tx,ty,tz, rx,ry,rz]`` (translation first) matches splatreg's contract.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..core.types import Frame, Gaussians
from .base import Residual


def _quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """Batched wxyz unit-quaternion → (N, 3, 3) rotation matrices (gsplat convention)."""
    q = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.empty((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - w * z)
    R[:, 0, 2] = 2.0 * (x * z + w * y)
    R[:, 1, 0] = 2.0 * (x * y + w * z)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - w * x)
    R[:, 2, 0] = 2.0 * (x * z - w * y)
    R[:, 2, 1] = 2.0 * (y * z + w * x)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R


def _gaussian_surface_normals(g: Gaussians) -> torch.Tensor:
    """Unit surface normals from each Gaussian's thinnest axis.

    A surface-fitted 3D Gaussian is a flattened ellipsoid; its smallest-scale principal
    axis is orthogonal to the local surface. We rotate that body axis into world by the
    Gaussian's orientation. Normals are returned un-oriented (sign is resolved per
    correspondence against the point-to-surface vector in :meth:`ICP.residual`).
    """
    scales = g.scales.exp() if g.log_scales else g.scales
    thin_axis = scales.argmin(dim=-1)  # (N,)
    R = _quat_to_rotmat(g.quats.to(dtype=g.means.dtype))  # (N, 3, 3)
    normals = R[torch.arange(R.shape[0], device=R.device), :, thin_axis]  # (N, 3)
    return normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _skew_rows(v: torch.Tensor) -> torch.Tensor:
    """Batched skew-symmetric matrices, ``v`` is (N, 3) → (N, 3, 3)."""
    n = v.shape[0]
    z = torch.zeros(n, device=v.device, dtype=v.dtype)
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    return torch.stack(
        [
            torch.stack([z, -vz, vy], dim=1),
            torch.stack([vz, z, -vx], dim=1),
            torch.stack([-vy, vx, z], dim=1),
        ],
        dim=1,
    )


class ICP(Residual):
    """Point-to-plane (default) / point-to-point ICP between ``source`` points and ``target``.

    The ``source`` is a :class:`Gaussians` (its means are the points) or a :class:`Frame`
    carrying ``point_cloud``. The ``target`` is the reference :class:`Gaussians`; its means
    are the candidate surface points and its thinnest-axis directions are the surface normals.
    Correspondences are nearest-neighbour by Euclidean distance, recomputed at the current
    ``T`` (standard ICP relinearization); pairs beyond ``max_correspondence_dist`` are dropped.

    Args:
        point_to_plane: ``True`` (default) for ``nᵀ(p−q)``; ``False`` for point-to-point.
        max_correspondence_dist: gate on the post-transform NN distance (metres). ``0`` keeps all.
        weight, robust: forwarded to :class:`Residual` (sqrt-weight / robust kernel handled by the solver).
    """

    def __init__(
        self,
        point_to_plane: bool = True,
        max_correspondence_dist: float = 0.0,
        weight: float = 1.0,
        robust: Optional[Any] = None,
    ):
        super().__init__(weight=weight, robust=robust)
        self.point_to_plane = bool(point_to_plane)
        self.max_correspondence_dist = float(max_correspondence_dist)
        self._n_corr = 0  # last correspondence count, exposed via dim()

    def requires(self) -> set:
        return {"source_gaussians"}

    # ── source point extraction ──────────────────────────────────────────────────

    @staticmethod
    def _source_points(source: Any) -> torch.Tensor:
        if isinstance(source, Gaussians):
            return source.means
        if isinstance(source, Frame):
            if source.point_cloud is None:
                raise ValueError("ICP source Frame has no point_cloud")
            return source.point_cloud
        if torch.is_tensor(source):
            return source
        raise TypeError("ICP source must be a Gaussians, a Frame with point_cloud, or an (N,3) tensor")

    def _correspondences(self, T: torch.Tensor, target: Gaussians, source: Any):
        """Return ``(p, q, n, src_pts, keep)`` at the current pose.

        ``p`` = transformed source points, ``q`` = matched target means, ``n`` = matched
        unit normals, ``src_pts`` = the original (untransformed) source points kept,
        ``keep`` = the boolean gate mask applied (for diagnostics / consistency).
        """
        src = self._source_points(source).to(device=target.means.device, dtype=target.means.dtype)
        R = T[:3, :3].to(device=src.device, dtype=src.dtype)
        t = T[:3, 3].to(device=src.device, dtype=src.dtype)
        p = (R @ src.T).T + t  # (N, 3)

        tgt = target.means.to(device=src.device, dtype=src.dtype)  # (M, 3)
        dists = torch.cdist(p, tgt)  # (N, M)
        min_dist, nn = dists.min(dim=1)  # (N,), (N,)

        if self.max_correspondence_dist > 0.0:
            keep = min_dist <= self.max_correspondence_dist
        else:
            keep = torch.ones_like(min_dist, dtype=torch.bool)

        q = tgt[nn]  # (N, 3)
        normals = _gaussian_surface_normals(target).to(device=src.device, dtype=src.dtype)
        n = normals[nn]  # (N, 3)
        # Orient each normal to face from the surface toward the (transformed) source point,
        # so the point-to-plane residual sign is consistent regardless of stored normal sign.
        sign = torch.sign(((p - q) * n).sum(dim=-1, keepdim=True))
        sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
        n = n * sign

        p, q, n, src = p[keep], q[keep], n[keep], src[keep]
        return p, q, n, src, keep

    # ── residual / jacobian ──────────────────────────────────────────────────────

    def residual(self, T: torch.Tensor, target: Gaussians, source: Any) -> torch.Tensor:
        p, q, n, _src, _keep = self._correspondences(T, target, source)
        self._n_corr = int(p.shape[0])
        if p.shape[0] == 0:
            return p.new_zeros(0)
        if self.point_to_plane:
            return ((p - q) * n).sum(dim=-1)  # (n,)
        return (p - q).norm(dim=-1)  # (n,)

    def jacobian(self, T: torch.Tensor, target: Gaussians, source: Any) -> Optional[torch.Tensor]:
        p, q, n, src, _keep = self._correspondences(T, target, source)
        self._n_corr = int(p.shape[0])
        if p.shape[0] == 0:
            return p.new_zeros(0, 6)

        R = T[:3, :3].to(device=p.device, dtype=p.dtype)
        # ∂p/∂v = R ; ∂p/∂ω = −R·[src]×    (right-perturbation: T·exp(ξ), step in source frame)
        skew = _skew_rows(src)  # (n, 3, 3)
        J_trans = R.unsqueeze(0).expand(p.shape[0], 3, 3)  # (n, 3, 3)
        J_rot = -(J_trans @ skew)  # (n, 3, 3)
        J_point = torch.cat([J_trans, J_rot], dim=2)  # (n, 3, 6)

        if self.point_to_plane:
            row = n  # ∂r/∂p = nᵀ
        else:
            row = (p - q) / (p - q).norm(dim=-1, keepdim=True).clamp_min(1e-12)  # ∂r/∂p = ûᵀ
        return (row.unsqueeze(1) @ J_point).squeeze(1)  # (n, 6)

    def dim(self) -> int:
        return self._n_corr
