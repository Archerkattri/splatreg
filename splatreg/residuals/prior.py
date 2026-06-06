"""Pose prior residual — a soft anchor on the pose under optimization.

Ports the "between" prior idea from GaussianFeels (``TemporalBetweenResidual`` /
``ICPBetweenResidual``): penalise how far the current pose ``T`` has moved from a reference
pose ``T_prior`` (a temporal warm-start, an ICP estimate, a centroid-implied pose, …). The
residual is the information-weighted se(3) tangent of the relative pose

    r = W · log( T_prior⁻¹ · T ) ∈ ℝ⁶,    W = diag(1/σ_t, 1/σ_t, 1/σ_t, 1/σ_r, 1/σ_r, 1/σ_r)

with ``log`` in the splatreg convention ``ξ = [tx,ty,tz, rx,ry,rz]`` (translation first).
``dim() == 6``.

Both the inverse and the matrix log are computed inline (self-contained — no GaussianFeels
imports). The analytic Jacobian uses ``∂r/∂ξ ≈ W`` (the right-Jacobian of SE(3)-log evaluated
at the relative pose, approximated by ``I`` near the anchor — the same first-order form the
production ``_BetweenResidual`` ships). This is exact at ``T = T_prior`` and accurate while the
relative rotation stays small, which is the regime a soft prior operates in.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..core.types import Gaussians
from .base import Residual


def _skew3(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros((), device=v.device, dtype=v.dtype)
    return torch.stack(
        [
            torch.stack([z, -v[2], v[1]]),
            torch.stack([v[2], z, -v[0]]),
            torch.stack([-v[1], v[0], z]),
        ]
    )


def _se3_log(T: torch.Tensor) -> torch.Tensor:
    """SE(3) logarithm: 4×4 → 6-vec ``[trans|rot]`` (right-perturbation convention)."""
    dev, dtype = T.device, T.dtype
    R = T[:3, :3]
    t = T[:3, 3]
    trace = ((R[0, 0] + R[1, 1] + R[2, 2]) - 1.0) * 0.5
    trace_c = trace.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(trace_c)
    theta_s = theta.clamp_min(1e-12)
    sin_t = torch.sin(theta_s)
    cos_t = torch.cos(theta_s)
    w_axis = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * sin_t)
    w = theta_s * w_axis
    W = _skew3(w_axis)
    W2 = W @ W
    I3 = torch.eye(3, device=dev, dtype=dtype)
    sin_t_safe = sin_t.clamp_min(1e-12)
    V_inv = I3 - (theta_s * 0.5) * W + (1.0 - theta_s * (1.0 + cos_t) / (2.0 * sin_t_safe)) * W2
    v = V_inv @ t
    near_zero = theta < 1e-10
    w = torch.where(near_zero, torch.zeros_like(w), w)
    v = torch.where(near_zero, t, v)
    return torch.cat([v, w])


def _se3_inv(T: torch.Tensor) -> torch.Tensor:
    """Inverse of an SE(3) matrix (LU-based; robust to fp32 drift in R)."""
    return torch.linalg.inv(T)


class Prior(Residual):
    """Soft SE(3) pose prior anchoring ``T`` to ``T_prior``.

    Args:
        T_prior: 4×4 reference pose to anchor toward.
        sigma_rot: rotation std (radians); larger ⇒ softer rotation prior.
        sigma_trans: translation std (metres); larger ⇒ softer translation prior.
        weight, robust: forwarded to :class:`Residual`.
    """

    def __init__(
        self,
        T_prior: torch.Tensor,
        sigma_rot: float = 0.1,
        sigma_trans: float = 0.01,
        weight: float = 1.0,
        robust: Optional[Any] = None,
    ):
        super().__init__(weight=weight, robust=robust)
        if T_prior is None:
            raise ValueError("Prior requires a 4x4 T_prior pose")
        self.T_prior = T_prior.to(dtype=torch.float32)
        self.sigma_rot = float(sigma_rot)
        self.sigma_trans = float(sigma_trans)
        dev = self.T_prior.device
        # Information sqrt-weights: [trans×3 | rot×3], matching ξ = [t | r].
        self._info_sqrt = torch.tensor(
            [1.0 / self.sigma_trans] * 3 + [1.0 / self.sigma_rot] * 3,
            device=dev,
            dtype=torch.float32,
        )
        self._J = torch.diag(self._info_sqrt)  # constant analytic Jacobian
        self._T_prior_inv = _se3_inv(self.T_prior)

    def requires(self) -> set:
        return set()

    def residual(self, T: torch.Tensor, target: Gaussians, source: Any) -> torch.Tensor:
        info = self._info_sqrt.to(device=T.device, dtype=T.dtype)
        rel = self._T_prior_inv.to(device=T.device, dtype=T.dtype) @ T
        return _se3_log(rel) * info  # (6,)

    def jacobian(self, T: torch.Tensor, target: Gaussians, source: Any) -> Optional[torch.Tensor]:
        return self._J.to(device=T.device, dtype=T.dtype)  # (6, 6) = diag(info_sqrt)

    def dim(self) -> int:
        return 6
