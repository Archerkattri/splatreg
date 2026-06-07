"""Core data contracts for splatreg.

Pure-python + torch tensors. NOTHING here imports gsplat/CUDA — these types are the stable
interface every module (solver, residuals, io, api) agrees on. Do not add heavy deps to this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class Gaussians:
    """A 3D Gaussian Splat, gsplat-compatible. All tensors share a device.

    Convention: ``quats`` are wxyz; ``scales`` are linear unless ``log_scales`` is True;
    ``colors`` are either (N,3) RGB or (N,K,3) SH coefficients.
    """

    means: torch.Tensor  # (N, 3)
    quats: torch.Tensor  # (N, 4) wxyz
    scales: torch.Tensor  # (N, 3)
    opacities: torch.Tensor  # (N,) or (N, 1)
    colors: Optional[torch.Tensor] = None  # (N, 3) RGB or (N, K, 3) SH
    log_scales: bool = False

    @property
    def device(self) -> torch.device:
        return self.means.device

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def to(self, device) -> "Gaussians":
        f = lambda t: t.to(device) if t is not None else None
        return Gaussians(
            f(self.means), f(self.quats), f(self.scales), f(self.opacities), f(self.colors), self.log_scales
        )


@dataclass
class Frame:
    """A single observation (for the camera/object tracking modes).

    For splat-to-splat registration the 'source' is another ``Gaussians``, not a Frame —
    residuals receive whichever is relevant via the ``source`` argument.
    """

    rgb: Optional[torch.Tensor] = None  # (H, W, 3)
    depth: Optional[torch.Tensor] = None  # (H, W)
    K: Optional[torch.Tensor] = None  # (3, 3) intrinsics
    mask: Optional[torch.Tensor] = None  # (H, W) bool
    point_cloud: Optional[torch.Tensor] = None  # (M, 3) points (cam/world frame)


@dataclass
class LinearizedProblem:
    """Assembled normal-equation inputs handed to a Solver backend.

    ``dof`` is 6 for SE(3) or 7 for Sim(3) (the 7th is log-scale). ``weight`` holds per-row
    sqrt-weights (a backend may apply them or assume J/r are already weighted — see the builtin LM).
    """

    J: torch.Tensor  # (R, dof) stacked Jacobian
    r: torch.Tensor  # (R,) stacked residual
    weight: torch.Tensor  # (R,) per-row sqrt weight
    dof: int = 6


@dataclass
class SE3Update:
    """A tangent-space step returned by a Solver (se(3) or sim(3))."""

    delta: torch.Tensor  # (dof,) tangent  [tx,ty,tz, rx,ry,rz, (log_s)]
    # float, or an on-device 0-dim tensor (the LM hot loop keeps cost on-GPU to avoid a per-iter
    # `.item()` sync; run_lm materialises it once after the loop). Either is accepted.
    cost: "float | torch.Tensor" = 0.0


@dataclass
class RegisterResult:
    """Output of ``register`` / ``Tracker.track``.

    ``T`` is the 4x4 transform aligning ``source`` to ``target`` (rotation*scale | translation
    for Sim(3); plain SE(3) when ``transform='se3'``). ``info`` carries diagnostics
    (per-iter cost, rmse, overlap, n_iters, timings, residual breakdown).
    """

    T: torch.Tensor  # (4, 4)
    scale: float = 1.0
    converged: bool = False
    info: dict = field(default_factory=dict)
