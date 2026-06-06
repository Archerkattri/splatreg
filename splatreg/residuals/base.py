"""The ``Residual`` plugin ABC — splatreg's public extension point ("bring your own residual").

Mirrors a Theseus cost function but specialized to a single SE(3)/Sim(3) pose variable, so a
subclass reads like math instead of factor-graph boilerplate. The optimization variable is the
transform ``T`` that maps ``source`` into ``target``'s frame.

Convention: right-perturbation, ``T_new = T @ exp(xi)``, ``xi = [tx,ty,tz, rx,ry,rz, (log_s)]``.
Implement ``residual``; optionally override ``jacobian`` for an analytic ``d r / d xi`` (else
splatreg autodiffs it). Override ``requires`` to declare needed inputs for validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from ..core.types import Gaussians


class Residual(ABC):
    def __init__(self, weight: float = 1.0, robust: Optional[Any] = None):
        self.weight = float(weight)  # multiplies the residual (sqrt-weight handled by the solver)
        self.robust = robust  # optional robust kernel (Huber/Cauchy), applied by the solver

    @abstractmethod
    def residual(self, T: torch.Tensor, target: "Gaussians", source: Any) -> torch.Tensor:
        """Return ``r`` with shape ``(..., dim)``.

        ``target`` is the reference splat; ``source`` is what is being aligned to it — a
        ``Gaussians`` (splat-to-splat registration) or a ``Frame`` (camera/object tracking),
        depending on the residual. Use ``self.requires()`` to declare which.
        """

    def jacobian(self, T: torch.Tensor, target: "Gaussians", source: Any) -> Optional[torch.Tensor]:
        """Optional analytic Jacobian, shape ``(..., dim, dof)`` wrt the se(3)/sim(3) tangent.

        Return ``None`` to let splatreg autodiff it (functorch ``jacrev``/``vmap``).
        """
        return None

    @abstractmethod
    def dim(self) -> int:
        """Residual output dimension (last axis of ``residual``)."""

    def requires(self) -> set:
        """Inputs this residual needs (e.g. ``{'depth', 'K'}`` or ``{'source_gaussians'}``)."""
        return set()
