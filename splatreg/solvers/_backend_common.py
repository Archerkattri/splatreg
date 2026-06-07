"""Shared glue for the external-engine solver backends (pypose / theseus / gtsam).

The builtin LM (:mod:`splatreg.solvers.lm`) assembles a :class:`LinearizedProblem` per iteration
from each residual's analytic-or-autodiff Jacobian and takes a damped normal-equation step. An
external backend instead runs its *own* nonlinear least-squares loop over splatreg's residuals, so
all it needs from splatreg is:

* a callable ``T(delta) -> residual_vector`` that evaluates every residual at the right-perturbed
  pose ``T0 @ exp(delta)`` and stacks them with each residual's sqrt-weight folded in — IDENTICAL
  rows to what :func:`splatreg.solvers.lm._assemble` would build, minus the Jacobian (the backend
  autodiffs / finite-differences its own);
* the matching tangent->matrix exp (:func:`se3_exp` / :func:`sim3_exp`) so the backend optimises in
  the same right-perturbation coordinates splatreg uses everywhere.

Parameterising by the tangent ``delta`` (re-based on a fixed ``T0`` per outer round) — rather than a
full pose manifold variable — keeps the convention bit-for-bit with the builtin core and works
uniformly for SE(3) (6 DoF) and Sim(3) (7 DoF, the 7th = log-scale) even though some engines lack a
native Sim(3) group. The outer re-basing (``T0 <- T0 @ exp(delta*)``) lets the local tangent
coordinate track large motion across rounds.
"""

from __future__ import annotations

from typing import Any, Callable, List

import torch

from ..core.lie import se3_exp, sim3_exp

_EXP = {"se3": se3_exp, "sim3": sim3_exp}
_DOF = {"se3": 6, "sim3": 7}


def exp_fn_for(transform: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Tangent->4x4 exponential for the active transform (right-perturbation convention)."""
    try:
        return _EXP[transform]
    except KeyError as exc:  # pragma: no cover - guarded upstream in register()
        raise ValueError(f"transform must be one of {sorted(_EXP)}, got {transform!r}") from exc


def dof_for(transform: str) -> int:
    return _DOF[transform]


def stacked_residual(
    T: torch.Tensor,
    residuals: List[Any],
    target: Any,
    source: Any,
) -> torch.Tensor:
    """Evaluate every residual at pose ``T`` and stack into one weighted ``(R,)`` vector.

    Each residual's rows are scaled by ``sqrt(weight)`` (and its optional ``robust`` IRLS kernel,
    matching :func:`splatreg.solvers.lm._assemble`) so that minimising ``||stacked||^2`` is exactly
    the builtin objective ``sum_i weight_i * ||r_i||^2``. Residuals returning ``None`` / empty are
    skipped. Returns a zero-length tensor when nothing contributes (the backend treats that as "no
    constraints this round" and stops).
    """
    rows: List[torch.Tensor] = []
    for res in residuals:
        r = res.residual(T, target, source)
        if r is None or r.numel() == 0:
            continue
        r = r.reshape(-1)
        sqrt_w = float(res.weight) ** 0.5
        w_row = torch.full((r.shape[0],), sqrt_w, device=r.device, dtype=r.dtype)
        robust = getattr(res, "robust", None)
        if callable(robust):
            w_row = w_row * robust(r.detach().abs())
        rows.append(r * w_row)
    if not rows:
        return torch.empty(0, device=T.device, dtype=T.dtype)
    return torch.cat(rows, dim=0)


def residual_count(
    T: torch.Tensor,
    residuals: List[Any],
    target: Any,
    source: Any,
) -> int:
    """Number of stacked residual rows at ``T`` (engines that pre-allocate need this up front)."""
    return int(stacked_residual(T, residuals, target, source).shape[0])
