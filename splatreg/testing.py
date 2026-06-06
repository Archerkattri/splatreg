"""Ship the Jacobian-audit discipline as a reusable check.

Every serious geometric-optimisation library checks each analytic Jacobian against a
numerical one (GTSAM `EXPECT_CORRECT_FACTOR_JACOBIANS`, SymForce manifold-native
`NumericalDerivative`, Theseus `autograd.functional.jacobian`). This exposes the same
for splatreg's `Residual` ABC: audit any residual — built-in or user/AI-added — in one
call, so a wrong gradient (which silently corrupts every pose estimate) is caught.

    from splatreg.testing import assert_residual_jacobian
    assert_residual_jacobian(ICP(point_to_plane=False), T, target, source)

The numerical Jacobian is a tangent-space central difference of the right-perturbation
`T @ se3_exp(xi)`. It re-evaluates `residual.residual` under the perturbed pose: exact
for field residuals (the SDF); for nearest-neighbour residuals (ICP) a small fraction of
rows near a correspondence-switch boundary may differ, which `max_mismatch` tolerates.
Audit in float64 for the tightest comparison.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import torch

from .core.lie import se3_exp

__all__ = ["numerical_jacobian", "check_residual_jacobian", "assert_residual_jacobian"]


def numerical_jacobian(
    residual_fn: Callable[[torch.Tensor], torch.Tensor], T: torch.Tensor, *, eps: float = 1e-6, dof: int = 6
) -> torch.Tensor:
    """Central-difference Jacobian of ``residual_fn(T_4x4) -> (n,)`` w.r.t. the
    right-perturbation tangent ``T @ se3_exp(xi)`` (xi = [tx,ty,tz, rx,ry,rz])."""
    cols = []
    for i in range(int(dof)):
        e = torch.zeros(int(dof), dtype=T.dtype, device=T.device)
        e[i] = eps
        cols.append((residual_fn(T @ se3_exp(e)) - residual_fn(T @ se3_exp(-e))) / (2.0 * eps))
    return torch.stack(cols, dim=1)


def check_residual_jacobian(
    residual: Any, T: torch.Tensor, target: Any, source: Any, *, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(per_row_max_error, analytic, numerical)`` for a residual at pose ``T``."""
    J_an = residual.jacobian(T, target, source).detach()
    Jn = numerical_jacobian(
        lambda Tp: residual.residual(Tp, target, source).detach(), T, eps=eps, dof=int(J_an.shape[1])
    )
    return (J_an - Jn).abs().max(dim=1).values, J_an, Jn


def assert_residual_jacobian(
    residual: Any,
    T: torch.Tensor,
    target: Any,
    source: Any,
    *,
    atol: float = 1e-4,
    eps: float = 1e-6,
    max_mismatch: float = 0.02,
) -> float:
    """Assert ``residual.jacobian`` matches the numerical Jacobian to ``atol`` for at least
    ``(1 - max_mismatch)`` of rows (a small NN-switch boundary fraction is tolerated for
    correspondence residuals). Returns the max per-row error. Raises ``AssertionError``."""
    row_err, _, _ = check_residual_jacobian(residual, T, target, source, eps=eps)
    bad = float((row_err > atol).float().mean().item())
    if bad > max_mismatch:
        raise AssertionError(
            f"{type(residual).__name__}.jacobian disagrees with numerical: "
            f"{bad * 100:.1f}% of rows exceed atol={atol:g} "
            f"(max row error {row_err.max().item():.2e})."
        )
    return float(row_err.max().item())
