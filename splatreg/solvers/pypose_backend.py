"""PyPose Levenberg-Marquardt backend for :func:`splatreg.register` (``backend="pypose"``).

Solves the SAME SE(3)/Sim(3) registration problem as the builtin core, but hands each nonlinear
least-squares step to `PyPose <https://pypose.org>`_'s ``pp.optim.LM`` (trust-region LM with its own
damping strategy and linear solver). splatreg supplies only the residual, PyPose autodiffs the
Jacobian, so a user can swap in PyPose's solver without writing a single Jacobian.

Design
------
The optimisation variable is the tangent ``delta`` (6-vector SE(3) / 7-vector Sim(3)), applied as a
right-perturbation ``T0 @ exp(delta)`` so the convention is bit-for-bit identical to the builtin LM
and every shipped analytic Jacobian. Each iteration takes ONE PyPose LM step on the local tangent,
then re-bases ``T0 <- T0 @ exp(delta*)`` and resets ``delta`` to zero. Taking a single inner step per
re-basing, rather than running PyPose's inner LM to convergence on frozen correspondences, is what
makes a correspondence residual (ICP) converge to machine precision instead of stalling: it matches
the builtin's "re-linearise (and re-match) every step" loop, just with PyPose computing the step.
This vector parameterisation also handles Sim(3) (7th DoF = log-scale) uniformly even though the
engine has no native Sim(3) group, because every step routes through splatreg's own ``exp``.

The builtin closed-form-Jacobian core stays the default and the fastest path; this backend is the
"bring your own solver" seam (PyPose users get their solver, splatreg's residual plugins, one call).
"""

from __future__ import annotations

from typing import Any, List

import torch

from ..core.types import RegisterResult
from ..core.lie import sim3_log
from ._backend_common import dof_for, exp_fn_for, residual_count, stacked_residual

_CONVERGENCE_TOL = 1e-8  # stop when a re-based tangent step's norm drops below this.


def solve(
    T0: torch.Tensor,
    residuals: List[Any],
    target: Any,
    source: Any,
    *,
    transform: str = "se3",
    max_iters: int = 20,
    **_ignored: Any,
) -> RegisterResult:
    """Register ``source`` onto ``target`` with PyPose's LM. Signature mirrors the builtin path.

    ``max_iters`` is the number of LM-stepped re-basing iterations (same role as the builtin
    ``n_iters``); each runs a single ``pp.optim.LM.step``.
    """
    try:
        import pypose as pp
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - import guarded by register() dispatch
        raise ImportError(
            "backend='pypose' needs the optional dependency PyPose. Install it with "
            "`pip install splatreg[pypose]` (or `pip install pypose`)."
        ) from exc

    dof = dof_for(transform)
    exp_fn = exp_fn_for(transform)
    device, dtype = T0.device, T0.dtype
    residuals = list(residuals)
    base_T = T0.clone()

    class _PoseResidual(nn.Module):
        """``forward() -> stacked weighted residual`` at ``base @ exp(delta)``; delta is the param."""

        def __init__(self) -> None:
            super().__init__()
            self.delta = nn.Parameter(torch.zeros(dof, device=device, dtype=dtype))
            self.base = base_T

        def forward(self, _input: Any = None) -> torch.Tensor:
            return stacked_residual(self.base @ exp_fn(self.delta), residuals, target, source)

    # Nothing to optimise if no residual fires at the init.
    if residual_count(base_T, residuals, target, source) == 0:
        return _result(base_T, transform, converged=False, info={"backend": "pypose", "n_iters": 0})

    model = _PoseResidual()
    step_norm = float("inf")
    converged = False
    iters_done = 0
    for i in range(max(1, int(max_iters))):
        model.base = base_T
        with torch.no_grad():
            model.delta.zero_()
        # One LM step on the current correspondences, then re-base — mirrors the builtin's
        # re-linearise-every-iteration loop, which is what drives ICP to machine precision.
        optimizer = pp.optim.LM(model, min=1e-8)
        optimizer.step(input=None)
        with torch.no_grad():
            step = model.delta.detach()
            base_T = base_T @ exp_fn(step)
            step_norm = float(step.norm().item())
        iters_done = i + 1
        if step_norm < _CONVERGENCE_TOL:
            converged = True
            break

    return _result(
        base_T,
        transform,
        converged=converged,
        info={"backend": "pypose", "n_iters": iters_done, "final_step_norm": step_norm},
    )


def _result(T: torch.Tensor, transform: str, *, converged: bool, info: dict) -> RegisterResult:
    scale = float(sim3_log(T)[6].exp().item()) if transform == "sim3" else 1.0
    return RegisterResult(T=T, scale=scale, converged=converged, info=info)
