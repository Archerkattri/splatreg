"""Theseus Levenberg-Marquardt backend for :func:`splatreg.register` (``backend="theseus"``).

Solves the SAME SE(3)/Sim(3) registration problem as the builtin core, but hands each nonlinear
least-squares step to `Theseus <https://sites.google.com/view/theseus-ai>`_'s
``th.LevenbergMarquardt`` (its differentiable nonlinear optimiser). splatreg supplies only the
residual via a single ``AutoDiffCostFunction``, Theseus autodiffs the Jacobian, so a user can
swap in Theseus's solver (and its end-to-end differentiability) without writing a Jacobian.

Design
------
Like the PyPose backend, the optimisation variable is the tangent ``delta`` (6-vector SE(3) /
7-vector Sim(3), Theseus has no native Sim(3) group), applied as a right-perturbation
``T0 @ exp(delta)`` through splatreg's own ``exp`` so the recovered pose matches the builtin path
exactly. Each iteration runs ONE Theseus LM step on the local tangent, then re-bases
``T0 <- T0 @ exp(delta*)`` and resets ``delta`` to zero. The single step per re-basing is what makes
a correspondence residual (ICP) converge to machine precision rather than stalling on frozen
correspondences, it reproduces the builtin's re-linearise (and re-match) every iteration.

Theseus is CPU-clean on this box; it builds an optional CUDA extension for its sparse
``baspacho``/``cusolver`` linear solvers, but the default dense ``CHOLESKY`` solver runs on either
device. The builtin closed-form-Jacobian core stays the default and the fastest path; this backend
is the "bring your own solver" seam.
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
    """Register ``source`` onto ``target`` with Theseus's LM. Signature mirrors the builtin path.

    ``max_iters`` is the number of LM-stepped re-basing iterations (same role as the builtin
    ``n_iters``); each runs a single Theseus LM step.
    """
    try:
        import theseus as th
    except Exception as exc:  # pragma: no cover - import guarded by register() dispatch
        raise ImportError(
            "backend='theseus' needs the optional dependency Theseus. Install it with "
            "`pip install splatreg[theseus]` (or `pip install theseus-ai`)."
        ) from exc

    dof = dof_for(transform)
    exp_fn = exp_fn_for(transform)
    device, dtype = T0.device, T0.dtype
    residuals = list(residuals)
    base_T = T0.clone()

    # Residual dimension is fixed across the run for splat-to-splat (point counts don't change), but
    # NN-based ICP can in principle drop a correspondence; Theseus needs a fixed cost dimension, so we
    # fix it at the init count and pad/clip later builds to it. (For the default residual sets the
    # count is constant.) Bail early if nothing fires.
    dim = residual_count(base_T, residuals, target, source)
    if dim == 0:
        return _result(base_T, transform, converged=False, info={"backend": "theseus", "n_iters": 0})

    # `base` is carried as a Theseus aux Variable so the cost re-reads it after each re-basing without
    # rebuilding the graph; `delta` is the optimisation Vector. Both must match the objective dtype.
    delta_var = th.Vector(dof, name="delta", dtype=dtype)
    base_var = th.Variable(base_T.reshape(1, 4, 4), name="base")
    weight = th.ScaleCostWeight(torch.tensor(1.0, dtype=dtype))

    def err_fn(optim_vars, aux_vars):
        (d,) = optim_vars
        (b,) = aux_vars
        T = b.tensor[0] @ exp_fn(d.tensor[0])
        r = stacked_residual(T, residuals, target, source)
        # Theseus expects a fixed (batch, dim) error; clip/pad to the init dimension if a
        # correspondence count drifted (rare; default residuals are constant-dimension).
        if r.shape[0] != dim:
            fixed = torch.zeros(dim, device=r.device, dtype=r.dtype)
            k = min(dim, r.shape[0])
            fixed[:k] = r[:k]
            r = fixed
        return r.reshape(1, dim)

    # autograd_mode='dense' avoids Theseus's default vmap Jacobian, which cannot run through
    # splatreg's in-place `se3_exp`/`sim3_exp` matrix construction (vmap rejects the `T[:3,:3] = R`
    # write). 'dense' uses torch.autograd.functional.jacobian — slower but convention-exact and the
    # external backend is the non-default path anyway.
    cost = th.AutoDiffCostFunction(
        [delta_var],
        err_fn,
        dim,
        aux_vars=[base_var],
        cost_weight=weight,
        name="splatreg_residual",
        autograd_mode="dense",
    )
    objective = th.Objective(dtype=dtype)
    objective.add(cost)
    # One inner LM step per re-basing; we drive the outer re-basing loop ourselves.
    optimizer = th.LevenbergMarquardt(objective, max_iterations=1, step_size=1.0)
    layer = th.TheseusLayer(optimizer)
    layer.to(device=device, dtype=dtype)

    step_norm = float("inf")
    converged = False
    iters_done = 0
    zero_delta = torch.zeros(1, dof, device=device, dtype=dtype)
    for i in range(max(1, int(max_iters))):
        inputs = {"delta": zero_delta, "base": base_T.reshape(1, 4, 4)}
        sol, _info = layer.forward(
            inputs, optimizer_kwargs={"damping": 1e-4, "verbose": False, "backward_mode": "implicit"}
        )
        with torch.no_grad():
            step = sol["delta"].reshape(-1).to(device=device, dtype=dtype)
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
        info={"backend": "theseus", "n_iters": iters_done, "final_step_norm": step_norm},
    )


def _result(T: torch.Tensor, transform: str, *, converged: bool, info: dict) -> RegisterResult:
    scale = float(sim3_log(T)[6].exp().item()) if transform == "sim3" else 1.0
    return RegisterResult(T=T, scale=scale, converged=converged, info=info)
