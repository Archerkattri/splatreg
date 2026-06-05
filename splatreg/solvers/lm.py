"""Builtin Levenberg-Marquardt registration core.

Two layers, kept separate so a backend can replace either:

* :func:`run_lm` is the *driver*. Each iteration it asks every :class:`~splatreg.residuals.base.Residual`
  for ``(r, J)`` at the current pose (autodiffing ``J`` when the residual returns ``None``), stacks
  them into a :class:`~splatreg.core.types.LinearizedProblem`, hands that to a :class:`Solver` for
  the damped linear step, then applies a right-perturbation update ``T = T @ se3_exp(delta)`` with a
  decoupled per-iteration step clamp, and checks convergence.
* :class:`LevenbergMarquardt` is the *step*. It implements the :class:`Solver` ABC: given the stacked
  ``(J, r, weight)`` it forms the Marquardt-damped normal equations ``(J^T W J + lambda diag) delta =
  -J^T W r`` and returns the tangent :class:`~splatreg.core.types.SE3Update`.

Convention: right-perturbation, tangent ``[tx, ty, tz, rx, ry, rz]`` (Sim(3): + ``log_s``). The style
is the GPU-sync-free, preallocated, analytic-Jacobian-first one ported from the GaussianFeels SE(3)
tracker; autodiff is strictly a fallback for residuals without a hand-written Jacobian.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
from torch.func import jacrev

from ..core.types import LinearizedProblem, RegisterResult, SE3Update
from ..core.lie import se3_exp, sim3_exp, sim3_log
from .base import Solver

DTYPE = torch.float32
_DOF = {"se3": 6, "sim3": 7}
# Reverse-mode `jacrev` materialises the entire (residual-rows x params) autograd graph at once.
# For the Gaussian-SDF residual every row carries a (1 x target-anchors) soft-weight subgraph, so a
# large residual makes the peak memory explode (this is the kind of blow-up that can OOM a box). We
# cap the Jacobian's reverse pass to this many output rows at a time via `jacrev(chunk_size=...)`,
# which chunks the internal vmap over residual rows — bounding the live graph to
# `jac_row_chunk x anchors` regardless of residual length, with a SINGLE forward pass and a result
# numerically IDENTICAL to the unchunked Jacobian (same function, same delta=0 linearisation point).
# This is the default; callers thread an explicit chunk through `run_lm(jac_row_chunk=...)` (e.g. the
# quality / auto-detect policy in `splatreg.quality`). 1024 rows is a good full-quality balance:
# big enough to keep the chunk loop short, small enough to bound peak memory.
_DEFAULT_JAC_ROW_CHUNK = 1024
# Per-transform tangent->matrix exponential. The right-perturbation update and the autodiff
# Jacobian both go through the same exp, so a transform never mixes conventions.
_EXP = {"se3": se3_exp, "sim3": sim3_exp}


def _flatten_rows(r: torch.Tensor, J: torch.Tensor, dof: int):
    """Collapse a residual's leading batch dims so ``r -> (R,)`` and ``J -> (R, dof)``."""
    r = r.reshape(-1)
    J = J.reshape(-1, dof)
    return r, J


def _autodiff_jacobian(
    residual, T: torch.Tensor, target, source, dof: int, exp_fn,
    jac_row_chunk: int = _DEFAULT_JAC_ROW_CHUNK,
) -> torch.Tensor:
    """Right-perturbation Jacobian ``d r / d delta`` at ``delta = 0`` via functorch ``jacrev``.

    Differentiates ``delta -> residual(T @ exp_fn(delta), ...)`` so the autodiffed Jacobian uses
    the exact same right-perturbation convention (and exp) as the matching update step. ``exp_fn`` is
    :func:`se3_exp` for the SE(3) path and :func:`sim3_exp` for Sim(3) — the latter carries the
    scale column ``d r / d rho`` analytically-correctly through ``s = exp(rho)``.

    Memory: ``jacrev`` builds the *whole* (rows x dof) autograd graph in one shot, so for the
    Gaussian-SDF residual — where every residual row owns a (1 x target-anchors) soft-weight subgraph
    — a long residual blows peak memory up. To bound it we pass ``chunk_size=jac_row_chunk``, which
    chunks ``jacrev``'s internal vmap over the residual rows so only ``jac_row_chunk`` rows' reverse
    graph is live at a time (peak ~``jac_row_chunk x anchors``, independent of residual length) while
    keeping a single forward pass. The result is numerically IDENTICAL to the unchunked Jacobian
    (same function, same ``delta=0`` point) — chunking trades nothing but peak memory. ``run_lm``
    threads ``jac_row_chunk`` from the quality policy. Returns ``(..., dim, dof)``, matching the
    analytic-Jacobian shape; ``run_lm`` flattens it afterwards.
    """
    def _r_of_delta(delta: torch.Tensor) -> torch.Tensor:
        return residual.residual(T @ exp_fn(delta), target, source)

    delta0 = torch.zeros(dof, device=T.device, dtype=T.dtype)
    chunk = max(1, int(jac_row_chunk))
    return jacrev(_r_of_delta, chunk_size=chunk)(delta0)


class LevenbergMarquardt(Solver):
    """Damped normal-equation step (the linear core of :func:`run_lm`).

    Stateful only in its damping/anisotropy *configuration*; :meth:`solve` itself allocates the
    6x6 (or 7x7) system fresh per call from the stacked problem. ``rot_damping_mult != 1.0`` applies
    a different Marquardt factor to the rotation channels (less damping -> larger rotation steps
    when translation is already close).
    """

    def __init__(self, damping: float = 1e-4, rot_damping_mult: float = 1.0):
        self.damping = float(damping)
        self.rot_damping_mult = float(rot_damping_mult)

    def solve(self, problem: LinearizedProblem) -> SE3Update:
        """Solve ``(J^T W J + lambda diag(J^T W J)) delta = -J^T W r`` and return the tangent step.

        ``problem.weight`` is per-row sqrt-weight: it is applied to ``J`` and ``r`` here (the
        contract lets a backend either apply or assume pre-weighted inputs; this builtin applies).
        """
        J = problem.J
        r = problem.r
        w = problem.weight
        dof = int(problem.dof)
        device, dtype = J.device, J.dtype

        if w is not None:
            w = w.reshape(-1, 1)
            Jw = J * w
            rw = r * w.reshape(-1)
        else:
            Jw = J
            rw = r

        H = Jw.transpose(-1, -2) @ Jw                 # (dof, dof)
        b = Jw.transpose(-1, -2) @ rw                 # (dof,)

        idx = torch.arange(dof, device=device)
        diag_H = H[idx, idx].clamp_min(1e-12)
        mult = self._damping_mults(dof, device, dtype)
        H_damped = H.clone()
        H_damped[idx, idx] = diag_H * mult
        try:
            delta = torch.linalg.solve(H_damped, -b)
        except torch.linalg.LinAlgError:
            eye = torch.eye(dof, device=device, dtype=dtype)
            delta = torch.linalg.solve(H + (self.damping * 100.0) * eye, -b)

        cost = float(0.5 * (rw * rw).sum().item())
        return SE3Update(delta=delta, cost=cost)

    def _damping_mults(self, dof: int, device, dtype) -> torch.Tensor:
        """Per-channel ``(1 + lambda*mult)`` Marquardt multiplier vector (trans 0:3, rot 3:6)."""
        d = self.damping
        rm = self.rot_damping_mult
        if abs(rm - 1.0) > 1e-9:
            vals = [1.0 + d] * 3 + [1.0 + d * rm] * 3 + [1.0 + d] * (dof - 6)
            return torch.tensor(vals, device=device, dtype=dtype)
        return torch.full((dof,), 1.0 + d, device=device, dtype=dtype)


def _assemble(
    T: torch.Tensor,
    residuals: list,
    target,
    source: Any,
    dof: int,
    exp_fn,
    autodiff_only: bool = False,
    jac_row_chunk: int = _DEFAULT_JAC_ROW_CHUNK,
) -> Optional[LinearizedProblem]:
    """Evaluate every residual at ``T`` and stack into one :class:`LinearizedProblem`.

    Per residual: ``r = res.residual(T, target, source)``; ``J = res.jacobian(...)`` or autodiff if
    ``None``. ``res.weight`` becomes a per-row sqrt-weight (folded with an optional callable
    ``res.robust`` IRLS kernel mapping ``|r| -> sqrt-weight``). Residuals returning an empty tensor
    are skipped. Returns ``None`` if nothing contributed this iteration.

    ``autodiff_only`` forces the autodiff Jacobian for *every* residual, ignoring any analytic
    ``jacobian()``. This is the Sim(3) path: the shipped analytic Jacobians are 6-DoF SE(3)-only
    (no scale column), so trusting them under a 7-DoF solve would silently drop ``d r / d rho``;
    autodiffing ``T @ sim3_exp(delta)`` yields the full, correct 7 columns. ``exp_fn`` is the
    matching tangent->matrix exp for the active transform.
    """
    J_rows = []
    r_rows = []
    w_rows = []
    for res in residuals:
        r = res.residual(T, target, source)
        if r is None or r.numel() == 0:
            continue
        J = None if autodiff_only else res.jacobian(T, target, source)
        if J is None:
            J = _autodiff_jacobian(res, T, target, source, dof, exp_fn, jac_row_chunk)
        r, J = _flatten_rows(r, J, dof)

        sqrt_w = float(res.weight) ** 0.5
        w_row = torch.full((r.shape[0],), sqrt_w, device=r.device, dtype=r.dtype)
        robust = getattr(res, "robust", None)
        if callable(robust):
            # IRLS: robust kernel maps residual magnitude to a multiplicative sqrt-weight.
            w_row = w_row * robust(r.detach().abs())

        J_rows.append(J)
        r_rows.append(r)
        w_rows.append(w_row)

    if not r_rows:
        return None
    return LinearizedProblem(
        J=torch.cat(J_rows, dim=0),
        r=torch.cat(r_rows, dim=0),
        weight=torch.cat(w_rows, dim=0),
        dof=dof,
    )


def run_lm(
    T0: torch.Tensor,
    residuals: Iterable,
    target,
    source: Any,
    *,
    transform: str = "se3",
    solver: Optional[Solver] = None,
    n_iters: int = 20,
    damping: float = 1e-4,
    max_trans_step: float = 0.01,
    max_rot_step: float = 0.15,
    convergence_tol: float = 1e-5,
    jac_row_chunk: int = _DEFAULT_JAC_ROW_CHUNK,
) -> RegisterResult:
    """Levenberg-Marquardt registration of ``source`` onto ``target`` over a list of residuals.

    Each iteration: assemble ``H = sum J^T W J`` / ``b = sum J^T W r`` from every residual (analytic
    Jacobian first, autodiff fallback), take the Marquardt-damped step from ``solver`` (a default
    :class:`LevenbergMarquardt` if ``None``), clamp translation/rotation independently, apply the
    right-perturbation update ``T = T @ se3_exp(delta)``, and stop when the step norm drops below
    ``convergence_tol``.

    Parameters
    ----------
    T0 : (4, 4) initial transform aligning ``source`` to ``target``.
    residuals : iterable of :class:`~splatreg.residuals.base.Residual`.
    target, source : passed straight through to each residual's ``residual`` / ``jacobian``.
    transform : ``"se3"`` (dof 6, analytic-Jacobian-first) or ``"sim3"`` (dof 7; every residual's
        Jacobian is autodiffed through ``T @ sim3_exp(delta)`` so the scale column ``d r / d rho`` is
        exact — the shipped analytic Jacobians are SE(3)-only and would drop it).
    solver : optional :class:`Solver` for the linear step; built from ``damping`` if omitted.
    n_iters, damping, max_trans_step, max_rot_step, convergence_tol : LM hyper-parameters.
    jac_row_chunk : row-chunk for the Sim(3) autodiff Jacobian (``jacrev`` reverse pass). Bounds
        peak autodiff memory to ``~jac_row_chunk x target-anchors`` with a result numerically
        identical to the unchunked Jacobian — quality is unaffected. Threaded from the quality
        policy (see :func:`splatreg.quality.resolve_quality`); defaults to ``_DEFAULT_JAC_ROW_CHUNK``.

    Returns
    -------
    :class:`~splatreg.core.types.RegisterResult` with the refined ``T`` and an ``info`` dict
    holding ``cost`` (final 0.5||Wr||^2), ``cost_history``, ``n_iters``, ``rmse``, and ``dof``.
    For ``transform="sim3"`` the returned ``T`` is the full similarity ``[[s*R, t], [0, 1]]`` and
    ``scale`` is the recovered ``s``; the SE(3) path keeps ``scale == 1``.
    """
    dof = _DOF.get(transform)
    if dof is None:
        raise ValueError(f"transform must be one of {sorted(_DOF)}, got {transform!r}")
    if solver is None:
        solver = LevenbergMarquardt(damping=damping)

    exp_fn = _EXP[transform]
    autodiff_only = transform == "sim3"

    residuals = list(residuals)
    device, dtype = T0.device, T0.dtype
    T = T0.clone()

    # On-device step-clamp scalars (avoid per-iter scalar->tensor syncs).
    mt = torch.tensor(max_trans_step, device=device, dtype=dtype)
    mr = torch.tensor(max_rot_step, device=device, dtype=dtype)
    # Per-iter cap on |rho| (log-scale) step: a runaway scale exponentiates, so a small trust
    # region keeps the similarity well-conditioned while the rotation/translation settle.
    ms = torch.tensor(0.1, device=device, dtype=dtype)
    one = torch.ones((), device=device, dtype=dtype)

    cost_history: list = []
    converged = False
    iters_done = n_iters
    last_cost = float("nan")
    for i in range(n_iters):
        problem = _assemble(T, residuals, target, source, dof, exp_fn, autodiff_only, jac_row_chunk)
        if problem is None:
            iters_done = i
            break

        update = solver.solve(problem)
        delta = update.delta
        last_cost = update.cost
        cost_history.append(last_cost)

        # Per-iter step clamp — translation and rotation decoupled so a large translation
        # gradient (common with ICP-style priors) does not suppress an otherwise reasonable
        # rotation step. The Sim(3) log-scale (7th) channel has its own |rho| <= ms clamp.
        t_norm = delta[:3].norm()
        r_norm = delta[3:6].norm()
        t_scale = torch.where(t_norm > mt, mt / t_norm.clamp_min(1e-12), one)
        r_scale = torch.where(r_norm > mr, mr / r_norm.clamp_min(1e-12), one)
        delta = delta.clone()
        delta[:3] *= t_scale
        delta[3:6] *= r_scale
        if dof == 7:
            delta[6] = delta[6].clamp(-ms, ms)

        T = T @ exp_fn(delta)

        if convergence_tol > 0 and float(delta.norm().item()) < convergence_tol:
            converged = True
            iters_done = i + 1
            break

    rmse = float("nan")
    if problem is not None:
        wr = problem.weight.reshape(-1) * problem.r
        rmse = float((wr * wr).mean().clamp_min(0.0).sqrt().item()) if wr.numel() else float("nan")

    # Recover the similarity scale from the final transform (cube-root of the 3x3 block det); for
    # SE(3) the block is a pure rotation so this is exactly 1.0.
    scale = float(sim3_log(T)[6].exp().item()) if dof == 7 else 1.0

    info = {
        "cost": last_cost,
        "cost_history": cost_history,
        "n_iters": iters_done,
        "rmse": rmse,
        "dof": dof,
        "transform": transform,
    }
    return RegisterResult(T=T, scale=scale, converged=converged, info=info)
