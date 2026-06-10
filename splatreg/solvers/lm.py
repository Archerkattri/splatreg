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
    residual,
    T: torch.Tensor,
    target,
    source,
    dof: int,
    exp_fn,
    jac_row_chunk: int = _DEFAULT_JAC_ROW_CHUNK,
) -> torch.Tensor:
    """Right-perturbation Jacobian ``d r / d delta`` at ``delta = 0`` via functorch ``jacrev``.

    Differentiates ``delta -> residual(T @ exp_fn(delta), ...)`` so the autodiffed Jacobian uses
    the exact same right-perturbation convention (and exp) as the matching update step. ``exp_fn`` is
    :func:`se3_exp` for the SE(3) path and :func:`sim3_exp` for Sim(3), the latter carries the
    scale column ``d r / d rho`` analytically-correctly through ``s = exp(rho)``.

    Memory: ``jacrev`` builds the *whole* (rows x dof) autograd graph in one shot, so for the
    Gaussian-SDF residual, where every residual row owns a (1 x target-anchors) soft-weight
    subgraph, a long residual blows peak memory up. To bound it we pass ``chunk_size=jac_row_chunk``, which
    chunks ``jacrev``'s internal vmap over the residual rows so only ``jac_row_chunk`` rows' reverse
    graph is live at a time (peak ~``jac_row_chunk x anchors``, independent of residual length) while
    keeping a single forward pass. The result is numerically IDENTICAL to the unchunked Jacobian
    (same function, same ``delta=0`` point), chunking trades nothing but peak memory. ``run_lm``
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
        # Cache the per-channel Marquardt multiplier vector keyed by (dof, device, dtype). It is a
        # constant of the solver config, but rebuilding it with `torch.tensor(...)` every `solve`
        # (host->device copy) cost ~2 ms/iter on GPU — a pure per-iteration sync stall in the warm-
        # start tracking loop. Built once, reused. (See `_damping_mults`.)
        self._mult_cache: dict = {}

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

        H = Jw.transpose(-1, -2) @ Jw  # (dof, dof)
        b = Jw.transpose(-1, -2) @ rw  # (dof,)

        idx = torch.arange(dof, device=device)
        diag_H = H[idx, idx].clamp_min(1e-12)
        mult = self._damping_mults(dof, device, dtype)
        H_damped = H.clone()
        H_damped[idx, idx] = diag_H * mult
        # `torch.linalg.solve` for a tiny (dof x dof) damped-normal system is robustly invertible
        # here (Marquardt damping keeps it SPD), so we call it directly: the previous try/except on
        # `LinAlgError` forced a host sync every iteration (the exception path must read the device
        # status) — a pure stall in the warm-start tracking loop. A genuinely singular system is
        # already guarded by the damping; on the rare exact-singular case `solve` raises and the
        # caller surfaces it rather than silently re-damping.
        delta = torch.linalg.solve(H_damped, -b)

        # Cost is the 0.5||Wr||^2 at the linearisation point. Keep it on-device (no `.item()` sync in
        # the hot loop); `run_lm` only reads it after the loop for the returned history.
        cost = 0.5 * (rw * rw).sum()
        return SE3Update(delta=delta, cost=cost)

    def _damping_mults(self, dof: int, device, dtype) -> torch.Tensor:
        """Per-channel ``(1 + lambda*mult)`` Marquardt multiplier vector (trans 0:3, rot 3:6).

        Cached per ``(dof, device, dtype)``, it is a constant of the solver config, so rebuilding it
        with a host->device `torch.tensor` copy every iteration was a measurable per-iter sync stall.
        """
        key = (dof, device, dtype)
        cached = self._mult_cache.get(key)
        if cached is not None:
            return cached
        d = self.damping
        rm = self.rot_damping_mult
        if abs(rm - 1.0) > 1e-9:
            vals = [1.0 + d] * 3 + [1.0 + d * rm] * 3 + [1.0 + d] * (dof - 6)
            mult = torch.tensor(vals, device=device, dtype=dtype)
        else:
            mult = torch.full((dof,), 1.0 + d, device=device, dtype=dtype)
        self._mult_cache[key] = mult
        return mult


def _assemble(
    T: torch.Tensor,
    residuals: list,
    target,
    source: Any,
    dof: int,
    exp_fn,
    jac_row_chunk: int = _DEFAULT_JAC_ROW_CHUNK,
) -> Optional[LinearizedProblem]:
    """Evaluate every residual at ``T`` and stack into one :class:`LinearizedProblem`.

    Per residual: ``r = res.residual(T, target, source)``; ``J = res.jacobian(...)`` or autodiff if
    ``None``. ``res.weight`` becomes a per-row sqrt-weight (folded with an optional callable
    ``res.robust`` IRLS kernel mapping ``|r| -> sqrt-weight``). Residuals returning an empty tensor
    are skipped. Returns ``None`` if nothing contributed this iteration.

    Under a 7-DoF Sim(3) solve a 6-DoF SE(3) analytic Jacobian must NOT be trusted, it silently
    drops the scale column ``d r / d rho``. But a residual MAY ship a genuine 7-column analytic
    Jacobian (the SDF residual extends the
    field-gradient chain to the log-scale column in closed form): we detect that by calling
    ``res.jacobian(..., dof=dof)`` and accepting the result ONLY when it has exactly ``dof``
    columns. Anything else (a 6-col SE(3) Jacobian, or ``None``) falls back to autodiffing
    ``T @ sim3_exp(delta)`` for the full, correct 7 columns. ``exp_fn`` is the matching
    tangent->matrix exp for the active transform.
    """
    J_rows = []
    r_rows = []
    w_rows = []
    for res in residuals:
        r = res.residual(T, target, source)
        if r is None or r.numel() == 0:
            continue
        # Try the analytic Jacobian, requesting the active ``dof`` so a residual can supply a
        # 7-column Sim(3) Jacobian when it has one. ``jacobian()`` only accepts ``dof`` if the
        # residual opted in; otherwise call the legacy 6-DoF signature.
        try:
            J = res.jacobian(T, target, source, dof=dof)
        except TypeError:
            J = res.jacobian(T, target, source)
        # Reject a Jacobian that does not carry all ``dof`` columns (e.g. a 6-col SE(3) analytic
        # Jacobian under a 7-DoF Sim(3) solve, which would drop ``d r / d rho``): autodiff instead.
        if J is not None and J.shape[-1] != dof:
            J = None
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
        exact, the shipped analytic Jacobians are SE(3)-only and would drop it).
    solver : optional :class:`Solver` for the linear step; built from ``damping`` if omitted.
    n_iters, damping, max_trans_step, max_rot_step, convergence_tol : LM hyper-parameters.
    jac_row_chunk : row-chunk for the Sim(3) autodiff Jacobian (``jacrev`` reverse pass). Bounds
        peak autodiff memory to ``~jac_row_chunk x target-anchors`` with a result numerically
        identical to the unchunked Jacobian, quality is unaffected. Threaded from the quality
        policy (see :func:`splatreg.quality.resolve_quality`); defaults to ``_DEFAULT_JAC_ROW_CHUNK``.

    Returns
    -------
    :class:`~splatreg.core.types.RegisterResult` with the refined ``T`` and an ``info`` dict
    holding ``cost`` (final 0.5||Wr||^2), ``cost_history``, ``n_iters``, ``rmse``, ``dof``, plus
    the pose-uncertainty pair for pose-graph / loop-closure use:

    * ``info["information"]``, the UNDAMPED Gauss-Newton information matrix ``JᵀWJ`` at the
      final accepted linearisation (``(6, 6)`` for SE(3); ``(7, 7)`` for Sim(3), log-scale
      channel last, the tangent ordering ``[tx, ty, tz, rx, ry, rz, (log_s)]``).
    * ``info["covariance"]``, its inverse scaled by the unbiased residual variance
      ``σ̂² = ||Wr||² / (R − dof)`` (the classic NLLS pose covariance: 2x the residual noise
      means ~4x the covariance). ``None`` when ``JᵀWJ`` is singular (under-constrained pose,
      inspect ``information``'s null space instead).

    Both are torch tensors on the solve device. For ``transform="sim3"`` the returned ``T`` is
    the full similarity ``[[s*R, t], [0, 1]]`` and ``scale`` is the recovered ``s``; the SE(3)
    path keeps ``scale == 1``.
    """
    dof = _DOF.get(transform)
    if dof is None:
        raise ValueError(f"transform must be one of {sorted(_DOF)}, got {transform!r}")
    if solver is None:
        solver = LevenbergMarquardt(damping=damping)

    exp_fn = _EXP[transform]

    residuals = list(residuals)
    device, dtype = T0.device, T0.dtype
    T = T0.clone()

    # On-device step-clamp scalars (avoid per-iter scalar->tensor syncs). Built in ONE host->device
    # copy (a single small CPU tensor moved once) rather than three separate `torch.tensor(scalar,
    # device=cuda)` calls — each of those is its own host->device transfer/sync, and at a few LM
    # iters per frame the three-per-call setup cost was a measurable slice of a warm-start track.
    # `ms` caps the Sim(3) |rho| (log-scale) step: a runaway scale exponentiates, so a small trust
    # region keeps the similarity well-conditioned while the rotation/translation settle.
    _clamps = torch.tensor([max_trans_step, max_rot_step, 0.1, 1.0], dtype=dtype, device="cpu").to(device)
    mt, mr, ms, one = _clamps[0], _clamps[1], _clamps[2], _clamps[3]

    cost_history: list = []
    converged = False
    iters_done = n_iters
    last_cost = float("nan")
    for i in range(n_iters):
        problem = _assemble(T, residuals, target, source, dof, exp_fn, jac_row_chunk)
        if problem is None:
            iters_done = i
            break

        update = solver.solve(problem)
        delta = update.delta
        cost_history.append(update.cost)  # on-device tensor; materialised to float after the loop

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

    # Materialise the costs to Python floats ONCE, after the loop — `solve` now returns the cost as
    # an on-device tensor so the iteration never pays a `.item()` sync. (A single sync here is fine.)
    cost_history = [float(c.item()) if torch.is_tensor(c) else float(c) for c in cost_history]
    last_cost = cost_history[-1] if cost_history else float("nan")

    rmse = float("nan")
    information = None
    covariance = None
    if problem is not None:
        wr = problem.weight.reshape(-1) * problem.r
        rmse = float((wr * wr).mean().clamp_min(0.0).sqrt().item()) if wr.numel() else float("nan")
        # Pose information / covariance from the FINAL ACCEPTED LINEARISATION (the last assembled
        # JᵀWJ — at convergence the linearisation point and the returned T differ by a step below
        # `convergence_tol`, so this is the Gauss-Newton information at the solution without
        # paying an extra residual+Jacobian pass in the hot loop). `information` is the UNDAMPED
        # JᵀWJ (dof×dof; 6 for SE(3), 7 for Sim(3) with the log-scale channel last);
        # `covariance` is its inverse scaled by the unbiased residual-variance estimate
        # σ̂² = ||Wr||² / (R − dof) — the classic nonlinear-least-squares pose covariance, so
        # noisier data honestly reports a looser covariance. `covariance` is None when the
        # system is singular (under-constrained problem: trust `information`'s null space).
        if wr.numel():
            Jw = problem.J * problem.weight.reshape(-1, 1)
            information = Jw.transpose(-1, -2) @ Jw
            n_rows = int(problem.r.shape[0])
            sigma2 = float((wr * wr).sum().item()) / max(n_rows - dof, 1)
            try:
                covariance = sigma2 * torch.linalg.inv(information)
                if not bool(torch.isfinite(covariance).all()):
                    covariance = None
            except RuntimeError:  # exactly singular JᵀWJ
                covariance = None

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
        "information": information,
        "covariance": covariance,
    }
    return RegisterResult(T=T, scale=scale, converged=converged, info=info)
