"""Warm-start pose tracking, the real-time entry point.

:func:`register` pays for a blind global init (a 256-seed super-Fibonacci SO(3) sweep + trimmed
ICP) that dominates its ~0.8 s (SE3) / ~1.3 s (Sim3) wall time. A *tracker* never does that: it
warm-starts from the previous frame's pose, so the only work left is a handful of closed-form-
Jacobian LM iterations from a pose that is already a few degrees / millimetres off. This module is
that fast path.

:func:`track` SKIPS the global init entirely, seeds the LM at ``prior_T`` (the last estimate), and
runs a few iterations of the same builtin Levenberg-Marquardt core (:mod:`splatreg.solvers.lm`) the
full registrar uses, so it inherits the EXACT closed-form SDF gradient (``residuals/sdf.py`` +
``geometry/gaussian_sdf.gaussian_sdf_grad``), including the SE(3) and the analytic Sim(3) log-scale
column. The defaults are tuned for tracking, not cold registration:

* **A tight per-query SDF truncation** (``trunc_sigmas``). With a warm start the source points start
  *near* the target surface, so only the handful of anchors within a few sigma matter; the truncated
  closed-form gradient costs ``N*k`` not ``N*M`` and stays exact for the truncated field (no autodiff).
* **Few source points** (``n_points``) and **few LM iterations** (``iters``). A warm start converges
  in 2-4 steps; ~300 anchors over-determine a 6/7-DoF pose comfortably.

This is a stateless function (the caller threads ``prior_T`` -> new ``T`` itself); for a stateful
sweep wrap it or use :class:`splatreg.api.Tracker`. It does not touch :func:`register` / the global
aligners and adds no new public types, it returns the same :class:`RegisterResult`.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from .core.types import Gaussians, RegisterResult
from .solvers.lm import LevenbergMarquardt, run_lm

__all__ = ["track", "make_track_residuals"]

_DOF = {"se3": 6, "sim3": 7}

# Tracking defaults (see the module docstring for the rationale).
# A warm start lands the source points near the target surface, so a tight truncation captures
# the field's full influence on the residual while collapsing the SDF cost to N*k.
_TRACK_TRUNC_SIGMAS = 3.0
# The truncated top-k anchor count per query. Small (vs the registrar's 50) because a warm-started
# query only sees a local surface patch; this is the ``k`` in the N*k SDF cost.
_TRACK_KNN = 16
# ~300 source anchors over-determine a 6/7-DoF pose when warm-started.
_TRACK_N_POINTS = 300
# A warm start converges in a few LM steps. 4 hits the <0.5 deg tracking-accuracy bar on the
# benchmark object (3 leaves a ~0.5 deg steady-state bias); each extra iter is only ~3 ms.
_TRACK_ITERS = 4
# SDF bandwidth as a multiple of the target's median Gaussian footprint (matches api._auto_sdf_sigma
# intent; kept local so track.py does not depend on api.py).
_SDF_SIGMA_SCALE_MULT = 2.0
_SDF_SIGMA_BBOX_FRAC = 1.0e-2
_EPS = 1.0e-12


def _auto_sdf_sigma(target: Gaussians) -> float:
    """Auto SDF bandwidth = ``~2x`` the target's median Gaussian footprint (bbox-frac fallback)."""
    scales = target.scales.exp() if target.log_scales else target.scales
    per_gauss = scales.mean(dim=-1)
    finite = per_gauss[torch.isfinite(per_gauss) & (per_gauss > 0.0)]
    if finite.numel() > 0:
        med = float(finite.median().item())
        if med > 0.0:
            return _SDF_SIGMA_SCALE_MULT * med
    extent = float((target.means.amax(dim=0) - target.means.amin(dim=0)).norm().item())
    return max(_SDF_SIGMA_BBOX_FRAC * extent, _EPS)


def _default_track_residuals(
    target: Gaussians, *, sigma: float, n_points: int, knn: int, trunc_sigmas: float
) -> list:
    """SDF-only tracking residual stack with truncation (cheap closed-form Jacobian, no autodiff).

    Tracking drops the registrar's ICP companion: ICP's per-iteration nearest-neighbour match over
    the full target is exactly the N*M cost the warm-start regime is built to avoid, and with a good
    prior the truncated Gaussian-SDF zero-level-set already pins the pose. One ``SDF`` with a tight
    ``trunc_sigmas`` keeps the whole step on the fast N*k closed-form gradient path.
    """
    from .residuals import SDF

    return [
        SDF(
            sigma=sigma,
            n_points=int(n_points),
            weight=1.0,
            trunc_sigmas=float(trunc_sigmas),
            knn=int(knn),
        )
    ]


def make_track_residuals(
    target: Gaussians,
    *,
    n_points: int = _TRACK_N_POINTS,
    knn: int = _TRACK_KNN,
    trunc_sigmas: float = _TRACK_TRUNC_SIGMAS,
    sigma: Optional[float] = None,
) -> list:
    """Build the default truncated-SDF tracking residual stack ONCE for reuse across frames.

    The target never moves during tracking, so its per-anchor PCA normals are constant; the ``SDF``
    residual caches them on the residual object keyed by target identity. Recomputing them every
    frame (a full ``cdist`` + per-anchor SVD over the target) dominates a warm-started track. A real
    tracker therefore builds this stack ONCE and threads it into :func:`track` via ``residuals=`` (or
    uses :class:`splatreg.api.Tracker`, which already holds a fixed residual list). Returns a list
    with a single tuned :class:`~splatreg.residuals.sdf.SDF`.
    """
    if not isinstance(target, Gaussians) or len(target) == 0:
        raise ValueError("make_track_residuals(): target must be a non-empty Gaussians.")
    sig = float(sigma) if sigma is not None else _auto_sdf_sigma(target)
    return _default_track_residuals(target, sigma=sig, n_points=n_points, knn=knn, trunc_sigmas=trunc_sigmas)


def track(
    target: Gaussians,
    source: Any,
    prior_T: torch.Tensor,
    *,
    transform: str = "se3",
    iters: int = _TRACK_ITERS,
    residuals: Optional[Sequence] = None,
    n_points: int = _TRACK_N_POINTS,
    knn: int = _TRACK_KNN,
    trunc_sigmas: float = _TRACK_TRUNC_SIGMAS,
    sigma: Optional[float] = None,
    damping: float = 1e-4,
    max_trans_step: float = 0.05,
    max_rot_step: float = 0.3,
    solver: Optional[Any] = None,
) -> RegisterResult:
    """Warm-started pose tracking: refine ``prior_T`` to align ``source`` onto ``target``.

    Unlike :func:`splatreg.register`, this performs **no** global init, it seeds the LM directly at
    ``prior_T`` (the previous frame's estimate) and runs ``iters`` closed-form-Jacobian LM steps.
    This is the only regime where sub-frame (<40 ms) tracking is reachable: the blind super-Fibonacci
    SO(3) sweep that dominates ``register`` is skipped entirely.

    Parameters
    ----------
    target : the fixed reference splat (:class:`~splatreg.core.types.Gaussians`).
    source : the moved splat to localise (a ``Gaussians``; a ``Frame`` works if the residuals accept
        one, the default SDF stack requires ``Gaussians``).
    prior_T : ``(4, 4)`` warm-start pose (e.g. last frame's estimate, or a constant-velocity guess).
        For ``transform="sim3"`` it may carry a scale; it is used as-is.
    transform : ``"se3"`` (6 DoF, the default, frame-to-frame rigid-object tracking) or ``"sim3"``
        (7 DoF; the analytic log-scale column is used, no autodiff, since the SDF residual ships a
        7-column Jacobian). NOTE: ``"sim3"`` is ill-conditioned in the sparse/tight-truncation
        tracking regime, the scale DoF needs far more anchor support to be observable, so the
        ``300``-point / ``knn=16`` defaults do NOT converge it. Use ``"se3"`` for tracking (a tracked
        rigid body does not change scale frame-to-frame); only reach for ``"sim3"`` with many more
        ``n_points`` and looser ``trunc_sigmas`` (and expect it to be slower and less accurate).
    iters : LM iterations (default 3, a warm start converges in a couple of steps).
    residuals : explicit residual stack; ``None`` builds a FRESH default truncated SDF-only tracker
        (``SDF(sigma=auto, n_points, trunc_sigmas, knn)``) on every call. For a real per-frame loop
        build the stack ONCE with :func:`make_track_residuals` and pass it here: the target never
        moves, so its per-anchor normals (a full cdist + SVD over the target) are cached on the
        residual and a fresh stack each frame would pay that dominant cost repeatedly.
    n_points, knn, trunc_sigmas, sigma : default-residual knobs (ignored if ``residuals`` is given).
        ``sigma=None`` auto-derives it from the target's median Gaussian footprint. ``trunc_sigmas``
        + ``knn`` set the per-query top-k truncation that makes the SDF Jacobian cost ``N*k``.
    damping, max_trans_step, max_rot_step : LM step controls. The step clamps are looser than the
        registrar's (a warm start takes larger, well-conditioned steps toward a nearby minimum).
    solver : optional :class:`~splatreg.solvers.base.Solver`; a default ``LevenbergMarquardt`` if None.

    Returns
    -------
    :class:`~splatreg.core.types.RegisterResult`, same contract as :func:`register`. The estimated
    ``T`` is the refined pose; the caller threads it back in as the next ``prior_T``.
    """
    if transform not in _DOF:
        raise ValueError(f"transform must be one of {sorted(_DOF)}, got {transform!r}")
    if not isinstance(prior_T, torch.Tensor) or prior_T.shape[-2:] != (4, 4):
        raise ValueError("track(): prior_T must be a (4, 4) tensor (the warm-start pose).")

    device, dtype = prior_T.device, (prior_T.dtype if prior_T.is_floating_point() else torch.float32)
    T0 = prior_T.to(device=device, dtype=dtype)

    if residuals is None:
        if not isinstance(target, Gaussians) or len(target) == 0:
            raise ValueError(
                "track(): the default residual stack needs `target` to be a non-empty Gaussians "
                "(to auto-derive the SDF sigma). Pass an explicit `residuals=[...]` otherwise."
            )
        sig = float(sigma) if sigma is not None else _auto_sdf_sigma(target)
        residuals = _default_track_residuals(
            target, sigma=sig, n_points=n_points, knn=knn, trunc_sigmas=trunc_sigmas
        )

    if solver is None:
        solver = LevenbergMarquardt(damping=damping)

    result = run_lm(
        T0,
        residuals,
        target,
        source,
        transform=transform,
        solver=solver,
        n_iters=int(iters),
        max_trans_step=max_trans_step,
        max_rot_step=max_rot_step,
        # No early-exit convergence test: with a fixed small `iters` the per-iteration
        # `delta.norm().item()` check would force a host<->device sync every step (a pure stall in
        # the warm-start loop) to save at most a fraction of an already-tiny iteration. Run all iters.
        convergence_tol=0.0,
    )
    result.info["mode"] = "track"
    return result
