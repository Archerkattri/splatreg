"""High-level registration API: :func:`register`, :func:`merge`, and :class:`Tracker`.

These wrap the builtin Levenberg-Marquardt core (:mod:`splatreg.solvers.lm`) behind the three verbs
in the package docstring. Everything is self-contained — torch + numpy only — and obeys the
right-perturbation convention ``T_new = T @ se3_exp(delta)``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Union

import torch

from .core.types import Gaussians, Frame, RegisterResult
from .core.lie import se3_exp  # noqa: F401  (re-exported convenience for downstream import)
from .fuse import voxel_dedupe_report, auto_voxel_size, knn_dedupe_report, auto_knn_radius
from .quality import QualityConfig, resolve_quality
from .solvers.base import Solver
from .solvers.lm import LevenbergMarquardt, run_lm

_DOF = {"se3": 6, "sim3": 7}

_log = logging.getLogger("splatreg")

# --- Default-residual constants -----------------------------------------------------------------
# Auto SDF bandwidth = this multiple of the target's median Gaussian scale. The soft sum-of-
# Gaussians SDF wants a sigma near the anchor footprint; ~2x the median scale keeps the field
# smooth without over-blurring (matches the bandwidth that recovered the Phase-2 SDF tests).
_SDF_SIGMA_SCALE_MULT = 2.0
# Fallback bandwidth as a fraction of the target bounding diagonal, used only when the target's
# scales are degenerate (zero / non-finite) so a sigma can still be derived.
_SDF_SIGMA_BBOX_FRAC = 1.0e-2
# Default residual mix: ICP-dominant. The soft-SDF zero-level-set carries a small surface bias
# that can *degrade* an already-good global init, so ICP (weight 1.0) leads and the SDF (0.3) only
# regularises. Point-to-point ICP — robust from the coarse init without trusting per-anchor normals.
_DEFAULT_ICP_WEIGHT = 1.0
_DEFAULT_SDF_WEIGHT = 0.3
# The default Gaussian-SDF residual point-sample size is NOT hardcoded to a low cap — it comes from
# the resolved quality policy (`splatreg.quality`): FULL by default (all source anchors), with the
# Sim(3) autodiff Jacobian row-chunked in `solvers/lm.py` so peak memory stays bounded WITHOUT
# throwing away points. A user can pick `quality="balanced"|"low"|<0..1>` to cap the sample, or
# `quality="auto"` to size it to the detected GPU/CPU memory; an explicit
# `residuals=[SDF(n_points=...)]` overrides everything.
_EPS = 1.0e-12


def _global_init(target, source: Any, transform: str, device, dtype) -> torch.Tensor:
    """Coarse 4x4 LM seed from :func:`splatreg.align.global_align`, or identity if unavailable.

    The ``align`` module is built in parallel, so the import is guarded: when it (or its
    ``global_align`` entry point) is missing the seed falls back to identity with a logged note,
    leaving registration correct (LM just starts cold). ``global_align`` is called with the same
    ``transform`` so a Sim(3) solve can be coarse-seeded with a scale as well as a pose.
    """
    try:
        from splatreg.align import global_align
    except Exception as exc:  # pragma: no cover - align module is optional/in-progress
        _log.info(
            "init='global' requested but splatreg.align.global_align is unavailable (%s); "
            "falling back to identity init.",
            exc,
        )
        return _identity(device, dtype)
    T = global_align(target, source, transform=transform)
    return T.to(device=device, dtype=dtype)


def _feature_init(target, source: Any, transform: str, device, dtype) -> torch.Tensor:
    """Coarse 4x4 LM seed from :func:`splatreg.align_features.feature_align`.

    Feature-based (FPFH-lite descriptors + mutual-NN + RANSAC) coarse init designed for
    partial-overlap scenarios.  Falls back to identity if the feature module is unavailable or
    too few correspondences are found (logged at DEBUG level).  When the RANSAC inlier count is
    low (< 6) the result is a poor init; callers needing reliability should combine this with
    a subsequent ``init="global"`` pass or check the returned inlier count directly via
    :func:`splatreg.align_features.feature_align`.
    """
    try:
        from splatreg.align_features import feature_align
    except Exception as exc:  # pragma: no cover
        _log.info(
            "init='features' requested but splatreg.align_features is unavailable (%s); "
            "falling back to identity init.",
            exc,
        )
        return _identity(device, dtype)
    T, n_inliers = feature_align(target, source, transform=transform)
    _log.debug("feature_align: %d RANSAC inliers", n_inliers)
    return T.to(device=device, dtype=dtype)


def _infer_device_dtype(*objs) -> tuple:
    """Best-effort (device, dtype) from the first tensor-bearing argument; CPU/float32 default."""
    for o in objs:
        t = None
        if isinstance(o, torch.Tensor):
            t = o
        elif isinstance(o, Gaussians):
            t = o.means
        elif isinstance(o, Frame):
            for cand in (o.point_cloud, o.depth, o.rgb, o.K):
                if cand is not None:
                    t = cand
                    break
        if t is not None:
            return t.device, (t.dtype if t.is_floating_point() else torch.float32)
    return torch.device("cpu"), torch.float32


def _identity(device, dtype) -> torch.Tensor:
    return torch.eye(4, device=device, dtype=dtype)


def _auto_sdf_sigma(target: Any) -> float:
    """Auto SDF bandwidth from the target geometry: ``~2x`` its median Gaussian scale.

    Reads the target's per-Gaussian mean linear scale (``exp``-ing log-scales), takes the median
    over the finite, positive ones, and returns ``_SDF_SIGMA_SCALE_MULT`` times it. When the
    scales are degenerate (a geometry-only splat with zero/non-finite scales) it falls back to a
    small fraction of the target's bounding diagonal so a usable, positive sigma is always returned.
    Same units as the target means. Only used to fill the default :class:`SDF` residual; an explicit
    ``residuals=[...]`` chooses its own sigma.
    """
    if not isinstance(target, Gaussians) or len(target) == 0:
        raise ValueError(
            "register(): default residuals need `target` to be a non-empty Gaussians "
            "(to auto-derive the SDF sigma). Pass an explicit `residuals=[...]` otherwise."
        )
    scales = target.scales.exp() if target.log_scales else target.scales
    per_gauss = scales.mean(dim=-1)  # (N,) anchor footprint
    finite = per_gauss[torch.isfinite(per_gauss) & (per_gauss > 0.0)]
    if finite.numel() > 0:
        med = float(finite.median().item())
        if med > 0.0:
            return _SDF_SIGMA_SCALE_MULT * med
    extent = float((target.means.amax(dim=0) - target.means.amin(dim=0)).norm().item())
    return max(_SDF_SIGMA_BBOX_FRAC * extent, _EPS)


def _default_residuals(target: Any, quality: QualityConfig) -> list:
    """The ICP-dominant default residual set for ``register`` / ``merge`` when none is given.

    ``[ICP(point_to_plane=False, weight=1.0), SDF(sigma=auto, weight=0.3, n_points=<quality>)]`` —
    ICP leads so the Gaussian-SDF's small surface bias only regularises an already-good global init
    (verified in Phase 2: an SDF-dominant fine step can pull a good init *off* the true pose).
    ``sigma`` is derived from the target via :func:`_auto_sdf_sigma`. The SDF point sample comes from
    the resolved ``quality``: FULL (``n_points <= 0`` -> all source anchors) by default, a bounded
    sample for ``balanced`` / ``low`` / a 0..1 scale, or a memory-fitted sample for ``auto``. The
    SDF per-query block (``chunk_size``) and the normal-estimation ``knn`` also follow ``quality``.
    Peak memory is bounded regardless by the row-chunked Sim(3) autodiff Jacobian (``solvers/lm.py``).
    """
    from .residuals import ICP, SDF  # local import: residuals are optional plugins

    sigma = _auto_sdf_sigma(target)
    # n_points=0 tells SDF to use ALL source anchors (its `m <= n_points` short-circuit). The
    # QualityConfig stores None for "full"; map that to 0 here.
    n_points = 0 if quality.n_points is None else int(quality.n_points)
    return [
        ICP(point_to_plane=False, weight=_DEFAULT_ICP_WEIGHT),
        SDF(
            sigma=sigma,
            weight=_DEFAULT_SDF_WEIGHT,
            n_points=n_points,
            knn=quality.knn,
            chunk_size=quality.sdf_chunk_size,
        ),
    ]


def _target_anchor_count(target: Any) -> Optional[int]:
    """Target splat Gaussian count (drives the ``quality='auto'`` memory budget), or ``None``."""
    return len(target) if isinstance(target, Gaussians) else None


def register(
    target,
    source: Any,
    *,
    residuals: Optional[Sequence] = None,
    init: Union[torch.Tensor, str, None] = None,
    transform: str = "se3",
    backend: str = "builtin",
    max_iters: Optional[int] = None,
    quality: Union[str, float, QualityConfig, None] = "full",
) -> RegisterResult:
    """Register ``source`` onto ``target`` over a list of residuals, returning the 4x4 transform.

    Parameters
    ----------
    target : the reference (a :class:`~splatreg.core.types.Gaussians`, usually).
    source : what is aligned to ``target`` — a ``Gaussians`` (splat-to-splat) or a ``Frame``
        (tracking); handed straight to each residual.
    residuals : sequence of :class:`~splatreg.residuals.base.Residual`, or ``None`` (default) to
        build an ICP-dominant default set ``[ICP(point_to_plane=False, weight=1.0),
        SDF(sigma=auto, weight=0.3, n_points=<quality>)]`` — the SDF ``sigma`` auto-derived from the
        target geometry (``~2x`` its median Gaussian scale; see :func:`_auto_sdf_sigma`) and the
        sample size / chunking from ``quality``. An explicit list is honoured unchanged (``quality``
        then only sets the autodiff row-chunk). ``None`` requires ``target`` to be a non-empty
        ``Gaussians``.
    init : initial 4x4 transform, or ``None`` for identity, or one of the strings:

        * ``"global"`` — coarse-init from :func:`splatreg.align.global_align` (super-Fibonacci
          SO(3) sweep + batched trimmed ICP; robust to noise/outliers and near-symmetric clouds;
          assumes full overlap).
        * ``"features"`` — coarse-init from :func:`splatreg.align_features.feature_align`
          (FPFH-lite descriptors + mutual-NN correspondences + RANSAC; designed for partial-overlap
          scenarios where the two clouds see different parts of the same object).  Falls back to
          identity when fewer than 6 mutual matches are found (featureless / ambiguous crop).

        Both string forms are guarded — fall back to identity with a logged note if the module
        is unavailable. The chosen transform seeds the LM.
    transform : ``"se3"`` (dof 6) or ``"sim3"`` (dof 7; the scale DoF is solved, autodiffed).
    backend : only ``"builtin"`` is implemented here (the builtin LM). The ``register`` surface
        keeps the argument so external-engine backends can be wired in without changing callers.
    max_iters : maximum LM iterations. ``None`` (default) takes the value from ``quality``
        (``QualityConfig.max_iters``); an explicit int always wins.
    quality : the quality / machine-adaptivity policy (see :func:`splatreg.quality.resolve_quality`):
        ``"full"`` (DEFAULT — nothing capped, all source anchors, full fidelity), ``"balanced"`` /
        ``"low"`` (bounded sample + tighter chunks), a ``0..1`` float (scaled), or ``"auto"`` (detect
        free GPU/CPU memory and pick the largest sizes that *fit* — full on a big machine, scaled
        down on a small GPU / CPU so it runs without OOM). A :class:`~splatreg.quality.QualityConfig`
        may be passed for full manual control. The Sim(3) autodiff Jacobian is always row-chunked
        (per ``quality``) so peak memory is bounded with **no** quality loss.

    Returns
    -------
    :class:`~splatreg.core.types.RegisterResult` whose ``T`` is the full transform (the similarity
    ``[[s*R, t], [0, 1]]`` for ``transform="sim3"``), ``scale`` the recovered ``s`` (``1.0`` for
    SE(3)), and ``info`` carries ``cost`` / ``n_iters`` / ``rmse`` (filled by
    :func:`~splatreg.solvers.lm.run_lm`) plus the resolved ``quality`` label.
    """
    if transform not in _DOF:
        raise ValueError(f"transform must be one of {sorted(_DOF)}, got {transform!r}")
    if backend != "builtin":
        raise NotImplementedError(
            f"backend={backend!r} is not available in the builtin core; only 'builtin' is wired up."
        )

    init_tensor = None if isinstance(init, str) else init
    device, dtype = _infer_device_dtype(init_tensor, target, source)
    q = resolve_quality(
        quality,
        device,
        target_anchors=_target_anchor_count(target),
        source_anchors=_target_anchor_count(source),
    )

    if residuals is None:
        residuals = _default_residuals(target, q)

    if isinstance(init, str):
        if init == "global":
            T0 = _global_init(target, source, transform, device, dtype)
        elif init == "features":
            T0 = _feature_init(target, source, transform, device, dtype)
        else:
            raise ValueError(f"init string must be 'global' or 'features', got {init!r}")
    elif init is None:
        T0 = _identity(device, dtype)
    else:
        T0 = init.to(device=device, dtype=dtype)

    n_iters = q.max_iters if max_iters is None else int(max_iters)
    solver: Solver = LevenbergMarquardt()
    result = run_lm(
        T0,
        residuals,
        target,
        source,
        transform=transform,
        solver=solver,
        n_iters=n_iters,
        jac_row_chunk=q.jac_row_chunk,
    )
    result.info["quality"] = q.label
    return result


def _apply_transform_to_gaussians(g: Gaussians, T: torch.Tensor, scale: float = 1.0) -> Gaussians:
    """Return a copy of ``g`` with the SE(3)/Sim(3) transform ``T`` baked into the Gaussians.

    ``T``'s top-left block is ``s * R`` (``s == scale``; ``1.0`` for SE(3)). The rotation is
    de-scaled out so the quaternion stays unit, then::

        means'  = s * (R @ means) + t          (= T applied to the homogeneous point)
        quats'  = quat(R) (x) quats            (compose the pure rotation onto each anchor)
        scales' = s * scales (linear)          (the similarity scales each anchor's extent)

    Opacities/colors carry through unchanged. For SE(3) (``scale == 1``) this is the rigid update
    (scales untouched). Log-scales are handled in log space (``log s`` added) so the splat's
    ``log_scales`` flag is preserved either way.
    """
    T = T.to(device=g.means.device, dtype=g.means.dtype)
    block = T[:3, :3]
    t = T[:3, 3]
    s = float(scale)
    R = block / s if abs(s - 1.0) > 1e-12 else block  # de-scale to a pure rotation
    means = s * (g.means @ R.transpose(-1, -2)) + t
    quats = _quat_mul_wxyz(_quat_from_matrix_wxyz(R), g.quats)

    if abs(s - 1.0) <= 1e-12:
        scales = g.scales.clone()
    elif g.log_scales:
        scales = g.scales + torch.log(g.scales.new_tensor(s))
    else:
        scales = g.scales * s
    return Gaussians(
        means=means,
        quats=quats,
        scales=scales,
        opacities=g.opacities.clone(),
        colors=None if g.colors is None else g.colors.clone(),
        log_scales=g.log_scales,
    )


def _quat_from_matrix_wxyz(R: torch.Tensor) -> torch.Tensor:
    """Rotation 3x3 -> unit quaternion (w, x, y, z). Branch-free (Shepperd-style)."""
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    t = m00 + m11 + m22
    w = torch.sqrt(torch.clamp(1.0 + t, min=1e-12)) * 0.5
    x = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-12)) * 0.5
    y = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-12)) * 0.5
    z = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-12)) * 0.5
    x = torch.copysign(x, R[2, 1] - R[1, 2])
    y = torch.copysign(y, R[0, 2] - R[2, 0])
    z = torch.copysign(z, R[1, 0] - R[0, 1])
    q = torch.stack([w, x, y, z])
    return q / q.norm().clamp_min(1e-12)


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of wxyz quaternions. ``a`` is (4,); ``b`` is (4,) or (N, 4) -> broadcast."""
    if b.dim() == 1:
        b = b.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    out = torch.stack([w, x, y, z], dim=-1)
    out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return out.squeeze(0) if squeeze else out


def merge(
    gaussians_list: Sequence[Gaussians],
    ref: int = 0,
    *,
    residuals: Optional[Sequence] = None,
    transform: str = "sim3",
    init: Union[torch.Tensor, str, None] = "global",
    dedupe: bool = True,
    dedupe_method: str = "voxel",
    voxel: Optional[float] = None,
    knn_radius: Optional[float] = None,
    max_iters: Optional[int] = None,
    quality: Union[str, float, QualityConfig, None] = "full",
) -> Gaussians:
    """Register every splat onto ``gaussians_list[ref]``, fuse, and return one ``Gaussians``.

    This is the v0.1 headline — a merge that is **not a naive cat**. For each non-reference splat
    it :func:`register`\\ s onto the reference (``init="global"`` so large offsets recover; default
    ICP-dominant ``residuals``; ``transform="sim3"`` so scale differences are absorbed), bakes the
    recovered Sim(3)/SE(3) into that splat's means / quats / scales, concatenates everything, then
    **dedupes the overlap**: a voxel-grid pass (:func:`splatreg.fuse.voxel_dedupe`) keeps the
    highest-opacity Gaussian per occupied voxel, so the double-density seam collapses to single
    density. The reference passes through unregistered. The result drops straight into
    :func:`splatreg.io.save_ply`.

    Parameters
    ----------
    gaussians_list : the splats to merge.
    ref : index of the reference splat (all others register onto it). Default 0.
    residuals : residual list per :func:`register` (``None`` -> the ICP-dominant default set).
    transform : ``"sim3"`` (default; recovers scale) or ``"se3"`` for each pairwise registration.
    init : initial transform / ``"global"`` (default coarse basin-finder) / a 4x4, per
        :func:`register`. ``"global"`` makes the merge robust to large inter-capture offsets.
    dedupe : when ``True`` (default) run the overlap dedupe after concatenation; when ``False`` the
        merge is a registered concatenation (still aligned, but double-density seam).
    dedupe_method : ``"voxel"`` (default) for the voxel-grid pass (:func:`splatreg.fuse.voxel_dedupe`)
        or ``"knn"`` for the cross-splat radius pass (:func:`splatreg.fuse.knn_dedupe`). KNN is
        translation-invariant, so it also removes the boundary-straddling duplicates a voxel grid
        leaves (~16% residual overlap on a registered seam) at the cost of an O(N) chunked
        nearest-neighbour scan. Ignored when ``dedupe=False``.
    voxel : voxel-pass edge in splat units; ``None`` auto-derives it from the anchor spacing (see
        :func:`splatreg.fuse.auto_voxel_size`). Used only when ``dedupe_method="voxel"``.
    knn_radius : KNN-pass suppression radius in splat units; ``None`` auto-derives it (half the
        anchor spacing, :func:`splatreg.fuse.auto_knn_radius`). Used only when ``dedupe_method="knn"``.
    max_iters : LM iterations per pairwise registration. ``None`` (default) takes the value from
        ``quality``; an explicit int wins.
    quality : quality / machine-adaptivity policy applied to every pairwise registration — ``"full"``
        (DEFAULT), ``"balanced"`` / ``"low"``, a ``0..1`` float, or ``"auto"`` (fit detected memory).
        See :func:`register` / :func:`splatreg.quality.resolve_quality`.

    Returns
    -------
    Gaussians : the fused splat. Raises ``ValueError`` on an empty list or out-of-range ``ref``.
    """
    splats = list(gaussians_list)
    if not splats:
        raise ValueError("merge() needs at least one Gaussians")
    if not (0 <= ref < len(splats)):
        raise ValueError(f"ref={ref} out of range for {len(splats)} splats")

    reference = splats[ref]
    pieces = []
    for i, g in enumerate(splats):
        if i == ref:
            pieces.append(g)
            continue
        result = register(
            reference,
            g,
            residuals=residuals,
            init=init,
            transform=transform,
            max_iters=max_iters,
            quality=quality,
        )
        pieces.append(_apply_transform_to_gaussians(g, result.T, result.scale))

    fused = Gaussians(
        means=torch.cat([p.means for p in pieces], dim=0),
        quats=torch.cat([p.quats for p in pieces], dim=0),
        scales=torch.cat([p.scales for p in pieces], dim=0),
        opacities=torch.cat([_as_col(p.opacities) for p in pieces], dim=0),
        colors=_cat_optional([p.colors for p in pieces]),
        log_scales=reference.log_scales,
    )
    if dedupe and len(fused) > 1:
        if dedupe_method not in ("voxel", "knn"):
            raise ValueError(f"dedupe_method must be 'voxel' or 'knn', got {dedupe_method!r}.")
        n_before = sum(len(p) for p in pieces)
        # Size the dedupe from the CLEAN reference splat's anchor spacing (not the merged splat,
        # whose overlap duplicates would corrupt the spacing estimate).
        if dedupe_method == "voxel":
            edge = voxel if voxel is not None else auto_voxel_size(reference)
            fused, used = voxel_dedupe_report(fused, edge)
        else:
            radius = knn_radius if knn_radius is not None else auto_knn_radius(reference)
            fused, used = knn_dedupe_report(fused, radius)
        _log.debug(
            "merge(): %s-dedupe at %.4g kept %d/%d Gaussians", dedupe_method, used, len(fused), n_before
        )
    return fused


def _as_col(o: torch.Tensor) -> torch.Tensor:
    """Flatten opacities to ``(N,)`` so heterogeneous-rank splats concatenate cleanly."""
    return o.reshape(-1)


def _cat_optional(cols):
    """Concatenate colors only if every splat has them; otherwise drop to ``None``."""
    if any(c is None for c in cols):
        return None
    return torch.cat(cols, dim=0)


class Tracker:
    """Stateful pose tracker: fixed ``target`` + residuals, warm-started across frames.

    Hold the reference splat and the residual list once, then call :meth:`track` per frame; each
    call seeds the LM from the previously estimated pose so small inter-frame motion converges in a
    few iterations.

    Parameters
    ----------
    target : the reference (a :class:`~splatreg.core.types.Gaussians`, usually).
    residuals : sequence of :class:`~splatreg.residuals.base.Residual`.
    transform : ``"se3"`` or ``"sim3"``.
    max_iters : LM iterations per :meth:`track` call. ``None`` takes the value from ``quality``.
    init : optional initial 4x4 pose (identity when ``None``).
    quality : quality / machine-adaptivity policy (see
        :func:`splatreg.quality.resolve_quality`). For the tracker the residual stack is fixed by the
        caller, so ``quality`` sets the Sim(3) autodiff row-chunk (peak-memory bound, no quality
        loss) and the default ``max_iters``; it is resolved against the target's device/anchors at
        construction. Default ``"full"``.
    """

    def __init__(
        self,
        target,
        residuals: Sequence,
        *,
        transform: str = "se3",
        max_iters: Optional[int] = None,
        init: Optional[torch.Tensor] = None,
        quality: Union[str, float, QualityConfig, None] = "full",
    ):
        if transform not in _DOF:
            raise ValueError(f"transform must be one of {sorted(_DOF)}, got {transform!r}")
        self.target = target
        self.residuals = list(residuals)
        self.transform = transform
        device, _ = _infer_device_dtype(init, target)
        self.quality: QualityConfig = resolve_quality(
            quality, device, target_anchors=_target_anchor_count(target)
        )
        self.max_iters = self.quality.max_iters if max_iters is None else int(max_iters)
        self.solver: Solver = LevenbergMarquardt()
        self._pose: Optional[torch.Tensor] = None if init is None else init.clone()

    @property
    def pose(self) -> Optional[torch.Tensor]:
        """The most recent estimated 4x4 pose (``None`` before the first :meth:`track`)."""
        return self._pose

    def reset(self, init: Optional[torch.Tensor] = None) -> None:
        """Drop the warm-start so the next :meth:`track` starts from ``init`` (or identity)."""
        self._pose = None if init is None else init.clone()

    def track(self, source_or_frame: Any) -> RegisterResult:
        """Estimate the pose of a new ``source`` (``Gaussians`` or ``Frame``), warm-started.

        Returns the :class:`~splatreg.core.types.RegisterResult`; the estimated ``T`` is retained
        as the warm-start for the next call.
        """
        device, dtype = _infer_device_dtype(self._pose, self.target, source_or_frame)
        T0 = self._pose.to(device=device, dtype=dtype) if self._pose is not None else _identity(device, dtype)
        result = run_lm(
            T0,
            self.residuals,
            self.target,
            source_or_frame,
            transform=self.transform,
            solver=self.solver,
            n_iters=self.max_iters,
            jac_row_chunk=self.quality.jac_row_chunk,
        )
        self._pose = result.T.detach().clone()
        return result
