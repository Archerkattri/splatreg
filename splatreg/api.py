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
# Solver backends `register(backend=...)` accepts. 'builtin' is the default closed-form-Jacobian LM
# (fastest); 'pypose'/'theseus' hand the nonlinear solve to an external engine (autodiffed); 'gtsam'
# is a recognised name but honestly not implemented (needs hand-written factor Jacobians).
_BACKENDS = {"builtin", "pypose", "theseus", "gtsam"}

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


def _feature_init(target, source: Any, transform: str, device, dtype) -> tuple[torch.Tensor, dict]:
    """Coarse 4x4 LM seed from :func:`splatreg.align_features.feature_align` + ambiguity info.

    Feature-based (FPFH descriptors + ratio-test mutual-NN + clique-prefiltered RANSAC) coarse init
    designed for partial-overlap scenarios.  Falls back to identity if the feature module is
    unavailable or too few correspondences are found (logged at DEBUG level).

    Returns ``(T, info)`` where ``info`` carries the honest diagnostics from ``feature_align``
    (``n_matches``, ``n_inliers``, ``ambiguous``, ``confidence``, ``ambiguity_deg``).  When the
    overlap does not constrain the pose — the disambiguating feature was cropped away — ``info``
    reports ``ambiguous=True`` / a low ``confidence`` rather than a misleadingly precise pose.
    """
    empty_info = {
        "n_matches": 0,
        "n_inliers": 0,
        "ambiguous": True,
        "confidence": 0.0,
        "ambiguity_deg": 180.0,
    }
    try:
        from splatreg.align_features import feature_align
    except Exception as exc:  # pragma: no cover
        _log.info(
            "init='features' requested but splatreg.align_features is unavailable (%s); "
            "falling back to identity init.",
            exc,
        )
        return _identity(device, dtype), dict(empty_info)
    T, info = feature_align(target, source, transform=transform)
    _log.debug(
        "feature_align: %d matches, %d inliers, ambiguous=%s, confidence=%.2f",
        info.get("n_matches", 0),
        info.get("n_inliers", 0),
        info.get("ambiguous"),
        info.get("confidence", 0.0),
    )
    return T.to(device=device, dtype=dtype), info


def _robust_init(target, source: Any, transform: str, device, dtype) -> tuple[torch.Tensor, dict]:
    """Scale-robust coarse init from :func:`splatreg.align_features.robust_feature_align` + info.

    Uses an Open3D FPFH+RANSAC seed (scale-correct, auto-voxelled — the robustness Open3D itself
    reports on real indoor scans like 3DMatch) refined by splatreg's overlap-aware ICP (and Sim(3)
    scale).  This is the recommended init for real metre-scale partial-overlap scans, where the
    object-tuned ``init="fast"``/``"features"`` FPFH seed collapses.  Falls back to the pure-splatreg
    feature path inside ``robust_feature_align`` when Open3D is unavailable; falls back to identity if
    the feature module itself cannot be imported.
    """
    empty_info = {"voxel": 0.0, "n_corr": 0, "used_open3d": False, "confidence": 0.0}
    try:
        from splatreg.align_features import robust_feature_align
    except Exception as exc:  # pragma: no cover
        _log.info(
            "init='robust' requested but splatreg.align_features is unavailable (%s); "
            "falling back to identity init.",
            exc,
        )
        return _identity(device, dtype), dict(empty_info)
    T, info = robust_feature_align(target, source, transform=transform)
    return T.to(device=device, dtype=dtype), info


def _learned_init(target, source: Any, transform: str, device, dtype) -> tuple[torch.Tensor, dict]:
    """Learned coarse init: pretrained GeoTransformer seed + splatreg overlap-aware refine + info.

    Mirrors :func:`_robust_init` but uses the LEARNED GeoTransformer 3DMatch correspondence model
    (CVPR 2022, ~92 % RR — past the classical FPFH ~77 % ceiling) for the seed, then splatreg's own
    overlap-aware ICP (and Sim(3) scale) on top.  Falls back inside ``learned_feature_align`` to the
    classical ``robust`` seed when GeoTransformer (its module / built CUDA-ext / pretrained weights)
    is unavailable; falls back to identity if the feature module itself cannot be imported.
    """
    empty_info = {
        "voxel": 0.0,
        "n_corr": 0,
        "used_open3d": False,
        "used_learned": False,
        "seed": "none",
        "confidence": 0.0,
    }
    try:
        from splatreg.align_features import learned_feature_align
    except Exception as exc:  # pragma: no cover
        _log.info(
            "init='learned' requested but splatreg.align_features is unavailable (%s); "
            "falling back to identity init.",
            exc,
        )
        return _identity(device, dtype), dict(empty_info)
    T, info = learned_feature_align(target, source, transform=transform)
    return T.to(device=device, dtype=dtype), info


def _mac_init(target, source: Any, transform: str, device, dtype) -> tuple[torch.Tensor, dict]:
    """MAC maximal-clique coarse init from :func:`splatreg.mac.mac_feature_align` + info.

    MAC (Zhang et al., CVPR 2023) replaces RANSAC-style hypothesis generation: a rigidity
    compatibility graph over the FPFH correspondences (SC^2 second-order weighted), maximal
    cliques as consensus hypotheses, a weighted SVD per clique, and the inlier-count winner —
    then the same overlap-aware ICP polish the other registrars use.  Falls back to identity
    (with ``ambiguous=True``) when the feature/mac modules are unavailable or when the
    correspondences carry no consistent consensus (honest ``success=False``, never silently
    wrong).  Needs networkx (``pip install "splatreg[mac]"``).
    """
    empty_info = {
        "n_matches": 0,
        "n_inliers": 0,
        "n_cliques": 0,
        "success": False,
        "ambiguous": True,
        "confidence": 0.0,
    }
    try:
        from splatreg.mac import mac_feature_align
    except Exception as exc:  # pragma: no cover
        _log.info(
            "init='mac' requested but splatreg.mac is unavailable (%s); "
            "falling back to identity init.",
            exc,
        )
        return _identity(device, dtype), dict(empty_info)
    T, info = mac_feature_align(target, source, transform=transform)
    return T.to(device=device, dtype=dtype), info


def _fast_init(target, source: Any, transform: str, device, dtype) -> torch.Tensor:
    """Coarse 4x4 LM seed from the FAST feature path (FPFH + GPU-batched RANSAC), or identity.

    Same FPFH-descriptor + batched-RANSAC registrar as ``init="features"`` (so it inherits the
    feature path's full-rotation robustness) but returns ONLY the 4x4 seed — the caller's LM then
    polishes it.  This is the default ``register`` init: ~20-35 ms versus the ~0.8-1.4 s blind
    super-Fibonacci sweep of ``init="global"``.  Guarded: falls back to ``init="global"`` (then
    identity) if the feature module is unavailable.
    """
    try:
        from splatreg.align_features import feature_align
    except Exception as exc:  # pragma: no cover
        _log.info(
            "init='fast' requested but splatreg.align_features is unavailable (%s); "
            "falling back to init='global'.",
            exc,
        )
        return _global_init(target, source, transform, device, dtype)
    T, _info = feature_align(target, source, transform=transform)
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


def _external_backend(backend: str):
    """Return the external engine's ``solve(T0, residuals, target, source, *, transform, max_iters)``.

    The backend modules import their (optional) engine lazily inside ``solve`` and raise a clear
    ``ImportError`` with the right ``pip install splatreg[<extra>]`` hint if it is missing, so this
    just selects the module. ``register`` has already validated ``backend`` against ``_BACKENDS``.
    """
    if backend == "pypose":
        from .solvers.pypose_backend import solve as _solve
    elif backend == "theseus":
        from .solvers.theseus_backend import solve as _solve
    else:  # pragma: no cover - register() validates backend before calling
        raise ValueError(f"no external backend for {backend!r}")
    return _solve


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
    refine: Optional[str] = None,
    refine_kwargs: Optional[dict] = None,
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

        * ``"fast"`` — the recommended fast coarse-init: FPFH descriptors + GPU-batched 3-point
          RANSAC (:func:`splatreg.align_features.feature_align`), returning ONLY the seed for the LM
          to polish. ~20-35 ms vs the ~0.8-1.4 s blind sweep, with the feature path's full-rotation
          robustness. Falls back to ``"global"`` then identity if the feature module is unavailable.
        * ``"global"`` — coarse-init from :func:`splatreg.align.global_align` (super-Fibonacci
          SO(3) sweep + batched trimmed ICP; robust to noise/outliers and near-symmetric clouds;
          assumes full overlap).
        * ``"features"`` — a complete partial-overlap registrar from
          :func:`splatreg.align_features.feature_align` (FPFH descriptors + ratio-test mutual-NN +
          clique-prefiltered RANSAC + an overlap-aware target->source ICP refine, with an
          overlap-aware *point-to-plane* super-Fibonacci basin-sweep fallback that recovers the pose
          from geometry alone on smooth splat surfaces where the descriptors are non-discriminative).
          Designed for the case where the two clouds see
          different parts of the same object.  It also returns honest diagnostics
          (``info['ambiguous']`` / ``info['confidence']`` / ``info['feature']``): when a crop removes
          the rotation-disambiguating geometry the pose is genuinely unrecoverable and the result is
          flagged rather than silently wrong.  Because the default residual set assumes FULL overlap
          (its ICP would pull a good partial-overlap init off-pose), ``init="features"`` with the
          default residuals returns the feature registration DIRECTLY and skips that LM; pass an
          explicit overlap-safe ``residuals=[...]`` to run the LM seeded from the feature init.
        * ``"robust"`` — scale-robust registrar (:func:`splatreg.align_features.robust_feature_align`):
          an Open3D FPFH+RANSAC seed (scale-correct, auto-voxelled — the ~77 % RR Open3D reports on
          real indoor scans like 3DMatch) refined by splatreg's overlap-aware ICP (+ Sim(3) scale).
          Like ``"features"`` it returns the registration DIRECTLY under the default residuals.
        * ``"learned"`` — LEARNED registrar (:func:`splatreg.align_features.learned_feature_align`):
          the pretrained GeoTransformer 3DMatch correspondence model (CVPR 2022, ~92 % RR — past the
          classical FPFH ~77 % ceiling) supplies the coarse seed, then the SAME overlap-aware ICP
          (+ Sim(3) scale) refine as ``"robust"``.  Falls back to ``"robust"`` (then identity) when
          GeoTransformer's module / built CUDA-ext / pretrained weights are unavailable.
        * ``"mac"`` — MAC maximal-clique registrar (:func:`splatreg.mac.mac_feature_align`;
          Zhang et al., CVPR 2023): instead of RANSAC minimal samples, hypotheses come from the
          **maximal cliques** of an SC²-weighted rigidity-compatibility graph over the FPFH
          correspondences — each clique gets a weighted-SVD pose, the inlier-count winner is
          refit on its consensus set, then the same overlap-aware ICP polish as ``"robust"``.
          Robust to heavily outlier-contaminated correspondence sets (including multi-consensus
          decoys that defeat a greedy prefilter + RANSAC); on an all-outlier set it returns an
          honest ``info['success']=False`` / ``ambiguous=True`` identity instead of a silent
          wrong pose.  Like the other registrars it returns the registration DIRECTLY under the
          default residuals.  Needs networkx (``pip install "splatreg[mac]"``).

        All string forms are guarded — fall back to identity with a logged note if the module
        is unavailable. For ``init="global"`` the chosen transform seeds the LM.
    transform : ``"se3"`` (dof 6) or ``"sim3"`` (dof 7; the scale DoF is solved, autodiffed).
    backend : the solver engine. ``"builtin"`` (DEFAULT, fastest) is splatreg's closed-form-Jacobian
        Levenberg-Marquardt core. ``"pypose"`` / ``"theseus"`` hand the whole nonlinear least-squares
        problem to that external engine instead — they optimise the same right-perturbation tangent
        through splatreg's own ``exp`` (so the recovered SE(3)/Sim(3) pose matches the builtin
        convention) and autodiff the Jacobian, so a user can bring their own solver without writing
        one. Both need the matching optional dependency (``pip install splatreg[pypose|theseus]``) and
        raise a clear ``ImportError`` otherwise. ``"gtsam"`` is recognised but raises
        ``NotImplementedError`` (a factor-graph backend needs hand-written analytic factor Jacobians).
    max_iters : maximum LM iterations. ``None`` (default) takes the value from ``quality``
        (``QualityConfig.max_iters``); an explicit int always wins.
    quality : the quality / machine-adaptivity policy (see :func:`splatreg.quality.resolve_quality`):
        ``"full"`` (DEFAULT — nothing capped, all source anchors, full fidelity), ``"balanced"`` /
        ``"low"`` (bounded sample + tighter chunks), a ``0..1`` float (scaled), or ``"auto"`` (detect
        free GPU/CPU memory and pick the largest sizes that *fit* — full on a big machine, scaled
        down on a small GPU / CPU so it runs without OOM). A :class:`~splatreg.quality.QualityConfig`
        may be passed for full manual control. The Sim(3) autodiff Jacobian is always row-chunked
        (per ``quality``) so peak memory is bounded with **no** quality loss.
    refine : optional OPT-IN second refinement stage run AFTER the geometric solve. The only value
        today is ``"photometric"`` — a short extra LM whose residual renders the SOURCE splat under
        the current ``T`` from a small synthetic camera ring around the target and compares it
        against renders of the TARGET splat from the same cameras
        (:func:`splatreg.residuals.photometric.refine_photometric`; PhotoReg-style, arXiv
        2410.05044, adapted splat-vs-splat so NO real images are needed). Geometric alignment
        leaves a visibly coloured seam that pure geometry cannot see; this stage fixes the part of
        it that is pose error. Requires BOTH splats to carry ``colors`` and gsplat to be installed
        (``pip install "splatreg[render]"``) unless ``refine_kwargs['render_fn']`` supplies a
        custom renderer — checked at call time with a clear ``ImportError``, never at import.
        Works for both ``transform="se3"`` and ``"sim3"``; iteration count defaults to the quality
        policy's ``refine_iters``.
    refine_kwargs : optional dict of keyword overrides for the refine stage (see
        :func:`splatreg.residuals.photometric.refine_photometric`: ``n_views``, ``width`` /
        ``height``, ``radius`` / ``radius_mult``, ``dssim_weight``, ``jac_mode``, ``max_iters``,
        ``render_fn``, ``sh_degree``, ...). Ignored when ``refine`` is ``None``.

    Returns
    -------
    :class:`~splatreg.core.types.RegisterResult` whose ``T`` is the full transform (the similarity
    ``[[s*R, t], [0, 1]]`` for ``transform="sim3"``), ``scale`` the recovered ``s`` (``1.0`` for
    SE(3)), and ``info`` carries ``cost`` / ``n_iters`` / ``rmse`` (filled by
    :func:`~splatreg.solvers.lm.run_lm`) plus the resolved ``quality`` label.
    """
    if transform not in _DOF:
        raise ValueError(f"transform must be one of {sorted(_DOF)}, got {transform!r}")
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}, got {backend!r}")
    if refine not in (None, "photometric"):
        # Validated up-front (fail fast, before the geometric solve burns time).
        raise ValueError(f"refine must be None or 'photometric', got {refine!r}")
    if backend == "gtsam":
        # gtsam is a factor-graph engine that needs hand-written analytic factor Jacobians (its
        # Python `CustomFactor` does not autodiff). Wiring splatreg's residual plugins into gtsam
        # factors is a heavier, separate piece of work than the autodiff backends (pypose/theseus),
        # so it is honestly not implemented rather than faked. Use backend='theseus'/'pypose' for an
        # external solver today, or 'builtin' (the default, fastest, closed-form-Jacobian core).
        raise NotImplementedError(
            "backend='gtsam' is not implemented: gtsam needs hand-written analytic factor Jacobians "
            "(its CustomFactor does not autodiff splatreg's residuals). Use backend='theseus' or "
            "'pypose' (both autodiff) for an external solver, or the default 'builtin'."
        )

    init_tensor = None if isinstance(init, str) else init
    device, dtype = _infer_device_dtype(init_tensor, target, source)
    q = resolve_quality(
        quality,
        device,
        target_anchors=_target_anchor_count(target),
        source_anchors=_target_anchor_count(source),
    )

    # The default residual set (auto ICP + SDF) assumes FULL overlap: its point-to-point ICP matches
    # every SOURCE anchor to the target, so when the target is a PARTIAL crop the anchors over the
    # missing region pull a good init off the true pose.  The ``init="features"`` aligner is itself a
    # complete partial-overlap registrar (FPFH -> RANSAC -> overlap-aware target->source ICP refine ->
    # basin-sweep fallback), so when the caller relies on the default residuals we must NOT run that
    # full-overlap LM on top of it — doing so re-introduces the original partial-overlap failure.
    # We therefore short-circuit: ``init="features"`` + default residuals returns the feature
    # registration directly.  An explicit ``residuals=[...]`` (the caller chose an overlap-safe set)
    # still runs the LM, seeded from the feature init.
    user_residuals = residuals is not None
    if residuals is None:
        residuals = _default_residuals(target, q)

    feature_info: Optional[dict] = None
    if isinstance(init, str):
        if init == "global":
            T0 = _global_init(target, source, transform, device, dtype)
        elif init == "fast":
            T0 = _fast_init(target, source, transform, device, dtype)
        elif init == "features":
            T0, feature_info = _feature_init(target, source, transform, device, dtype)
        elif init == "robust":
            T0, feature_info = _robust_init(target, source, transform, device, dtype)
        elif init == "learned":
            T0, feature_info = _learned_init(target, source, transform, device, dtype)
        elif init == "mac":
            T0, feature_info = _mac_init(target, source, transform, device, dtype)
        else:
            raise ValueError(
                "init string must be 'fast', 'robust', 'learned', 'mac', 'global', or 'features', "
                f"got {init!r}"
            )
    elif init is None:
        # Default to the FAST feature init (FPFH + GPU-batched RANSAC, ~20-35 ms) so a bare
        # register(target, source) is fast out of the box; it self-falls-back to global->identity
        # if the feature module is unavailable.
        T0 = _fast_init(target, source, transform, device, dtype)
    else:
        T0 = init.to(device=device, dtype=dtype)

    feature_only = init in ("features", "robust", "learned", "mac") and not user_residuals
    if feature_only:
        # Return the self-contained feature registration (no full-overlap LM that would corrupt a
        # partial-overlap init).  Recover the scale from the transform block for Sim(3).
        scale = 1.0
        if transform == "sim3":
            det = float(torch.linalg.det(T0[:3, :3]).abs().clamp_min(1e-18).item())
            scale = det ** (1.0 / 3.0)
        result = RegisterResult(T=T0, scale=scale, converged=True, info={})
    else:
        n_iters = q.max_iters if max_iters is None else int(max_iters)
        if backend == "builtin":
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
        else:
            # External-engine backend (pypose / theseus): hand the whole nonlinear problem to the
            # engine. It optimises the same right-perturbation tangent through splatreg's `exp`, so
            # the recovered pose matches the builtin convention; the engine autodiffs the Jacobian
            # (no analytic Jacobian needed). The builtin stays the default + fastest path.
            backend_solve = _external_backend(backend)
            result = backend_solve(
                T0, list(residuals), target, source, transform=transform, max_iters=n_iters
            )
    result.info["quality"] = q.label
    if feature_info is not None:
        # Surface the honest partial-overlap diagnostics on the result so callers can trust (or
        # distrust) the recovered pose.  ``ambiguous=True`` means the overlap did not constrain the
        # pose (the disambiguating feature was cropped away) — the returned T is the best feasible
        # guess but should be treated as unreliable; ``confidence`` (0..1) grades it.
        result.info["feature"] = feature_info
        result.info["ambiguous"] = bool(feature_info.get("ambiguous", False))
        result.info["confidence"] = float(feature_info.get("confidence", 0.0))
    if refine == "photometric":
        result = _refine_photometric_stage(
            result, target, source, transform=transform, quality=q, refine_kwargs=refine_kwargs
        )
    return result


def _refine_photometric_stage(
    result: RegisterResult,
    target,
    source: Any,
    *,
    transform: str,
    quality: QualityConfig,
    refine_kwargs: Optional[dict],
) -> RegisterResult:
    """Run the opt-in PhotoReg-style photometric stage on top of a geometric ``RegisterResult``.

    Seeds :func:`splatreg.residuals.photometric.refine_photometric` with the geometric ``T`` and a
    short iteration budget from the quality policy (``quality.refine_iters``; an explicit
    ``max_iters`` in ``refine_kwargs`` wins). The refined pose/scale replace the geometric ones and
    the stage's LM diagnostics land under ``info["refine"]`` — the geometric diagnostics (cost,
    feature confidence, ...) stay where they were, so callers can see both stages honestly.

    gsplat availability (or a custom ``render_fn``) is enforced inside the refine call — at CALL
    time, with the install hint — keeping ``refine`` honest-optional like every other extra.
    """
    from .residuals.photometric import refine_photometric  # local: keeps gsplat fully optional

    kwargs = dict(refine_kwargs or {})
    kwargs.setdefault("max_iters", quality.refine_iters)
    kwargs.setdefault("jac_row_chunk", quality.jac_row_chunk)
    refined = refine_photometric(target, source, result.T, transform=transform, **kwargs)
    info = dict(result.info)
    info["refine"] = dict(refined.info)
    info["refine"]["converged"] = bool(refined.converged)
    return RegisterResult(T=refined.T, scale=refined.scale, converged=result.converged, info=info)


def _apply_transform_to_gaussians(g: Gaussians, T: torch.Tensor, scale: float = 1.0) -> Gaussians:
    """Return a copy of ``g`` with the SE(3)/Sim(3) transform ``T`` baked into the Gaussians.

    ``T``'s top-left block is ``s * R`` (``s == scale``; ``1.0`` for SE(3)). The rotation is
    de-scaled out so the quaternion stays unit, then::

        means'  = s * (R @ means) + t          (= T applied to the homogeneous point)
        quats'  = quat(R) (x) quats            (compose the pure rotation onto each anchor)
        scales' = s * scales (linear)          (the similarity scales each anchor's extent)
        SH'     = D(R) @ SH                    (real-SH Wigner-D rotation of the colour lobes)

    Opacities carry through unchanged; so do plain ``(N, 3)`` RGB colours (view-independent).
    ``(N, K, 3)`` spherical-harmonic colours are a *function on the view sphere*, so their
    higher-order bands are rotated with the splat via the block-diagonal real-SH Wigner-D matrix
    (:func:`splatreg.sh.rotate_sh`, Ivanic–Ruedenberg recurrence; the DC band is rotation-
    invariant and the Sim(3) scale does not touch colour). For SE(3) (``scale == 1``) the
    geometric update is rigid (scales untouched). Log-scales are handled in log space (``log s``
    added) so the splat's ``log_scales`` flag is preserved either way.
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

    if g.colors is None:
        colors = None
    elif g.colors.dim() == 3 and g.colors.shape[1] > 1:
        # SH stack with higher-order bands: rotate the view-dependent lobes WITH the splat.
        from .sh import rotate_sh  # local import keeps api import-light

        colors = rotate_sh(g.colors, R)
    else:
        colors = g.colors.clone()  # RGB / DC-only: rotation-invariant
    return Gaussians(
        means=means,
        quats=quats,
        scales=scales,
        opacities=g.opacities.clone(),
        colors=colors,
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



def apply_transform(gaussians: Gaussians, T: torch.Tensor, scale: float = 1.0) -> Gaussians:
    """Bake an SE(3)/Sim(3) transform into a splat and return the transformed copy.

    The align-without-merging workflow: :func:`register` two scans, apply the recovered
    transform to the source, and save each scan as its **own** PLY — aligned into a common
    frame but never fused::

        result = register(target, source, transform="sim3")
        save_ply(apply_transform(source, result.T, result.scale), "source_aligned.ply")
        # target.ply stays untouched; the two files now sit registered in any viewer.

    Equivalent to the CLI: ``splatreg align target.ply source.ply -o source_aligned.ply``.
    ``T`` is the 4x4 from :class:`RegisterResult` (top-left block ``scale * R``); pass
    ``result.scale`` for Sim(3) (``1.0`` for SE(3)). Colour is handled fully: DC is
    rotation-invariant, quats are composed, and higher-order SH bands (``f_rest``) are
    Wigner-rotated in the real basis (:mod:`splatreg.sh`) so view-dependent colour turns
    with the splat.
    """
    return _apply_transform_to_gaussians(gaussians, T, scale)

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
    refine: Optional[str] = None,
    refine_kwargs: Optional[dict] = None,
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
    refine : optional opt-in per-pair refinement stage, forwarded to :func:`register`. The merge
        seam is exactly where ``refine="photometric"`` earns its keep: after the geometric solve a
        short photometric LM renders each moving splat against the reference from a synthetic
        camera ring and polishes the pose so the seam *looks* aligned, not just measures aligned
        (PhotoReg-style; needs ``colors`` on the splats + gsplat or a custom render_fn).
    refine_kwargs : keyword overrides for the refine stage, forwarded to :func:`register`.

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
            refine=refine,
            refine_kwargs=refine_kwargs,
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
