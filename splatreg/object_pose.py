"""6-DoF object-pose estimation against a known object splat (v0.2).

This is the *known-model* framing of registration, the one FoundationPose / YCB-Video evaluate:
you hold a **canonical model** of an object (here a :class:`~splatreg.core.types.Gaussians` splat,
the inverse of a CAD mesh in the splat world) and an **observation** of that same object in some
scene (a cropped point cloud or splat), and you recover the rigid pose ``T_SO`` that places the
model into the scene/camera frame::

    p_scene  =  T_SO @ p_model            (T_SO is the object's 6-DoF pose in the scene frame)

It reuses the existing :func:`splatreg.register` machinery wholesale — the global init basin-finder,
the ICP + Gaussian-SDF residual stack, and the closed-form-Jacobian LM core — and adds only the two
things the *object-pose* task needs on top of plain registration:

1. a thin API (:func:`estimate_object_pose` / :class:`ObjectPoseEstimator`) that names the inputs
   ``model`` / ``observation`` and returns an :class:`ObjectPose` carrying ``T_SO`` directly, with a
   warm-started multi-frame path (the same object tracked across a video, the FoundationPose regime);
2. the **ADD / ADD-S** metrics every 6-DoF-pose paper reports (Hinterstoisser ADD, its symmetric
   ADD-S variant, and the area-under-curve recall over a distance threshold), so a result can be
   scored against ground truth exactly the way YCB-Video / FoundationPose do.

The pose is SE(3) by default (a rigid object does not change size); ``transform="sim3"`` is exposed
for the case where the observation is at an unknown metric scale (e.g. a monocular reconstruction).

This module adds **no** new solver or residual — it is a task-level wrapper. The honest limit is the
same as :func:`register`: under heavy occlusion / partial views the geometry-only basin can be
ambiguous, which the underlying feature init flags via ``info['ambiguous']``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

import torch

from .api import register
from .core.types import Gaussians, RegisterResult
from .quality import QualityConfig
from .track import make_track_residuals, track

__all__ = [
    "ObjectPose",
    "ObjectPoseEstimator",
    "estimate_object_pose",
    "add_metric",
    "adds_metric",
    "add_auc",
]


@dataclass
class ObjectPose:
    """Result of :func:`estimate_object_pose`.

    ``T_SO`` (4x4) places the canonical ``model`` into the scene/observation frame
    (``p_scene = T_SO @ p_model``). ``scale`` is the recovered similarity scale (``1.0`` for the
    default SE(3)). ``info`` carries the underlying :class:`RegisterResult` diagnostics
    (``cost`` / ``rmse`` / ``ambiguous`` / ``confidence`` when a feature init was used).
    """

    T_SO: torch.Tensor
    scale: float = 1.0
    converged: bool = False
    info: dict = field(default_factory=dict)

    @classmethod
    def _from_register(cls, r: RegisterResult) -> "ObjectPose":
        return cls(T_SO=r.T, scale=r.scale, converged=r.converged, info=dict(r.info))


def estimate_object_pose(
    model: Gaussians,
    observation: Any,
    *,
    init: Union[torch.Tensor, str, None] = "fast",
    transform: str = "se3",
    residuals: Optional[Sequence] = None,
    backend: str = "builtin",
    max_iters: Optional[int] = None,
    quality: Union[str, float, QualityConfig, None] = "full",
) -> ObjectPose:
    """Estimate the 6-DoF pose ``T_SO`` of a known ``model`` splat from an ``observation``.

    Parameters
    ----------
    model : the canonical object splat (the "CAD model" of the splat world), in its own object frame.
    observation : the observed instance — a :class:`~splatreg.core.types.Gaussians` (an observed
        splat / crop) or a :class:`~splatreg.core.types.Frame` carrying a ``point_cloud`` — in the
        scene / camera frame.
    init : coarse-init mode, per :func:`splatreg.register`. Default ``"fast"`` (FPFH + GPU-batched
        RANSAC) finds the rotation basin for an arbitrarily-rotated object; pass a 4x4 to warm-start
        from a prior pose, or ``"global"`` for the blind super-Fibonacci sweep.
    transform : ``"se3"`` (default, rigid 6-DoF — a real object does not change size) or ``"sim3"``
        when the observation is at an unknown metric scale.
    residuals, backend, max_iters, quality : forwarded to :func:`splatreg.register`.

    Returns
    -------
    :class:`ObjectPose` whose ``T_SO`` maps model points into the observation frame.

    Notes
    -----
    Internally this is ``register(target=observation, source=model)`` — the *observation* is the
    target so the recovered transform moves the *model* onto the observation, i.e. exactly the object
    pose ``T_SO`` the caller wants. The honest occlusion / partial-view limit of ``register`` carries
    over: a heavily cropped observation can leave the pose ambiguous, surfaced as ``info['ambiguous']``.
    """
    if not isinstance(model, Gaussians) or len(model) == 0:
        raise ValueError("estimate_object_pose(): `model` must be a non-empty Gaussians.")
    r = register(
        observation,
        model,
        residuals=residuals,
        init=init,
        transform=transform,
        backend=backend,
        max_iters=max_iters,
        quality=quality,
    )
    return ObjectPose._from_register(r)


class ObjectPoseEstimator:
    """Stateful 6-DoF object-pose tracker for a *fixed* known ``model`` (the FoundationPose regime).

    Build once with the canonical ``model`` splat, then call :meth:`estimate` per frame. The first
    frame pays the global/feature init to find the pose basin; every subsequent frame warm-starts
    from the previous pose via the fast :func:`splatreg.track` path (skip-global-init + truncated-SDF
    closed-form LM), so a tracked object updates in a few LM iterations rather than re-searching SO(3).

    Parameters
    ----------
    model : the canonical object splat (held fixed for the estimator's life).
    transform : ``"se3"`` (default) or ``"sim3"``.
    init : the FIRST-frame coarse init mode (per :func:`estimate_object_pose`); later frames ignore it.
    track_iters : LM iterations per warm-started frame (default from :func:`splatreg.track`).
    quality : quality / machine-adaptivity policy for the first-frame :func:`register`.
    """

    def __init__(
        self,
        model: Gaussians,
        *,
        transform: str = "se3",
        init: Union[torch.Tensor, str] = "fast",
        track_iters: int = 4,
        quality: Union[str, float, QualityConfig, None] = "full",
    ):
        if not isinstance(model, Gaussians) or len(model) == 0:
            raise ValueError("ObjectPoseEstimator(): `model` must be a non-empty Gaussians.")
        self.model = model
        self.transform = transform
        self.init = init
        self.track_iters = int(track_iters)
        self.quality = quality
        self._pose: Optional[torch.Tensor] = None
        # Tracking residuals depend only on the (fixed) model surface, so build them ONCE: the SDF
        # residual caches the model's per-anchor normals, which would otherwise be recomputed each
        # frame (a full cdist + per-anchor SVD over the model — the dominant per-frame cost).
        self._track_residuals = make_track_residuals(model)

    @property
    def pose(self) -> Optional[torch.Tensor]:
        """The most recent estimated ``T_SO`` (``None`` before the first :meth:`estimate`)."""
        return self._pose

    def reset(self) -> None:
        """Drop the warm-start so the next :meth:`estimate` re-runs the cold init."""
        self._pose = None

    def estimate(self, observation: Any) -> ObjectPose:
        """Estimate ``T_SO`` for a new observation, warm-started after the first frame."""
        if self._pose is None:
            op = estimate_object_pose(
                self.model,
                observation,
                init=self.init,
                transform=self.transform,
                quality=self.quality,
            )
            self._pose = op.T_SO.detach().clone()
            return op
        # Warm-started frame: track the MODEL onto the observation from the previous pose.
        r = track(
            observation,
            self.model,
            self._pose,
            transform=self.transform,
            iters=self.track_iters,
            residuals=self._track_residuals,
        )
        self._pose = r.T.detach().clone()
        return ObjectPose._from_register(r)


# --------------------------------------------------------------------------- ADD / ADD-S metrics


def _model_points(model: Union[Gaussians, torch.Tensor]) -> torch.Tensor:
    """Pull the (N, 3) model points from a Gaussians (its means) or accept a raw point tensor."""
    if isinstance(model, Gaussians):
        return model.means
    pts = torch.as_tensor(model)
    if pts.dim() != 2 or pts.shape[-1] != 3:
        raise ValueError(f"model points must be (N, 3), got {tuple(pts.shape)}")
    return pts


def _transform_points(pts: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Apply a 4x4 (SE(3) or Sim(3)) transform to (N, 3) points: ``(block @ x) + t``."""
    T = T.to(device=pts.device, dtype=pts.dtype)
    return pts @ T[:3, :3].transpose(-1, -2) + T[:3, 3]


def add_metric(
    model: Union[Gaussians, torch.Tensor], T_pred: torch.Tensor, T_gt: torch.Tensor
) -> float:
    """Hinterstoisser **ADD**: mean model-point distance between the predicted and GT pose.

    ``ADD = mean_i || T_pred · x_i − T_gt · x_i ||`` over the model points ``x_i``. The standard
    non-symmetric pose-error metric (asymmetric objects). Returned in the model's units (metres).
    """
    x = _model_points(model)
    a = _transform_points(x, T_pred)
    b = _transform_points(x, T_gt)
    return float((a - b).norm(dim=-1).mean())


def adds_metric(
    model: Union[Gaussians, torch.Tensor],
    T_pred: torch.Tensor,
    T_gt: torch.Tensor,
    *,
    max_pts: int = 4000,
) -> float:
    """**ADD-S**: symmetric (closest-point) variant of ADD for symmetric objects.

    ``ADD-S = mean_i min_j || T_pred · x_i − T_gt · x_j ||`` — each transformed-by-prediction point is
    matched to its *nearest* GT-transformed model point, so a rotation about a symmetry axis (which
    maps the model onto itself) is not penalised. This is the metric YCB-Video / FoundationPose use
    for symmetric objects. Both sides are deterministically strided to ``max_pts`` to bound the
    pairwise distance. Returned in model units (metres).
    """
    x = _model_points(model)

    def sub(p):
        if p.shape[0] <= max_pts:
            return p
        sel = torch.linspace(0, p.shape[0] - 1, max_pts, device=p.device).round().long()
        return p[sel]

    a = sub(_transform_points(x, T_pred))
    b = sub(_transform_points(x, T_gt))
    d = torch.cdist(a, b)  # (Na, Nb)
    return float(d.min(dim=1).values.mean())


def add_auc(
    errors: Sequence[float], *, max_threshold: float = 0.1, n_steps: int = 1000
) -> float:
    """Area-under-curve of the ADD/ADD-S **accuracy-threshold** recall (the YCB-Video AUC).

    Sweeps a distance threshold ``d`` from 0 to ``max_threshold`` (default 0.1 m = 10 cm, the
    YCB-Video convention), at each ``d`` measures the fraction of poses whose error ``≤ d``, and
    returns the normalised area under that curve in ``[0, 1]`` (×100 gives the reported AUC). A higher
    AUC means more poses are accurate at a tighter threshold.
    """
    errs = torch.as_tensor(list(errors), dtype=torch.float64)
    if errs.numel() == 0:
        return 0.0
    thr = torch.linspace(0.0, float(max_threshold), int(n_steps) + 1, dtype=torch.float64)
    # recall(d) = fraction of errors <= d, evaluated on the threshold grid.
    recall = (errs.unsqueeze(0) <= thr.unsqueeze(1)).double().mean(dim=1)  # (n_steps+1,)
    # Trapezoidal area, normalised by max_threshold so the result is in [0, 1].
    auc = torch.trapz(recall, thr) / float(max_threshold)
    return float(auc.clamp(0.0, 1.0))
