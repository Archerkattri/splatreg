"""Ambiguity-aware pose alternatives and held-out fusion decisions.

Registration residuals and a single optimizer covariance are not enough to
decide whether two independently reconstructed splats should be fused.  This
module keeps a small finite set of candidate poses, clusters genuinely
different poses, and evaluates the chosen action on held-out target anchors.

The result is an evidence object for an application policy.  It is not a
formal bound on rendering quality: the default scorer is geometric nearest
neighbour evidence, and callers must calibrate its threshold on scene-disjoint
data before attaching a risk number.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import torch


def _rotation_part(T: torch.Tensor) -> torch.Tensor:
    block = T[:3, :3]
    scale = torch.linalg.det(block).abs().clamp_min(1e-18).pow(1.0 / 3.0)
    return block / scale


def _rotation_distance_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    rel = _rotation_part(a).double() @ _rotation_part(b).double().transpose(-1, -2)
    cosine = ((torch.trace(rel) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.arccos(cosine)).item())


def _translation_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a[:3, 3].double() - b[:3, 3].double()).item())


def _scale(T: torch.Tensor) -> float:
    return float(torch.linalg.det(T[:3, :3].double()).abs().clamp_min(1e-18).pow(1.0 / 3.0).item())


def _normalise(T: torch.Tensor) -> torch.Tensor:
    if T.shape != (4, 4) or not torch.isfinite(T).all():
        raise ValueError("each pose hypothesis must be a finite 4x4 transform")
    return T


@dataclass(frozen=True)
class PoseCluster:
    members: tuple[int, ...]
    representative: int


@dataclass(frozen=True)
class HypothesisScore:
    index: int
    fit_mean: float
    heldout_mean: float
    heldout_p95: float
    heldout_inlier_rate: float
    n_fit: int
    n_heldout: int
    cluster: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FusionDecision:
    fuse: bool
    selected: int | None
    reason: str
    ambiguity: bool
    clusters: tuple[PoseCluster, ...]
    scores: tuple[HypothesisScore, ...]
    max_heldout_p95: float

    def as_dict(self) -> dict:
        out = asdict(self)
        out["clusters"] = [asdict(cluster) for cluster in self.clusters]
        out["scores"] = [asdict(score) for score in self.scores]
        return out


def cluster_hypotheses(
    hypotheses: Sequence[torch.Tensor],
    *,
    rotation_tol_deg: float = 5.0,
    translation_tol: float = 0.02,
    scale_tol: float = 0.03,
) -> tuple[PoseCluster, ...]:
    """Cluster poses that are equivalent within declared SE(3)/Sim(3) tolerances."""
    if not hypotheses:
        raise ValueError("at least one pose hypothesis is required")
    if rotation_tol_deg < 0 or translation_tol < 0 or scale_tol < 0:
        raise ValueError("pose cluster tolerances must be non-negative")
    poses = [_normalise(T) for T in hypotheses]
    unseen = set(range(len(poses)))
    clusters: list[PoseCluster] = []
    while unseen:
        seed = min(unseen)
        members = {
            j
            for j in unseen
            if _rotation_distance_deg(poses[seed], poses[j]) <= rotation_tol_deg
            and _translation_distance(poses[seed], poses[j]) <= translation_tol
            and abs(_scale(poses[seed]) - _scale(poses[j])) <= scale_tol
        }
        unseen -= members
        clusters.append(PoseCluster(tuple(sorted(members)), seed))
    return tuple(clusters)


def _apply(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return points @ T[:3, :3].transpose(-1, -2) + T[:3, 3]


def _nearest_residual(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.cdist(query, reference).amin(dim=1)


def score_hypotheses(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    hypotheses: Sequence[torch.Tensor],
    clusters: Sequence[PoseCluster] | None = None,
    *,
    inlier_tol: float = 0.02,
) -> tuple[HypothesisScore, ...]:
    """Score source→target poses with deterministic fit/held-out target anchors."""
    if source_points.ndim != 2 or target_points.ndim != 2:
        raise ValueError("source_points and target_points must be rank-2 tensors")
    if source_points.shape[1] != 3 or target_points.shape[1] != 3:
        raise ValueError("source_points and target_points must have shape (N, 3)")
    if min(source_points.shape[0], target_points.shape[0]) < 2:
        raise ValueError("at least two source and target points are required")
    if inlier_tol <= 0 or not math.isfinite(float(inlier_tol)):
        raise ValueError("inlier_tol must be finite and positive")
    poses = [_normalise(T) for T in hypotheses]
    if clusters is None:
        clusters = cluster_hypotheses(poses)
    cluster_for: dict[int, int] = {
        member: cluster_id
        for cluster_id, cluster in enumerate(clusters)
        for member in cluster.members
    }
    # Alternating anchors make the split deterministic and prevent a caller
    # from accidentally selecting the best pose on exactly the same points it
    # later reports as held-out evidence.
    fit = target_points[::2]
    heldout = target_points[1::2]
    if heldout.shape[0] == 0:
        heldout = fit
    scores: list[HypothesisScore] = []
    for index, pose in enumerate(poses):
        transformed = _apply(source_points, pose)
        fit_err = _nearest_residual(fit, transformed)
        held_err = _nearest_residual(heldout, transformed)
        scores.append(
            HypothesisScore(
                index=index,
                fit_mean=float(fit_err.mean().item()),
                heldout_mean=float(held_err.mean().item()),
                heldout_p95=float(torch.quantile(held_err, 0.95).item()),
                heldout_inlier_rate=float((held_err <= inlier_tol).double().mean().item()),
                n_fit=int(fit_err.numel()),
                n_heldout=int(held_err.numel()),
                cluster=int(cluster_for.get(index, -1)),
            )
        )
    return tuple(scores)


def decide_fusion(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    hypotheses: Sequence[torch.Tensor],
    *,
    max_heldout_p95: float,
    ambiguity_margin: float = 0.05,
    inlier_tol: float = 0.02,
    cluster_kwargs: dict | None = None,
) -> FusionDecision:
    """Return a conservative fuse/no-fuse decision from held-out evidence.

    ``max_heldout_p95`` and ``ambiguity_margin`` are policy parameters.  They
    must be selected on independent calibration scenes; this function does not
    turn them into a universal probability guarantee.
    """
    if max_heldout_p95 <= 0 or not math.isfinite(float(max_heldout_p95)):
        raise ValueError("max_heldout_p95 must be finite and positive")
    if ambiguity_margin < 0 or not math.isfinite(float(ambiguity_margin)):
        raise ValueError("ambiguity_margin must be finite and non-negative")
    clusters = cluster_hypotheses(hypotheses, **(cluster_kwargs or {}))
    scores = score_hypotheses(
        source_points, target_points, hypotheses, clusters, inlier_tol=inlier_tol
    )
    ordered = sorted(scores, key=lambda row: (row.heldout_p95, row.heldout_mean, row.index))
    best = ordered[0]
    near_cutoff = max(
        best.heldout_p95 * (1.0 + ambiguity_margin),
        float(inlier_tol) * 0.05,
        1e-12,
    )
    near = [
        row
        for row in ordered[1:]
        if row.heldout_p95 <= near_cutoff
        and row.cluster != best.cluster
    ]
    ambiguous = bool(near)
    if best.heldout_p95 > max_heldout_p95:
        return FusionDecision(
            False, None, "heldout_geometry_exceeds_policy", False, tuple(clusters), scores, max_heldout_p95
        )
    if ambiguous:
        return FusionDecision(
            False, None, "multiple_pose_clusters_fit_heldout_evidence", True,
            tuple(clusters), scores, max_heldout_p95
        )
    return FusionDecision(
        True, best.index, "single_pose_cluster_passes_heldout_geometry", False,
        tuple(clusters), scores, max_heldout_p95
    )


def calibrate_p95_threshold(errors: Sequence[float], *, alpha: float = 0.1) -> float:
    """Fit a finite-sample order-statistic threshold for held-out p95 errors.

    This returns a calibration threshold only.  Its interpretation depends on
    independent calibration units and the caller's declared exchangeability;
    scene/view pixels are not interchangeable independent samples.
    """
    if not errors:
        raise ValueError("at least one calibration error is required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    values = torch.tensor(list(errors), dtype=torch.float64)
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("calibration errors must be finite and non-negative")
    rank = min(values.numel(), max(1, math.ceil((values.numel() + 1) * (1.0 - alpha))))
    return float(torch.sort(values).values[rank - 1].item())
