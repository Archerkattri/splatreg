"""Feature-based global aligner — partial-overlap robust coarse init.

``feature_align`` is an alternative to :func:`splatreg.align.global_align` designed for
partial-overlap scenarios (two splat captures of the same object that see different parts of it).
The centroid-ICP sweep in ``align.py`` assumes full overlap and fails when 20–60 % of the target
is missing (0/9 in the robustness bench).  Feature-based matching sidesteps that by finding point
pairs that look locally similar before estimating the transform.

Algorithm
---------
1. **Sub-sample** both clouds to a manageable size (strided, deterministic).
2. **Local PCA descriptors** (``_fpfh_lite``): for each anchor compute its ``k``-nearest
   neighbours' point-pair statistics — mean direction histogram, angular spread, PCA eigenvalue
   ratios.  These are FPFH-inspired but do NOT require normals: they are derived entirely from
   positional geometry, so they work on plain Gaussian means with no per-Gaussian metadata.
   Each anchor gets a ``descriptor_dim``-vector; geometrically distinctive anchors (e.g. the
   lobe in the test object) have unique descriptors; flat regions are ambiguous.
3. **Mutual-nearest-neighbour (MNN) correspondences** in feature space: for each source anchor
   find its nearest target descriptor, then verify the match is mutual (reciprocal).  MNN
   rejects false positives aggressively; the residual match set is small and high-confidence.
4. **RANSAC over 3-point samples** (``_ransac_se3``): draw random triplets from the MNN
   matches, fit an SE(3)/Sim(3) with Umeyama, count 3-D inliers.  Keep the hypothesis with the
   most inliers.  3-point sampling is the classical RANSAC minimal set for a rigid body.
5. Return the winner as a 4×4 transform (on the source's device), or fall back to identity
   when fewer than 6 MNN matches exist (insufficient for reliable RANSAC).

Why this helps with partial overlap
------------------------------------
The MNN step in feature space uses ONLY the region that *is* present in both clouds.  Anchors
in the source that sit over the target's missing region will have no mutual match (the target has
no descriptor near them), so they are automatically excluded before RANSAC.  The full-overlap
centroid assumption never enters.

Limitations (honest)
---------------------
* When the crop removes the geometrically distinctive region (e.g. the +x lobe is in the 60 %
  that was removed), the descriptors in the *surviving* region are nearly flat → few or no MNN
  matches → fall back to identity → still fails.  This is *inherently ambiguous*: no descriptor
  method can recover a rotation from a featureless region.  ``feature_align`` correctly reports
  low inlier counts in that case (visible via ``verbose=True``).
* The descriptors are positional only (no colour/opacity).  Real splat captures with colour
  would benefit from adding colour-based features; that is a future extension.
* Pure torch / numpy, CPU-runnable, no additional dependencies.

Symmetric-object improvement in ``align.py``
---------------------------------------------
Two fixes are applied to the existing ``_pca_seed_rotations`` and ``global_align`` in
``splatreg/align.py``:
  * ``_pca_axes_safe``: if the PCA eigenvalue spread is below a threshold (isotropic cloud) the
    PCA axes are arbitrary/unstable, so the PCA seeds are SKIPPED and only the super-Fibonacci
    grid is used.  This addresses RC-S1 from docs/03.
  * ``_pick_winner``: among seeds within ``_SCORE_EPS`` of the best score, pick the one with
    scale closest to 1.0 (stability tie-break) instead of the first index.  Addresses RC-S2.
Those fixes live in ``align.py``; this file only documents their rationale.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .core.types import Gaussians
from .align import _stride_subsample, _umeyama

# ── tuneable defaults ─────────────────────────────────────────────────────────

# Sub-sample sizes.  Keep small for CPU feasibility; partial-overlap scenes
# need denser descriptors than the full-overlap centroid sweep.
_FS_N_SOURCE = 512
_FS_N_TARGET = 512

# k-NN for local descriptors.  Larger k → smoother descriptor, fewer outliers,
# but also blurs sharp features.  12 is a good compromise for objects ~ 1400 pts.
_FS_KNN = 12

# descriptor dimension (set by _fpfh_lite's design — 9 bins; must match _fpfh_lite)
_FS_DESC_DIM = 9

# RANSAC
_FS_RANSAC_ITERS = 512       # number of 3-point hypotheses
_FS_INLIER_TOL = 0.015       # 3-D inlier distance threshold (metres); ~1.5 cm
_FS_MIN_MATCHES = 6          # fewer MNN matches → fall back to identity

# Score epsilon for stability tie-break in global_align (used in align.py)
_SCORE_EPS = 1e-3


# ── local geometric descriptors ──────────────────────────────────────────────


def _knn_indices(pts: torch.Tensor, k: int) -> torch.Tensor:
    """For each point return the indices of its ``k`` nearest neighbours (excluding self).

    Uses chunked ``cdist`` to bound memory.  Returns ``(N, k)`` int64 indices.
    ``pts`` must be on the same device; CPU-friendly (no cuda required).
    """
    N = pts.shape[0]
    k = min(k, N - 1)
    out = pts.new_empty((N, k), dtype=torch.int64)
    step = max(1, int(4_000_000 // max(N, 1)))
    for lo in range(0, N, step):
        hi = min(lo + step, N)
        d = torch.cdist(pts[lo:hi], pts)                  # (chunk, N)
        d[:, lo:hi].fill_diagonal_(float("inf"))          # exclude self
        _, idx = torch.topk(d, k, dim=1, largest=False)
        out[lo:hi] = idx
    return out


def _fpfh_lite(pts: torch.Tensor, k: int = _FS_KNN) -> torch.Tensor:
    """FPFH-inspired positional-only descriptor: ``(N, 9)`` float32 per anchor.

    For each anchor compute:
    * 3 PCA eigenvalue ratios of its k-NN neighbourhood (linearity, planarity, scattering).
    * 3 mean point-pair angle bins (elevation, azimuth mean/std relative to the local frame).
    * 3 mean distance statistics (mean NN dist, std, max/mean ratio).

    All values are in [0, 1] or normalised; concatenated into a 9-vector.
    No normals required.  Degenerate (flat) neighbourhoods get a zero descriptor.

    Args:
        pts: ``(N, 3)`` point cloud on any device/dtype.
        k: neighbourhood size.

    Returns:
        ``(N, 9)`` float32 descriptor matrix on the same device.
    """
    N = pts.shape[0]
    k = min(k, N - 1)
    pts32 = pts.to(torch.float32)
    idx = _knn_indices(pts32, k)               # (N, k)

    # Neighbourhood vectors: (N, k, 3)
    nbrs = pts32[idx]                           # (N, k, 3)
    centred = nbrs - pts32.unsqueeze(1)         # (N, k, 3) — vectors from anchor to neighbours

    # ─── PCA eigenvalue ratios ───────────────────────────────────────────────
    # Covariance (N, 3, 3) — summed outer products, no-bias (divide by k)
    cov = torch.einsum("nki,nkj->nij", centred, centred) / k
    # SVD → singular values (= eigenvalues of a PSD matrix, desc sorted)
    try:
        sv = torch.linalg.svdvals(cov)                  # (N, 3), descending
    except Exception:
        sv = torch.ones(N, 3, device=pts.device, dtype=torch.float32) * (1.0 / 3.0)
    total = sv.sum(dim=1, keepdim=True).clamp_min(1e-10)
    sv_norm = sv / total                                # (N, 3), sums to 1

    # linearity / planarity / scattering  (Weinmann et al., 2014)
    l_feat = (sv_norm[:, 0] - sv_norm[:, 1]).clamp(0.0, 1.0)          # (N,)
    p_feat = (sv_norm[:, 1] - sv_norm[:, 2]).clamp(0.0, 1.0)          # (N,)
    s_feat = sv_norm[:, 2].clamp(0.0, 1.0)                             # (N,)
    pca_feats = torch.stack([l_feat, p_feat, s_feat], dim=1)           # (N, 3)

    # ─── distance statistics ──────────────────────────────────────────────────
    dists = centred.norm(dim=2)                                         # (N, k)
    d_mean = dists.mean(dim=1)                                          # (N,)
    d_std = dists.std(dim=1).clamp_min(0.0)                            # (N,)
    d_max = dists.amax(dim=1)                                           # (N,)
    d_ratio = (d_max / d_mean.clamp_min(1e-10)).clamp(1.0, 5.0) / 5.0  # normalise [0,1]
    # normalise mean/std by their dataset-wide max so values are in [0,1]
    d_mean_n = d_mean / d_mean.amax().clamp_min(1e-10)
    d_std_n = d_std / d_std.amax().clamp_min(1e-10)
    dist_feats = torch.stack([d_mean_n, d_std_n, d_ratio], dim=1)      # (N, 3)

    # ─── angular statistics ───────────────────────────────────────────────────
    # Local frame from the first PCA axis (most-variant direction); project
    # neighbour vectors and compute elevation + azimuth.
    # U[:, :, 0] = principal direction (N, 3) from SVD of cov
    # Use svd for the full rotation matrix
    try:
        U, _, _ = torch.linalg.svd(cov)                # U: (N, 3, 3)
    except Exception:
        U = torch.eye(3, device=pts.device, dtype=torch.float32).unsqueeze(0).expand(N, 3, 3)

    axis1 = U[:, :, 0]                                  # (N, 3) first principal axis
    # elevation: angle between neighbour vector and axis1
    c_norm = centred / (centred.norm(dim=2, keepdim=True).clamp_min(1e-10))  # unit vectors
    elev = torch.einsum("nki,ni->nk", c_norm, axis1).clamp(-1.0, 1.0)       # (N, k) cos θ
    elev_mean = elev.mean(dim=1) * 0.5 + 0.5           # shift to [0, 1]
    elev_std = elev.std(dim=1).clamp(0.0, 1.0)
    # azimuth: angle of the projection onto the plane perpendicular to axis1
    proj_plane = c_norm - elev.unsqueeze(2) * axis1.unsqueeze(1)             # (N, k, 3)
    axis2 = U[:, :, 1]                                  # (N, 3) second principal axis
    az_cos = torch.einsum("nki,ni->nk", proj_plane, axis2).clamp(-1.0, 1.0)
    az_mean = az_cos.mean(dim=1) * 0.5 + 0.5
    ang_feats = torch.stack([elev_mean, elev_std, az_mean], dim=1)           # (N, 3)

    desc = torch.cat([pca_feats, dist_feats, ang_feats], dim=1)              # (N, 9)
    return desc.to(torch.float32)


# ── mutual-nearest-neighbour correspondences ─────────────────────────────────


def _mnn_correspondences(
    desc_src: torch.Tensor,
    desc_tgt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mutual nearest-neighbour matching in feature space.

    Args:
        desc_src: ``(Ns, D)`` source descriptors.
        desc_tgt: ``(Nt, D)`` target descriptors.

    Returns:
        ``(src_idx, tgt_idx)`` int64 tensors of matching anchor indices, both ``(M,)``.
        M is the number of mutual matches; may be 0 if no mutual pairs exist.
    """
    # Pairwise L2 squared distances in feature space
    dist = torch.cdist(desc_src, desc_tgt)           # (Ns, Nt)
    # Source → target nearest
    s2t = dist.argmin(dim=1)                         # (Ns,)
    # Target → source nearest
    t2s = dist.argmin(dim=0)                         # (Nt,)
    # Mutual: src_i -> tgt_j and tgt_j -> src_i
    src_range = torch.arange(desc_src.shape[0], device=desc_src.device)
    mutual = t2s[s2t] == src_range                   # (Ns,) bool
    src_idx = src_range[mutual]
    tgt_idx = s2t[mutual]
    return src_idx, tgt_idx


# ── 3-point RANSAC (SE3 / Sim3) ───────────────────────────────────────────────


def _ransac_se3(
    src_pts: torch.Tensor,
    tgt_pts: torch.Tensor,
    src_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    *,
    n_iters: int = _FS_RANSAC_ITERS,
    inlier_tol: float = _FS_INLIER_TOL,
    with_scale: bool = False,
    rng_seed: int = 42,
) -> tuple[torch.Tensor, int]:
    """RANSAC over 3-point minimal samples; returns best 4×4 transform + inlier count.

    Each iteration draws 3 correspondence pairs uniformly at random, fits an SE(3)/Sim(3)
    via Umeyama (closed-form, exact for 3 points), and counts how many of the *all* MNN
    matches are inliers under ``inlier_tol``.  The hypothesis with the most inliers wins.
    Ties broken by lower residual.

    Args:
        src_pts: ``(Ns, 3)`` sub-sampled source cloud.
        tgt_pts: ``(Nt, 3)`` sub-sampled target cloud.
        src_idx: ``(M,)`` source MNN indices (into src_pts).
        tgt_idx: ``(M,)`` target MNN indices (into tgt_pts).
        n_iters: number of RANSAC hypotheses.
        inlier_tol: 3-D point distance threshold for an inlier.
        with_scale: if True, fit Sim(3) (estimate scale).
        rng_seed: seed for the hypothesis sampler.

    Returns:
        ``(T_4x4, n_inliers)`` — best 4×4 on ``src_pts``'s device, plus the inlier count.
        On failure (M < 3) returns ``(identity, 0)``.
    """
    M = src_idx.shape[0]
    dev = src_pts.device
    dtype = src_pts.dtype
    eye4 = torch.eye(4, device=dev, dtype=dtype)

    if M < 3:
        return eye4, 0

    # Correspondence point sets (float64 for Umeyama stability)
    cs = src_pts[src_idx].double()    # (M, 3) — source points in matched correspondences
    ct = tgt_pts[tgt_idx].double()    # (M, 3) — matching target points

    best_T = eye4.clone()
    best_n = 0
    best_res = float("inf")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(rng_seed)

    for _ in range(n_iters):
        # Sample 3 distinct correspondence indices
        perm = torch.randperm(M, generator=rng)[:3]
        s_s = cs[perm]     # (3, 3) source sample
        s_t = ct[perm]     # (3, 3) target sample

        # Fit similarity / rigid (Umeyama; float64 for SVD precision)
        try:
            s_est, R_est, t_est = _umeyama(s_s, s_t, with_scale)
        except Exception:
            continue

        # Build 4×4 candidate
        T_cand = torch.eye(4, device=dev, dtype=torch.float64)
        T_cand[:3, :3] = s_est * R_est
        T_cand[:3, 3] = t_est

        # Count inliers over all MNN correspondences
        cs_t = cs @ R_est.transpose(-1, -2) * s_est + t_est   # (M, 3)
        res = (cs_t - ct).norm(dim=1)                          # (M,)
        n_in = int((res < inlier_tol).sum().item())
        mean_res = float(res[res < inlier_tol].mean().item()) if n_in > 0 else float("inf")

        if n_in > best_n or (n_in == best_n and mean_res < best_res):
            best_n = n_in
            best_res = mean_res
            best_T = T_cand.to(dtype=dtype, device=dev)

    return best_T, best_n


# ── public entry ──────────────────────────────────────────────────────────────


@torch.no_grad()
def feature_align(
    target: Gaussians,
    source: Gaussians,
    *,
    transform: str = "sim3",
    n_source: int = _FS_N_SOURCE,
    n_target: int = _FS_N_TARGET,
    knn: int = _FS_KNN,
    ransac_iters: int = _FS_RANSAC_ITERS,
    inlier_tol: float = _FS_INLIER_TOL,
    min_matches: int = _FS_MIN_MATCHES,
    verbose: bool = False,
) -> tuple[torch.Tensor, int]:
    """Feature-based coarse global init, robust to partial overlap.

    Computes FPFH-lite positional descriptors, finds mutual-NN correspondences in feature
    space, and runs RANSAC over 3-point samples to recover the coarse SE(3)/Sim(3) transform.
    Falls back to the identity when too few mutual matches are found (e.g. featureless / fully
    ambiguous crop).

    Args:
        target: the reference splat.  Only ``.means`` is read.
        source: the splat to align.  Only ``.means`` is read.
        transform: ``"se3"`` (rigid) or ``"sim3"`` (similarity, estimates scale).
        n_source: source sub-sample size for descriptors.
        n_target: target sub-sample size for descriptors.
        knn: neighbourhood size for local descriptors.
        ransac_iters: number of RANSAC 3-point hypotheses.
        inlier_tol: 3-D inlier threshold in the same units as the means.
        min_matches: minimum MNN matches required before attempting RANSAC
            (fewer → fallback to identity).
        verbose: if True, print match count and inlier count.

    Returns:
        ``(T_4x4, n_inliers)`` — the estimated 4×4 transform (``source``'s device/dtype) and
        the RANSAC inlier count (0 on fallback).  A low inlier count signals an ambiguous
        overlap (feature-poor region; rotation unrecoverable).
    """
    if transform not in ("se3", "sim3"):
        raise ValueError(f"transform must be 'se3' or 'sim3', got {transform!r}")
    with_scale = transform == "sim3"

    dev = source.means.device
    dtype = source.means.dtype
    src_full = source.means.to(torch.float32)
    tgt_full = target.means.to(device=dev, dtype=torch.float32)

    if src_full.shape[0] < 3 or tgt_full.shape[0] < 3:
        return torch.eye(4, device=dev, dtype=dtype), 0

    # Sub-sample deterministically (strided)
    src_sub = _stride_subsample(src_full, n_source)
    tgt_sub = _stride_subsample(tgt_full, n_target)

    # Compute local geometric descriptors
    desc_src = _fpfh_lite(src_sub, k=knn)    # (Ns, D)
    desc_tgt = _fpfh_lite(tgt_sub, k=knn)    # (Nt, D)

    # Mutual-NN correspondences in feature space
    src_idx, tgt_idx = _mnn_correspondences(desc_src, desc_tgt)
    n_matches = int(src_idx.shape[0])

    if verbose:
        print(f"[feature_align] MNN matches: {n_matches} / min {min_matches}")

    if n_matches < min_matches:
        if verbose:
            print("[feature_align] too few matches — falling back to identity")
        return torch.eye(4, device=dev, dtype=dtype), 0

    # RANSAC over 3-point samples
    T_feat, n_inliers = _ransac_se3(
        src_sub, tgt_sub, src_idx, tgt_idx,
        n_iters=ransac_iters,
        inlier_tol=inlier_tol,
        with_scale=with_scale,
    )

    if verbose:
        print(f"[feature_align] RANSAC inliers: {n_inliers} / {n_matches}")

    return T_feat.to(device=dev, dtype=dtype), n_inliers
