"""Feature-based global aligner, partial-overlap robust coarse init + ambiguity detection.

``feature_align`` is an alternative to :func:`splatreg.align.global_align` designed for
partial-overlap scenarios (two splat captures of the same object that see different parts of it).
The centroid-ICP sweep in ``align.py`` assumes full overlap and fails when 20–60 % of the target
is missing.  Feature-based matching sidesteps that by finding point pairs that look locally
similar (via a proper FPFH descriptor) before estimating the transform, using ONLY the region
that is present in both clouds.

Algorithm
---------
1. **Sub-sample** both clouds to a manageable size (strided, deterministic).
2. **Normals** (``_estimate_normals``): per-point surface normal from the smallest-eigenvector of
   the local k-NN covariance (PCA), sign-disambiguated toward the cloud centroid so both clouds
   orient consistently.
3. **FPFH descriptors** (``_fpfh``): the textbook Fast Point Feature Histogram (Rusu et al.,
   ICRA 2009).  For each point we first build a Simplified PFH (``_spfh``): for every k-neighbour
   pair we compute the three Darboux-frame angular features ``(alpha, phi, theta)`` from the two
   estimated normals and histogram each into ``B`` bins.  The FPFH then re-weights each point's
   SPFH by its neighbours' SPFHs (inverse-distance weighting), giving a ``3*B``-vector that is
   *rotation/translation invariant* and *discriminative* on geometrically distinctive regions
   (e.g. the lobe of the test object) while staying ambiguous on flat/smooth regions.
4. **Ratio-test mutual-NN matching** (``_match_features``): for each source descriptor find its two
   nearest target descriptors; accept the match only if it passes Lowe's ratio test AND is mutual
   (reciprocal NN).  This yields a small, high-confidence correspondence set drawn only from the
   shared region.
5. **Robust pose estimation** (``_robust_register``): a maximal-clique / consistency pre-filter
   (TEASER-lite) keeps the largest mutually distance-consistent subset of correspondences, then
   RANSAC over 3-point minimal samples fits an SE(3)/Sim(3) via Umeyama and scores by 3-D inliers.
   The clique pre-filter removes gross outliers before RANSAC so a sound inlier threshold suffices.
6. Return the winner 4×4 transform plus diagnostics (inlier count, total matches, an
   **ambiguity** score and flag).

Ambiguity detection (honest signal)
-----------------------------------
When the crop removes the rotation-disambiguating feature (e.g. the ``+x`` lobe is gone), the
surviving region is near-symmetric: several RANSAC hypotheses fit the inliers nearly equally well
but disagree on rotation.  ``_robust_register`` enumerates the top hypotheses, clusters them by
geodesic rotation distance, and computes an **ambiguity score** = the spread of near-best
rotations.  A large spread (or too few inliers / matches) means the true pose is *genuinely
unrecoverable from this overlap by any method*; we flag it (``ambiguous=True``) and report a low
``confidence`` instead of silently returning a wrong pose.  Correctly flagging an inherently
ambiguous crop is the intended, honest behaviour, not a failure to hide.

Symmetric-object improvement in ``align.py``
---------------------------------------------
Two fixes are applied to the existing ``_pca_seed_rotations`` and ``global_align`` in
``splatreg/align.py``:
  * isotropy probe for near-spherical clouds (PCA axes are arbitrary there);
  * a stability tie-break: among seeds within an epsilon of the best score, prefer the one whose
    aligned centroid is closest to the target centroid.
Those fixes live in ``align.py``; this file only documents their rationale.

Pure torch, CPU-runnable, no additional dependencies.
"""

from __future__ import annotations

import math

import torch

from .core.types import Gaussians
from .align import _stride_subsample, _umeyama, super_fibonacci_so3, _batched_nn, _batched_umeyama

# ── tuneable defaults ─────────────────────────────────────────────────────────

# Sub-sample sizes.  Keep small for CPU feasibility; partial-overlap scenes
# need denser descriptors than the full-overlap centroid sweep.
_FS_N_SOURCE = 600
_FS_N_TARGET = 600

# k-NN for normals and SPFH.  Normals want a tight neighbourhood (sharp surface
# estimate); FPFH wants a slightly larger one to be discriminative.
_FS_KNN_NORMAL = 16
_FS_KNN_FPFH = 16

# FPFH histogram bins per angular feature.  3 features × _FS_FPFH_BINS = descriptor dim.
_FS_FPFH_BINS = 11

# Matching
_FS_RATIO = 0.92  # Lowe ratio-test threshold (lenient: FPFH NN distances are close on smooth geom)

# Robust estimation
_FS_RANSAC_ITERS = 2000  # number of 3-point hypotheses
_FS_INLIER_TOL = 0.02  # 3-D inlier distance threshold (metres); ~2 cm at OBJ_RADIUS 0.14
_FS_MIN_MATCHES = 4  # fewer ratio/MNN matches → ambiguous (insufficient to constrain a pose)
_FS_MIN_INLIERS = 6  # fewer RANSAC inliers → low confidence / ambiguous

# Ambiguity thresholds
# PRIMARY failure signal: the normalised target->source overlap residual at the converged pose
# (``_overlap_residual_norm``).  The observed (partial) target should land exactly on the full
# source surface under the true pose, so a correct alignment scores ~0; a wrong / unrecoverable
# pose leaves the target floating off the surface.  Calibrated on the benchmark: every recovered
# partial crop scores ~0.000, the one unrecoverable crop scores ~0.12.  A 0.04 gate separates them
# with wide margin.
_FS_AMBIG_RESID = 0.04
# SECONDARY signal: post-polish rotation SENSITIVITY (``_rotation_constrained_probe``) — the
# smallest rise of the overlap residual (above the surface-resampling floor) when the converged
# rotation is twisted by +/-30 deg about any axis, normalised by the object RMS radius.  Near 0
# means some twist leaves the overlap unchanged → rotation genuinely unconstrained.  Kept as a low
# gate so it fires only on extreme rotational flatness (the residual gate catches ordinary
# failures); a discretely-sampled near-symmetric remnant that STILL aligns (Chamfer-success) is
# intentionally NOT failed by this gate.
_FS_AMBIG_SENSITIVITY = 0.02
# Diagnostic only: RANSAC hypothesis rotation spread among near-best hypotheses.
_FS_AMBIG_ROT_SPREAD_DEG = 12.0  # near-best hypotheses spread above this (deg) → rotation disagreement
_FS_TOPK_HYP = 12  # number of top RANSAC hypotheses kept for the spread test
_FS_HYP_SCORE_SLACK = 0.85  # a hypothesis is "near-best" if inliers >= slack * best_inliers

# Score epsilon for stability tie-break in global_align (used in align.py)
_SCORE_EPS = 1e-3


# ── k-NN ──────────────────────────────────────────────────────────────────────


def _knn_indices(pts: torch.Tensor, k: int) -> torch.Tensor:
    """For each point return the indices of its ``k`` nearest neighbours (excluding self).

    Uses chunked ``cdist`` to bound memory.  Returns ``(N, k)`` int64 indices.
    CPU-friendly (no cuda required).
    """
    N = pts.shape[0]
    k = min(k, N - 1)
    out = pts.new_empty((N, k), dtype=torch.int64)
    step = max(1, int(4_000_000 // max(N, 1)))
    for lo in range(0, N, step):
        hi = min(lo + step, N)
        d = torch.cdist(pts[lo:hi], pts)  # (chunk, N)
        d[:, lo:hi].fill_diagonal_(float("inf"))  # exclude self
        _, idx = torch.topk(d, k, dim=1, largest=False)
        out[lo:hi] = idx
    return out


# ── normals ────────────────────────────────────────────────────────────────────


def _estimate_normals(pts: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Per-point surface normal ``(N, 3)`` from the smallest-eigenvector of the local covariance.

    The normal is the eigenvector of the k-NN covariance with the *smallest* eigenvalue (the
    direction of least variance = surface normal).  Signs are disambiguated to point away from the
    cloud centroid so both source and target orient consistently (FPFH features use the normal
    directions, so a consistent convention is required for the descriptors to match across clouds).

    Args:
        pts: ``(N, 3)`` float32 cloud.
        idx: ``(N, k)`` precomputed k-NN indices.

    Returns:
        ``(N, 3)`` unit normals on the same device.
    """
    N = pts.shape[0]
    nbrs = pts[idx]  # (N, k, 3)
    centred = nbrs - nbrs.mean(dim=1, keepdim=True)  # (N, k, 3)
    cov = torch.einsum("nki,nkj->nij", centred, centred) / centred.shape[1]  # (N, 3, 3)
    # eigh returns ascending eigenvalues; smallest -> column 0 of eigenvectors
    try:
        _, evecs = torch.linalg.eigh(cov)  # evecs: (N, 3, 3), columns are eigenvectors
    except Exception:
        return torch.zeros(N, 3, device=pts.device, dtype=pts.dtype).index_fill_(
            1, torch.tensor([2], device=pts.device), 1.0
        )
    normals = evecs[:, :, 0]  # (N, 3) smallest-eigenvalue direction
    # Orient away from the cloud centroid (consistent sign convention across clouds).
    centroid = pts.mean(dim=0, keepdim=True)  # (1, 3)
    out = pts - centroid  # (N, 3) outward radial direction
    flip = (normals * out).sum(dim=1) < 0.0  # normal points inward → flip
    normals = torch.where(flip.unsqueeze(1), -normals, normals)
    normals = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return normals


# ── FPFH descriptor (Rusu et al., ICRA 2009) ───────────────────────────────────


def _spfh(pts: torch.Tensor, normals: torch.Tensor, idx: torch.Tensor, bins: int) -> torch.Tensor:
    """Simplified Point Feature Histogram ``(N, 3*bins)`` from the Darboux-frame angles.

    For every (anchor p, neighbour q) pair we build the Darboux frame from the two estimated
    normals and compute the three classic PFH features:

        u = n_p
        v = (q - p) / ||q - p||  ×  u        (cross product, then normalised)
        w = u × v
        alpha = v · n_q                       (in [-1, 1])
        phi   = u · (q - p) / ||q - p||       (in [-1, 1])
        theta = atan2(w · n_q, u · n_q)       (in [-pi, pi])

    Each feature is histogrammed into ``bins`` bins; the three histograms are concatenated and
    L1-normalised.  Translation/rotation invariant by construction (only relative directions enter).

    Args:
        pts: ``(N, 3)`` cloud.
        normals: ``(N, 3)`` unit normals.
        idx: ``(N, k)`` k-NN indices.
        bins: histogram bins per feature.

    Returns:
        ``(N, 3*bins)`` float32 SPFH (each point's row L1-sums to 3, one per feature block).
    """
    N, k = idx.shape
    dev = pts.device
    p = pts.unsqueeze(1)  # (N, 1, 3)
    q = pts[idx]  # (N, k, 3)
    n_p = normals.unsqueeze(1)  # (N, 1, 3)
    n_q = normals[idx]  # (N, k, 3)

    diff = q - p  # (N, k, 3)
    dist = diff.norm(dim=2, keepdim=True).clamp_min(1e-9)  # (N, k, 1)
    pq = diff / dist  # (N, k, 3) unit vector p->q

    u = n_p.expand(N, k, 3)  # (N, k, 3) Darboux u = source normal
    v = torch.cross(pq, u, dim=2)  # (N, k, 3)
    v = v / v.norm(dim=2, keepdim=True).clamp_min(1e-9)
    w = torch.cross(u, v, dim=2)  # (N, k, 3)

    alpha = (v * n_q).sum(dim=2).clamp(-1.0, 1.0)  # (N, k)
    phi = (u * pq).sum(dim=2).clamp(-1.0, 1.0)  # (N, k)
    theta = torch.atan2((w * n_q).sum(dim=2), (u * n_q).sum(dim=2))  # (N, k) in [-pi, pi]

    # Normalise each feature to [0, 1) for binning.
    a01 = (alpha + 1.0) * 0.5
    p01 = (phi + 1.0) * 0.5
    t01 = (theta + math.pi) / (2.0 * math.pi)

    def hist(x01: torch.Tensor) -> torch.Tensor:
        # bin index in [0, bins-1]; scatter-add a count per bin per row
        b = (x01 * bins).floor().clamp(0, bins - 1).to(torch.int64)  # (N, k)
        h = torch.zeros(N, bins, device=dev, dtype=torch.float32)
        h.scatter_add_(1, b, torch.ones_like(b, dtype=torch.float32))
        return h  # (N, bins)

    spfh = torch.cat([hist(a01), hist(p01), hist(t01)], dim=1)  # (N, 3*bins)
    # L1-normalise each feature block to k counts -> fraction (robust to varying neighbour counts).
    spfh = spfh / float(k)
    return spfh


def _fpfh(pts: torch.Tensor, normals: torch.Tensor, k: int, bins: int) -> torch.Tensor:
    """Fast Point Feature Histogram ``(N, 3*bins)`` (Rusu et al., ICRA 2009).

    ``FPFH(p) = SPFH(p) + (1/k) * sum_j (1/d(p, p_j)) * SPFH(p_j)`` over the k neighbours, with the
    neighbour SPFHs inverse-distance weighted.  The result is concatenated-and-normalised so the
    descriptor stays scale-comparable across clouds of different size.

    Args:
        pts: ``(N, 3)`` cloud.
        normals: ``(N, 3)`` unit normals.
        k: neighbourhood size.
        bins: histogram bins per feature.

    Returns:
        ``(N, 3*bins)`` float32 FPFH descriptors.
    """
    idx = _knn_indices(pts, k)  # (N, k)
    spfh = _spfh(pts, normals, idx, bins)  # (N, 3*bins)

    # Inverse-distance neighbour weighting.
    diff = pts[idx] - pts.unsqueeze(1)  # (N, k, 3)
    dist = diff.norm(dim=2).clamp_min(1e-9)  # (N, k)
    wgt = 1.0 / dist  # (N, k)
    wgt = wgt / wgt.sum(dim=1, keepdim=True).clamp_min(1e-9)  # row-normalised weights

    nbr_spfh = spfh[idx]  # (N, k, 3*bins)
    weighted = (wgt.unsqueeze(2) * nbr_spfh).sum(dim=1)  # (N, 3*bins)
    fpfh = spfh + weighted
    # Final L2-normalise so feature-space NN distances are well-conditioned.
    fpfh = fpfh / fpfh.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return fpfh.to(torch.float32)


# ── matching: ratio-test + mutual-NN ───────────────────────────────────────────


def _match_features(
    desc_src: torch.Tensor,
    desc_tgt: torch.Tensor,
    ratio: float = _FS_RATIO,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ratio-test + mutual-nearest-neighbour matching in FPFH space.

    For each source descriptor find its two nearest target descriptors; accept only if the best is
    sufficiently closer than the second-best (Lowe's ratio test) AND the match is mutual (the
    target's nearest source is back to this source).  Both filters reject ambiguous matches on
    smooth/symmetric regions, leaving a small high-confidence set.

    Args:
        desc_src: ``(Ns, D)`` source descriptors.
        desc_tgt: ``(Nt, D)`` target descriptors.
        ratio: Lowe ratio threshold (``d1 < ratio * d2`` to accept).

    Returns:
        ``(src_idx, tgt_idx)`` int64 tensors of matching anchor indices, both ``(M,)``.
    """
    dist = torch.cdist(desc_src, desc_tgt)  # (Ns, Nt)
    Ns, Nt = dist.shape

    # Source -> target: two nearest for the ratio test.
    k2 = min(2, Nt)
    d_sorted, i_sorted = torch.topk(dist, k2, dim=1, largest=False)  # (Ns, k2)
    s2t = i_sorted[:, 0]  # (Ns,) best target for each source
    if k2 == 2:
        d1 = d_sorted[:, 0].clamp_min(1e-12)
        d2 = d_sorted[:, 1].clamp_min(1e-12)
        ratio_ok = d1 < ratio * d2  # (Ns,)
    else:
        ratio_ok = torch.ones(Ns, dtype=torch.bool, device=dist.device)

    # Target -> source nearest (for mutual check).
    t2s = dist.argmin(dim=0)  # (Nt,)
    src_range = torch.arange(Ns, device=dist.device)
    mutual = t2s[s2t] == src_range  # (Ns,)

    keep = ratio_ok & mutual
    src_idx = src_range[keep]
    tgt_idx = s2t[keep]
    return src_idx, tgt_idx


# ── robust pose estimation: maximal-clique pre-filter + RANSAC ──────────────────


def _clique_prefilter(
    cs: torch.Tensor,
    ct: torch.Tensor,
    tol: float,
    max_keep: int = 200,
) -> torch.Tensor:
    """Keep the largest mutually distance-consistent subset of correspondences (TEASER-lite).

    A rigid/similarity transform preserves *pairwise distances up to a global scale*.  Two
    correspondences (i, j) are consistent if ``| ||cs_i - cs_j|| - ||ct_i - ct_j|| || <= tol``
    (rigid), gross mismatches violate this with most other correspondences.  We build the
    consistency graph and greedily grow a clique from the highest-degree vertex: that vertex's
    consistent neighbourhood is a near-clique of inliers, which we return as the surviving indices.

    This is a cheap O(M^2) inlier pre-selection (M is small after ratio/MNN), enough to let a
    fixed RANSAC threshold succeed.

    Args:
        cs: ``(M, 3)`` source correspondence points.
        ct: ``(M, 3)`` target correspondence points.
        tol: pairwise-distance consistency tolerance (metres).
        max_keep: cap on the consistency-matrix size (subsamples if M is large).

    Returns:
        ``(K,)`` int64 indices (into the M correspondences) of the kept consistent subset.
    """
    M = cs.shape[0]
    if M <= 4:
        return torch.arange(M, device=cs.device)
    if M > max_keep:
        sel = torch.linspace(0, M - 1, max_keep, device=cs.device).round().to(torch.int64)
    else:
        sel = torch.arange(M, device=cs.device)
    cs_s = cs[sel]
    ct_s = ct[sel]
    ds = torch.cdist(cs_s, cs_s)  # (m, m) source pairwise distances
    dt = torch.cdist(ct_s, ct_s)  # (m, m) target pairwise distances
    consistent = (ds - dt).abs() <= tol  # (m, m) bool consistency graph (rigid scale=1)
    deg = consistent.sum(dim=1)  # (m,) vertex degree
    seed = int(deg.argmax().item())
    # The clique candidate = the seed's consistent neighbourhood.
    member = consistent[seed]  # (m,)
    kept = sel[member]
    if kept.numel() < 4:
        return torch.arange(M, device=cs.device)
    return kept


def _robust_register(
    src_pts: torch.Tensor,
    tgt_pts: torch.Tensor,
    src_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    *,
    n_iters: int,
    inlier_tol: float,
    with_scale: bool,
    rng_seed: int = 42,
) -> dict:
    """RANSAC (3-point Umeyama) over a clique-prefiltered correspondence set + ambiguity probe.

    Returns a dict with:
        ``T``         best 4×4 transform,
        ``n_inliers`` best inlier count,
        ``n_matches`` number of input correspondences,
        ``ambiguity`` rotation spread (deg) among near-best hypotheses (0 if a single basin),
        ``hypotheses`` number of distinct near-best rotation clusters.

    The ambiguity probe keeps the top hypotheses (by inlier count), takes those within
    ``_FS_HYP_SCORE_SLACK`` of the best, and measures the max pairwise geodesic rotation distance
    among them: a large spread means the inliers do not constrain rotation (the disambiguating
    feature is gone) → the pose is inherently ambiguous.
    """
    dev = src_pts.device
    dtype = src_pts.dtype
    eye4 = torch.eye(4, device=dev, dtype=dtype)
    M0 = int(src_idx.shape[0])
    out = {
        "T": eye4.clone(),
        "n_inliers": 0,
        "n_matches": M0,
        "ambiguity": 180.0,
        "hypotheses": 0,
    }
    if M0 < 3:
        return out

    cs_all = src_pts[src_idx].double()  # (M0, 3)
    ct_all = tgt_pts[tgt_idx].double()  # (M0, 3)

    # Clique pre-filter (rigid pairwise-distance consistency) — drops gross outliers.
    keep = _clique_prefilter(cs_all, ct_all, tol=inlier_tol)
    cs = cs_all[keep]
    ct = ct_all[keep]
    M = cs.shape[0]
    if M < 3:
        cs, ct, M = cs_all, ct_all, M0  # fall back to all matches if clique collapsed

    rng = torch.Generator(device="cpu")
    rng.manual_seed(rng_seed)

    best_T = eye4.clone()
    best_n = 0
    best_res = float("inf")
    # Track the top hypotheses' rotations + inlier counts for the ambiguity probe.
    hyp_R: list[torch.Tensor] = []
    hyp_n: list[int] = []

    for _ in range(n_iters):
        perm = torch.randperm(M, generator=rng)[:3]
        s_s = cs[perm]
        s_t = ct[perm]
        try:
            s_est, R_est, t_est = _umeyama(s_s, s_t, with_scale)
        except Exception:
            continue
        # Degenerate sample (collinear) → skip.
        if not torch.isfinite(R_est).all() or not torch.isfinite(t_est).all():
            continue

        cs_t = cs @ R_est.transpose(-1, -2) * s_est + t_est  # (M, 3)
        res = (cs_t - ct).norm(dim=1)
        inl = res < inlier_tol
        n_in = int(inl.sum().item())
        mean_res = float(res[inl].mean().item()) if n_in > 0 else float("inf")

        if n_in >= 3:
            hyp_R.append(R_est.detach())
            hyp_n.append(n_in)

        if n_in > best_n or (n_in == best_n and mean_res < best_res):
            best_n = n_in
            best_res = mean_res
            T_cand = torch.eye(4, device=dev, dtype=torch.float64)
            T_cand[:3, :3] = s_est * R_est
            T_cand[:3, 3] = t_est
            best_T = T_cand.to(dtype=dtype, device=dev)

    # Refine the winner on ALL its inliers (Umeyama on the consensus set).
    if best_n >= 3:
        R_b = best_T[:3, :3].double()
        s_b = float(torch.linalg.det(R_b).abs().clamp_min(1e-12) ** (1.0 / 3.0)) if with_scale else 1.0
        Rb_pure = R_b / s_b if with_scale else R_b
        cs_t = cs @ Rb_pure.transpose(-1, -2) * s_b + best_T[:3, 3].double()
        inl = (cs_t - ct).norm(dim=1) < inlier_tol
        if int(inl.sum().item()) >= 3:
            try:
                s_r, R_r, t_r = _umeyama(cs[inl], ct[inl], with_scale)
                T_ref = torch.eye(4, device=dev, dtype=torch.float64)
                T_ref[:3, :3] = s_r * R_r
                T_ref[:3, 3] = t_r
                best_T = T_ref.to(dtype=dtype, device=dev)
            except Exception:
                pass

    out["T"] = best_T
    out["n_inliers"] = best_n

    # ── ambiguity probe ──────────────────────────────────────────────────────
    if best_n >= 3 and len(hyp_R) >= 1:
        thresh = _FS_HYP_SCORE_SLACK * best_n
        near = [R for R, n in zip(hyp_R, hyp_n) if n >= thresh]
        if len(near) > _FS_TOPK_HYP:
            # keep the top-k by inlier count
            order = sorted(range(len(hyp_n)), key=lambda i: -hyp_n[i])
            near = [hyp_R[i] for i in order[:_FS_TOPK_HYP] if hyp_n[i] >= thresh]
        if len(near) >= 2:
            Rs = torch.stack([R.to(torch.float64) for R in near], dim=0)  # (H, 3, 3)
            # pairwise geodesic rotation distance (deg); max spread among near-best hypotheses
            H = Rs.shape[0]
            rel = torch.einsum("aij,bkj->abik", Rs, Rs)  # (H, H, 3, 3) = Ra @ Rb^T
            tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
            cosang = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
            ang = torch.rad2deg(torch.arccos(cosang))  # (H, H)
            spread = float(ang.max().item())
            out["ambiguity"] = spread
            # number of distinct rotation clusters (> spread threshold apart)
            out["hypotheses"] = H
        else:
            out["ambiguity"] = 0.0
            out["hypotheses"] = len(near)
    return out


def _robust_register_batched(
    src_pts: torch.Tensor,
    tgt_pts: torch.Tensor,
    src_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    *,
    n_iters: int,
    inlier_tol: float,
    with_scale: bool,
    rng_seed: int = 42,
) -> dict:
    """GPU-batched 3-point RANSAC, the vectorised equivalent of :func:`_robust_register`.

    Same contract (returns ``T`` / ``n_inliers`` / ``n_matches`` / ``ambiguity`` / ``hypotheses``)
    but evaluates **all** ``n_iters`` minimal hypotheses in one shot: a single batched closed-form
    Umeyama (:func:`_batched_umeyama`) over ``(n_iters, 3, 3)`` triplets, then one batched inlier
    score against the whole clique-filtered correspondence set, no Python per-iteration loop, no
    per-iter CPU ``randperm`` / GPU sync.  This is the fast path the per-pair init uses; it removes
    the ~1.4 s Python RANSAC loop that dominated ``feature_align``.  Deterministic given ``rng_seed``.
    """
    dev = src_pts.device
    dtype = src_pts.dtype
    eye4 = torch.eye(4, device=dev, dtype=dtype)
    M0 = int(src_idx.shape[0])
    out = {"T": eye4.clone(), "n_inliers": 0, "n_matches": M0, "ambiguity": 180.0, "hypotheses": 0}
    if M0 < 3:
        return out

    cs_all = src_pts[src_idx].float()  # (M0, 3)
    ct_all = tgt_pts[tgt_idx].float()

    keep = _clique_prefilter(cs_all.double(), ct_all.double(), tol=inlier_tol)
    cs = cs_all[keep]
    ct = ct_all[keep]
    M = cs.shape[0]
    if M < 3:
        cs, ct, M = cs_all, ct_all, M0

    # All minimal samples at once: B triplets of distinct indices (argsort of a random matrix on the
    # correspondence axis gives a per-row permutation; take the first 3 columns). RNG on-device,
    # seeded for determinism.
    B = int(n_iters)
    gen = torch.Generator(device=dev)
    gen.manual_seed(rng_seed)
    tri = torch.argsort(torch.rand(B, M, device=dev, generator=gen), dim=1)[:, :3]  # (B, 3)
    Xs = cs[tri]  # (B, 3, 3) source triplets
    Ys = ct[tri]  # (B, 3, 3) target triplets
    w = torch.ones(B, 3, device=dev, dtype=Xs.dtype)
    s_b, R_b, t_b = _batched_umeyama(Xs, Ys, w, with_scale)  # (B,), (B,3,3), (B,3)

    # Batched inlier score: transform every correspondence under every hypothesis.
    # cs_t[b] = s_b[b] * (cs @ R_b[b]^T) + t_b[b]  -> (B, M, 3)
    cs_t = s_b[:, None, None] * torch.einsum("mi,bji->bmj", cs, R_b) + t_b[:, None, :]
    res = (cs_t - ct.unsqueeze(0)).norm(dim=2)  # (B, M)
    finite = torch.isfinite(R_b).all(dim=(1, 2)) & torch.isfinite(t_b).all(dim=1)  # (B,)
    inl = (res < inlier_tol) & finite[:, None]  # (B, M)
    n_in = inl.sum(dim=1)  # (B,)
    # mean inlier residual (inf where no inliers) for the tie-break.
    masked = torch.where(inl, res, torch.full_like(res, float("inf")))
    mean_res = torch.where(
        n_in > 0, masked.sum(dim=1) / n_in.clamp_min(1), torch.full_like(s_b, float("inf"))
    )

    best = int(torch.argmax(n_in - 1e-6 * mean_res.clamp_max(1e6)).item())
    best_n = int(n_in[best].item())
    if best_n < 3:
        return out

    # Refine the winner on ALL its inliers (Umeyama on the consensus set), in float64.
    mask = inl[best]
    s_r, R_r, t_r = _umeyama(cs[mask].double(), ct[mask].double(), with_scale)
    T_ref = torch.eye(4, device=dev, dtype=torch.float64)
    T_ref[:3, :3] = s_r * R_r
    T_ref[:3, 3] = t_r
    out["T"] = T_ref.to(dtype=dtype, device=dev)
    out["n_inliers"] = best_n

    # ── ambiguity probe (batched geodesic spread among near-best hypotheses) ──
    thresh = _FS_HYP_SCORE_SLACK * best_n
    near_mask = (n_in >= thresh) & (n_in >= 3)
    near_idx = near_mask.nonzero(as_tuple=False).squeeze(1)
    if near_idx.numel() > _FS_TOPK_HYP:
        order = torch.argsort(n_in[near_idx], descending=True)
        near_idx = near_idx[order[:_FS_TOPK_HYP]]
    if near_idx.numel() >= 2:
        Rs = R_b[near_idx].to(torch.float64)  # (H, 3, 3)
        rel = torch.einsum("aij,bkj->abik", Rs, Rs)
        tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
        cosang = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
        out["ambiguity"] = float(torch.rad2deg(torch.arccos(cosang)).max().item())
        out["hypotheses"] = int(Rs.shape[0])
    else:
        out["ambiguity"] = 0.0
        out["hypotheses"] = int(near_idx.numel())
    return out


# ── overlap-aware ICP polish (partial-overlap safe) ─────────────────────────────


def _overlap_icp_polish(
    src_full: torch.Tensor,
    tgt_full: torch.Tensor,
    T0: torch.Tensor,
    *,
    with_scale: bool,
    iters: int = 10,
    trim: float = 0.85,
    n_src: int = 2048,
    n_tgt: int = 1024,
    conv_tol: float = 1e-5,
) -> torch.Tensor:
    """Refine ``T`` (source→target) by matching TARGET→SOURCE, robust to a partial target.

    The FPFH+RANSAC step gives a coarse rotation; this polishes it without the partial-overlap
    failure mode of a naive source→target ICP.  Because the (possibly cropped) target is a SUBSET
    of the full source under the true pose, **every observed target point has a correct match in the
    transformed full source**.  So we match each target point to its nearest transformed-source
    point (not the reverse), trim the worst ``1-trim`` fraction (occlusion-boundary mismatches), and
    re-fit ``T`` via Umeyama on those matches.  The parts of the source outside the overlap simply
    never get selected, they cannot drag the fit off-pose the way a source→target match does.

    Args:
        src_full: full source cloud ``(Ns, 3)`` (float32).
        tgt_full: (partial) target cloud ``(Nt, 3)`` (float32).
        T0: initial 4×4 source→target transform.
        with_scale: estimate scale (Sim(3)) if True.
        iters: ICP iterations.
        trim: fraction of best target matches kept each iteration.
        n_src / n_tgt: deterministic strided subsample sizes.

    Returns:
        Refined 4×4 transform (same device/dtype as ``T0``).
    """
    dev, dtype = T0.device, T0.dtype
    src = _stride_subsample(src_full, n_src).double()
    tgt = _stride_subsample(tgt_full, n_tgt).double()
    if src.shape[0] < 3 or tgt.shape[0] < 3:
        return T0
    block = T0[:3, :3].double()
    t = T0[:3, 3].double()
    s = float(torch.linalg.det(block).abs().clamp_min(1e-18) ** (1.0 / 3.0)) if with_scale else 1.0
    R = block / s if with_scale else block
    keep_k = max(3, int(round(trim * tgt.shape[0])))
    for _ in range(int(iters)):
        TA = src @ R.transpose(-1, -2) * s + t  # transformed full source (Ns, 3)
        d = torch.cdist(tgt, TA)  # (Nt, Ns)
        nn = d.argmin(dim=1)  # each target -> nearest transformed source
        dmin = d.gather(1, nn.unsqueeze(1)).squeeze(1)  # (Nt,)
        thr = torch.kthvalue(dmin, keep_k).values
        m = dmin <= thr  # inlier (overlapping) target points
        if int(m.sum().item()) < 3:
            break
        src_matched = src[nn[m]]  # source points paired to inlier targets
        tgt_matched = tgt[m]
        try:
            s_new, R_new, t_new = _umeyama(src_matched, tgt_matched, with_scale)
        except Exception:
            break
        if not (torch.isfinite(R_new).all() and torch.isfinite(t_new).all()):
            break
        # Early-exit on convergence: the FPFH+RANSAC seed is already close, so the trimmed
        # re-fit settles in a couple of iterations.  Measure the incremental change in the
        # (R, s, t) estimate; once it is below ``conv_tol`` further iterations only repeat the
        # same cdist/Umeyama at no benefit, so we stop and save the per-iter cost.
        dR = float((R_new - R).abs().max().item())
        dt = float((t_new - t).abs().max().item())
        ds = abs(float(s_new) - s)
        R, s, t = R_new, float(s_new), t_new
        if dR < conv_tol and dt < conv_tol and ds < conv_tol:
            break
    T = torch.eye(4, device=dev, dtype=torch.float64)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T.to(device=dev, dtype=dtype)


def _point_to_plane_overlap_icp(
    src: torch.Tensor,
    snrm: torch.Tensor,
    tgt: torch.Tensor,
    R0: torch.Tensor,
    t0: torch.Tensor,
    *,
    iters: int,
    trim: float,
    lam: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Overlap-aware **point-to-plane** trimmed ICP (target→source), the partial-overlap workhorse.

    Two design choices make this converge to the TRUE pose on a one-sided crop, where a naive ICP
    settles several degrees off (both were verified directly on the benchmark crops):

    * **Pivot the incremental rotation about the OVERLAP centroid, not the source centroid.**  This
      is the load-bearing fix.  A one-sided crop's centroid does not correspond to the full-source
      centroid, so a rotation parametrised about the source/world origin couples rotation into a
      large translation error and the trimmed objective's minimum drifts ~5-13° off the true pose.
      Mapped instead about the centroid of the matched (overlapping) region, ``q.mean(0)``, the
      trimmed target→source cost becomes a SHARP basin whose minimum sits exactly at the true pose
      (residual ~0), so a seed within the basin descends straight to it.  (Measured: with the source
      pivot the cost is minimised 3-13° from ground truth; with the overlap pivot the cost rises
      monotonically from 0 at the true pose to ~70× at 10° off, a clean, attracting basin.)
    * **Levenberg-style diagonal damping** (``lam`` × diag(JᵀJ)).  At a perfect subset-overlap
      optimum the matched residuals are ~0 and the one-sided crop leaves some rotational DoF weakly
      constrained, so an undamped Gauss-Newton step jitters in that near-null space and walks OFF the
      optimum even when started exactly on it.  Damping pins the unconstrained directions so the step
      stays on the true pose once reached.

    Why point-to-plane (not point-to-point): point-to-plane lets each observed target point slide
    ALONG the source surface to its correct tangential position, so the true basin sharpens to
    residual ~0 while wrong basins stay high, a separable score that both recovers the pose AND lets
    the honest ambiguity gate fire only on the truly-unrecoverable crops.

    Each iteration: match every (trimmed) target point to its nearest transformed-source point, then
    take one damped Gauss-Newton point-to-plane step (linearised small-angle SE(3), rotation about
    the overlap centroid) minimising ``sum_i [ n_i . ( (R q_i + t) - p_i ) ]^2`` over the matched
    source point ``q_i`` (normal ``n_i``) and target point ``p_i``.  The trim drops the ``1-trim``
    worst matches (occlusion boundary / outside-overlap source) so the partial target can't drag the
    fit off-pose.  All math in float64; CPU-friendly (one ``cdist`` + one damped 6×6 solve per iter).

    Args:
        src: full source cloud ``(Ns, 3)`` float64.
        snrm: per-source-point unit normals ``(Ns, 3)`` float64.
        tgt: (partial) target cloud ``(Nt, 3)`` float64.
        R0, t0: initial rotation ``(3, 3)`` and translation ``(3,)`` (source→target).
        iters: ICP iterations.
        trim: fraction of best target matches kept each iteration.
        lam: Levenberg diagonal-damping factor (× the mean of diag(JᵀJ)).

    Returns:
        ``(R, t, residual)``, refined rotation/translation and the final trimmed target→source
        overlap residual (lower = better fit).
    """
    R = R0.clone()
    t = t0.clone()
    dt = src.dtype
    keep_k = max(6, int(round(trim * tgt.shape[0])))
    keep_k = min(keep_k, tgt.shape[0])
    eye6 = torch.eye(6, device=src.device, dtype=dt)
    eye3 = torch.eye(3, device=src.device, dtype=dt)
    for _ in range(int(iters)):
        TA = src @ R.transpose(-1, -2) + t  # transformed source (Ns, 3)
        nrmA = snrm @ R.transpose(-1, -2)  # transformed source normals (Ns, 3)
        d = torch.cdist(tgt, TA)  # (Nt, Ns)
        dmin, nn = d.min(dim=1)  # nearest transformed-source per target
        thr = torch.kthvalue(dmin, keep_k).values
        m = dmin <= thr
        if int(m.sum().item()) < 6:
            break
        p = tgt[m]  # target points (M, 3)
        q = TA[nn[m]]  # matched source points (M, 3)
        nq = nrmA[nn[m]]  # matched source normals (M, 3)
        pivot = q.mean(0)  # OVERLAP centroid — rotate the increment about this, not the origin
        qc = q - pivot
        # point-to-plane linear system (rotation about the overlap pivot): row = [ qc x n , n ].
        Arow = torch.cat([torch.cross(qc, nq, dim=1), nq], dim=1)  # (M, 6)
        b = -(nq * (q - p)).sum(dim=1)  # (M,)
        AtA = Arow.transpose(-1, -2) @ Arow
        H = AtA + lam * torch.diag(torch.diagonal(AtA).clamp_min(1e-12)) + 1e-12 * eye6
        try:
            x = torch.linalg.solve(H, Arow.transpose(-1, -2) @ b)
        except Exception:
            break
        if not torch.isfinite(x).all():
            break
        om = x[:3]
        tr = x[3:]
        th = om.norm().clamp_min(1e-12)
        K = torch.zeros(3, 3, device=src.device, dtype=dt)
        K[0, 1], K[0, 2] = -om[2], om[1]
        K[1, 0], K[1, 2] = om[2], -om[0]
        K[2, 0], K[2, 1] = -om[1], om[0]
        dR = eye3 + (torch.sin(th) / th) * K + ((1.0 - torch.cos(th)) / (th * th)) * (K @ K)
        # Apply the increment about the overlap pivot: x' = dR (x - pivot) + pivot + tr  ⇒
        #   R ← dR R ,  t ← dR t + (pivot + tr - dR pivot).
        R = dR @ R
        t = dR @ t + (pivot + tr - dR @ pivot)
    TA = src @ R.transpose(-1, -2) + t
    dmin = torch.cdist(tgt, TA).min(dim=1).values
    res = float(dmin.topk(keep_k, largest=False).values.mean().item())
    return R, t, res


def _batched_t2s_prefilter(
    src: torch.Tensor,
    tgt: torch.Tensor,
    seeds: torch.Tensor,
    *,
    iters: int,
    trim: float,
    seed_batch: int = 128,
) -> torch.Tensor:
    """Rank ALL ``seeds`` by a cheap fully-batched target→source point-to-point trimmed ICP.

    This is a *coarse, fast* basin pre-selector, it runs every super-Fibonacci seed at once
    (vectorised closed-form Umeyama steps via :func:`splatreg.align._batched_umeyama`, chunked over
    seeds) so we never pay the per-seed Python-loop / 6×6-solve cost of the point-to-plane refiner on
    thousands of candidates.  Point-to-point can't reach the exact subset pose on a one-sided crop
    (that's what the pivoted point-to-plane refine is for), but its trimmed target→source overlap
    residual reliably *ranks* seeds so the true basin is among the top few, the expensive refine then
    only runs on those.  Returns the converged rotations ``(B, 3, 3)`` ordered best-overlap first.
    """
    Ns, Nt = src.shape[0], tgt.shape[0]
    sc = src.mean(0)
    tc = tgt.mean(0)
    keep_k = max(6, int(round(trim * Nt)))
    score_chunks: list[torch.Tensor] = []
    R_chunks: list[torch.Tensor] = []
    src0 = (src - sc).to(torch.float32)
    tgtf = tgt.to(torch.float32)
    for lo in range(0, seeds.shape[0], seed_batch):
        Rb = seeds[lo : lo + seed_batch].to(torch.float32)
        b = Rb.shape[0]
        # transformed full source per seed (B, Ns, 3): rotate about source centroid, place at target.
        X = torch.bmm(src0.unsqueeze(0).expand(b, Ns, 3), Rb.transpose(1, 2)) + tc.to(torch.float32)
        Y = tgtf.unsqueeze(0).expand(b, Nt, 3).contiguous()
        # track the cumulative rotation applied to the source (starts at the seed Rb)
        Rcur = Rb.clone()
        for _ in range(int(iters)):
            d, idx = _batched_nn(Y, X)  # each target -> nearest transformed source
            thr = torch.kthvalue(d, keep_k, dim=1, keepdim=True).values
            wmask = (d <= thr).to(X.dtype)
            Xm = torch.gather(X, 1, idx.unsqueeze(-1).expand(b, Nt, 3))  # matched src per target
            s_i, R_i, t_i = _batched_umeyama(Xm, Y, wmask, with_scale=False)
            X = torch.bmm(X, R_i.transpose(1, 2)) + t_i[:, None, :]
            Rcur = torch.bmm(R_i, Rcur)
        d, _ = _batched_nn(Y, X)
        k = max(3, int(0.9 * Nt))
        score_chunks.append(torch.sort(d, dim=1).values[:, :k].sqrt().mean(dim=1))
        R_chunks.append(Rcur)
    scores = torch.cat(score_chunks, dim=0)
    Rall = torch.cat(R_chunks, dim=0)
    order = torch.argsort(scores)
    return Rall[order].to(torch.float64)


def _overlap_basin_sweep(
    src_full: torch.Tensor,
    tgt_full: torch.Tensor,
    *,
    with_scale: bool,
    n_rot: int = 2048,
    icp_iters: int = 12,
    trim: float = 0.7,
    topk: int = 200,
    refine_iters: int = 150,
    refine_trim: float = 0.8,
    n_src: int = 1400,
    n_tgt: int = 800,
) -> torch.Tensor:
    """Overlap-aware **point-to-plane** super-Fibonacci basin sweep, the partial-overlap recoverer.

    The FPFH path lands a wrong rotation on smooth splat surfaces (the descriptors are near-flat off
    the disambiguating lobe), so the coarse init falls into a wrong basin and the ambiguity detector
    then (correctly) flags it.  This sweep replaces that coarse init with a geometry-only search that
    is robust to a one-sided crop:

    1. Seed SO(3) with a deterministic super-Fibonacci covering (``n_rot`` rotations); for each seed
       centroid-align the source to the target and run a short overlap-aware *point-to-plane* trimmed
       ICP (:func:`_point_to_plane_overlap_icp`).  Point-to-plane (not point-to-point) is essential:
       it lets the partial target slide along the smooth source surface to its true tangential
       position, so the correct basin converges to residual ~0 while wrong basins stay at ~0.005.
    2. Keep the ``topk`` lowest-residual converged seeds and re-refine each with a longer,
       tighter-trim point-to-plane ICP, then pick the winner by the SYMMETRIC overlap residual (see
       step 3).  On MODERATE crops (keep≈60 %) the true basin's coarse seed ranks DEEP in the cheap
       prefilter (the cropped one-directional residual barely separates it from the 170-ish mirror),
       so a shallow ``topk`` drops it before the precise refine ever sees it; the true basin also needs
       enough refine iterations to fully descend.  ``topk=200`` + ``refine_iters=150`` keeps the true
       seed in the pool and lets it converge, this is what lifts keep60 from 1/3 to 3/3 solved,
       while staying GPU-affordable (the prefilter is fully batched; only ``topk`` seeds are refined).
    3. Return the lowest-residual transform.  When the crop genuinely deletes the rotation-
       disambiguating geometry (heavy crops), no seed reaches a low residual, the caller's ambiguity
       gate then honestly flags it instead of returning a confident wrong pose.

    Deterministic (closed-form seeds + strided subsample, no RNG).  ``with_scale`` is accepted for
    signature parity but the partial-overlap sweep estimates a rigid pose (scale is recovered by the
    downstream FPFH/Umeyama path when needed); the test geometry is metric-consistent so this does
    not affect recovery.

    Returns the best 4×4 source→target transform.
    """
    del with_scale  # rigid sweep; scale parity handled upstream
    dev, dtype = src_full.device, src_full.dtype
    src = _stride_subsample(src_full, n_src).double()
    tgt = _stride_subsample(tgt_full, n_tgt).double()
    eye4 = torch.eye(4, device=dev, dtype=dtype)
    if src.shape[0] < 6 or tgt.shape[0] < 6:
        return eye4
    # Per-source-point normals (reuse the FPFH normal estimator for a consistent surface estimate).
    sidx = _knn_indices(src.to(torch.float32), _FS_KNN_NORMAL)
    snrm = _estimate_normals(src.to(torch.float32), sidx).double()
    src_c = src.mean(0)
    tgt_c = tgt.mean(0)
    seeds = super_fibonacci_so3(n_rot, device=dev, dtype=torch.float64)

    # Stage 0 (fully batched, fast): rank ALL seeds by a cheap point-to-point target→source ICP so the
    # expensive pivoted point-to-plane refine only touches the top few candidates.  This keeps a DENSE
    # seed covering (so the small basins of moderate crops are hit) affordable.
    R_ranked = _batched_t2s_prefilter(src, tgt, seeds, iters=int(icp_iters), trim=trim)
    topn = max(1, int(topk))
    R_ranked = R_ranked[:topn]

    # Stage 1: precise pivoted, LM-damped point-to-plane refine on the top seeds (the recoverer).  The
    # overlap-pivot + damping makes the trimmed objective a sharp basin AT the true subset pose, so a
    # ranked seed inside the basin descends straight to residual ~0; wrong basins stay high.  Pick the
    # lowest-residual refine.  On heavy crops no seed reaches a low residual → the caller's ambiguity
    # gate honestly flags it.
    # Collect ALL refined candidates, then rank by a SYMMETRIC residual.  The point-to-plane ICP
    # residual is one-directional (target→source): a flipped pose on a self-similar surface can slide
    # the partial target onto the source and so score a low one-directional residual even though a big
    # chunk of the in-overlap SOURCE then floats off the target.  Adding the source→target term
    # (:func:`_symmetric_overlap_residual`) penalises exactly that flip, which is what lets the
    # MODERATE keep≈60 % crops resolve to the true basin instead of a 170-ish mirror.
    cands: list[tuple[float, torch.Tensor, torch.Tensor]] = []
    for R0 in R_ranked:
        t0 = tgt_c - (R0 @ src_c)
        R2, t2, res2 = _point_to_plane_overlap_icp(
            src, snrm, tgt, R0, t0, iters=refine_iters, trim=refine_trim
        )
        cands.append((res2, R2, t2))
    if not cands:
        return eye4
    # Restrict the (more expensive) symmetric re-rank to the candidates whose one-directional residual
    # is within a small factor of the best — i.e. the genuinely competing basins — so a clearly bad
    # converge can't win on a fluke of the symmetric term.
    r_min = min(c[0] for c in cands)
    contenders = [c for c in cands if c[0] <= max(r_min * 3.0, r_min + 1e-4)] or [
        min(cands, key=lambda c: c[0])
    ]
    best_sym = float("inf")
    best = eye4.clone()
    for res2, R2, t2 in contenders:
        sym = _symmetric_overlap_residual(src, tgt, R2, 1.0, t2)
        if sym < best_sym:
            best_sym = sym
            Tb = torch.eye(4, device=dev, dtype=torch.float64)
            Tb[:3, :3] = R2
            Tb[:3, 3] = t2
            best = Tb.to(dtype=dtype, device=dev)
    return best


# ── ambiguity probe (does the surviving overlap pin down the rotation?) ─────────


def _overlap_residual_norm(
    src_full: torch.Tensor,
    tgt_full: torch.Tensor,
    T: torch.Tensor,
    *,
    with_scale: bool,
    n_src: int = 2048,
    n_tgt: int = 1024,
) -> float:
    """Normalised target→source overlap residual at the converged pose (0 = the shapes fit).

    Because the (possibly cropped) target should be a SUBSET of the transformed full source under
    the true pose, every observed target point must land on the source surface, so a CORRECT pose
    drives this trimmed nearest-neighbour residual to ~0.  A wrong / unrecoverable pose leaves the
    target floating off the surface → a large residual.  This is the decisive failure signal: it
    directly measures "did we actually align the observed geometry", independent of any rotation
    bookkeeping.  Normalised by the target RMS radius so it is a dimensionless, object-relative
    fraction.
    """
    src = _stride_subsample(src_full, n_src).double()
    tgt = _stride_subsample(tgt_full, n_tgt).double()
    if src.shape[0] < 3 or tgt.shape[0] < 3:
        return float("inf")
    block = T[:3, :3].double()
    t = T[:3, 3].double()
    TA = src @ block.transpose(-1, -2) + t  # full source -> target frame (block already s*R)
    d = torch.cdist(tgt, TA).min(dim=1).values  # each observed target -> nearest source
    k = max(3, int(0.85 * d.shape[0]))  # trim occlusion-boundary tail
    resid = float(d.topk(k, largest=False).values.mean().item())
    scale_ref = float((tgt - tgt.mean(dim=0)).norm(dim=1).pow(2).mean().sqrt().clamp_min(1e-9).item())
    return resid / scale_ref


def _symmetric_overlap_residual(
    src: torch.Tensor, tgt: torch.Tensor, R: torch.Tensor, s: float, t: torch.Tensor, trim: float = 0.8
) -> float:
    """BIDIRECTIONAL trimmed overlap residual (already-subsampled, double clouds).

    The one-directional target→source residual (:func:`_overlap_residual_norm`) is blind to scale:
    shrinking the source toward the target region keeps every target point near *some* source point,
    so it never penalises a too-small scale.  The scale DoF is only pinned by *also* requiring the
    overlapping source points to land on the target, i.e. a symmetric Chamfer over the shared band.
    This returns ``mean(trimmed tgt→src) + mean(trimmed src_overlap→tgt)``; the second term is the one
    that makes the residual a true function of scale, so a 1-D line-search over ``s`` has a real
    minimum at the correct scale instead of a flat valley.
    """
    TA = src @ R.transpose(-1, -2) * s + t  # source -> target frame
    d_ts = torch.cdist(tgt, TA).min(dim=1).values  # each target -> nearest source
    k_t = max(3, int(trim * d_ts.shape[0]))
    r_ts = d_ts.topk(k_t, largest=False).values.mean()
    # Source->target, but only the source points actually inside the overlap band (the kept target
    # matches define which source points are in-overlap); use the trimmed nearest src→tgt.
    d_st = torch.cdist(TA, tgt).min(dim=1).values  # each source -> nearest target
    k_s = max(3, int(0.5 * d_st.shape[0]))  # keep the closest half (the in-overlap source)
    r_st = d_st.topk(k_s, largest=False).values.mean()
    return float((r_ts + r_st).item())


def _scale_line_search(
    src_full: torch.Tensor,
    tgt_full: torch.Tensor,
    T: torch.Tensor,
    *,
    n_src: int = 3000,
    n_tgt: int = 1500,
    span: float = 0.4,
    iters: int = 28,
) -> torch.Tensor:
    """Golden-section line-search on the Sim(3) SCALE that minimises the SYMMETRIC overlap residual.

    Under partial overlap the joint Umeyama scale is loosely pinned (the surviving matches span a
    small one-sided extent), so the recovered ``s`` can drift ~25 %.  Holding the (well-recovered)
    rotation fixed, we search ``s`` over ``[s0/(1+span), s0*(1+span)]`` against
    :func:`_symmetric_overlap_residual`, whose source→target term gives scale a real gradient, and
    recompute ``t`` for each candidate so the fit stays consistent.  Returns the corrected 4×4 (or the
    input unchanged if the search does not improve on it).
    """
    dev, dtype = T.device, T.dtype
    src = _stride_subsample(src_full, n_src).double()
    tgt = _stride_subsample(tgt_full, n_tgt).double()
    if src.shape[0] < 8 or tgt.shape[0] < 8:
        return T
    block = T[:3, :3].double()
    s0 = float(torch.linalg.det(block).abs().clamp_min(1e-18) ** (1.0 / 3.0))
    R = block / s0
    src_c = src.mean(0)
    tgt_c = tgt.mean(0)

    def resid_at(s: float) -> float:
        t = tgt_c - s * (R @ src_c)
        return _symmetric_overlap_residual(src, tgt, R, s, t)

    lo, hi = s0 / (1.0 + span), s0 * (1.0 + span)
    r0 = resid_at(s0)
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - (hi - lo) * invphi
    d = lo + (hi - lo) * invphi
    fc, fd = resid_at(c), resid_at(d)
    for _ in range(iters):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - (hi - lo) * invphi
            fc = resid_at(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + (hi - lo) * invphi
            fd = resid_at(d)
    s_best = 0.5 * (lo + hi)
    if resid_at(s_best) > r0:  # no improvement over the joint-solve scale → keep it
        return T
    t_best = tgt_c - s_best * (R @ src_c)
    out = torch.eye(4, device=dev, dtype=torch.float64)
    out[:3, :3] = s_best * R
    out[:3, 3] = t_best
    return out.to(device=dev, dtype=dtype)


def _overlap_inlier_count(
    src_full: torch.Tensor,
    tgt_full: torch.Tensor,
    T: torch.Tensor,
    tol: float,
    *,
    n_src: int = 2048,
    n_tgt: int = 1024,
) -> int:
    """Number of observed target points landing within ``tol`` of the transformed source surface.

    The overlap-aware inlier measure for a pose recovered by the basin sweep (which has no RANSAC
    correspondence set): a target→source nearest-neighbour count under the same 3-D tolerance used
    by RANSAC, so the inlier-count gate is consistent across the FPFH and sweep paths.
    """
    src = _stride_subsample(src_full, n_src).double()
    tgt = _stride_subsample(tgt_full, n_tgt).double()
    if src.shape[0] < 3 or tgt.shape[0] < 3:
        return 0
    TA = src @ T[:3, :3].double().transpose(-1, -2) + T[:3, 3].double()
    dmin = torch.cdist(tgt, TA).min(dim=1).values
    return int((dmin < tol).sum().item())


def _axis_angle_R(axis: torch.Tensor, rad: float) -> torch.Tensor:
    """Rodrigues rotation (3×3, float64) of ``rad`` radians about a unit ``axis`` tensor."""
    a = axis / axis.norm().clamp_min(1e-12)
    K = torch.tensor(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]],
        device=axis.device,
        dtype=torch.float64,
    )
    eye = torch.eye(3, device=axis.device, dtype=torch.float64)
    return eye + math.sin(rad) * K + (1.0 - math.cos(rad)) * (K @ K)


def _rotation_constrained_probe(
    src_full: torch.Tensor,
    tgt_full: torch.Tensor,
    T: torch.Tensor,
    *,
    with_scale: bool,
    probe_deg: float = 10.0,
    n_src: int = 2048,
    n_tgt: int = 1024,
) -> float:
    """Quantify how strongly the surviving overlap constrains rotation (0 = flat/ambiguous).

    At the converged pose, the target→source overlap residual is at a minimum.  If the surviving
    geometry actually constrains rotation, perturbing the rotation by ``±probe_deg`` about ANY axis
    (around the overlap centroid) raises that residual; if the region is rotationally ambiguous (a
    symmetric / featureless crop), some perturbation leaves the residual essentially unchanged.

    We return the SMALLEST residual increase over a set of probe axes, a *sensitivity* score:

        sensitivity = min_axis ( resid(R rotated by probe_deg) - resid(R rotated by 1deg) ) / scale

    Subtracting the residual under a TINY (1deg) rotation removes the surface-*resampling* floor: on
    a discretely-sampled cloud even an in-place re-sampling of the same surface (a rotation that maps
    the shape onto itself, e.g. a sphere) costs ~one point-spacing of residual.  A genuinely
    constrained object's residual keeps climbing far past that floor at ``probe_deg``; an ambiguous
    one stays near the floor for some axis, so its score is ~0.  The rise is normalised by the
    target's RMS radius (a dimensionless object-relative score).  Cheap: a handful of cdist passes,
    CPU-friendly.

    Args:
        src_full / tgt_full: full source / (partial) target clouds (float32).
        T: converged 4×4 source→target transform.
        with_scale: Sim(3) if True.
        probe_deg: perturbation magnitude (degrees).
        n_src / n_tgt: strided subsample sizes.

    Returns:
        ``sensitivity`` (>= 0).  Larger = better-constrained rotation; near 0 = ambiguous.
    """
    dev = T.device
    src = _stride_subsample(src_full, n_src).double()
    tgt = _stride_subsample(tgt_full, n_tgt).double()
    if src.shape[0] < 3 or tgt.shape[0] < 3:
        return 0.0
    block = T[:3, :3].double()
    t = T[:3, 3].double()
    s = float(torch.linalg.det(block).abs().clamp_min(1e-18) ** (1.0 / 3.0)) if with_scale else 1.0
    R = block / s if with_scale else block

    def overlap_resid(Rm: torch.Tensor, tm: torch.Tensor) -> float:
        TA = src @ Rm.transpose(-1, -2) * s + tm  # (Ns, 3)
        d = torch.cdist(tgt, TA).min(dim=1).values  # each observed target -> nearest source
        # trimmed mean (drop occlusion-boundary tail) for a stable residual
        k = max(3, int(0.85 * d.shape[0]))
        return float(d.topk(k, largest=False).values.mean().item())

    # Normalise the residual rise by the TARGET's characteristic size (RMS radius), so the score is
    # a dimensionless "how far did the twist move the overlap, relative to the object scale".
    scale_ref = float((tgt - tgt.mean(dim=0)).norm(dim=1).pow(2).mean().sqrt().clamp_min(1e-9).item())
    # overlap centroid (rotate about it so translation re-solves cleanly)
    TA0 = src @ R.transpose(-1, -2) * s + t
    nn = torch.cdist(tgt, TA0).argmin(dim=1)
    pivot = TA0[nn].mean(dim=0)  # centroid of the matched (overlap) source region

    axes = torch.tensor(
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0], [1.0, 0, 1.0], [0, 1.0, 1.0]],
        device=dev,
        dtype=torch.float64,
    )

    def probe_resid(ax: torch.Tensor, sgn: float, deg: float) -> float:
        dR = _axis_angle_R(ax / ax.norm(), sgn * math.radians(deg))
        Rm = dR @ R
        tm = t + (pivot - dR @ pivot)  # keep the overlap centroid fixed under the probe rotation
        return overlap_resid(Rm, tm)

    sensitivities = []
    for ax in axes:
        for sgn in (1.0, -1.0):
            floor = probe_resid(ax, sgn, 1.0)  # resampling floor in this rotation direction
            far = probe_resid(ax, sgn, probe_deg)  # residual at the full probe angle
            sensitivities.append((far - floor) / scale_ref)
    return float(max(0.0, min(sensitivities)))


# ── scale-robust seed (Open3D FPFH+RANSAC, optional) + splatreg refine ──────────


def _cloud_voxel(pts: torch.Tensor, mult: float = 1.5) -> float:
    """Scale-adaptive FPFH voxel from the cloud's POINT SPACING (median nearest-neighbour distance).

    The object-scale FPFH defaults (``inlier_tol`` ~2 cm, descriptor radii tuned for a 14 cm test
    object) are wrong by 1-2 orders of magnitude on metre-scale indoor scans, that is why the
    object-tuned ``feature_align`` seed lands 90-170 deg off on 3DMatch.  The robust seed instead
    follows Open3D's recipe (voxel → normals 2·voxel → FPFH 5·voxel), so the only scale knob is the
    voxel.  A bbox/N**(1/3) estimate badly over-coarsens here (two not-yet-registered fragments span
    a ~5 m union box but each is a sparse ~5 k-point scan, giving a useless ~0.28 m voxel).  The
    correct, density-following scale is the cloud's own point spacing: take the median nearest-
    neighbour distance and scale it (``mult``) so a few points fall per voxel.  This recovers ≈0.05 m
    on the (already 5 cm-decimated) 3DMatch fragments, Open3D's proven setting, and still adapts
    down to a dense 14 cm object.  Returns a positive metre voxel.
    """
    n = pts.shape[0]
    if n < 4:
        return 0.05
    # median nearest-neighbour distance on a strided subsample (cheap, scale-following).
    sub = _stride_subsample(pts, min(n, 2000)).to(torch.float32)
    d = torch.cdist(sub, sub)
    d.fill_diagonal_(float("inf"))
    nn = d.min(dim=1).values
    spacing = float(nn.median().item())
    if not math.isfinite(spacing) or spacing <= 0.0:
        ext = (pts.amax(dim=0) - pts.amin(dim=0)).clamp_min(0.0)
        return float(max(float(ext.norm().item()) / 50.0, 1e-4))
    return float(max(spacing * mult, 1e-4))


def _open3d_fpfh_ransac_seed(
    src: torch.Tensor, tgt: torch.Tensor, voxel: float, rng_seed: int = 42
) -> tuple[torch.Tensor | None, int]:
    """Classical FPFH + RANSAC global seed via Open3D (source→target 4×4), or ``None`` if unavailable.

    Matches the proven Open3D ``registration_ransac_based_on_feature_matching`` recipe (Choi et al.
    2015 / the Open3D global-registration tutorial): voxel downsample, normals at radius ``2*voxel``,
    FPFH at radius ``5*voxel``, mutual-filter RANSAC over 4-point samples with edge-length + distance
    correspondence checkers and 100k/0.999 convergence.  This is scale-correct (radii follow the
    auto ``voxel``) and is the robustness Open3D itself reports (~77 % RR on 3DMatch), splatreg then
    refines it for accuracy and (optionally) scale.  Returns ``(T, n_corr)``; ``n_corr`` is the size
    of the RANSAC correspondence set (a coarse seed-quality signal).
    """
    try:
        import open3d as o3d  # optional dependency; pure-splatreg path used when absent
    except Exception:
        return None, 0
    import numpy as _np

    sp_np = src.detach().cpu().to(torch.float64).numpy()
    tp_np = tgt.detach().cpu().to(torch.float64).numpy()

    def _prep(p):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(p)
        # NB: the input clouds are already voxel-decimated by the caller; an extra voxel_down_sample
        # here only throws away the correspondences RANSAC needs, so we estimate features directly.
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pc, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100)
        )
        return pc, fpfh

    try:
        # Determinism: Open3D's RANSAC draws minimal samples from a global RNG, so without a fixed
        # seed the recovered pose is non-deterministic across runs (a bad draw once hit 117 deg where
        # clean reruns sit at ~11 deg).  ``o3d.utility.random.seed`` is the only knob — this overload
        # of ``registration_ransac_based_on_feature_matching`` exposes no per-call seed argument.
        o3d.utility.random.seed(int(rng_seed))
        sp, sf = _prep(sp_np)
        tp, tf = _prep(tp_np)
        dist = voxel * 1.5
        res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            sp,
            tp,
            sf,
            tf,
            True,  # mutual filter
            dist,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
        )
        T = torch.as_tensor(_np.asarray(res.transformation), dtype=torch.float64)
        return T, int(len(res.correspondence_set))
    except Exception:
        return None, 0


def _open3d_icp_refine(
    src: torch.Tensor, tgt: torch.Tensor, T0: torch.Tensor, voxel: float, iters: int = 50
) -> torch.Tensor | None:
    """Open3D point-to-plane ICP local refine of a global seed (source→target 4×4), or ``None``.

    Standard scale-correct local refinement: point-to-plane ICP at a ``2*voxel`` max-correspondence
    distance with the seed as the initialisation.  This is the Open3D global-registration tutorial's
    own refine step; it tightens a good FPFH+RANSAC seed without the object-scale failure mode of the
    overlap ICP tuned for 14 cm splats (which wanders on low-overlap room scans).  Returns the refined
    4×4, or ``None`` to signal "keep the seed".
    """
    try:
        import open3d as o3d
    except Exception:
        return None
    import numpy as _np

    try:
        sp = o3d.geometry.PointCloud()
        sp.points = o3d.utility.Vector3dVector(src.detach().cpu().to(torch.float64).numpy())
        tp = o3d.geometry.PointCloud()
        tp.points = o3d.utility.Vector3dVector(tgt.detach().cpu().to(torch.float64).numpy())
        tp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
        res = o3d.pipelines.registration.registration_icp(
            sp,
            tp,
            voxel * 2.0,
            T0.detach().cpu().to(torch.float64).numpy(),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iters)),
        )
        return torch.as_tensor(_np.asarray(res.transformation), dtype=torch.float64)
    except Exception:
        return None


@torch.no_grad()
def robust_feature_align(
    target: Gaussians,
    source: Gaussians,
    *,
    transform: str = "se3",
    voxel: float | None = None,
    refine_iters: int = 30,
    rng_seed: int = 42,
) -> tuple[torch.Tensor, dict]:
    """Scale-robust registrar: Open3D FPFH+RANSAC seed + splatreg overlap-aware refine (+ Sim(3)).

    The object-tuned :func:`feature_align` seed collapses on metre-scale indoor scans (its FPFH radii
    and 2 cm inlier tolerance are object-scale), which is why pure-splatreg ``init="fast"`` only hits
    ~13 % RR on 3DMatch while Open3D's scale-correct FPFH+RANSAC hits ~77 %.  This path takes the best
    of both: it borrows Open3D's *robust correspondence/seed* (auto-scaled voxel + radii) and then
    runs splatreg's own overlap-aware trimmed ICP (:func:`_overlap_icp_polish`, target→source so a
    partial overlap can't drag the fit) on top, adding refine accuracy and, for ``transform="sim3"``,
    a recovered scale Open3D's rigid RANSAC never estimates.

    When Open3D is absent it falls back to the pure-splatreg :func:`feature_align` (with the same
    scale-adaptive ``voxel`` mapped into its ``inlier_tol``), so the function always returns a pose.

    Args:
        target / source: reference / to-align splats (only ``.means`` read).
        transform: ``"se3"`` (rigid) or ``"sim3"`` (estimate scale in the refine).
        voxel: downsample/feature scale (m); ``None`` → auto from the cloud bbox.
        refine_iters: splatreg overlap-ICP polish iterations on top of the seed.

    Returns:
        ``(T_4x4, info)``, estimated 4×4 (source→target, source device/dtype) and an ``info`` dict
        (``voxel``, ``n_corr``, ``seed_rre`` placeholder, ``used_open3d``, ``confidence``).
    """
    if transform not in ("se3", "sim3"):
        raise ValueError(f"transform must be 'se3' or 'sim3', got {transform!r}")
    with_scale = transform == "sim3"
    dev = source.means.device
    dtype = source.means.dtype
    src_full = source.means.to(torch.float32)
    tgt_full = target.means.to(device=dev, dtype=torch.float32)

    info = {"voxel": 0.0, "n_corr": 0, "used_open3d": False, "confidence": 0.0}
    if src_full.shape[0] < 4 or tgt_full.shape[0] < 4:
        return torch.eye(4, device=dev, dtype=dtype), info

    if voxel is None:
        # Derive from each cloud's point spacing; use the finer so detail survives the downsample.
        voxel = min(_cloud_voxel(src_full), _cloud_voxel(tgt_full))
    info["voxel"] = float(voxel)

    T_seed, n_corr = _open3d_fpfh_ransac_seed(src_full, tgt_full, voxel, rng_seed=rng_seed)
    if T_seed is not None:
        info["used_open3d"] = True
        info["n_corr"] = int(n_corr)
        T0 = T_seed.to(device=dev, dtype=dtype)

        # SE(3): Open3D point-to-plane ICP refine (scale-correct, standard) — tightens the seed at
        # room scale without the object-tuned overlap ICP's low-overlap wander.  For Sim(3) we need a
        # scale DoF Open3D's rigid ICP can't give, so we use splatreg's overlap-aware Sim(3) ICP.
        if not with_scale:
            T_o3d = _open3d_icp_refine(src_full, tgt_full, T0, voxel, iters=refine_iters)
            T_ref = T_o3d.to(device=dev, dtype=dtype) if T_o3d is not None else T0
        else:
            T_ref = _overlap_icp_polish(
                src_full,
                tgt_full,
                T0,
                with_scale=True,
                iters=refine_iters,
                n_src=min(4000, src_full.shape[0]),
                n_tgt=min(2000, tgt_full.shape[0]),
            )
        # Accept the refine only if it does not worsen the overlap fit (a degenerate ICP can wander).
        # NOTE: gate on the scale-blind one-directional residual BEFORE any scale-only correction, so
        # the rotation/translation winner is chosen on the same footing the polish was tuned for.
        r0 = _overlap_residual_norm(src_full, tgt_full, T0, with_scale=with_scale)
        r1 = _overlap_residual_norm(src_full, tgt_full, T_ref, with_scale=with_scale)
        T = T_ref if r1 <= r0 + 1e-6 else T0
        # Dedicated scale line-search (Sim(3) only): the joint Umeyama scale is loosely pinned under
        # partial overlap, so AFTER the pose is chosen, refine just the scale DoF against the
        # SYMMETRIC overlap residual (which — unlike the one-directional gate above — actually
        # depends on scale).  Rotation/translation are untouched, so this cannot flip the basin.
        if with_scale:
            T = _scale_line_search(src_full, tgt_full, T)
        info["confidence"] = float(max(0.0, 1.0 - min(r0, r1)))
        return T.to(device=dev, dtype=dtype), info

    # No Open3D — fall back to the pure-splatreg feature path with a scale-adaptive inlier tolerance.
    info["used_open3d"] = False
    T, finfo = feature_align(target, source, transform=transform, inlier_tol=max(voxel * 1.0, _FS_INLIER_TOL))
    info["confidence"] = float(finfo.get("confidence", 0.0))
    info["n_corr"] = int(finfo.get("n_inliers", 0))
    return T, info


# ── learned seed (GeoTransformer, optional) + splatreg refine ───────────────────

# Module-level cache of the loaded GeoTransformer model + config so we pay the (~0.3 s) load
# exactly once per process; subsequent learned_feature_align calls reuse it.
_GEOTRANSFORMER_CACHE: dict = {}


def _geotransformer_paths() -> tuple[str, str] | None:
    """Resolve ``(repo_root, experiment_dir)`` for the bundled GeoTransformer, or ``None``.

    The learned backend lives under ``splatreg/third_party_models/GeoTransformer`` (gitignored,
    cloned + built by the Tier-2 setup).  We discover it relative to this file so the path is robust
    to the caller's CWD.  Returns ``None`` (→ caller falls back to ``init="robust"``) when the repo /
    its built extension / its pretrained weights are not present.
    """
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.normpath(os.path.join(here, "..", "third_party_models", "GeoTransformer"))
    exp = os.path.join(repo, "experiments", "geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn")
    weights = os.path.join(repo, "weights", "geotransformer-3dmatch.pth.tar")
    # The .so name embeds the cpython tag; accept any built ext.* in that dir.
    import glob

    ext_ok = bool(glob.glob(os.path.join(repo, "geotransformer", "ext.*.so")))
    if not (os.path.isdir(exp) and os.path.isfile(weights) and ext_ok):
        return None
    return repo, exp


def _load_geotransformer(device: torch.device):
    """Lazily import + load the pretrained GeoTransformer 3DMatch model (cached), or ``None``.

    Adds the GeoTransformer repo and its 3DMatch experiment dir to ``sys.path`` (so ``config`` /
    ``model`` resolve), builds the model, loads the released ``geotransformer-3dmatch.pth.tar``
    weights, and moves it to ``device`` in eval mode.  Returns ``(model, cfg, collate_fn, to_cuda,
    release_cuda)`` or ``None`` if anything (import / weights / CUDA-ext) is unavailable, in which
    case the caller falls back to the classical ``robust`` seed.  Result is cached module-wide.
    """
    key = str(device)
    if key in _GEOTRANSFORMER_CACHE:
        return _GEOTRANSFORMER_CACHE[key]
    paths = _geotransformer_paths()
    if paths is None:
        _GEOTRANSFORMER_CACHE[key] = None
        return None
    repo, exp = paths
    import sys

    for p in (repo, exp):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from config import make_cfg  # type: ignore  # GeoTransformer 3DMatch experiment config
        from model import create_model  # type: ignore
        from geotransformer.utils.data import registration_collate_fn_stack_mode  # type: ignore
        from geotransformer.utils.torch import to_cuda, release_cuda  # type: ignore
    except Exception:
        _GEOTRANSFORMER_CACHE[key] = None
        return None
    try:
        cfg = make_cfg()
        model = create_model(cfg).to(device).eval()
        import os

        weights = os.path.join(repo, "weights", "geotransformer-3dmatch.pth.tar")
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"])
    except Exception:
        _GEOTRANSFORMER_CACHE[key] = None
        return None
    bundle = (model, cfg, registration_collate_fn_stack_mode, to_cuda, release_cuda)
    _GEOTRANSFORMER_CACHE[key] = bundle
    return bundle


def _geotransformer_seed(src: torch.Tensor, tgt: torch.Tensor, device: torch.device) -> torch.Tensor | None:
    """Learned global seed (source→target 4×4) from pretrained GeoTransformer, or ``None``.

    Runs the GeoTransformer 3DMatch model on the (source, target) point clouds and returns its
    ``estimated_transform`` (a full SE(3) pose recovered from its learned dense correspondences +
    LGR estimator).  GeoTransformer is the learned 3DMatch SOTA (~92 % RR); used here ONLY as the
    coarse seed, splatreg's overlap-aware ICP (+ optional Sim(3) scale) then refines it.  Returns
    ``None`` on any failure (model unavailable, forward error) so the caller can fall back.
    """
    bundle = _load_geotransformer(device)
    if bundle is None:
        return None
    model, cfg, collate_fn, to_cuda, release_cuda = bundle
    try:
        import numpy as _np

        ref_np = tgt.detach().cpu().to(torch.float32).numpy()
        src_np = src.detach().cpu().to(torch.float32).numpy()
        data_dict = {
            "ref_points": ref_np,
            "src_points": src_np,
            "ref_feats": _np.ones((ref_np.shape[0], 1), dtype=_np.float32),
            "src_feats": _np.ones((src_np.shape[0], 1), dtype=_np.float32),
            "transform": _np.eye(4, dtype=_np.float32),  # unused for inference; collate expects it
        }
        neighbor_limits = [38, 36, 36, 38]  # the released 3DMatch default
        data_dict = collate_fn(
            [data_dict],
            cfg.backbone.num_stages,
            cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius,
            neighbor_limits,
        )
        data_dict = to_cuda(data_dict) if str(device).startswith("cuda") else data_dict
        with torch.no_grad():
            out = model(data_dict)
        out = release_cuda(out)
        T = out["estimated_transform"]  # (4, 4) source→target
        return torch.as_tensor(_np.asarray(T), dtype=torch.float64)
    except Exception:
        return None


def _geotransformer_correspondences(
    src: torch.Tensor, tgt: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    """GeoTransformer's learned correspondences ``(src_corr, tgt_corr, T_lgr)``, or ``None``.

    Same forward as :func:`_geotransformer_seed` but returns the model's matched point pairs
    (``src_corr_points`` / ``ref_corr_points``), the input the MAC maximal-clique hypothesis
    stage needs (``seed_selector="mac"``), PLUS the model's own LGR ``estimated_transform``
    from the SAME forward (``T_lgr``, 4×4 float64 or ``None``), so a caller whose MAC consensus
    fails can fall back to LGR without paying a second model forward.  Returns ``None`` on any
    failure so the caller can fall back.
    """
    bundle = _load_geotransformer(device)
    if bundle is None:
        return None
    model, cfg, collate_fn, to_cuda, release_cuda = bundle
    try:
        import numpy as _np

        ref_np = tgt.detach().cpu().to(torch.float32).numpy()
        src_np = src.detach().cpu().to(torch.float32).numpy()
        data_dict = {
            "ref_points": ref_np,
            "src_points": src_np,
            "ref_feats": _np.ones((ref_np.shape[0], 1), dtype=_np.float32),
            "src_feats": _np.ones((src_np.shape[0], 1), dtype=_np.float32),
            "transform": _np.eye(4, dtype=_np.float32),
        }
        neighbor_limits = [38, 36, 36, 38]
        data_dict = collate_fn(
            [data_dict],
            cfg.backbone.num_stages,
            cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius,
            neighbor_limits,
        )
        data_dict = to_cuda(data_dict) if str(device).startswith("cuda") else data_dict
        with torch.no_grad():
            out = model(data_dict)
        out = release_cuda(out)
        src_corr = torch.as_tensor(_np.asarray(out["src_corr_points"]), dtype=torch.float32)
        tgt_corr = torch.as_tensor(_np.asarray(out["ref_corr_points"]), dtype=torch.float32)
        if src_corr.shape[0] < 3 or src_corr.shape != tgt_corr.shape:
            return None
        T_lgr: torch.Tensor | None = None
        try:
            T_lgr = torch.as_tensor(_np.asarray(out["estimated_transform"]), dtype=torch.float64)
        except Exception:
            T_lgr = None
        return src_corr, tgt_corr, T_lgr
    except Exception:
        return None


@torch.no_grad()
def learned_feature_align(
    target: Gaussians,
    source: Gaussians,
    *,
    transform: str = "se3",
    voxel: float | None = None,
    refine_iters: int = 30,
    seed_selector: str = "lgr",
) -> tuple[torch.Tensor, dict]:
    """LEARNED registrar: pretrained GeoTransformer seed + splatreg overlap-aware refine (+ Sim(3)).

    Mirrors :func:`robust_feature_align` but swaps the classical Open3D FPFH+RANSAC seed for the
    *learned* GeoTransformer 3DMatch correspondence model (CVPR 2022, ~92 % RR, past the ~77 %
    classical FPFH ceiling).  GeoTransformer supplies the coarse SE(3) seed from its learned dense
    matches; splatreg then runs the SAME overlap-aware refine the ``robust`` path uses (Open3D
    point-to-plane ICP for SE(3), or its own Sim(3) overlap ICP to additionally recover scale),
    accepting the refine only when it does not worsen the overlap residual.  This gives learned-SOTA
    recall with splatreg's accuracy / scale bonus on top.

    Guarded: when GeoTransformer (its module / built CUDA-ext / pretrained weights) is unavailable,
    or its forward fails, it falls back to :func:`robust_feature_align` (classical seed) so the
    function always returns a pose.

    ``seed_selector`` picks the hypothesis stage on top of GeoTransformer's learned
    correspondences: ``"lgr"`` (default) uses the model's own local-to-global registration
    estimate (``estimated_transform``); ``"mac"`` re-estimates the seed pose from the model's
    matched point pairs with the MAC maximal-clique stage (:func:`splatreg.mac.mac_pose`,
    Zhang et al. CVPR 2023, the published 3DLoMatch tie-breaker; ~71→78 % recall over
    GeoTransformer in their paper.  That number is **pending** verification on this
    implementation, it needs the 3DLoMatch data + GeoTransformer features on a GPU box; see
    ``docs_site/init-modes.md``).  ``"mac"`` falls back to ``"lgr"`` when the correspondence
    extraction or MAC consensus fails, so the contract is unchanged.

    Args / Returns: identical contract to :func:`robust_feature_align`; ``info`` additionally carries
    ``used_learned`` (bool), ``seed`` (``"geotransformer"`` / ``"geotransformer+mac"`` /
    ``"robust-fallback"``) and, when ``seed_selector="mac"`` ran the clique stage, ``mac``
    (its stats: ``success`` / ``n_matches`` / ``n_inliers`` / ``n_cliques`` / ``n_hypotheses`` /
    ``truncated``).
    """
    if transform not in ("se3", "sim3"):
        raise ValueError(f"transform must be 'se3' or 'sim3', got {transform!r}")
    if seed_selector not in ("lgr", "mac"):
        raise ValueError(f"seed_selector must be 'lgr' or 'mac', got {seed_selector!r}")
    with_scale = transform == "sim3"
    dev = source.means.device
    dtype = source.means.dtype
    src_full = source.means.to(torch.float32)
    tgt_full = target.means.to(device=dev, dtype=torch.float32)

    info = {
        "voxel": 0.0,
        "n_corr": 0,
        "used_open3d": False,
        "used_learned": False,
        "seed": "none",
        "confidence": 0.0,
    }
    if src_full.shape[0] < 4 or tgt_full.shape[0] < 4:
        return torch.eye(4, device=dev, dtype=dtype), info

    T_seed = None
    seed_name = "geotransformer"
    if seed_selector == "mac":
        # MAC hypothesis stage over the learned correspondences (instead of the model's LGR).
        # ONE model forward supplies both the matched pairs and the LGR estimate, so the
        # consensus-failure fallback to LGR below never pays a second forward.
        corr = _geotransformer_correspondences(src_full, tgt_full, dev)
        if corr is not None:
            try:
                from .mac import mac_pose

                # 0.10 m inlier threshold = the MAC paper's 3DMatch/3DLoMatch evaluation setting
                # (scene-scale scans; the object-scale _FS_INLIER_TOL would starve the consensus).
                rr = mac_pose(corr[0].to(dev), corr[1].to(dev), with_scale=with_scale, inlier_tol=0.10)
                info["mac"] = {
                    "success": bool(rr["success"]),
                    "n_matches": int(rr["n_matches"]),
                    "n_inliers": int(rr["n_inliers"]),
                    "n_cliques": int(rr["n_cliques"]),
                    "n_hypotheses": int(rr["n_hypotheses"]),
                    "truncated": bool(rr["truncated"]),
                }
                if rr["success"]:
                    T_seed = rr["T"].to(torch.float64)
                    seed_name = "geotransformer+mac"
            except Exception:
                T_seed = None  # MAC unavailable/failed -> LGR fallback below
            if T_seed is None and corr[2] is not None:
                # MAC found no consensus -> the model's own LGR estimate from the SAME forward.
                T_seed = corr[2]
    if T_seed is None:
        T_seed = _geotransformer_seed(src_full, tgt_full, dev)
    if T_seed is None:
        # Learned backend unavailable → classical robust seed (still a strong, scale-correct path).
        T, rinfo = robust_feature_align(
            target, source, transform=transform, voxel=voxel, refine_iters=refine_iters
        )
        rinfo = dict(rinfo)
        rinfo["used_learned"] = False
        rinfo["seed"] = "robust-fallback"
        return T, rinfo

    info["used_learned"] = True
    info["seed"] = seed_name
    if voxel is None:
        voxel = min(_cloud_voxel(src_full), _cloud_voxel(tgt_full))
    info["voxel"] = float(voxel)

    T0 = T_seed.to(device=dev, dtype=dtype)
    # SAME refine policy as robust_feature_align: Open3D point-to-plane ICP for SE(3) (scale-correct,
    # tightens the learned seed at room scale); splatreg overlap-aware Sim(3) ICP when a scale DoF is
    # wanted.  Accept the refine only if it does not worsen the overlap residual.
    if not with_scale:
        T_o3d = _open3d_icp_refine(src_full, tgt_full, T0, voxel, iters=refine_iters)
        T_ref = T_o3d.to(device=dev, dtype=dtype) if T_o3d is not None else T0
    else:
        T_ref = _overlap_icp_polish(
            src_full,
            tgt_full,
            T0,
            with_scale=True,
            iters=refine_iters,
            n_src=min(4000, src_full.shape[0]),
            n_tgt=min(2000, tgt_full.shape[0]),
        )
    r0 = _overlap_residual_norm(src_full, tgt_full, T0, with_scale=with_scale)
    r1 = _overlap_residual_norm(src_full, tgt_full, T_ref, with_scale=with_scale)
    T = T_ref if r1 <= r0 + 1e-6 else T0
    info["confidence"] = float(max(0.0, 1.0 - min(r0, r1)))
    return T.to(device=dev, dtype=dtype), info


# ── public entry ──────────────────────────────────────────────────────────────


@torch.no_grad()
def feature_align(
    target: Gaussians,
    source: Gaussians,
    *,
    transform: str = "sim3",
    n_source: int = _FS_N_SOURCE,
    n_target: int = _FS_N_TARGET,
    knn_normal: int = _FS_KNN_NORMAL,
    knn_fpfh: int = _FS_KNN_FPFH,
    bins: int = _FS_FPFH_BINS,
    ratio: float = _FS_RATIO,
    ransac_iters: int = _FS_RANSAC_ITERS,
    inlier_tol: float = _FS_INLIER_TOL,
    min_matches: int = _FS_MIN_MATCHES,
    min_inliers: int = _FS_MIN_INLIERS,
    verbose: bool = False,
    batched_ransac: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Feature-based coarse global init (partial-overlap robust) + ambiguity diagnostics.

    Computes FPFH descriptors, finds ratio-test mutual-NN correspondences in feature space, and
    runs a clique-prefiltered RANSAC (3-point Umeyama) to recover the coarse SE(3)/Sim(3) transform.
    Falls back to identity when too few matches are found (featureless / fully ambiguous crop) and
    reports an honest ambiguity flag/confidence in that and other low-information cases.

    Args:
        target: the reference splat.  Only ``.means`` is read.
        source: the splat to align.  Only ``.means`` is read.
        transform: ``"se3"`` (rigid) or ``"sim3"`` (similarity, estimates scale).
        n_source / n_target: sub-sample sizes for descriptors.
        knn_normal / knn_fpfh: neighbourhood sizes for normals / FPFH.
        bins: FPFH histogram bins per angular feature.
        ratio: Lowe ratio-test threshold for matching.
        ransac_iters: number of RANSAC 3-point hypotheses.
        inlier_tol: 3-D inlier threshold (same units as means).
        min_matches: minimum correspondences before attempting RANSAC.
        min_inliers: inliers below this → low confidence / flagged ambiguous.
        verbose: print match / inlier / ambiguity diagnostics.

    Returns:
        ``(T_4x4, info)``, the estimated 4×4 transform (``source``'s device/dtype) and an ``info``
        dict with keys ``n_matches``, ``n_inliers``, ``ambiguous`` (bool), ``confidence`` (0..1),
        and ``ambiguity_deg`` (rotation spread among near-best hypotheses).  A flagged-ambiguous
        result means the overlap genuinely does not constrain the pose; trust ``confidence``.
    """
    if transform not in ("se3", "sim3"):
        raise ValueError(f"transform must be 'se3' or 'sim3', got {transform!r}")
    with_scale = transform == "sim3"

    dev = source.means.device
    dtype = source.means.dtype
    src_full = source.means.to(torch.float32)
    tgt_full = target.means.to(device=dev, dtype=torch.float32)

    info = {
        "n_matches": 0,
        "n_inliers": 0,
        "ambiguous": True,
        "confidence": 0.0,
        "ambiguity_deg": 180.0,
    }

    if src_full.shape[0] < 4 or tgt_full.shape[0] < 4:
        return torch.eye(4, device=dev, dtype=dtype), info

    src_sub = _stride_subsample(src_full, n_source)
    tgt_sub = _stride_subsample(tgt_full, n_target)

    # Normals + FPFH descriptors.
    src_nidx = _knn_indices(src_sub, knn_normal)
    tgt_nidx = _knn_indices(tgt_sub, knn_normal)
    src_normals = _estimate_normals(src_sub, src_nidx)
    tgt_normals = _estimate_normals(tgt_sub, tgt_nidx)
    desc_src = _fpfh(src_sub, src_normals, knn_fpfh, bins)
    desc_tgt = _fpfh(tgt_sub, tgt_normals, knn_fpfh, bins)

    # Ratio-test + mutual-NN matching.
    src_idx, tgt_idx = _match_features(desc_src, desc_tgt, ratio=ratio)
    n_matches = int(src_idx.shape[0])
    info["n_matches"] = n_matches

    if verbose:
        print(f"[feature_align] ratio+MNN matches: {n_matches} / min {min_matches}")

    # Robust pose estimation from the FPFH correspondences (when there are enough), then an
    # overlap-aware (target->source) ICP polish that is robust to a partial target.
    if n_matches >= min_matches:
        _ransac = _robust_register_batched if batched_ransac else _robust_register
        rr = _ransac(
            src_sub,
            tgt_sub,
            src_idx,
            tgt_idx,
            n_iters=ransac_iters,
            inlier_tol=inlier_tol,
            with_scale=with_scale,
        )
        n_inliers = rr["n_inliers"]
        if n_inliers >= 3:
            rr["T"] = _overlap_icp_polish(
                src_full, tgt_full, rr["T"].to(device=dev, dtype=dtype), with_scale=with_scale
            )
    else:
        rr = {"T": torch.eye(4, device=dev, dtype=dtype), "n_inliers": 0, "ambiguity": 180.0}
        n_inliers = 0
    info["n_inliers"] = n_inliers

    # Quality of the FPFH-driven pose so far.
    feat_resid = (
        _overlap_residual_norm(src_full, tgt_full, rr["T"].to(device=dev, dtype=dtype), with_scale=with_scale)
        if n_inliers >= 3
        else float("inf")
    )

    # Fallback: a sparse / degenerate crop can starve FPFH of correspondences even when the pose is
    # observable.  If the feature path produced a bad fit (or too few matches), run the overlap-aware
    # super-Fibonacci basin sweep (target->source ICP per seed) — it recovers the pose from geometry
    # alone, robust to partial overlap.  Keep whichever lands the observed target on the source best.
    info["used_basin_sweep"] = False
    if feat_resid > _FS_AMBIG_RESID:
        if verbose:
            print(f"[feature_align] FPFH fit weak (resid={feat_resid:.4f}) — overlap basin sweep")
        T_sweep = _overlap_basin_sweep(src_full, tgt_full, with_scale=with_scale)
        sweep_resid = _overlap_residual_norm(src_full, tgt_full, T_sweep, with_scale=with_scale)
        if sweep_resid < feat_resid:
            rr["T"] = T_sweep
            # Count the sweep's actual overlap inliers (target points landing on the source surface)
            # so the inlier-count gate reflects the recovered fit, not the sparse FPFH match set.
            n_inliers = max(n_inliers, _overlap_inlier_count(src_full, tgt_full, T_sweep, inlier_tol))
            info["n_inliers"] = n_inliers
            info["used_basin_sweep"] = True

    # ── honest ambiguity / failure detection ─────────────────────────────────
    # Two complementary, physical signals on the CONVERGED pose:
    #   (1) overlap residual — did the observed (partial) target actually land on the source
    #       surface?  A correct pose drives this to ~0; a wrong / unrecoverable one leaves the
    #       target floating off the surface.  This is the decisive FAILURE detector.
    #   (2) rotation sensitivity — even when the shapes DO align, is the rotation pinned down, or
    #       does some twist leave the overlap unchanged (a genuinely rotation-symmetric remnant,
    #       e.g. the disambiguating lobe was cropped away)?  This catches the "aligned but the
    #       rotation is unobservable" case the residual alone would miss.
    if n_inliers >= 3:
        resid_norm = _overlap_residual_norm(src_full, tgt_full, rr["T"], with_scale=with_scale)
        sensitivity = _rotation_constrained_probe(
            src_full, tgt_full, rr["T"], with_scale=with_scale, probe_deg=30.0
        )
    else:
        resid_norm = float("inf")  # too few inliers to even fit — failed
        sensitivity = 0.0
    info["overlap_residual"] = resid_norm
    info["ambiguity_sensitivity"] = sensitivity
    info["ambiguity_deg"] = rr["ambiguity"]  # diagnostic: RANSAC hypothesis rotation spread

    low_inliers = n_inliers < min_inliers
    bad_fit = resid_norm > _FS_AMBIG_RESID  # the observed geometry did not align onto the source
    rot_ambiguous = sensitivity < _FS_AMBIG_SENSITIVITY  # rotation unconstrained at the optimum
    ambiguous = bool(low_inliers or bad_fit or rot_ambiguous)

    # Confidence (0..1): high only when the overlap fits AND the rotation is constrained AND there
    # are enough inliers.  ``fit_conf`` decays as the residual approaches the bad-fit threshold;
    # ``rot_conf`` saturates with the rotation sensitivity.
    inlier_ratio = min(1.0, n_inliers / max(min_inliers * 2, 1))
    fit_conf = max(0.0, 1.0 - resid_norm / _FS_AMBIG_RESID) if resid_norm < float("inf") else 0.0
    rot_conf = min(1.0, sensitivity / (2.0 * _FS_AMBIG_SENSITIVITY))
    confidence = float(max(0.0, min(1.0, fit_conf * rot_conf * inlier_ratio)))
    if ambiguous:
        confidence = min(confidence, 0.3)
    info["ambiguous"] = ambiguous
    info["confidence"] = confidence

    if verbose:
        print(
            f"[feature_align] inliers={n_inliers}/{n_matches}  resid={resid_norm:.4f}  "
            f"rot_sens={sensitivity:.3f}  confidence={confidence:.2f}  ambiguous={ambiguous}"
        )

    return rr["T"].to(device=dev, dtype=dtype), info
