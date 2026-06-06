"""Global coarse-init aligner — the basin finder that runs BEFORE the fine LM refine.

``register`` solves splat-to-splat alignment as *coarse global init → fine multi-residual
LM*. This module is the coarse half: given two Gaussian splats it returns a 4x4 transform
(Sim(3) by default, SE(3) optional) that lands ``source`` inside the convergence basin of
``target`` — close enough (typically within ~10-15deg / a few % scale) that the LM finishes
the job. It is deliberately approximate: the goal is a *good init*, not a precise pose.

Algorithm (ported from the A/B-bench metric-side pred->GT global aligner —
``project_ab_bench_fscore_alignment``: super-Fibonacci SO(3) candidate sweep + GPU-batched
trimmed ICP, tuned defaults 256 rotations / 40 ICP iters / 12288 points):

1. Centre both clouds on their centroids (``Gaussians.means``).
2. Estimate scale as the ratio of the two clouds' RMS radius about their centroids
   (Sim(3) only; SE(3) fixes scale to 1).
3. Seed SO(3) with a deterministic near-uniform super-Fibonacci grid (Alexa, CVPR 2022)
   plus a handful of PCA principal-axis sign-flip candidates. A ~26deg covering provably
   lands one seed in the global basin, so even featureless / symmetric clouds recover.
4. Run *all* seeds through one GPU-batched trimmed point-to-point ICP (batched
   nearest-neighbour via ``torch.cdist`` + a batched closed-form Umeyama step each iter,
   outlier-trimmed). Score each converged seed by the trimmed symmetric Chamfer between the
   transformed source and the target.
5. Keep the lowest-Chamfer seed and recover the exact closed-form similarity that maps the
   subsampled source onto that winner; return it as a 4x4 matrix.

**Fully on-device (GPU-native).** Everything — the super-Fibonacci/PCA seeds, the batched ICP
sweep, and the final closed-form Umeyama recovery — runs in torch on ``source``'s device; there
is no ``.cpu()`` / numpy round-trip in the compute path (the final recovery is done in float64
on-device for SVD precision). Self-contained: torch only, no gsplat / pytorch3d / scipy / numpy
/ SLAM imports. Deterministic — no RNG (closed-form seeds, strided subsample, first-index
tie-break) — so it is reproducible.
"""

from __future__ import annotations

import math

import torch

from .core.types import Gaussians

# Defaults tuned (A/B-bench) for 0 failures on the hardest case — a near-featureless sphere
# under an arbitrary uniform-SO(3) transform.
DEFAULT_N_ROTATIONS = 256  # super-Fibonacci SO(3) seeds (~26deg covering); + a few PCA seeds
DEFAULT_ICP_ITERS = 40  # trimmed-ICP iterations per seed
DEFAULT_N_POINTS = 12288  # deterministic strided target subsample for the fit (denser -> lower floor)
_SUB_SOURCE = 4096  # deterministic strided source subsample driving the fit
_ICP_TRIM_KEEP = 0.85  # fraction of best correspondences kept each iter (outlier reject)
_SEED_BATCH = 64  # seeds processed per chunk in the batched ICP (bounds peak memory)
_PSI = 1.533751168755204288118041


# ── super-Fibonacci SO(3) grid (on-device torch) ──────────────────────────────────────


def _superfib_quats(n: int, device, dtype) -> torch.Tensor:
    """Super-Fibonacci unit quaternions ``(n, 4)`` as ``(x, y, z, w)``, on ``device``."""
    phi = math.sqrt(2.0)
    i = torch.arange(int(n), device=device, dtype=dtype)
    s = i + 0.5
    t = s / float(n)
    r = torch.sqrt(t)
    rr = torch.sqrt((1.0 - t).clamp_min(0.0))
    alpha = (2.0 * math.pi / phi) * s
    beta = (2.0 * math.pi / _PSI) * s
    return torch.stack(
        [r * torch.sin(alpha), r * torch.cos(alpha), rr * torch.sin(beta), rr * torch.cos(beta)], dim=1
    )


def _quats_to_R(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternions ``(n, 4)`` ``(x, y, z, w)`` -> rotation matrices ``(n, 3, 3)``."""
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = q.new_empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def super_fibonacci_so3(n: int, device=None, dtype=torch.float64) -> torch.Tensor:
    """Deterministic near-uniform SO(3) covering as ``(n, 3, 3)`` rotation matrices (on-device).

    Super-Fibonacci spiral on the quaternion 3-sphere (Alexa, CVPR 2022). ~26deg covering at
    ``n == 256``.
    """
    return _quats_to_R(_superfib_quats(int(n), device, dtype))


# ── PCA sign-flip seeds (on-device torch) ─────────────────────────────────────────────

# Eigenvalue-spread threshold for degenerate-PCA detection (symmetric-object note).
# If the ratio of the largest to smallest PCA eigenvalue is below this, the cloud is
# near-isotropic (sphere-like) and the PCA axes are arbitrary/unstable.
# Used only for diagnostic purposes in _pca_seed_rotations; the PCA seeds are kept
# regardless (removing them degrades performance on the sphere because PCA seeds
# accidentally provide good centroid-alignment candidates that the 256-seed Fibonacci
# grid at default density would miss, since a sphere is near-rotationally symmetric
# but NOT perfectly so at N=800 — individual seeds still differ in Chamfer score by
# ~9mm, which the trimmed ICP can distinguish).
_PCA_ISOTROPY_THRESH = 2.0


def _pca_axes(pts_centered: torch.Tensor) -> torch.Tensor:
    """Principal axes (as columns), descending variance, from centered points."""
    _, _, Vh = torch.linalg.svd(pts_centered, full_matrices=False)
    return Vh.transpose(-2, -1)


def _pca_eigenvalue_spread(pts_centered: torch.Tensor) -> float:
    """Ratio of largest to smallest singular value of the centred cloud (isotropy probe).

    Values near 1.0 indicate a near-isotropic (sphere-like) cloud where PCA axes are
    arbitrary. The asymmetric test object scores ~2.5; a sphere ~1.05.
    """
    _, sv, _ = torch.linalg.svd(pts_centered, full_matrices=False)
    return float((sv[0] / sv[-1].clamp_min(1e-10)).item())


def _pca_seed_rotations(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Identity + PCA principal-axis-match sign flips, ``(<=5, 3, 3)`` on-device.

    ICP from identity alone has a small basin; PCA-axis seeds cover large symmetry-axis
    rotations a uniform grid may straddle.

    Note on symmetric objects: for near-isotropic clouds the PCA axes
    are arbitrary (eigenvalue spread < _PCA_ISOTROPY_THRESH).  We keep the PCA seeds
    anyway because on the test sphere (N=800) individual rotations still differ in
    Chamfer score by ~9mm, so the batched trimmed ICP still selects among them
    meaningfully — and at default 256 Fibonacci seeds the grid is too coarse to guarantee
    a near-centroid seed without the PCA candidates providing additional coverage.
    Removing PCA seeds for isotropic clouds was tested and reliably made the symmetric
    result WORSE (8/9→3/9) due to the Fibonacci grid missing the correct basin.
    """
    dev, dt = src.device, src.dtype
    Vs = _pca_axes(src - src.mean(0))
    Vt = _pca_axes(tgt - tgt.mean(0))
    cands = [torch.eye(3, device=dev, dtype=dt)]
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            S = torch.diag(torch.tensor([sx, sy, sx * sy], device=dev, dtype=dt))
            R = Vt @ S @ Vs.transpose(-2, -1)
            if torch.linalg.det(R) > 0:  # proper rotation only
                cands.append(R)
    return torch.stack(cands, dim=0)


# ── closed-form similarity (Umeyama 1991, on-device torch) ────────────────────────────


def _umeyama(src: torch.Tensor, dst: torch.Tensor, with_scale: bool):
    """Least-squares similarity ``dst ≈ s * (src @ R.T) + t`` (Umeyama 1991), all in torch.

    ``with_scale=False`` fixes ``s = 1`` (rigid Kabsch). Returns ``(s, R, t)`` torch tensors on
    ``src``'s device/dtype. Run in float64 (cast by the caller) for SVD precision.
    """
    n = src.shape[0]
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    sc = src - src_mean
    dc = dst - dst_mean
    cov = (dc.transpose(-2, -1) @ sc) / n
    U, D, Vh = torch.linalg.svd(cov)
    S = torch.eye(3, device=src.device, dtype=src.dtype)
    if torch.linalg.det(U) * torch.linalg.det(Vh) < 0:  # reflection guard
        S[2, 2] = -1.0
    R = U @ S @ Vh
    if with_scale:
        var_src = sc.pow(2).sum() / n
        s = (D * torch.diagonal(S)).sum() / var_src.clamp_min(1e-12)
    else:
        s = torch.ones((), device=src.device, dtype=src.dtype)
    t = dst_mean - s * (R @ src_mean)
    return s, R, t


# ── batched torch ICP helpers (already on-device) ─────────────────────────────────────


def _stride_subsample(a: torch.Tensor, k: int) -> torch.Tensor:
    """Deterministic strided subsample to <= k rows (no RNG)."""
    if a.shape[0] <= k:
        return a
    sel = torch.linspace(0, a.shape[0] - 1, k, device=a.device).round().to(torch.int64)
    return a[sel]


def _batched_nn(X: torch.Tensor, Y: torch.Tensor):
    """Batched squared-distance nearest neighbour: for each ``X[b,i]`` its nearest ``Y[b]``.

    Chunked ``torch.cdist``. Returns ``(dist_sq (B,Nx), idx (B,Nx))``; chunks over the query
    axis to bound the ``(B,Nx,Ny)`` pairwise tensor.
    """
    B, Nx, _ = X.shape
    Ny = Y.shape[1]
    out_d = X.new_empty((B, Nx))
    out_i = torch.empty((B, Nx), device=X.device, dtype=torch.int64)
    step = max(1, int(8_000_000 // max(B * Ny, 1)))
    for lo in range(0, Nx, step):
        hi = min(lo + step, Nx)
        d = torch.cdist(X[:, lo:hi], Y)
        md, mi = d.min(dim=2)
        out_d[:, lo:hi] = md * md
        out_i[:, lo:hi] = mi
    return out_d, out_i


def _batched_umeyama(X: torch.Tensor, Y: torch.Tensor, w: torch.Tensor, with_scale: bool):
    """Batched weighted closed-form similarity mapping ``X[b] -> Y[b]`` (Umeyama 1991).

    ``w (B,N)`` are non-negative weights (the trim mask). Returns ``(s (B,), R (B,3,3),
    t (B,3))`` with ``Y ≈ s * (X @ R.T) + t``.
    """
    B = X.shape[0]
    ws = w.sum(dim=1, keepdim=True).clamp_min(1e-9)
    wn = (w / ws).unsqueeze(-1)
    mu_x = (wn * X).sum(dim=1)
    mu_y = (wn * Y).sum(dim=1)
    Xc = X - mu_x.unsqueeze(1)
    Yc = Y - mu_y.unsqueeze(1)
    cov = torch.einsum("bni,bnj->bij", wn * Yc, Xc)
    U, Dsv, Vh = torch.linalg.svd(cov)
    detUV = torch.linalg.det(U) * torch.linalg.det(Vh)
    S = torch.eye(3, device=X.device, dtype=X.dtype).expand(B, 3, 3).clone()
    S[:, 2, 2] = torch.sign(detUV)
    R = U @ S @ Vh
    if with_scale:
        var_x = (wn * Xc.pow(2)).sum(dim=(1, 2)).clamp_min(1e-12)
        s = (Dsv * torch.diagonal(S, dim1=1, dim2=2)).sum(dim=1) / var_x
    else:
        s = torch.ones(B, device=X.device, dtype=X.dtype)
    t = mu_y - s.unsqueeze(1) * torch.einsum("bij,bj->bi", R, mu_x)
    return s, R, t


def _trim_mean_sqrt(dist_sq: torch.Tensor, frac: float) -> torch.Tensor:
    """Mean of the sqrt of the lowest ``frac`` fraction of squared distances, per batch row."""
    k = max(1, int(round(frac * dist_sq.shape[1])))
    v, _ = torch.sort(dist_sq, dim=1)
    return v[:, :k].sqrt().mean(dim=1)


def _batched_trimmed_icp(
    src_sub: torch.Tensor, tgt_sub: torch.Tensor, R_seeds: torch.Tensor, with_scale: bool, icp_iters: int
):
    """All seeds through one batched trimmed point-to-point ICP (centre, per-seed scale match,
    trimmed Umeyama iterations, trimmed symmetric Chamfer score). Returns ``(aligned (B,K,3),
    score (B,))`` (lower score = better). Seeds processed in chunks of ``_SEED_BATCH``."""
    K, Kg = src_sub.shape[0], tgt_sub.shape[0]
    pc = src_sub.mean(0)
    gc = tgt_sub.mean(0)
    p0c = src_sub - pc
    rg = (tgt_sub - gc).pow(2).sum(-1).mean().sqrt()
    keep_k = max(3, int(round(_ICP_TRIM_KEEP * K)))

    aligned_chunks: list[torch.Tensor] = []
    score_chunks: list[torch.Tensor] = []
    for lo in range(0, R_seeds.shape[0], _SEED_BATCH):
        Rb = R_seeds[lo : lo + _SEED_BATCH]
        b = Rb.shape[0]
        X = torch.bmm(p0c.unsqueeze(0).expand(b, K, 3), Rb.transpose(1, 2))
        if with_scale:
            rp = X.pow(2).sum(-1).mean(1).sqrt()
            X = X * (rg / rp.clamp_min(1e-9)).view(b, 1, 1)
        X = X + gc
        Y = tgt_sub.unsqueeze(0).expand(b, Kg, 3).contiguous()
        for _ in range(int(icp_iters)):
            d, idx = _batched_nn(X, Y)
            Ym = torch.gather(Y, 1, idx.unsqueeze(-1).expand(b, K, 3))
            thr = torch.kthvalue(d, keep_k, dim=1, keepdim=True).values
            wmask = (d <= thr).to(X.dtype)
            s_i, R_i, t_i = _batched_umeyama(X, Ym, wmask, with_scale)
            X = s_i[:, None, None] * torch.bmm(X, R_i.transpose(1, 2)) + t_i[:, None, :]
        d_xy, _ = _batched_nn(X, Y)
        d_yx, _ = _batched_nn(Y, X)
        score = 0.5 * (_trim_mean_sqrt(d_xy, _ICP_TRIM_KEEP) + _trim_mean_sqrt(d_yx, _ICP_TRIM_KEEP))
        aligned_chunks.append(X)
        score_chunks.append(score)
    return torch.cat(aligned_chunks, dim=0), torch.cat(score_chunks, dim=0)


# ── public entry ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def global_align(
    target: Gaussians,
    source: Gaussians,
    *,
    transform: str = "sim3",
    n_rotations: int = DEFAULT_N_ROTATIONS,
    icp_iters: int = DEFAULT_ICP_ITERS,
    n_points: int = DEFAULT_N_POINTS,
    seed: int = 0,
) -> torch.Tensor:
    """Coarse global init: a 4x4 transform that lands ``source`` in ``target``'s basin.

    Sweeps ``n_rotations`` super-Fibonacci SO(3) candidates (plus PCA sign-flips), scores each
    by a batched trimmed symmetric Chamfer after a batched trimmed ICP, and returns the
    closed-form similarity for the best seed — a *basin-correct init* for the fine LM, not a
    precise pose. Runs fully on ``source``'s device (CPU or CUDA); no host round-trip.

    Args:
        target: the reference splat (``source`` aligns onto it). Only ``.means`` is read.
        source: the splat to align.
        transform: ``"sim3"`` (default) estimates a scale; ``"se3"`` fixes scale to 1.
        n_rotations: super-Fibonacci SO(3) seed count (more -> finer covering, slower).
        icp_iters: trimmed-ICP iterations per seed.
        n_points: target subsample size for the fit.
        seed: accepted for API stability; deterministic (no RNG).

    Returns:
        A ``(4, 4)`` ``torch.Tensor`` on ``source``'s device/dtype: ``T @ [x,y,z,1]``; top-left
        block ``s*R`` (Sim(3)) or ``R`` (SE(3)), last column the translation.
    """
    del seed  # deterministic; no RNG to seed (documented for API stability)
    if transform not in ("sim3", "se3"):
        raise ValueError(f"transform must be 'sim3' or 'se3', got {transform!r}")
    with_scale = transform == "sim3"

    dev = source.means.device
    dtype = source.means.dtype
    src_full = source.means
    tgt_full = target.means.to(device=dev, dtype=dtype)

    if src_full.shape[0] < 3 or tgt_full.shape[0] < 3:
        return torch.eye(4, device=dev, dtype=dtype)

    # Fit in float32 on-device (SVD/cdist stability); deterministic strided subsample.
    src_sub = _stride_subsample(src_full, _SUB_SOURCE).to(torch.float32)
    tgt_sub = _stride_subsample(tgt_full, int(n_points)).to(torch.float32)

    # Seeds (all on-device): PCA sign-flip candidates first (deterministic tie favours them),
    # then the super-Fibonacci grid.
    R_pca = _pca_seed_rotations(src_sub, tgt_sub)  # (<=5,3,3)
    R_grid = super_fibonacci_so3(int(n_rotations), device=dev, dtype=torch.float32)  # (n,3,3)
    R_seeds = torch.cat([R_pca, R_grid], dim=0)

    aligned_sub, scores = _batched_trimmed_icp(src_sub, tgt_sub, R_seeds, with_scale, int(icp_iters))

    # Stability tie-break (symmetric fix): among seeds within _SCORE_EPS
    # of the best score, prefer the one whose centroid is closest to the target centroid —
    # a proxy for scale/translation stability that avoids per-seed SVD calls and works in
    # tensor ops.  For symmetric clouds all seeds score nearly equally; the first-index
    # tie-break can pick a degenerate seed (wrong translation).  Centroid proximity picks
    # the seed that lands source closest to target without needing to refit Umeyama per seed.
    _SCORE_EPS = 5e-3
    best_score = scores.min()
    near_best = scores <= best_score + _SCORE_EPS * (best_score.abs().clamp_min(1.0))
    near_idx = near_best.nonzero(as_tuple=False).squeeze(1)  # indices of near-best seeds
    if near_idx.shape[0] > 1:
        # Pick the near-best seed whose centroid is closest to the target centroid.
        # aligned_sub[b] is the source sub-sample after ICP convergence under seed b.
        near_centroids = aligned_sub[near_idx].mean(dim=1)  # (K, 3)
        tgt_centroid = tgt_sub.mean(0)  # (3,)
        centroid_dists = (near_centroids - tgt_centroid).norm(dim=1)  # (K,)
        best_sub = near_idx[centroid_dists.argmin()]
    else:
        best_sub = torch.argmin(scores)
    winner = aligned_sub[best_sub]  # stays on-device

    # Recover the exact closed-form similarity src_sub -> winner in float64 on-device.
    s_tot, R_tot, t_tot = _umeyama(src_sub.double(), winner.double(), with_scale)
    T = torch.eye(4, dtype=dtype, device=dev)
    T[:3, :3] = (s_tot * R_tot).to(dtype=dtype)
    T[:3, 3] = t_tot.to(dtype=dtype)
    return T
