"""MAC maximal-clique correspondence seed, Zhang et al., CVPR 2023, reimplemented in torch.

``init="mac"`` replaces RANSAC-style hypothesis generation with the **MAC** (3D Registration
with Maximal Cliques, Zhang, Sun, Wang & Guo, CVPR 2023, https://github.com/zhangxy0517/
3D-Registration-with-Maximal-Cliques) scheme:

1. **Compatibility graph.** Two correspondences ``(p_i, q_i)`` / ``(p_j, q_j)`` are first-order
   compatible when the rigidity constraint holds: ``| ||p_i - p_j|| - ||q_i - q_j|| | < gamma``.
   Each surviving edge carries the soft score ``s(i,j) = exp(-d^2 / (2 gamma^2))`` and is then
   re-weighted by the paper's **second-order (SC^2) measure** (from Chen et al., SC^2-PCR):
   ``w2 = s * (S @ S)`` element-wise, an edge is only as strong as the *common compatible
   neighbourhood* the two correspondences share. Chance-compatible outlier pairs have no common
   neighbours, so SC^2 drives their weight to ~0 and they are dropped from the graph.
2. **Maximal cliques** of the SC^2 graph (networkx ``find_cliques``, Bron-Kerbosch with
   pivoting). Every maximal clique is a mutually-rigidity-consistent correspondence subset, i.e.
   one *consensus hypothesis*, including consensus sets a greedy max-degree pre-filter or a
   minimal-sample RANSAC draw would never isolate.
3. **Per-clique weighted SVD.** Each selected clique fits a pose by Kabsch/Umeyama with each
   correspondence weighted by its summed SC^2 edge weight inside the clique; hypotheses are
   scored by inlier count over *all* correspondences and the consensus winner is refit on its
   full inlier set.

Worst-case control (the paper applies the same kind of caps; ours are explicit constants):

* correspondences capped at ``max_corr`` (deterministic strided subsample),
* per-node **degree cap** (keep each node's top-``degree_cap`` edges by SC^2 weight, AND-
  symmetrised so the cap is a hard guarantee), Bron-Kerbosch blowup is driven by dense graphs,
* **clique-count cap** + **wall-clock budget** on the enumeration generator (networkx
  ``find_cliques`` yields lazily, so both caps are exact, not post-hoc),
* **hypothesis cap**: node-guided selection (per node keep its heaviest clique, the paper's
  trick), dedupe, then top-``max_hyps`` by clique weight.

Sim(3) note (the paper is SE(3)-only)
-------------------------------------
A similarity breaks the rigidity constraint in (1), so for ``with_scale=True`` the scale is
estimated **first** from the correspondence pairwise-distance ratios (median of
``||q_i - q_j|| / ||p_i - p_j||`` over subsampled pairs, robust to moderate outlier rates
because the median of a contaminated ratio set still sits in the inlier mode), the source is
de-scaled, and SE(3) MAC runs on the de-scaled set; the consensus refit then re-estimates a
residual scale on the winning inlier set (Umeyama ``with_scale=True``) so the median seed only
has to be approximately right. At very high outlier rates (>~70 %) the median ratio itself
degrades, the honest ``success`` flag covers that case.

Honesty
-------
The MAC paper reports lifting GeoTransformer's 3DLoMatch registration recall ~71 % -> ~78 %
when MAC replaces its hypothesis generation. splatreg's implementation is validated here on
synthetic correspondence sets (see ``tests/test_mac.py``); the 3DLoMatch number requires the
GeoTransformer features + dataset on a GPU box and is **pending**, we cite the paper for the
expectation and do not claim the number.

Pure torch + numpy + networkx (clique enumeration only); CPU-runnable.
"""

from __future__ import annotations

import math
import time

import torch

from .core.types import Gaussians
from .align import _stride_subsample, _umeyama

__all__ = ["mac_pose", "mac_feature_align"]

# ── tuneable defaults (worst-case caps — see module docstring) ─────────────────

_MAC_MAX_CORR = 1000  # correspondence cap before graph construction (paper subsamples too)
_MAC_DEGREE_CAP = 48  # per-node edge cap (top-k by SC^2 weight, AND-symmetrised)
_MAC_MAX_CLIQUES = 10000  # maximal-clique enumeration cap (generator, exact)
_MAC_TIME_BUDGET_S = 4.0  # wall-clock budget for the enumeration (generator, exact)
_MAC_MAX_HYPS = 64  # pose hypotheses after node-guided selection + weight ranking
_MAC_MIN_CLIQUE = 3  # a pose needs >= 3 correspondences
_MAC_MIN_INLIERS = 6  # consensus below this -> honest failure (chance cliques score 3-5 on
#                       all-outlier sets; matches the feature path's _FS_MIN_INLIERS gate)
_MAC_GAMMA_MULT = 2.0  # default gamma = mult * inlier_tol (rigidity tolerance ~2x noise)
_MAC_RATIO_PAIRS = 2048  # pair sample for the Sim(3) median distance-ratio scale seed
_EPS = 1e-12


# ── compatibility graph ────────────────────────────────────────────────────────


def compatibility_graph(
    cs: torch.Tensor, ct: torch.Tensor, gamma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order rigidity graph + SC^2 second-order weights over a correspondence set.

    Args:
        cs: ``(M, 3)`` source correspondence points.
        ct: ``(M, 3)`` target correspondence points (``cs[i] <-> ct[i]``).
        gamma: rigidity tolerance, edge iff ``| ||cs_i-cs_j|| - ||ct_i-ct_j|| | < gamma``.

    Returns:
        ``(adj, w2)``, ``(M, M)`` bool adjacency (SC^2-pruned: first-order edges whose
        second-order weight is positive) and the ``(M, M)`` SC^2 weight matrix
        ``w2 = s * (S @ S)`` with ``s = exp(-d^2 / (2 gamma^2))`` on first-order edges.
        Both symmetric with a zero/false diagonal.
    """
    d = (torch.cdist(cs, cs) - torch.cdist(ct, ct)).abs()  # (M, M) rigidity violation
    adj1 = d < gamma
    adj1.fill_diagonal_(False)
    s = torch.exp(-(d * d) / (2.0 * gamma * gamma)) * adj1  # soft first-order score
    w2 = s * (s @ s)  # SC^2: re-weight by the shared compatible neighbourhood
    adj = adj1 & (w2 > 0.0)
    return adj, w2 * adj


def _cap_degree(adj: torch.Tensor, w2: torch.Tensor, cap: int) -> torch.Tensor:
    """Cap each node's degree at ``cap`` by keeping its top-``cap`` edges by SC^2 weight.

    The kept set is AND-symmetrised (an edge survives only if it is in BOTH endpoints' top
    lists), so ``cap`` is a hard per-node guarantee, that bounds the Bron-Kerbosch branching.
    """
    M = adj.shape[0]
    if cap <= 0 or M <= cap + 1:
        return adj
    deg = adj.sum(dim=1)
    if int(deg.max().item()) <= cap:
        return adj
    masked = torch.where(adj, w2, torch.full_like(w2, -1.0))
    top = masked.topk(cap, dim=1).indices  # (M, cap) best edges per node
    keep = torch.zeros_like(adj)
    keep.scatter_(1, top, True)
    keep &= adj  # never resurrect a non-edge (rows with degree < cap pad with junk indices)
    return keep & keep.T  # AND-symmetrise -> hard degree bound


def enumerate_maximal_cliques(
    adj: torch.Tensor,
    *,
    max_cliques: int = _MAC_MAX_CLIQUES,
    time_budget_s: float = _MAC_TIME_BUDGET_S,
    min_size: int = _MAC_MIN_CLIQUE,
) -> tuple[list[list[int]], bool]:
    """All maximal cliques of ``adj`` (size >= ``min_size``), capped by count and wall clock.

    networkx ``find_cliques`` is a lazy Bron-Kerbosch-with-pivoting generator, so both caps cut
    the enumeration itself (not just the returned list). Returns ``(cliques, truncated)``,
    ``truncated=True`` when either cap fired (the search was not exhaustive).
    """
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "init='mac' needs networkx for maximal-clique enumeration: "
            "pip install 'splatreg[mac]' (or pip install networkx)."
        ) from exc

    G = nx.from_numpy_array(adj.cpu().numpy())
    out: list[list[int]] = []
    truncated = False
    t0 = time.monotonic()
    for clique in nx.find_cliques(G):
        if len(clique) >= min_size:
            out.append(clique)
            if len(out) >= max_cliques:
                truncated = True
                break
        if time.monotonic() - t0 > time_budget_s:
            truncated = True
            break
    return out, truncated


def _select_cliques(
    cliques: list[list[int]], w2: torch.Tensor, max_hyps: int
) -> tuple[list[list[int]], list[float]]:
    """Node-guided clique selection (the paper's hypothesis pruning) + top-``max_hyps``.

    Each clique is weighted by the sum of its internal SC^2 edge weights. For every node we keep
    only the heaviest clique containing it (node-guided selection), dedupe, and rank the
    survivors by weight, returning at most ``max_hyps`` cliques and their weights.
    """
    if not cliques:
        return [], []
    weights = []
    for c in cliques:
        idx = torch.as_tensor(c, dtype=torch.int64, device=w2.device)
        weights.append(float(w2[idx][:, idx].sum().item()) * 0.5)  # each edge counted twice
    # Node-guided: best clique per node.
    best_for_node: dict[int, int] = {}
    for ci, c in enumerate(cliques):
        for v in c:
            if v not in best_for_node or weights[ci] > weights[best_for_node[v]]:
                best_for_node[v] = ci
    keep = sorted(set(best_for_node.values()), key=lambda ci: -weights[ci])[:max_hyps]
    return [cliques[ci] for ci in keep], [weights[ci] for ci in keep]


# ── Sim(3) scale seed (median pairwise distance ratio) ─────────────────────────


def _median_ratio_scale(cs: torch.Tensor, ct: torch.Tensor, n_pairs: int = _MAC_RATIO_PAIRS) -> float:
    """Median ``||ct_i - ct_j|| / ||cs_i - cs_j||`` over sampled correspondence pairs.

    A similarity scales every source pairwise distance by the same ``s``, so among inlier pairs
    the ratio is constant; the median over a contaminated set stays in the inlier mode for
    moderate outlier rates. Deterministic (seeded). Returns 1.0 when degenerate.
    """
    M = cs.shape[0]
    if M < 2:
        return 1.0
    gen = torch.Generator(device="cpu").manual_seed(0)
    i = torch.randint(0, M, (n_pairs,), generator=gen)
    j = torch.randint(0, M, (n_pairs,), generator=gen)
    ok = i != j
    i, j = i[ok], j[ok]
    ds = (cs[i] - cs[j]).norm(dim=1)
    dt = (ct[i] - ct[j]).norm(dim=1)
    valid = ds > _EPS
    if int(valid.sum().item()) == 0:
        return 1.0
    ratio = (dt[valid] / ds[valid]).clamp(1e-6, 1e6)
    s = float(ratio.median().item())
    return s if math.isfinite(s) and s > _EPS else 1.0


# ── MAC pose estimation ────────────────────────────────────────────────────────


def mac_pose(
    src_corr: torch.Tensor,
    tgt_corr: torch.Tensor,
    *,
    with_scale: bool = False,
    inlier_tol: float = 0.02,
    gamma: float | None = None,
    max_corr: int = _MAC_MAX_CORR,
    degree_cap: int = _MAC_DEGREE_CAP,
    max_cliques: int = _MAC_MAX_CLIQUES,
    max_hyps: int = _MAC_MAX_HYPS,
    time_budget_s: float = _MAC_TIME_BUDGET_S,
    min_inliers: int = _MAC_MIN_INLIERS,
) -> dict:
    """MAC pose from a putative correspondence set: SC^2 graph -> maximal cliques -> weighted SVD.

    Args:
        src_corr / tgt_corr: ``(M, 3)`` matched point pairs (``src_corr[i] <-> tgt_corr[i]``).
        with_scale: Sim(3), median-ratio scale seed first, SE(3) MAC on the de-scaled set,
            residual scale re-fit on the consensus inliers (see module docstring).
        inlier_tol: 3-D inlier distance for hypothesis scoring (target units).
        gamma: rigidity tolerance for the compatibility graph; ``None`` ->
            ``_MAC_GAMMA_MULT * inlier_tol``.
        max_corr / degree_cap / max_cliques / max_hyps / time_budget_s: worst-case caps
            (module docstring).
        min_inliers: consensus floor, a winning hypothesis below this is reported as an
            honest failure (chance-compatible cliques score 3-5 inliers even on all-outlier
            sets; the same gate the feature path's ``_FS_MIN_INLIERS`` applies).

    Returns:
        dict with ``T`` (4x4, ``src -> tgt``, source dtype/device; for Sim(3) the top-left block
        is ``s*R``), ``success`` (bool, ``False`` means the correspondences carried no
        consistent consensus: T is identity, do NOT trust it), ``n_inliers``, ``n_matches``,
        ``n_cliques`` (enumerated), ``n_hypotheses`` (evaluated), ``scale`` (1.0 for SE(3)),
        ``truncated`` (a cap/time budget cut the clique enumeration).
    """
    dev = src_corr.device
    dtype = src_corr.dtype
    out = {
        "T": torch.eye(4, device=dev, dtype=dtype),
        "success": False,
        "n_inliers": 0,
        "n_matches": int(src_corr.shape[0]),
        "n_cliques": 0,
        "n_hypotheses": 0,
        "scale": 1.0,
        "truncated": False,
    }
    M0 = int(src_corr.shape[0])
    if M0 < _MAC_MIN_CLIQUE:
        return out
    if src_corr.shape != tgt_corr.shape:
        raise ValueError(
            f"src_corr/tgt_corr must be matched (M, 3) pairs, got {tuple(src_corr.shape)} vs "
            f"{tuple(tgt_corr.shape)}"
        )

    cs_all = src_corr.detach().double()
    ct_all = tgt_corr.detach().double()
    # Cap M before the O(M^2) graph (deterministic strided subsample, like the rest of splatreg).
    if M0 > max_corr:
        sel = torch.linspace(0, M0 - 1, max_corr, device=dev).round().to(torch.int64)
        cs_all, ct_all = cs_all[sel], ct_all[sel]

    # Sim(3): scale-first (median pairwise-distance ratio), then rigid MAC on the de-scaled set.
    s0 = _median_ratio_scale(cs_all, ct_all) if with_scale else 1.0
    cs_w = cs_all * s0  # de-scaled source (target units)

    g = float(gamma) if gamma is not None else _MAC_GAMMA_MULT * float(inlier_tol)
    adj, w2 = compatibility_graph(cs_w.float(), ct_all.float(), g)
    adj = _cap_degree(adj, w2, degree_cap)
    w2 = w2 * adj
    if int(adj.sum().item()) == 0:
        return out  # honest failure: no rigidity-consistent pair at all

    cliques, truncated = enumerate_maximal_cliques(
        adj, max_cliques=max_cliques, time_budget_s=time_budget_s, min_size=_MAC_MIN_CLIQUE
    )
    out["n_cliques"] = len(cliques)
    out["truncated"] = truncated
    if not cliques:
        return out

    hyps, _hyp_w = _select_cliques(cliques, w2, max_hyps)
    out["n_hypotheses"] = len(hyps)

    # Per-clique weighted Kabsch (correspondence weight = its summed SC^2 edge weight inside the
    # clique), scored by inlier count over ALL correspondences.
    best_n, best_res, best_T = 0, float("inf"), None
    for c in hyps:
        idx = torch.as_tensor(c, dtype=torch.int64, device=dev)
        X = cs_w[idx]
        Y = ct_all[idx]
        w = w2[idx][:, idx].sum(dim=1).double().clamp_min(_EPS)
        try:
            R, t = _weighted_kabsch(X, Y, w)
        except Exception:  # pragma: no cover - SVD failure on a degenerate clique
            continue
        if not (torch.isfinite(R).all() and torch.isfinite(t).all()):
            continue
        res = (cs_w @ R.T + t - ct_all).norm(dim=1)
        inl = res < inlier_tol
        n_in = int(inl.sum().item())
        mean_res = float(res[inl].mean().item()) if n_in else float("inf")
        if n_in > best_n or (n_in == best_n and mean_res < best_res):
            best_n, best_res = n_in, mean_res
            best_T = (R, t, inl)

    if best_T is None or best_n < max(int(min_inliers), _MAC_MIN_CLIQUE):
        out["n_inliers"] = int(best_n)  # diagnostic: the sub-floor consensus that was rejected
        return out  # honest failure: no consensus above the chance-clique floor (T stays identity)

    # Consensus refit on the winner's full inlier set. For Sim(3) re-estimate a residual scale
    # there (Umeyama with_scale) so the median seed only needs to be approximately right.
    R, t, inl = best_T
    s_r = 1.0
    try:
        s_fit, R_fit, t_fit = _umeyama(cs_w[inl], ct_all[inl], with_scale)
        if torch.isfinite(R_fit).all() and torch.isfinite(t_fit).all():
            R, t, s_r = R_fit, t_fit, float(s_fit)
            res = (s_r * (cs_w @ R.T) + t - ct_all).norm(dim=1)
            best_n = int((res < inlier_tol).sum().item())
    except Exception:  # pragma: no cover - keep the clique pose if the refit SVD fails
        pass

    s_total = s0 * s_r  # total similarity scale (1.0 * 1.0 for SE(3))
    T = torch.eye(4, dtype=torch.float64, device=dev)
    T[:3, :3] = s_total * R
    T[:3, 3] = t
    out["T"] = T.to(dtype=dtype, device=dev)
    out["scale"] = float(s_total)
    out["n_inliers"] = int(best_n)
    out["success"] = True
    return out


def _weighted_kabsch(X: torch.Tensor, Y: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted rigid Kabsch ``Y ~= X @ R.T + t`` (float64; reflection-guarded)."""
    wn = (w / w.sum().clamp_min(_EPS)).unsqueeze(-1)
    mu_x = (wn * X).sum(dim=0)
    mu_y = (wn * Y).sum(dim=0)
    Xc, Yc = X - mu_x, Y - mu_y
    cov = (wn * Yc).T @ Xc
    U, _D, Vh = torch.linalg.svd(cov)
    S = torch.eye(3, dtype=X.dtype, device=X.device)
    if torch.linalg.det(U) * torch.linalg.det(Vh) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vh
    t = mu_y - R @ mu_x
    return R, t


# ── splat-level entry: FPFH correspondences (or injected ones) -> MAC -> ICP polish ─


@torch.no_grad()
def mac_feature_align(
    target: Gaussians,
    source: Gaussians,
    *,
    transform: str = "se3",
    correspondences: tuple[torch.Tensor, torch.Tensor] | None = None,
    inlier_tol: float | None = None,
    refine_iters: int = 30,
    **mac_kwargs,
) -> tuple[torch.Tensor, dict]:
    """MAC registrar over splats: FPFH correspondences -> :func:`mac_pose` -> overlap-aware polish.

    Mirrors :func:`splatreg.align_features.feature_align` but replaces the clique-prefiltered
    RANSAC hypothesis stage with MAC maximal-clique hypothesis generation (module docstring).
    The recovered seed is polished by the same overlap-aware trimmed ICP the feature path uses
    and gated on the overlap residual, so a degenerate polish can never worsen the seed.

    Args:
        target / source: reference / to-align splats (only ``.means`` read).
        transform: ``"se3"`` or ``"sim3"`` (median-ratio scale seed + residual refit; see
            :func:`mac_pose`).
        correspondences: optional pre-matched ``(src_pts (M, 3), tgt_pts (M, 3))`` pairs, the
            injection seam for learned correspondences (e.g. GeoTransformer's) or tests; when
            ``None`` the FPFH ratio-test mutual-NN matcher of the feature path supplies them.
        inlier_tol: hypothesis-scoring inlier distance; ``None`` -> the feature path default.
        refine_iters: overlap-ICP polish iterations on the MAC seed.
        **mac_kwargs: forwarded to :func:`mac_pose` (``gamma``, the caps, ...).

    Returns:
        ``(T_4x4, info)``, ``info`` carries ``n_matches`` / ``n_inliers`` / ``n_cliques`` /
        ``n_hypotheses`` / ``truncated`` / ``success`` plus the honest ``ambiguous`` /
        ``confidence`` the other registrars report. ``success=False`` (all-outlier / no
        consensus) returns identity with ``ambiguous=True`` and ``confidence=0.0``, never a
        silent wrong pose.
    """
    from .align_features import (
        _FS_INLIER_TOL,
        _FS_KNN_FPFH,
        _FS_KNN_NORMAL,
        _FS_FPFH_BINS,
        _FS_N_SOURCE,
        _FS_N_TARGET,
        _FS_AMBIG_RESID,
        _estimate_normals,
        _fpfh,
        _knn_indices,
        _match_features,
        _overlap_icp_polish,
        _overlap_residual_norm,
    )

    if transform not in ("se3", "sim3"):
        raise ValueError(f"transform must be 'se3' or 'sim3', got {transform!r}")
    with_scale = transform == "sim3"
    tol = float(inlier_tol) if inlier_tol is not None else _FS_INLIER_TOL

    dev = source.means.device
    dtype = source.means.dtype
    src_full = source.means.to(torch.float32)
    tgt_full = target.means.to(device=dev, dtype=torch.float32)

    info: dict = {
        "n_matches": 0,
        "n_inliers": 0,
        "n_cliques": 0,
        "n_hypotheses": 0,
        "truncated": False,
        "success": False,
        "ambiguous": True,
        "confidence": 0.0,
    }
    if src_full.shape[0] < 4 or tgt_full.shape[0] < 4:
        return torch.eye(4, device=dev, dtype=dtype), info

    if correspondences is not None:
        cs = correspondences[0].to(device=dev, dtype=torch.float32)
        ct = correspondences[1].to(device=dev, dtype=torch.float32)
    else:
        # The feature path's FPFH + ratio-test mutual-NN matcher (same defaults).
        src_sub = _stride_subsample(src_full, _FS_N_SOURCE)
        tgt_sub = _stride_subsample(tgt_full, _FS_N_TARGET)
        src_normals = _estimate_normals(src_sub, _knn_indices(src_sub, _FS_KNN_NORMAL))
        tgt_normals = _estimate_normals(tgt_sub, _knn_indices(tgt_sub, _FS_KNN_NORMAL))
        desc_src = _fpfh(src_sub, src_normals, _FS_KNN_FPFH, _FS_FPFH_BINS)
        desc_tgt = _fpfh(tgt_sub, tgt_normals, _FS_KNN_FPFH, _FS_FPFH_BINS)
        si, ti = _match_features(desc_src, desc_tgt)
        cs, ct = src_sub[si], tgt_sub[ti]

    rr = mac_pose(cs, ct, with_scale=with_scale, inlier_tol=tol, **mac_kwargs)
    info.update(
        n_matches=rr["n_matches"],
        n_inliers=rr["n_inliers"],
        n_cliques=rr["n_cliques"],
        n_hypotheses=rr["n_hypotheses"],
        truncated=rr["truncated"],
        success=rr["success"],
    )
    if not rr["success"]:
        return torch.eye(4, device=dev, dtype=dtype), info  # honest failure, never silent-wrong

    T0 = rr["T"].to(device=dev, dtype=dtype)
    # Overlap-aware polish + residual gate (the same accept-only-if-not-worse policy as the
    # robust/learned registrars).
    T_ref = _overlap_icp_polish(src_full, tgt_full, T0, with_scale=with_scale, iters=refine_iters)
    r0 = _overlap_residual_norm(src_full, tgt_full, T0, with_scale=with_scale)
    r1 = _overlap_residual_norm(src_full, tgt_full, T_ref, with_scale=with_scale)
    T = T_ref if r1 <= r0 + 1e-6 else T0
    resid = min(r0, r1)
    info["overlap_residual"] = float(resid)
    info["ambiguous"] = bool(resid > _FS_AMBIG_RESID)
    info["confidence"] = float(max(0.0, 1.0 - resid / _FS_AMBIG_RESID)) if resid < float("inf") else 0.0
    if info["ambiguous"]:
        info["confidence"] = min(info["confidence"], 0.3)
    return T.to(device=dev, dtype=dtype), info
