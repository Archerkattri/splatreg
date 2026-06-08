"""Multi-splat joint / bundle registration — N splats into one loop-consistent frame.

:func:`splatreg.merge` registers every splat onto a single reference *sequentially* and independently.
That is correct for a star topology, but when the splats form a **loop** (capture 0 overlaps 1, 1
overlaps 2, …, and the last wraps back to 0) the independent pairwise solves do not agree around the
loop: each carries its own small error and they **accumulate** into a visible mis-closure at the seam
(the classic open-loop drift). This module closes the loop by treating the alignment as a **pose
graph**:

1. **Pairwise constraints.** For every overlapping pair ``(i, j)`` run the existing
   :func:`splatreg.register` to get the relative transform ``T_ij`` aligning splat ``j`` into splat
   ``i``'s frame (so a ``j``-point maps to ``i`` as ``T_ij @ p``). Each is one *edge measurement*.
2. **Joint solve.** Optimise all ``N`` **absolute** poses ``T_i`` (mapping splat ``i`` into the
   common reference frame) so that every edge is satisfied simultaneously:
   ``T_i @ T_ij ≈ T_j``. The edge residual is the tangent-space error
   ``e_ij = log( (T_i @ T_ij)^{-1} @ T_j )`` (6-vector for SE(3), 7 for Sim(3)); the joint cost
   ``Σ_ij ‖e_ij‖²`` is minimised by **Gauss-Newton in the SE(3)/Sim(3) tangent**, reusing
   :mod:`splatreg.core.lie` (``se3_exp`` / ``sim3_exp`` / the matching ``log``) for the manifold
   retraction. The reference pose is held at identity to fix the **gauge** (the global frame is
   otherwise free).

Spreading each edge's error over the whole loop is exactly what makes the result loop-consistent:
the joint optimum distributes the closure error instead of dumping it all at the last edge, so the
maximum pairwise inconsistency drops well below the sequential chain's.

Pure ``torch`` (+ the existing register/lie/solver stack). The Jacobian of each edge residual w.r.t.
the two incident pose increments is obtained by autodiff through the lie ``exp``/``log`` (the same
right-perturbation convention as the rest of splatreg), so no factor Jacobian is hand-derived.

**Robust outlier-edge rejection.** A single wrong pairwise ``register`` result (a bad edge) would, in
a plain least-squares pose graph, corrupt every absolute pose. The joint solve is therefore an
**IRLS** with a per-edge **Huber / Cauchy** kernel (``robust="huber"`` by default) and a
**graduated-non-convexity (GNC)** schedule (start near-convex so the outlier surfaces as the largest
residual, then anneal the robust scale down to reject it) — so one bad edge is down-weighted out
instead of dragging the solution. ``robust=None`` recovers the plain solve. Note a *bare ring* (every
node degree 2) has no redundancy to localise an outlier — the error spreads evenly and is
indistinguishable at the optimum; rejection needs a redundant graph (chords / multiple loops), the
realistic loop-closure case.

API
---
    T_list = bundle_register(splats, ref=0, pairs="auto")           # list of N absolute 4x4
    T_list, fused = bundle_register(splats, fuse=True)              # + one merged Gaussians

``pairs="auto"`` builds the loop/chain edges from overlap; pass an explicit ``[(i, j), ...]`` to
control the graph (e.g. add the loop-closure edge a pure chain omits).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple, Union

import torch

from .core.types import Gaussians
from .core.lie import se3_exp, se3_log, sim3_exp, sim3_log
from .api import register, _apply_transform_to_gaussians

__all__ = ["bundle_register", "pairwise_consistency", "BundleResult", "solve_pose_graph"]

_log = logging.getLogger("splatreg")
_DOF = {"se3": 6, "sim3": 7}
_EXP = {"se3": se3_exp, "sim3": sim3_exp}


def _log_map(transform: str):
    """The tangent ``log`` matching ``transform`` (``se3_log`` returns 6, ``sim3_log`` 7)."""
    if transform == "sim3":
        return sim3_log
    return lambda T: se3_log(T, dof=6)


class BundleResult:
    """Diagnostics from :func:`bundle_register` (returned on ``info``-style access).

    Attributes
    ----------
    poses : the ``N`` absolute 4x4 transforms (splat ``i`` -> reference frame).
    max_edge_err / mean_edge_err : the worst / mean per-edge tangent inconsistency after the solve.
    cost_history : ``Σ‖e_ij‖²`` per Gauss-Newton iteration.
    edges : the pair list actually used.
    edge_weights : final robust (IRLS) weight per edge ``(i, j) -> w in [0, 1]``; ``1.0`` for an
        inlier, ``-> 0`` for an outlier the robust kernel down-weighted out. Empty when
        ``robust=None``.
    rejected_edges : edges whose final weight fell below ``reject_threshold`` (treated as outliers).
    """

    def __init__(
        self,
        poses,
        max_edge_err,
        mean_edge_err,
        cost_history,
        edges,
        edge_weights=None,
        rejected_edges=None,
    ):
        self.poses = poses
        self.max_edge_err = max_edge_err
        self.mean_edge_err = mean_edge_err
        self.cost_history = cost_history
        self.edges = edges
        self.edge_weights = edge_weights or {}
        self.rejected_edges = rejected_edges or []


def _auto_pairs(n: int) -> List[Tuple[int, int]]:
    """Default graph: the chain ``0-1-2-…-(n-1)`` plus the loop-closure edge ``(n-1, 0)``.

    This is the topology the module is built for — a ring of overlapping captures. For ``n == 2``
    it is the single pair ``(0, 1)`` (no loop). The caller can override with an explicit pair list
    for star / arbitrary graphs.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    edges = [(i, i + 1) for i in range(n - 1)]
    edges.append((n - 1, 0))  # loop closure
    return edges


def _edge_residual(
    T_i: torch.Tensor,
    T_j: torch.Tensor,
    T_ij: torch.Tensor,
    log_map,
) -> torch.Tensor:
    """Tangent error of one edge: ``log( (T_i @ T_ij)^{-1} @ T_j )``.

    Zero exactly when the absolute poses satisfy the relative measurement ``T_i @ T_ij == T_j``
    (mapping a ``j``-point j->i->ref equals mapping it j->ref directly).
    """
    pred_j = T_i @ T_ij  # where T_j "should" be, per this edge
    err = torch.linalg.inv(pred_j) @ T_j
    return log_map(err)


def _robust_weight(norm: float, scale: float, kernel: Optional[str]) -> float:
    """IRLS weight ``w(r)`` such that ``w·r²`` is the robust loss of a residual of norm ``r``.

    The Gauss-Newton/IRLS reweighting: minimising ``Σ ρ(‖e‖)`` is, at each iteration, a weighted
    least-squares ``Σ w·‖e‖²`` with ``w = ρ'(r)/r``. For Huber ``w = 1`` while ``r ≤ c`` and
    ``w = c/r`` beyond (so a gross outlier's pull falls off as ``1/r`` instead of growing); for
    Cauchy ``w = 1/(1 + (r/c)²)`` (redescending — a far outlier is suppressed even harder).
    ``kernel=None`` returns 1 (plain least squares). ``scale = c`` is the inlier/outlier knee.
    """
    if kernel is None:
        return 1.0
    c = max(float(scale), 1e-12)
    r = float(norm)
    if kernel == "huber":
        return 1.0 if r <= c else c / max(r, 1e-12)
    if kernel == "cauchy":
        return 1.0 / (1.0 + (r / c) ** 2)
    raise ValueError(f"robust kernel must be 'huber', 'cauchy' or None, got {kernel!r}")


def _auto_robust_scale(norms: Sequence[float]) -> float:
    """Robust spread estimate ``1.4826·median(‖e‖)`` (MAD-style) for the IRLS knee ``c``.

    Using the median (not the mean) of the edge-residual norms makes the scale itself outlier-proof:
    one huge bad-edge residual barely moves the median, so the knee stays at the inlier level and the
    bad edge lands firmly in the down-weighted tail. ``1.4826`` is the Gaussian-consistency factor.
    """
    if not norms:
        return 1.0
    t = torch.tensor(list(norms), dtype=torch.float64)
    med = float(t.median())
    return max(1.4826 * med, 1e-9)


def solve_pose_graph(
    poses: Sequence[torch.Tensor],
    rel: dict,
    ref: int,
    *,
    transform: str = "se3",
    robust: Optional[str] = "huber",
    robust_scale: Union[str, float] = "auto",
    max_iters: int = 30,
    damping: float = 1e-6,
    convergence_tol: float = 1e-9,
) -> Tuple[List[torch.Tensor], List[float], dict]:
    """Robust (IRLS) pose-graph Gauss-Newton over absolute poses given relative measurements.

    The numeric core of :func:`bundle_register`, exposed so it can be driven with *arbitrary*
    relative measurements ``rel`` (e.g. to inject a deliberately corrupt edge in a test). Optimises
    every pose except ``ref`` (held at its seeded value to fix the gauge) so that each edge
    ``T_i @ T_ij ≈ T_j`` holds. With ``robust`` set, each edge carries a per-iteration IRLS weight
    ``w(‖e_ij‖)`` (Huber / Cauchy) folded into the normal equations, so a single bad edge is
    down-weighted instead of corrupting the global solution.

    Parameters
    ----------
    poses : seed absolute poses (mutated copies are returned, the input list is not changed
        in place beyond what the caller passes). ``poses[ref]`` is held fixed.
    rel : ``(i, j) -> T_ij`` relative measurements (edges).
    ref : reference index, held fixed (gauge).
    transform / robust / robust_scale / max_iters / damping / convergence_tol : see
        :func:`bundle_register`.

    Returns
    -------
    ``(poses, cost_history, edge_weights)`` — refined absolute poses, the per-iteration robust cost
    (plus a final-cost tail), and the final per-edge IRLS weight dict.
    """
    poses = list(poses)
    n = len(poses)
    device, dtype = poses[ref].device, poses[ref].dtype
    dof = _DOF[transform]
    exp_fn = _EXP[transform]
    log_map = _log_map(transform)

    free = [k for k in range(n) if k != ref]
    slot = {k: idx for idx, k in enumerate(free)}
    n_free = len(free)
    cost_history: List[float] = []
    edge_weights: dict = {}

    # Graduated non-convexity (GNC) schedule. IRLS alone is init-dependent: if the seed already
    # *satisfies* a bad edge (e.g. the sequential chain was routed through it), that edge has a tiny
    # residual at iter 0 and IRLS happily keeps it while down-weighting an innocent neighbour. GNC
    # cures this by starting near-convex — a large scale so every edge keeps ~unit weight and plain
    # GN spreads the closure error, surfacing the true outlier as the *largest* residual — then
    # annealing the scale down to the target MAD knee so the now-obvious outlier is rejected.
    gnc_iters = min(8, int(max_iters)) if robust is not None else 0
    gnc_inflate0 = 16.0  # initial scale multiplier (≈ convex); decays geometrically to 1.

    if n_free > 0:
        for it in range(int(max_iters)):
            H = torch.zeros((n_free * dof, n_free * dof), device=device, dtype=dtype)
            b = torch.zeros(n_free * dof, device=device, dtype=dtype)
            total_cost = 0.0
            # First pass: residuals + Jacobians + per-edge robust weights for THIS iterate.
            cache = []
            norms = []
            for (i, j), T_ij in rel.items():
                e, Ji, Jj = _edge_jac(poses[i], poses[j], T_ij, exp_fn, log_map, dof, device, dtype)
                cache.append(((i, j), e, Ji, Jj))
                norms.append(float(e.norm()))
            # Adaptive scale (MAD of the current edge norms) keeps the knee at the inlier level.
            if robust is not None:
                base_c = _auto_robust_scale(norms) if robust_scale == "auto" else float(robust_scale)
                # GNC: inflate the knee early (near-convex), decay geometrically toward base_c.
                if it < gnc_iters and gnc_iters > 0:
                    inflate = gnc_inflate0 ** (1.0 - it / float(gnc_iters))
                else:
                    inflate = 1.0
                c = base_c * inflate
            else:
                c = 1.0
            for ((i, j), e, Ji, Jj), nrm in zip(cache, norms):
                w = _robust_weight(nrm, c, robust)
                edge_weights[(i, j)] = w
                # Report the raw (unweighted) pose-graph cost so the history is comparable across
                # iterations even as the IRLS weights change; the weights only steer H/b.
                total_cost += float((e * e).sum())
                blocks = []
                if i in slot:
                    blocks.append((slot[i], Ji))
                if j in slot:
                    blocks.append((slot[j], Jj))
                for (bi, Jb) in blocks:
                    rb = slice(bi * dof, (bi + 1) * dof)
                    b[rb] -= w * (Jb.T @ e)
                    for (bj, Jc) in blocks:
                        cb = slice(bj * dof, (bj + 1) * dof)
                        H[rb, cb] += w * (Jb.T @ Jc)
            cost_history.append(total_cost)
            # Levenberg damping -> SPD; solve for the stacked tangent step.
            H = H + damping * torch.eye(n_free * dof, device=device, dtype=dtype)
            try:
                delta = torch.linalg.solve(H, b)
            except Exception:  # pragma: no cover - damping normally keeps H invertible
                break
            for k in free:
                dk = delta[slot[k] * dof : (slot[k] + 1) * dof]
                poses[k] = poses[k] @ exp_fn(dk)
            if float(delta.norm()) < convergence_tol:
                break
        # Final cost tail + edge weights refreshed at the optimum.
        fc = 0.0
        final_norms = {}
        for (i, j), T_ij in rel.items():
            e = _edge_residual(poses[i], poses[j], T_ij, log_map)
            fc += float((e * e).sum())
            final_norms[(i, j)] = float(e.norm())
        cost_history.append(fc)
        if robust is not None:
            c = (
                _auto_robust_scale(list(final_norms.values()))
                if robust_scale == "auto"
                else float(robust_scale)
            )
            edge_weights = {k: _robust_weight(v, c, robust) for k, v in final_norms.items()}

    return poses, cost_history, edge_weights


def pairwise_consistency(
    poses: Sequence[torch.Tensor],
    rel: dict,
    transform: str = "se3",
) -> Tuple[float, float]:
    """Max and mean per-edge tangent inconsistency of an absolute-pose set against the measurements.

    ``rel`` maps ``(i, j) -> T_ij``. Returns ``(max||e_ij||, mean||e_ij||)`` — the metric the joint
    solve drives down and the headline "loop closes" number. The tangent norm mixes rotation
    (radians) and translation (splat units); for the synthetic loop both are small so it is a fair
    single scalar, but callers comparing regimes should look at the components.
    """
    log_map = _log_map(transform)
    norms = []
    for (i, j), T_ij in rel.items():
        e = _edge_residual(poses[i], poses[j], T_ij, log_map)
        norms.append(float(e.norm()))
    if not norms:
        return 0.0, 0.0
    t = torch.tensor(norms)
    return float(t.max()), float(t.mean())


def _sequential_poses(
    splats: Sequence[Gaussians],
    rel: dict,
    ref: int,
    n: int,
    device,
    dtype,
) -> List[torch.Tensor]:
    """Open-loop absolute poses by chaining the chain edges out from ``ref`` (the merge-style baseline).

    Breadth-first from the reference over the chain edges (ignores the loop-closure edge), composing
    ``T_child = T_parent @ T_(parent,child)`` — exactly the independent sequential alignment
    :func:`splatreg.merge` performs, so it is the drift baseline the joint solve is compared against.
    """
    poses: List[Optional[torch.Tensor]] = [None] * n
    poses[ref] = torch.eye(4, device=device, dtype=dtype)
    # Adjacency from the relative measurements (both directions available via inverse).
    adj: dict = {k: [] for k in range(n)}
    for (i, j) in rel:
        adj[i].append(j)
        adj[j].append(i)
    queue = [ref]
    while queue:
        a = queue.pop(0)
        for b in adj[a]:
            if poses[b] is not None:
                continue
            if (a, b) in rel:
                T_ab = rel[(a, b)]
            else:
                T_ab = torch.linalg.inv(rel[(b, a)])
            poses[b] = poses[a] @ T_ab
            queue.append(b)
    # Any unreachable splat (disconnected) falls back to identity.
    return [p if p is not None else torch.eye(4, device=device, dtype=dtype) for p in poses]


def bundle_register(
    splats: Sequence[Gaussians],
    ref: int = 0,
    pairs: Union[str, Sequence[Tuple[int, int]]] = "auto",
    *,
    transform: str = "se3",
    init: Union[str, torch.Tensor, None] = "global",
    register_kwargs: Optional[dict] = None,
    max_iters: int = 30,
    damping: float = 1e-6,
    convergence_tol: float = 1e-9,
    robust: Optional[str] = "huber",
    robust_scale: Union[str, float] = "auto",
    reject_threshold: float = 0.1,
    fuse: bool = False,
    return_info: bool = False,
    dedupe: bool = True,
):
    """Jointly register ``N`` splats into one loop-consistent frame via a pose-graph solve.

    Builds a relative-pose constraint ``T_ij`` for every edge in ``pairs`` (each from
    :func:`splatreg.register`), then optimises all ``N`` absolute poses ``T_i`` so every edge
    ``T_i @ T_ij ≈ T_j`` holds *simultaneously* — Gauss-Newton in the SE(3)/Sim(3) tangent, with the
    reference pose pinned to identity to fix the gauge. Unlike the sequential merge (which chains the
    edges and accumulates drift around a loop) the joint optimum spreads the closure error over the
    whole graph, so the max pairwise inconsistency drops.

    Parameters
    ----------
    splats : the ``N`` splats to register together.
    ref : index of the reference splat, held at identity (the global frame). Default 0.
    pairs : ``"auto"`` (chain ``0-1-…-(N-1)`` + loop-closure ``(N-1, 0)``) or an explicit
        ``[(i, j), ...]`` edge list. Each pair becomes one relative-pose constraint via
        ``register(splats[i], splats[j])`` (``j`` aligned into ``i``'s frame).
    transform : ``"se3"`` (6-DoF edges/poses) or ``"sim3"`` (7-DoF, scale included).
    init / register_kwargs : forwarded to each pairwise :func:`splatreg.register` (``init`` defaults
        to ``"global"`` so large inter-capture offsets recover; ``register_kwargs`` can set
        residuals / quality / max_iters of the *pairwise* solves).
    max_iters / damping / convergence_tol : Gauss-Newton hyper-parameters for the **joint** solve
        (a small Levenberg damping keeps the normal equations SPD; the solve stops when the pose
        update norm drops below ``convergence_tol``).
    robust : per-edge robust kernel applied in the joint solve (IRLS). ``"huber"`` (default) or
        ``"cauchy"`` down-weight an edge whose tangent residual exceeds ``robust_scale`` so a single
        wrong pairwise :func:`register` result (a bad edge) cannot corrupt the global poses; ``None``
        recovers the plain least-squares solve. The edge weight is folded into the normal equations
        (``H += w·JᵀJ``, ``b -= w·Jᵀe``) and recomputed every iteration from the current residual.
    robust_scale : the kernel scale ``c`` (the residual norm at which an edge starts to be
        down-weighted). ``"auto"`` (default) sets ``c = 1.4826·median(‖e_ij‖)`` (a robust MAD
        estimate of the inlier spread) each iteration, so it adapts to the loop's noise level; pass a
        float to pin it.
    reject_threshold : an edge whose final IRLS weight is below this is reported in
        ``info.rejected_edges`` (it was effectively excluded from the solution). Diagnostic only —
        the down-weighting itself is continuous, not a hard cut.
    fuse : also return one merged :class:`~splatreg.core.types.Gaussians` — every splat baked into
        its recovered absolute pose, concatenated, and (``dedupe``) overlap-deduped, like
        :func:`splatreg.merge` but with the *jointly* optimised poses.
    return_info : also return a :class:`BundleResult` with the consistency diagnostics.
    dedupe : voxel-dedupe the fused splat when ``fuse=True`` (default ``True``).

    Returns
    -------
    ``list[Tensor]`` of the ``N`` absolute 4x4 poses, by default. With ``fuse=True`` a
    ``(poses, fused)`` tuple; with ``return_info=True`` a trailing :class:`BundleResult` is appended.
    """
    if transform not in _DOF:
        raise ValueError(f"transform must be one of {sorted(_DOF)}, got {transform!r}")
    splats = list(splats)
    n = len(splats)
    if n == 0:
        raise ValueError("bundle_register needs at least one splat.")
    if not (0 <= ref < n):
        raise ValueError(f"ref={ref} out of range for {n} splats")
    device, dtype = splats[ref].means.device, splats[ref].means.dtype
    reg_kw = dict(register_kwargs or {})

    if robust not in (None, "huber", "cauchy"):
        raise ValueError(f"robust must be None, 'huber' or 'cauchy', got {robust!r}")

    if n == 1:
        poses = [torch.eye(4, device=device, dtype=dtype)]
        return _finish(
            poses, splats, ref, {}, transform, [0.0], fuse, return_info, dedupe, device, dtype, {}, []
        )

    edges = _auto_pairs(n) if pairs == "auto" else [tuple(p) for p in pairs]
    for (i, j) in edges:
        if not (0 <= i < n and 0 <= j < n) or i == j:
            raise ValueError(f"invalid edge {(i, j)} for {n} splats")

    # 1) Pairwise relative-pose measurements T_ij (j aligned into i's frame).
    rel: dict = {}
    for (i, j) in edges:
        res = register(splats[i], splats[j], init=init, transform=transform, **reg_kw)
        rel[(i, j)] = res.T.detach().to(device=device, dtype=dtype)
        _log.debug("bundle edge (%d,%d): rmse=%.4g", i, j, res.info.get("rmse", float("nan")))

    # 2) Open-loop seed (the sequential / merge-style chain) — also the drift baseline.
    poses = _sequential_poses(splats, rel, ref, n, device, dtype)

    # 3) Joint (optionally robust) Gauss-Newton / IRLS over the free poses.
    poses, cost_history, edge_weights = solve_pose_graph(
        poses, rel, ref, transform=transform, robust=robust, robust_scale=robust_scale,
        max_iters=max_iters, damping=damping, convergence_tol=convergence_tol,
    )

    rejected = [k for k, w in edge_weights.items() if w < float(reject_threshold)]
    return _finish(
        poses, splats, ref, rel, transform, cost_history, fuse, return_info, dedupe, device, dtype,
        edge_weights, rejected,
    )


def _edge_jac(T_i, T_j, T_ij, exp_fn, log_map, dof, device, dtype):
    """Edge residual ``e`` and its Jacobians ``(Ji, Jj)`` w.r.t. the right-increments of ``T_i``/``T_j``.

    Differentiates ``delta -> log( (T_i exp(d_i) T_ij)^{-1} (T_j exp(d_j)) )`` at ``delta = 0`` via
    autodiff through splatreg's own ``exp``/``log`` — the same right-perturbation convention as the
    pairwise solves, so the joint solve composes cleanly. Returns ``e`` (dof,), ``Ji`` (dof, dof),
    ``Jj`` (dof, dof).
    """
    zero = torch.zeros(dof, device=device, dtype=dtype, requires_grad=True)

    def res_of(di, dj):
        pred = (T_i @ exp_fn(di)) @ T_ij
        err = torch.linalg.inv(pred) @ (T_j @ exp_fn(dj))
        return log_map(err)

    # Jacobians via functional autodiff (small dof, CPU — cheap and exact).
    from torch.autograd.functional import jacobian as _afjac

    di0 = torch.zeros(dof, device=device, dtype=dtype)
    dj0 = torch.zeros(dof, device=device, dtype=dtype)
    Ji, Jj = _afjac(res_of, (di0, dj0), vectorize=True)
    e = res_of(di0, dj0).detach()
    return e, Ji.detach(), Jj.detach()


def _finish(
    poses, splats, ref, rel, transform, cost_history, fuse, return_info, dedupe, device, dtype,
    edge_weights=None, rejected_edges=None,
):
    """Assemble the return value: poses (+ optional fused splat, + optional BundleResult)."""
    max_e, mean_e = pairwise_consistency(poses, rel, transform=transform) if rel else (0.0, 0.0)
    out: list = [poses]
    if fuse:
        out.append(_fuse_with_poses(splats, poses, transform, dedupe))
    if return_info:
        out.append(
            BundleResult(
                poses, max_e, mean_e, cost_history, list(rel.keys()),
                edge_weights=edge_weights, rejected_edges=rejected_edges,
            )
        )
    if len(out) == 1:
        return out[0]
    return tuple(out)


def _fuse_with_poses(
    splats: Sequence[Gaussians], poses: Sequence[torch.Tensor], transform: str, dedupe: bool
) -> Gaussians:
    """Bake each splat into its absolute pose, concatenate, and (optionally) overlap-dedupe."""
    from .fuse import voxel_dedupe, auto_voxel_size

    pieces = []
    for g, T in zip(splats, poses):
        scale = 1.0
        if transform == "sim3":
            det = float(torch.linalg.det(T[:3, :3]).abs().clamp_min(1e-18))
            scale = det ** (1.0 / 3.0)
        pieces.append(_apply_transform_to_gaussians(g, T, scale))
    fused = Gaussians(
        means=torch.cat([p.means for p in pieces], dim=0),
        quats=torch.cat([p.quats for p in pieces], dim=0),
        scales=torch.cat([p.scales for p in pieces], dim=0),
        opacities=torch.cat([p.opacities.reshape(-1) for p in pieces], dim=0),
        colors=None if any(p.colors is None for p in pieces) else torch.cat([p.colors for p in pieces], 0),
        log_scales=splats[0].log_scales,
    )
    if dedupe and len(fused) > 1:
        fused = voxel_dedupe(fused, auto_voxel_size(splats[0]))
    return fused
