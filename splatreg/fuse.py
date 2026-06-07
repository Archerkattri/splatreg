"""Overlap dedupe for merged Gaussian splats — the geometry behind ``merge``'s "not naive cat".

When two registered splats are concatenated their overlap region holds *double* the Gaussian
density (every surface patch is covered by anchors from both captures). A naive ``torch.cat``
keeps that doubling, so the merged splat renders over-bright / over-dense in the seam. This
module collapses the overlap back to single density with a **voxel-grid dedupe**: snap every
Gaussian centre to a regular voxel grid and keep exactly one Gaussian — the highest-opacity
representative — per occupied voxel.

Keeping the *highest-opacity* survivor (rather than averaging or first-wins) is the safe choice
for a registration merge: opacity is the splat's own confidence in an anchor, so the densest,
most-certain Gaussian wins the voxel and faint duplicates are dropped. Non-overlapping regions
have at most one Gaussian per voxel already, so they pass through untouched (the dedupe only
ever removes near-coincident duplicates, never thins genuine geometry — pick ``voxel`` no larger
than the splat's own anchor spacing and it is loss-free outside the overlap).

Voxel size
----------
``voxel`` is the grid edge length in the splat's own units. When ``None`` it is derived from the
splat geometry as a small multiple of the median Gaussian scale (the typical anchor footprint),
so a voxel holds roughly one anchor's worth of surface and only genuine duplicates collide. The
chosen value is returned alongside the deduped splat by :func:`voxel_dedupe_report` for logging.

Self-contained: torch + numpy only. Deterministic — the per-voxel winner is chosen by a stable
(voxel-key, descending-opacity) ordering, so the survivor set is reproducible across runs.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .core.types import Gaussians

__all__ = [
    "voxel_dedupe",
    "voxel_dedupe_report",
    "auto_voxel_size",
    "knn_dedupe",
    "knn_dedupe_report",
    "auto_knn_radius",
]

# Voxel edge as a fraction of the median nearest-neighbour ANCHOR SPACING when ``voxel`` is not
# given. ~0.5x the spacing snaps each surface patch's near-coincident cross-capture duplicates into
# one cell while keeping genuinely distinct neighbours (a full spacing apart) in separate cells.
# Scale is deliberately NOT the basis: a splat can carry large footprints on a fine point lattice
# (or vice-versa); the *spacing* is what distinguishes a duplicate from real geometry.
_VOXEL_SPACING_MULT = 0.5
# Fallbacks when the spacing estimate is unavailable (degenerate / single-axis clouds): a multiple
# of the median Gaussian scale, then a fraction of the bounding diagonal.
_VOXEL_SCALE_FALLBACK_MULT = 1.0
_VOXEL_BBOX_FRAC = 1.0e-2
# Cross-splat KNN-dedupe radius as a fraction of the median anchor spacing. Matched to
# ``_VOXEL_SPACING_MULT`` so the KNN pass removes the SAME near-coincident duplicates as the voxel
# pass — minus the grid-alignment artefact: voxel-snapping keeps any duplicate pair that straddles a
# cell boundary (leaving ~16% residual overlap), whereas a radius test is translation-invariant and
# catches them. Half the spacing never reaches a genuine one-spacing-apart neighbour, so it stays
# loss-free outside the overlap.
_KNN_RADIUS_MULT = 0.5
_EPS = 1.0e-12


def _linear_scales(g: Gaussians) -> torch.Tensor:
    """Per-Gaussian linear scales ``(N, 3)`` (``exp`` applied when the splat holds log-scales)."""
    return g.scales.exp() if g.log_scales else g.scales


def _median_anchor_spacing(g: Gaussians) -> Optional[float]:
    """Median nearest-neighbour ANCHOR SPACING of the splat, or ``None`` if degenerate.

    Strided-subsamples the means (bounded ``cdist``), drops self / exact coincidences, and returns
    the median of the per-anchor nearest non-coincident-neighbour distance. Shared by
    :func:`auto_voxel_size` and :func:`auto_knn_radius` so both size off the same spacing estimate.
    """
    n = len(g)
    if n <= 1:
        return None
    means = g.means
    m = min(n, 4096)
    step = max(1, n // m)
    sample = means[::step][:m]  # deterministic strided subsample
    ref = means if n <= 20000 else means[:: max(1, n // 20000)][:20000]
    d = torch.cdist(sample, ref)
    d = d.masked_fill(d <= _EPS, float("inf"))  # drop self / exact coincidence
    nn = d.min(dim=1).values  # nearest non-coincident neighbour
    finite = nn[torch.isfinite(nn) & (nn > 0.0)]
    if finite.numel() > 0:
        med_nn = float(finite.median().item())
        if med_nn > 0.0:
            return med_nn
    return None


def _spacing_fallback(g: Gaussians) -> float:
    """Spacing fallback for degenerate clouds: median linear scale, then a bbox-diagonal fraction."""
    scales = _linear_scales(g)
    per = scales.mean(dim=-1)
    fs = per[torch.isfinite(per) & (per > 0.0)]
    if fs.numel() > 0:
        med = float(fs.median().item())
        if med > 0.0:
            return _VOXEL_SCALE_FALLBACK_MULT * med
    means = g.means
    extent = float((means.amax(dim=0) - means.amin(dim=0)).norm().item())
    return max(_VOXEL_BBOX_FRAC * extent, _EPS)


def auto_voxel_size(g: Gaussians) -> float:
    """Derive a dedupe voxel edge from the splat's anchor SPACING (median nearest-neighbour
    distance) -- ``_VOXEL_SPACING_MULT x`` it.

    NOTE: call this on a single, duplicate-free splat (e.g. the merge *reference*). Running it on an
    already-concatenated splat would read the overlap duplicates themselves as the spacing and
    under-dedupe. Falls back to the median scale, then a bbox fraction, for degenerate inputs.
    """
    if len(g) <= 1:
        return 1.0
    med_nn = _median_anchor_spacing(g)
    if med_nn is not None:
        return _VOXEL_SPACING_MULT * med_nn
    return _spacing_fallback(g)


def auto_knn_radius(g: Gaussians) -> float:
    """Derive a cross-splat KNN-dedupe radius from the anchor SPACING -- ``_KNN_RADIUS_MULT x`` it.

    Same spacing basis (and the same "call on the clean reference, not the merged splat" caveat) as
    :func:`auto_voxel_size`; only the multiplier differs. A radius of half the anchor spacing
    suppresses a duplicate that landed *anywhere* within half a lattice step of a kept anchor —
    including the sub-voxel-boundary pairs a grid snap leaves behind — while never reaching a genuine
    one-spacing-away neighbour. Falls back to the median scale, then a bbox fraction, when degenerate.
    """
    if len(g) <= 1:
        return _EPS
    med_nn = _median_anchor_spacing(g)
    if med_nn is not None:
        return _KNN_RADIUS_MULT * med_nn
    return _spacing_fallback(g)


def _opacity_key(opacities: torch.Tensor, n: int) -> torch.Tensor:
    """Flatten opacities to a ``(N,)`` comparable score for the per-voxel winner choice."""
    o = opacities.reshape(-1) if opacities.dim() > 1 else opacities
    if o.shape[0] != n:
        o = o.reshape(n)
    return o


def _voxel_winner_indices(means: torch.Tensor, opacities: torch.Tensor, voxel: float) -> torch.Tensor:
    """Indices of the highest-opacity Gaussian in each occupied voxel (stable, sorted ascending).

    Means are snapped to integer voxel coordinates and hashed to one key per cell; within a cell
    the largest-opacity row wins (first-row tie-break). Returns the kept indices sorted ascending
    so the survivors preserve the input ordering.
    """
    n = means.shape[0]
    coords = torch.floor(means / voxel).to(torch.int64)  # (N, 3) voxel indices
    # Shift to a non-negative origin so a positional hash stays monotone and collision-free.
    coords = coords - coords.amin(dim=0, keepdim=True)
    dims = coords.amax(dim=0) + 1  # (3,) grid extent per axis
    stride_y = dims[2]
    stride_x = dims[1] * dims[2]
    keys = coords[:, 0] * stride_x + coords[:, 1] * stride_y + coords[:, 2]  # (N,) unique per cell

    opa = _opacity_key(opacities, n)
    # Stable per-voxel argmax-opacity: sort rows by (key asc, opacity desc); the first row of each
    # key-run is that voxel's highest-opacity survivor. ``stable`` keeps the original index order as
    # the final tie-break, so the result is deterministic.
    order = torch.argsort(opa, descending=True, stable=True)  # opacity desc (primary content)
    keys_o = keys[order]
    key_sorted, key_order = torch.sort(keys_o, stable=True)  # keys asc, opacity order kept
    order = order[key_order]
    # First occurrence of each distinct key in the (key asc, opacity desc) ordering.
    first_mask = torch.ones_like(key_sorted, dtype=torch.bool)
    first_mask[1:] = key_sorted[1:] != key_sorted[:-1]
    keep = order[first_mask]
    return torch.sort(keep).values  # ascending -> stable survivor set


def _index_gaussians(g: Gaussians, idx: torch.Tensor) -> Gaussians:
    """Gather a subset of ``g`` by row index, carrying every field (and optional colors)."""
    opac = g.opacities
    opac = opac[idx] if opac.dim() == 1 else opac[idx]
    return Gaussians(
        means=g.means[idx],
        quats=g.quats[idx],
        scales=g.scales[idx],
        opacities=opac,
        colors=None if g.colors is None else g.colors[idx],
        log_scales=g.log_scales,
    )


def voxel_dedupe(g: Gaussians, voxel: Optional[float] = None) -> Gaussians:
    """Collapse near-coincident Gaussians to one-per-voxel, keeping the highest-opacity survivor.

    Args:
        g: the (typically concatenated) splat to dedupe.
        voxel: grid edge length in the splat's units; ``None`` derives it via
            :func:`auto_voxel_size` (a small multiple of the median Gaussian scale).

    Returns:
        Gaussians: a subset of ``g`` with at most one Gaussian per occupied voxel. Fields,
        ``colors`` (if present), ``log_scales``, device and dtype are preserved. An empty or
        single-Gaussian input is returned unchanged.
    """
    return voxel_dedupe_report(g, voxel)[0]


def voxel_dedupe_report(g: Gaussians, voxel: Optional[float] = None) -> Tuple[Gaussians, float]:
    """Like :func:`voxel_dedupe` but also returns the voxel edge actually used (for logging)."""
    n = len(g)
    if n <= 1:
        return g, float(voxel) if voxel is not None else auto_voxel_size(g)
    v = float(voxel) if voxel is not None else auto_voxel_size(g)
    if not (v > 0.0):
        raise ValueError(f"voxel_dedupe: voxel size must be > 0, got {v}.")
    keep = _voxel_winner_indices(g.means, g.opacities, v)
    return _index_gaussians(g, keep), v


# ── cross-splat KNN (radius) dedupe ───────────────────────────────────────────────────
# A grid snap keeps any near-coincident pair that happens to straddle a voxel boundary, so after a
# voxel dedupe a registered overlap still carries ~16% residual duplicates. This radius pass closes
# that gap: a Gaussian is dropped when a STRICTLY-HIGHER-PRIORITY Gaussian (higher opacity; lower
# index breaks ties) sits within ``radius`` of it — translation-invariant non-max-suppression, so
# boundary-straddling duplicates are removed too. The "higher opacity wins" survivor rule matches
# the voxel pass exactly. Deterministic (stable opacity+index priority). Single forward pass,
# chunked over the query axis so peak memory is ``_KNN_CHUNK x N`` regardless of splat size.
_KNN_CHUNK = 2048


def _knn_keep_mask(means: torch.Tensor, opacities: torch.Tensor, radius: float) -> torch.Tensor:
    """Boolean keep-mask: drop a point iff a higher-priority point lies within ``radius``.

    Priority is ``(opacity desc, index asc)``. A point ``i`` is suppressed when there exists ``j != i``
    with ``dist(i, j) <= radius`` and ``j`` higher-priority than ``i`` — i.e. ``opa[j] > opa[i]`` or
    (``opa[j] == opa[i]`` and ``j < i``). With a strict priority every duplicate cluster keeps exactly
    its highest-opacity / lowest-index member (mutual suppression is impossible), so the result is a
    well-defined single-density survivor set. Chunked over the query rows to bound peak memory.
    """
    n = means.shape[0]
    opa = _opacity_key(opacities, n)
    idx_all = torch.arange(n, device=means.device)
    r2 = float(radius) * float(radius)
    keep = torch.ones(n, dtype=torch.bool, device=means.device)
    chunk = max(1, _KNN_CHUNK)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        rows = idx_all[start:end]  # (c,) absolute query indices
        d2 = torch.cdist(means[start:end], means) ** 2  # (c, N)
        within = d2 <= r2
        within[torch.arange(end - start, device=means.device), rows] = False  # exclude self
        oi = opa[start:end].unsqueeze(1)  # (c, 1)
        higher = (opa.unsqueeze(0) > oi) | (
            (opa.unsqueeze(0) == oi) & (idx_all.unsqueeze(0) < rows.unsqueeze(1))
        )
        keep[start:end] = ~(within & higher).any(dim=1)
    return keep


def _knn_keep_mask_indexed(means: torch.Tensor, opacities: torch.Tensor, radius: float) -> torch.Tensor:
    """Index-accelerated equivalent of :func:`_knn_keep_mask` — same survivor set, near-O(N).

    Builds a :class:`splatreg.spatial_index.SpatialIndex` over the means and asks it for every
    in-radius neighbour of each point (the grid touches only the local cells, not all N anchors),
    then applies the identical ``(opacity desc, index asc)`` priority suppression. The result is the
    SAME boolean keep-mask as the brute-force pass — only the neighbour search is pruned — so a
    scene-scale splat dedupes without the O(N^2) ``cdist``. Falls back to the brute-force mask if the
    spatial index module is unavailable.
    """
    try:
        from .spatial_index import SpatialIndex
    except Exception:  # pragma: no cover - index is optional
        return _knn_keep_mask(means, opacities, radius)
    n = means.shape[0]
    opa = _opacity_key(opacities, n)
    idx = SpatialIndex(means, cell=float(radius))
    # Query a hair beyond `radius` so the grid returns a SUPERSET of the brute-force in-radius pairs;
    # the squared re-filter below is then the sole (and identical) boundary arbiter.
    qi, ai = idx.radius(means, float(radius) * (1.0 + 1e-6))  # (P,) query / anchor index pairs
    if qi.numel() == 0:
        return torch.ones(n, dtype=torch.bool, device=means.device)
    self_pair = qi == ai
    qi, ai = qi[~self_pair], ai[~self_pair]
    # Re-filter with the SAME distance kernel the brute path uses — ``cdist(...)**2 <= r2`` — so the
    # survivor set is bit-identical. ``cdist**2`` (a fused kernel) and a manual ``(p-q)^2.sum()`` round
    # differently for a pair at distance exactly r, which would otherwise flip a handful of
    # on-the-boundary duplicates. Computed per-pair via a batched 1-row cdist to stay O(P).
    pair_d = torch.linalg.vector_norm(means[ai] - means[qi], dim=-1)  # matches cdist's norm reduction
    within = pair_d * pair_d <= float(radius) * float(radius)
    qi, ai = qi[within], ai[within]
    # Neighbour ai is strictly higher-priority than query qi?
    higher = (opa[ai] > opa[qi]) | ((opa[ai] == opa[qi]) & (ai < qi))
    suppressed_q = qi[higher]
    keep = torch.ones(n, dtype=torch.bool, device=means.device)
    keep[suppressed_q] = False
    return keep


def knn_dedupe(g: Gaussians, radius: Optional[float] = None, *, use_index: bool = False) -> Gaussians:
    """Cross-splat radius dedupe: keep the highest-opacity survivor within every ``radius`` ball.

    The translation-invariant complement to :func:`voxel_dedupe` — it removes the residual overlap a
    voxel grid leaves at cell boundaries (see this section's note). Use it (e.g. via
    ``merge(..., dedupe_method="knn")``) when the voxel pass under-dedupes a registered seam.

    Args:
        g: the (typically concatenated) splat to dedupe.
        radius: suppression ball radius in the splat's units; ``None`` derives it via
            :func:`auto_knn_radius` (half the median anchor spacing).
        use_index: when ``True`` route the neighbour search through the voxel-hash
            :class:`splatreg.spatial_index.SpatialIndex` (near-O(N) on scene-scale splats) instead of
            the default O(N^2) chunked ``cdist`` scan. The survivor set is IDENTICAL either way — only
            the neighbour search is pruned. Default ``False`` (the original brute-force path).

    Returns:
        Gaussians: a subset of ``g`` with no two survivors closer than ``radius`` to a
        higher-priority neighbour. Fields, ``colors``, ``log_scales``, device and dtype are
        preserved. An empty / single-Gaussian input is returned unchanged.
    """
    return knn_dedupe_report(g, radius, use_index=use_index)[0]


def knn_dedupe_report(
    g: Gaussians, radius: Optional[float] = None, *, use_index: bool = False
) -> Tuple[Gaussians, float]:
    """Like :func:`knn_dedupe` but also returns the radius actually used (for logging).

    ``use_index`` selects the voxel-hash-accelerated neighbour search (identical survivors,
    near-O(N) on large splats); the default brute-force ``cdist`` path is unchanged.
    """
    n = len(g)
    if n <= 1:
        return g, float(radius) if radius is not None else auto_knn_radius(g)
    r = float(radius) if radius is not None else auto_knn_radius(g)
    if not (r > 0.0):
        raise ValueError(f"knn_dedupe: radius must be > 0, got {r}.")
    if use_index:
        keep_mask = _knn_keep_mask_indexed(g.means, g.opacities, r)
    else:
        keep_mask = _knn_keep_mask(g.means, g.opacities, r)
    keep = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)
    return _index_gaussians(g, keep), r
