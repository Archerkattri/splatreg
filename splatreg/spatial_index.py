"""Scene-scale spatial index over Gaussian means — O(1)-ish neighbour queries for big splats.

The Gaussian-SDF / dedupe / merge query path is a *spatial* one: "which anchors lie near this
query point?". The naive answer forms the full ``(queries, anchors)`` distance matrix — an O(N·M)
scan that is fine for an object splat (a few thousand anchors) but quadratic-blows-up on a
scene-scale splat (hundreds of thousands of anchors). This module builds a **voxel-hash grid**
over the anchor means once, then answers the three queries the rest of splatreg needs against only
the anchors in the relevant cells:

* :meth:`SpatialIndex.radius` — every anchor within ``r`` of each query (the dedupe / SDF support).
* :meth:`SpatialIndex.knn` — the ``k`` nearest anchors to each query (the truncated-SDF support).
* :meth:`SpatialIndex.region` — every anchor inside an axis-aligned box (region / crop queries).

Why a voxel hash (not a kd-tree)
--------------------------------
A uniform voxel grid is the natural structure for a Gaussian splat: the anchors already trace a
surface at a roughly uniform spacing (the Gaussian footprint), so a grid sized to that spacing
holds ~O(1) anchors per occupied cell and a radius/knn query only has to look at the query's cell
plus its neighbours — independent of the total anchor count. It is also pure ``torch`` (a
``sort`` + bucketing, no recursion, no Python per-node walk), so it stays vectorised and
device-agnostic, and it is **deterministic** (a stable sort keys the buckets). A balanced kd-tree
would give the same asymptotics but needs a Python build/descent that does not vectorise on a
splat this size.

Design
------
The grid is built by snapping each anchor to integer voxel coordinates ``floor((x - origin) /
cell)`` and bucketing anchors by a flattened, collision-free cell key (an argsort gives contiguous
per-cell runs; a hash map ``cell_key -> (run_start, run_len)`` then locates a cell in O(1)). A
query gathers the candidate anchors from the query cell's 3x3x3 (radius) or expanding-ring (knn)
neighbourhood and does the *exact* distance test only on those — so the result is exact (identical
to brute force), just without touching the far anchors. ``cell`` defaults to the splat's median
anchor spacing (so a radius up to one cell only ever needs the 27-neighbourhood), auto-derived via
:func:`splatreg.fuse.auto_voxel_size`-style spacing.

This is an *optional acceleration*: the SDF / dedupe paths keep their existing brute-force default
and only route through the index when explicitly handed one (or when ``use_index=True`` asks for
an auto-built one), so behaviour is unchanged and the index is pure upside on large splats.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .core.types import Gaussians

__all__ = ["SpatialIndex", "build_index"]

_EPS = 1.0e-12


def _median_spacing(means: torch.Tensor) -> float:
    """Median nearest-neighbour anchor spacing (bounded subsample), or a bbox fallback.

    Mirrors :func:`splatreg.fuse._median_anchor_spacing` but takes the raw means tensor so the
    index has no import cycle with ``fuse``. Strided-subsamples to bound the ``cdist``.
    """
    n = int(means.shape[0])
    if n <= 1:
        extent = float((means.amax(dim=0) - means.amin(dim=0)).norm()) if n == 1 else 1.0
        return max(extent, 1.0)
    m = min(n, 2048)
    step = max(1, n // m)
    sample = means[::step][:m]
    ref = means if n <= 8000 else means[:: max(1, n // 8000)][:8000]
    d = torch.cdist(sample, ref)
    d = d.masked_fill(d <= _EPS, float("inf"))
    nn = d.min(dim=1).values
    finite = nn[torch.isfinite(nn) & (nn > 0.0)]
    if finite.numel() > 0:
        med = float(finite.median())
        if med > 0.0:
            return med
    extent = float((means.amax(dim=0) - means.amin(dim=0)).norm())
    return max(1.0e-2 * extent, _EPS)


class SpatialIndex:
    """A voxel-hash grid over a point set supporting radius / knn / region queries.

    Build once from a ``(M, 3)`` point tensor (or a :class:`~splatreg.core.types.Gaussians` via
    :func:`build_index`), then query repeatedly. All queries return results IDENTICAL to a
    brute-force scan — the grid only prunes which anchors are distance-tested, never the answer.

    Parameters
    ----------
    points : ``(M, 3)`` positions to index (the anchor means).
    cell : voxel edge in the points' units. ``None`` auto-derives it from the median anchor
        spacing, so a radius up to one cell needs only the 27-cell neighbourhood. Must be > 0.
    """

    def __init__(self, points: torch.Tensor, cell: Optional[float] = None):
        if points.dim() != 2 or points.shape[-1] != 3:
            raise ValueError(f"SpatialIndex: points must be (M, 3), got {tuple(points.shape)}.")
        self.points = points
        self.device = points.device
        self.dtype = points.dtype
        self.n = int(points.shape[0])
        c = float(cell) if cell is not None else _median_spacing(points)
        if not (c > 0.0):
            raise ValueError(f"SpatialIndex: cell must be > 0, got {c}.")
        self.cell = c
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        """Bucket anchors by voxel cell: sort by a flat cell key into contiguous per-cell runs."""
        if self.n == 0:
            self.origin = torch.zeros(3, device=self.device, dtype=self.dtype)
            self.dims = torch.ones(3, device=self.device, dtype=torch.int64)
            self._order = torch.zeros(0, dtype=torch.int64, device=self.device)
            self._cell_keys = torch.zeros(0, dtype=torch.int64, device=self.device)
            self._run_start = {}
            return
        self.origin = self.points.amin(dim=0)
        coords = self._coords(self.points)  # (M, 3) int64 voxel coords >= 0
        self.dims = coords.amax(dim=0) + 1  # (3,) grid extent
        keys = self._flatten(coords)  # (M,) one int per cell
        # Stable sort -> contiguous per-cell runs; the survivor order is deterministic.
        order = torch.argsort(keys, stable=True)
        self._order = order
        self._cell_keys = keys[order]
        # cell_key -> (run_start, run_len). Built on CPU once (host map); query gathers slices.
        keys_cpu = self._cell_keys.cpu()
        uniq, first_idx, counts = torch.unique_consecutive(
            keys_cpu, return_inverse=False, return_counts=True
        ), None, None
        # unique_consecutive with return_counts gives run lengths in order; recompute starts.
        u, cnt = torch.unique_consecutive(keys_cpu, return_counts=True)
        starts = torch.cumsum(torch.cat([torch.zeros(1, dtype=cnt.dtype), cnt[:-1]]), dim=0)
        self._run_start = {
            int(k): (int(s), int(c)) for k, s, c in zip(u.tolist(), starts.tolist(), cnt.tolist())
        }

    def _coords(self, pts: torch.Tensor) -> torch.Tensor:
        """Snap points to non-negative integer voxel coordinates relative to the grid origin."""
        c = torch.floor((pts - self.origin) / self.cell).to(torch.int64)
        return c.clamp_min(0)

    def _flatten(self, coords: torch.Tensor) -> torch.Tensor:
        """Flatten (…, 3) integer voxel coords to a single collision-free key per cell."""
        sx = self.dims[1] * self.dims[2]
        sy = self.dims[2]
        return coords[..., 0] * sx + coords[..., 1] * sy + coords[..., 2]

    def _gather_cells(self, cell_keys) -> torch.Tensor:
        """Original-index anchors living in any of the given (host int) cell keys."""
        slices = []
        for k in cell_keys:
            run = self._run_start.get(int(k))
            if run is not None:
                s, ln = run
                slices.append(self._order[s : s + ln])
        if not slices:
            return torch.zeros(0, dtype=torch.int64, device=self.device)
        return torch.cat(slices, dim=0)

    def _neighbour_keys(self, coord: torch.Tensor, ring: int) -> list:
        """Flat keys of every cell within a ``±ring`` cube around an integer voxel ``coord``."""
        rng = range(-ring, ring + 1)
        cx, cy, cz = int(coord[0]), int(coord[1]), int(coord[2])
        sx = int(self.dims[1] * self.dims[2])
        sy = int(self.dims[2])
        dimx, dimy, dimz = int(self.dims[0]), int(self.dims[1]), int(self.dims[2])
        keys = []
        for dx in rng:
            x = cx + dx
            if x < 0 or x >= dimx:
                continue
            for dy in rng:
                y = cy + dy
                if y < 0 or y >= dimy:
                    continue
                for dz in rng:
                    z = cz + dz
                    if z < 0 or z >= dimz:
                        continue
                    keys.append(x * sx + y * sy + z)
        return keys

    # ------------------------------------------------------------------ queries
    def radius(self, queries: torch.Tensor, r: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """All anchors within distance ``r`` of each query, in flat (query, anchor) pair form.

        Returns ``(pair_query_idx, pair_anchor_idx)`` — two ``(P,)`` long tensors so that
        ``points[pair_anchor_idx[j]]`` is within ``r`` of ``queries[pair_query_idx[j]]``. This flat
        form composes directly with the chunked SDF / dedupe code (no ragged padding). The result is
        EXACT: every in-radius anchor is returned and no out-of-radius anchor is, identical to a
        brute-force ``cdist <= r`` — the grid only limits which anchors are distance-tested.
        """
        if queries.dim() != 2 or queries.shape[-1] != 3:
            raise ValueError(f"radius: queries must be (Q, 3), got {tuple(queries.shape)}.")
        if not (r > 0.0):
            raise ValueError(f"radius: r must be > 0, got {r}.")
        q_idx_parts: list = []
        a_idx_parts: list = []
        if self.n == 0:
            empty = torch.zeros(0, dtype=torch.int64, device=self.device)
            return empty, empty.clone()
        # A radius r spans ceil(r / cell) cells in each direction.
        ring = int(torch.ceil(torch.tensor(r / self.cell)).item())
        q_coords = self._coords(queries)
        r2 = float(r) * float(r)
        for qi in range(int(queries.shape[0])):
            cand = self._gather_cells(self._neighbour_keys(q_coords[qi], ring))
            if cand.numel() == 0:
                continue
            # Use cdist for the exact distance so the boundary test matches a brute-force
            # ``cdist(queries, points) <= r`` bit-for-bit (a manual (p-q)^2 sum rounds differently
            # right at d == r, which would flip a handful of exactly-on-the-radius pairs).
            d = torch.cdist(queries[qi : qi + 1], self.points[cand]).reshape(-1)
            keep = cand[d <= float(r)]
            if keep.numel():
                q_idx_parts.append(torch.full((keep.numel(),), qi, dtype=torch.int64, device=self.device))
                a_idx_parts.append(keep)
        if not a_idx_parts:
            empty = torch.zeros(0, dtype=torch.int64, device=self.device)
            return empty, empty.clone()
        return torch.cat(q_idx_parts), torch.cat(a_idx_parts)

    def knn(self, queries: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """The ``k`` nearest anchors to each query.

        Returns ``(idx, dist)`` of shape ``(Q, k)`` (``idx`` into ``points``, ``dist`` Euclidean),
        sorted nearest-first per query — EXACTLY the brute-force ``cdist``-topk result. Expands the
        searched cell ring until at least ``k`` candidates are found AND the ring radius safely
        exceeds the k-th distance (so no closer anchor in an unsearched outer cell is missed), then
        does the exact top-k over the candidates. ``k`` is clamped to the anchor count.
        """
        if queries.dim() != 2 or queries.shape[-1] != 3:
            raise ValueError(f"knn: queries must be (Q, 3), got {tuple(queries.shape)}.")
        kk = max(1, min(int(k), self.n))
        q = int(queries.shape[0])
        out_idx = torch.empty((q, kk), dtype=torch.int64, device=self.device)
        out_dist = torch.empty((q, kk), dtype=self.dtype, device=self.device)
        q_coords = self._coords(queries)
        max_ring = int(self.dims.amax().item())
        for qi in range(q):
            ring = 1
            while True:
                cand = self._gather_cells(self._neighbour_keys(q_coords[qi], ring))
                # Enough candidates AND the searched cube guarantees the k-th NN is inside it:
                # any anchor outside a `ring`-cube is at least `ring*cell` away (minus the query's
                # offset within its own cell, bounded by one cell), so once the k-th candidate
                # distance < (ring) * cell we can stop. A conservative (ring) margin is used.
                if cand.numel() >= kk:
                    d = torch.cdist(queries[qi : qi + 1], self.points[cand]).reshape(-1)
                    top_d, top_i = torch.topk(d, k=kk, largest=False)
                    if float(top_d[-1]) <= ring * self.cell or ring >= max_ring:
                        out_idx[qi] = cand[top_i]
                        out_dist[qi] = top_d
                        break
                elif ring >= max_ring:
                    # Fewer than k anchors exist in the whole grid reach: pad with the closest.
                    if cand.numel() == 0:
                        cand = torch.arange(self.n, device=self.device)
                    d = torch.cdist(queries[qi : qi + 1], self.points[cand]).reshape(-1)
                    kq = min(kk, cand.numel())
                    top_d, top_i = torch.topk(d, k=kq, largest=False)
                    out_idx[qi, :kq] = cand[top_i]
                    out_dist[qi, :kq] = top_d
                    if kq < kk:  # repeat the last (degenerate tiny cloud)
                        out_idx[qi, kq:] = cand[top_i[-1]]
                        out_dist[qi, kq:] = top_d[-1]
                    break
                ring += 1
        return out_idx, out_dist

    def region(self, lo, hi) -> torch.Tensor:
        """Indices of every anchor inside the axis-aligned box ``[lo, hi]`` (inclusive).

        ``lo`` / ``hi`` are length-3 (tensor or sequence). Gathers the box's covered cells, then
        does the exact per-axis bound test on those candidates — the EXACT set a brute-force
        ``((points >= lo) & (points <= hi)).all(-1)`` would return.
        """
        lo = torch.as_tensor(lo, device=self.device, dtype=self.dtype).reshape(3)
        hi = torch.as_tensor(hi, device=self.device, dtype=self.dtype).reshape(3)
        if self.n == 0:
            return torch.zeros(0, dtype=torch.int64, device=self.device)
        lo_c = self._coords(lo.unsqueeze(0))[0]
        hi_c = self._coords(hi.unsqueeze(0))[0]
        keys = []
        sx = int(self.dims[1] * self.dims[2])
        sy = int(self.dims[2])
        dimx, dimy, dimz = int(self.dims[0]), int(self.dims[1]), int(self.dims[2])
        for x in range(max(0, int(lo_c[0])), min(dimx, int(hi_c[0]) + 1)):
            for y in range(max(0, int(lo_c[1])), min(dimy, int(hi_c[1]) + 1)):
                for z in range(max(0, int(lo_c[2])), min(dimz, int(hi_c[2]) + 1)):
                    keys.append(x * sx + y * sy + z)
        cand = self._gather_cells(keys)
        if cand.numel() == 0:
            return cand
        p = self.points[cand]
        inside = ((p >= lo) & (p <= hi)).all(dim=-1)
        return torch.sort(cand[inside]).values


def build_index(g, cell: Optional[float] = None) -> SpatialIndex:
    """Build a :class:`SpatialIndex` over a splat's means (or a raw ``(M, 3)`` tensor).

    Convenience constructor: accepts a :class:`~splatreg.core.types.Gaussians` (indexes its
    ``means``) or a point tensor directly. ``cell`` ``None`` auto-sizes to the median anchor spacing.
    """
    pts = g.means if isinstance(g, Gaussians) else g
    return SpatialIndex(pts, cell=cell)
