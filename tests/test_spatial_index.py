#!/usr/bin/env python
"""CPU performance checks for spatial-index acceleration; no unit suite."""

from __future__ import annotations

import time

import pytest
import torch

from splatreg.spatial_index import SpatialIndex
from splatreg.fuse import _knn_keep_mask, _knn_keep_mask_indexed


def _rand_points(count: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(count, 3, generator=generator)


@pytest.mark.parametrize("nq", [4000])
@pytest.mark.performance
def test_batch_query_speedup_vs_loop(device, nq):
    """Benchmark loop-free batch queries against Python per-query loops on CPU."""
    if device != "cpu":
        pytest.skip("speedup asserted on CPU")
    torch.set_num_threads(2)
    points = _rand_points(8000).to(device)
    queries = _rand_points(nq, seed=5).to(device)
    index = SpatialIndex(points)
    radius = 0.03

    index.radius(queries[:50], radius)
    index.radius_batch(queries[:50], radius)
    index.knn(queries[:50], 8)
    index.knn_batch(queries[:50], 8)

    started = time.time()
    index.radius(queries, radius)
    loop_radius = time.time() - started
    started = time.time()
    index.radius_batch(queries, radius)
    batch_radius = time.time() - started
    started = time.time()
    index.knn(queries, 8)
    loop_knn = time.time() - started
    started = time.time()
    index.knn_batch(queries, 8)
    batch_knn = time.time() - started

    radius_speedup = loop_radius / max(batch_radius, 1e-9)
    knn_speedup = loop_knn / max(batch_knn, 1e-9)
    print(
        f"\n[spatial-index batch] Q={nq} | radius loop={loop_radius:.3f}s "
        f"batch={batch_radius:.3f}s ({radius_speedup:.1f}x) "
        f"| knn loop={loop_knn:.3f}s batch={batch_knn:.3f}s ({knn_speedup:.1f}x)"
    )
    assert radius_speedup >= 2.0, f"radius_batch expected >=2x over loop, got {radius_speedup:.2f}x"
    assert knn_speedup >= 2.0, f"knn_batch expected >=2x over loop, got {knn_speedup:.2f}x"


@pytest.mark.parametrize("count", [40000])
@pytest.mark.performance
def test_dedupe_speedup_on_scene_scale(device, count):
    """Benchmark indexed Gaussian dedupe against the O(N²) brute scan on CPU."""
    if device != "cpu":
        pytest.skip("speedup asserted on CPU")
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(0)
    base = torch.rand(count, 3, generator=generator)
    radius = 0.002
    duplicates = base[: count // 5] + (radius / 4.0) * torch.randn(
        count // 5, 3, generator=generator
    )
    means = torch.cat([base, duplicates], dim=0)
    opacity = torch.rand(means.shape[0], generator=generator)

    started = time.time()
    brute = _knn_keep_mask(means, opacity, radius)
    brute_seconds = time.time() - started
    started = time.time()
    indexed = _knn_keep_mask_indexed(means, opacity, radius)
    indexed_seconds = time.time() - started

    speedup = brute_seconds / max(indexed_seconds, 1e-9)
    difference_fraction = float((brute != indexed).float().mean())
    print(
        f"\n[spatial-index dedupe] N={means.shape[0]} brute={brute_seconds:.2f}s "
        f"index={indexed_seconds:.2f}s speedup={speedup:.1f}x "
        f"diff_frac={difference_fraction:.2e}"
    )
    assert difference_fraction < 1e-3, (
        f"index dedupe diverged from brute by {difference_fraction:.2e} (> float-boundary)"
    )
    assert speedup >= 1.5, f"expected >=1.5x speedup at N={means.shape[0]}, got {speedup:.2f}x"
