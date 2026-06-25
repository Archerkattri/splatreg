#!/usr/bin/env python
"""Regression tests for ICP memory scaling.

Large production splats can have millions of anchors. The default registration
stack must not build a dense source x target distance matrix when quality has
explicitly capped the working sample.
"""

from __future__ import annotations

import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from splatreg.api import _default_residuals  # noqa: E402
from splatreg.core.types import Gaussians  # noqa: E402
from splatreg.quality import QualityConfig  # noqa: E402
from splatreg.residuals import ICP, SDF  # noqa: E402


def _make_splat(n: int, seed: int = 0) -> Gaussians:
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n, 3, generator=g)
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    return Gaussians(
        means=means,
        quats=q,
        scales=torch.full((n, 3), 0.01),
        opacities=torch.ones(n),
        log_scales=False,
    )


def test_default_residuals_pass_quality_sample_cap_to_icp():
    target = _make_splat(200, seed=1)
    q = QualityConfig(n_points=37, sdf_chunk_size=19, label="test")

    residuals = _default_residuals(target, q)
    icp = next(r for r in residuals if isinstance(r, ICP))

    assert icp.n_points == 37
    assert icp.nn_chunk_size == 19


def test_default_residuals_give_sdf_precomputed_splat_normals():
    target = _make_splat(200, seed=1)
    q = QualityConfig(n_points=37, sdf_chunk_size=19, label="test")

    residuals = _default_residuals(target, q)
    sdf = next(r for r in residuals if isinstance(r, SDF))

    assert sdf.target_normals is not None
    assert sdf.target_normals.shape == target.means.shape


def test_icp_correspondences_sample_source_and_chunk_cdist(monkeypatch):
    target = _make_splat(120, seed=2)
    source = _make_splat(90, seed=3)
    icp = ICP(point_to_plane=False, n_points=25, nn_chunk_size=7)
    cdist_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    real_cdist = torch.cdist

    def tracking_cdist(x1, x2, *args, **kwargs):
        cdist_shapes.append((tuple(x1.shape), tuple(x2.shape)))
        assert x1.shape[0] <= 7
        assert x2.shape[0] == len(target)
        return real_cdist(x1, x2, *args, **kwargs)

    monkeypatch.setattr(torch, "cdist", tracking_cdist)

    p, q, n, src, keep = icp._correspondences(torch.eye(4), target, source)

    assert p.shape == q.shape == n.shape == src.shape == (25, 3)
    assert keep.shape == (25,)
    assert len(cdist_shapes) == 4
