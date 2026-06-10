#!/usr/bin/env python
"""Regression tests for three correctness fixes (CPU-only, deterministic).

1. ``test_sdf_weight_applied_once`` — the SDF residual's contribution to the assembled
   normal equations scales LINEARLY with ``weight`` (the standard least-squares convention),
   not as ``weight**1.5`` / ``weight**3``. Before the fix ``SDF.residual``/``SDF.jacobian``
   pre-multiplied by ``self.weight`` AND the solver folded in ``sqrt(weight)`` again, so the
   effective objective scaled as ``weight**3``.

2. ``test_merge_preserves_scales_across_log_convention`` — merging a linear-scale splat with a
   log-scale splat decodes back to the SAME real-world scales (no mis-exponentiation). Before
   the fix the fused splat labelled every piece with the reference's ``log_scales`` flag while
   concatenating their raw, mixed-convention ``scales`` tensors.

3. ``test_empty_index_knn_returns_empty`` — ``knn`` on an index built from zero points returns
   empty results (matching ``radius``'s contract) instead of crashing in ``topk``.
"""

from __future__ import annotations

import torch

from splatreg.core.types import Gaussians
from splatreg.core.lie import se3_exp
from splatreg.residuals.sdf import SDF
from splatreg.solvers.lm import _assemble
from splatreg.spatial_index import SpatialIndex


def _tiny_splat(seed: int, n: int = 40, log_scales: bool = False) -> Gaussians:
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 3, generator=g)
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    scales = 0.05 + 0.02 * torch.rand(n, 3, generator=g)
    if log_scales:
        scales = torch.log(scales)
    opacities = torch.rand(n, generator=g)
    return Gaussians(means=means, quats=quats, scales=scales, opacities=opacities, log_scales=log_scales)


# --------------------------------------------------------------------------- #1 SDF weighting
def _sdf_gram_contribution(weight: float) -> torch.Tensor:
    """Assemble a one-residual SDF problem at the given weight; return the weighted Gram J^T W J.

    The solver folds each row's ``sqrt(weight)`` into ``problem.weight``; the normal-equation
    contribution the LM step actually uses is ``(sqrt_w * J)^T (sqrt_w * J)``.
    """
    target = _tiny_splat(0)
    source = _tiny_splat(1)
    # Offset the source a touch so the residual is non-zero and the Jacobian non-degenerate.
    T = se3_exp(torch.tensor([0.03, -0.02, 0.01, 0.0, 0.0, 0.0]))
    res = SDF(sigma=0.05, n_points=40, weight=weight)
    prob = _assemble(T, [res], target, source, dof=6, exp_fn=se3_exp)
    assert prob is not None
    Jw = prob.J * prob.weight.reshape(-1, 1)
    return Jw.transpose(0, 1) @ Jw


def test_sdf_weight_applied_once():
    """The SDF normal-equation contribution scales as weight^1, not weight^1.5 / weight^3."""
    g_lo = _sdf_gram_contribution(0.3)
    g_hi = _sdf_gram_contribution(1.0)
    # Linear-in-weight convention: G(1.0) / G(0.3) == 1.0 / 0.3 elementwise (same geometry, same
    # sampled points, only the weight differs). A weight^3 bug would give (1/0.3)^3 ≈ 37x.
    ratio_expected = 1.0 / 0.3
    mask = g_lo.abs() > 1e-9
    ratios = (g_hi[mask] / g_lo[mask])
    assert torch.allclose(ratios, torch.full_like(ratios, ratio_expected), rtol=1e-4), (
        f"SDF weight not applied linearly: got Gram ratio ~{ratios.mean():.3f}, "
        f"expected {ratio_expected:.3f} (a weight^3 bug gives ~{ratio_expected**3:.1f})."
    )


# --------------------------------------------------------------------------- #2 log_scales fusion
def test_merge_preserves_scales_across_log_convention():
    """Fusing opposite log_scales conventions decodes to the inputs' true real-world scales."""
    from splatreg.api import _cat_scales

    lin = _tiny_splat(2, n=10, log_scales=False)
    log = _tiny_splat(3, n=10, log_scales=True)
    pieces = [lin, log]

    # Decode each input piece to LINEAR real-world scale (ground truth).
    lin_true = lin.scales
    log_true = torch.exp(log.scales)
    expected_linear = torch.cat([lin_true, log_true], dim=0)

    # Fuse under the reference's (linear) convention, then decode.
    fused_scales = _cat_scales(pieces, log_scales=False)
    assert torch.allclose(fused_scales, expected_linear, atol=1e-6), (
        "merge fused-scale decode mismatch under mixed log_scales (linear reference)."
    )

    # And under a log reference: decode = exp(fused).
    fused_log = _cat_scales(pieces, log_scales=True)
    assert torch.allclose(torch.exp(fused_log), expected_linear, atol=1e-5), (
        "merge fused-scale decode mismatch under mixed log_scales (log reference)."
    )


# --------------------------------------------------------------------------- #3 empty-index knn
def test_empty_index_knn_returns_empty():
    """knn() on an empty index returns empty (Q, 0) results, matching radius()'s contract."""
    idx = SpatialIndex(torch.zeros(0, 3))
    q = torch.rand(5, 3)
    ii, dd = idx.knn(q, 4)
    assert ii.shape == (5, 0) and dd.shape == (5, 0), f"expected empty (5,0), got {ii.shape}/{dd.shape}"
    assert ii.dtype == torch.int64
    # radius() already returns empty on an empty index — knn() now matches.
    qi, ai = idx.radius(q, 0.1)
    assert qi.numel() == 0 and ai.numel() == 0
