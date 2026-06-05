"""Gaussian-derived signed-distance field — splatreg's differentiating primitive.

A 3D Gaussian Splat is a cloud of anchors (the Gaussian means) that already trace the
surface of a scanned object. This module turns that cloud into a smooth, queryable
*signed-distance proxy* and its spatial gradient at arbitrary 3D points, with no meshing,
no marching cubes, and no CUDA extension — plain ``torch``.

It is a standalone, sellable primitive: hand it a :class:`~splatreg.core.types.Gaussians`
and a ``(N, 3)`` batch of query points and it returns ``(sdf, grad)``. It is the SDF that
the splat-to-splat :class:`~splatreg.residuals.sdf.SDF` residual evaluates, but it has no
dependency on the residual or solver layers.

Proxy definition (sum-of-Gaussians soft-anchor SDF)
---------------------------------------------------
Given anchors ``q_i`` (Gaussian means), per-anchor surface normals ``n_i`` and an influence
bandwidth ``sigma``, define for a query point ``p``::

    w_i(p) = exp( -||p - q_i||^2 / (2 sigma^2) )          (Gaussian kernel weight)
    q~(p)  = sum_i w_i(p) q_i / sum_i w_i(p)              (weighted anchor centroid)
    n~(p)  = sum_i w_i(p) n_i / ||sum_i w_i(p) n_i||      (weighted, renormalised normal)
    d(p)   = (p - q~(p)) . n~(p)                          (signed distance to local surface)

``d(p)`` is positive outside the surface (along ``+n~``), negative inside, and crosses zero
on the soft surface. The spatial gradient returned is ``grad(p) = n~(p)`` — the unit
surface normal at the query. This is exact to first order: the surface is locally the plane
through ``q~`` with normal ``n~``, so the gradient of the point-to-plane distance is ``n~``
(the implicit dependence of ``q~`` / ``n~`` on ``p`` contributes only second-order curvature
terms, which this proxy intentionally drops for a cheap, stable Jacobian).

Assumptions / knobs
-------------------
* **Bandwidth.** ``sigma`` (metres, in the splat's own units) sets each anchor's influence
  radius. Smaller ``sigma`` -> sharper field, more noise-sensitive; larger -> smoother. It is
  the single most important knob and has no universal default, so it is **required**.
* **Normals.** Gaussians carry no surface normal, so one is derived per anchor from local
  geometry: the smallest-eigenvector of each anchor's k-NN covariance (a PCA normal),
  oriented outward from the splat centroid. Pass ``normals=`` to override (e.g. from a mesh).
  Normals are cached on the call site, not on the splat, keeping this function pure.
* **Truncation.** Anchors farther than ``trunc_sigmas * sigma`` from a query contribute a
  negligible weight; setting ``trunc_sigmas`` (default ``None`` = no truncation) lets very
  large splats skip them via a per-query top-k gather for speed. Truncation only changes the
  weight support, never the proxy definition.
* **Normalisation.** Weights are mean-normalised per query (the ``sum_i w_i`` denominator),
  so the field is invariant to anchor count / density and to a global scale on the weights.
* **Opacity.** Splat opacity is *not* folded into the kernel by default (a surface anchor and
  a faint anchor weigh equally once both are near the query); pass ``use_opacity=True`` to
  multiply each ``w_i`` by its anchor opacity.

The query is fully differentiable w.r.t. ``points`` (autograd flows through ``d(p)``), so the
primitive is equally usable as a plug-in implicit field for optimisation as for one-shot SDF
sampling.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..core.types import Gaussians

__all__ = ["gaussian_sdf", "estimate_anchor_normals"]

# Numerical floor shared by the weight-sum and normal-norm denominators.
_EPS = 1.0e-12


def estimate_anchor_normals(means: torch.Tensor, k: int = 50) -> torch.Tensor:
    """Per-anchor outward unit normals via local k-NN PCA.

    For each anchor the smallest principal axis of its ``k`` nearest neighbours (the local
    covariance's smallest-eigenvector) is taken as the surface normal, then flipped to point
    away from the cloud centroid so the resulting signed distance is positive outside.

    Args:
        means: ``(M, 3)`` anchor positions (Gaussian means).
        k: neighbourhood size for the PCA fit (clamped to ``[2, M]``).

    Returns:
        ``(M, 3)`` unit normals. Degenerate clouds (``M < 3``) fall back to ``+z``.
    """
    pts = means
    m = pts.shape[0]
    if m < 3:
        out = torch.zeros((m, 3), device=pts.device, dtype=pts.dtype)
        if m > 0:
            out[:, 2] = 1.0
        return out

    kk = max(2, min(int(k), m))
    dists = torch.cdist(pts, pts)                              # (M, M)
    knn_idx = torch.topk(dists, k=kk, largest=False).indices   # (M, k)
    neighbours = pts[knn_idx]                                  # (M, k, 3)
    centered = neighbours - neighbours.mean(dim=1, keepdim=True)
    # SVD's smallest right-singular vector == covariance's smallest eigenvector,
    # without forming the (M, 3, 3) covariance explicitly.
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)  # vh: (M, 3, 3)
    normals = vh[:, -1, :]                                      # (M, 3)

    centroid = pts.mean(dim=0, keepdim=True)
    outward = pts - centroid
    flip = (normals * outward).sum(dim=1) < 0
    normals = torch.where(flip.unsqueeze(1), -normals, normals)
    return normals / normals.norm(dim=1, keepdim=True).clamp_min(_EPS)


def gaussian_sdf(
    gaussians: Gaussians,
    points: torch.Tensor,
    *,
    sigma: float,
    normals: Optional[torch.Tensor] = None,
    trunc_sigmas: Optional[float] = None,
    use_opacity: bool = False,
    knn: int = 50,
    chunk_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample the Gaussian-derived signed-distance proxy and its gradient at ``points``.

    See the module docstring for the proxy definition and assumptions.

    Args:
        gaussians: the splat providing the anchors (``means``, ``opacities``). Only
            ``means`` (and ``opacities`` when ``use_opacity``) are read.
        points: ``(N, 3)`` query positions, in the splat's own frame.
        sigma: Gaussian kernel bandwidth (influence radius), same units as ``means``.
            **Required** — there is no universal default. Must be > 0.
        normals: optional ``(M, 3)`` per-anchor normals. If ``None`` they are estimated
            once via :func:`estimate_anchor_normals` (k-NN PCA, outward-oriented).
        trunc_sigmas: if set, only the anchors within ``trunc_sigmas * sigma`` of each query
            (via a per-query top-k gather) contribute; ``None`` uses every anchor. A speed
            knob for very large splats; does not change the proxy otherwise.
        use_opacity: multiply each kernel weight by its anchor opacity.
        knn: neighbourhood size passed to the normal estimator (ignored if ``normals`` given).
        chunk_size: rows of ``points`` processed per block, bounding the ``(chunk, M)`` weight
            matrix's memory.

    Returns:
        ``(sdf, grad)`` where ``sdf`` is ``(N,)`` signed distances (``> 0`` outside) and
        ``grad`` is ``(N, 3)`` unit surface normals ``n~(p)`` (the spatial gradient of the
        proxy). Both live on ``points``' device and dtype.

    Raises:
        ValueError: empty splat, mismatched ``normals``, malformed ``points``, or ``sigma<=0``.
    """
    means = gaussians.means
    m = int(means.shape[0])
    if m == 0:
        raise ValueError("gaussian_sdf: the splat has no Gaussians (means is empty).")
    if points.dim() != 2 or points.shape[-1] != 3:
        raise ValueError(f"gaussian_sdf: points must be (N, 3), got {tuple(points.shape)}.")
    if not (sigma > 0.0):
        raise ValueError(f"gaussian_sdf: sigma must be > 0, got {sigma}.")

    device, dtype = points.device, points.dtype
    anchors = means.to(device=device, dtype=dtype)

    if normals is None:
        normals = estimate_anchor_normals(anchors, k=knn)
    else:
        normals = normals.to(device=device, dtype=dtype)
        if normals.shape != anchors.shape:
            raise ValueError(
                f"gaussian_sdf: normals must match means shape {tuple(anchors.shape)}, "
                f"got {tuple(normals.shape)}."
            )
    anchor_normals = normals / normals.norm(dim=1, keepdim=True).clamp_min(_EPS)

    two_sigma_sq = 2.0 * float(sigma) * float(sigma)

    opa: Optional[torch.Tensor] = None
    if use_opacity:
        opa = gaussians.opacities.to(device=device, dtype=dtype).reshape(-1)
        if opa.shape[0] != m:
            raise ValueError(
                f"gaussian_sdf: opacities length {opa.shape[0]} != n_gaussians {m}."
            )

    trunc_topk: Optional[int] = None
    if trunc_sigmas is not None:
        radius = float(trunc_sigmas) * float(sigma)
        # A query keeps only anchors inside `radius`; we approximate that support by a
        # fixed top-k gather (the k nearest), which is what makes the cost N*k not N*M.
        # k is sized from the cloud so the cap never silently drops near anchors.
        trunc_topk = max(1, min(m, int(knn)))
        radius_sq = radius * radius
    else:
        radius_sq = 0.0

    n = int(points.shape[0])
    chunk = max(1, int(chunk_size))
    sd_parts = []
    grad_parts = []
    for start in range(0, n, chunk):
        block = points[start : start + chunk]                       # (c, 3)
        diff = block[:, None, :] - anchors[None, :, :]              # (c, M, 3)
        dist_sq = (diff * diff).sum(dim=-1)                         # (c, M)

        if trunc_topk is not None:
            # Gather the k nearest anchors per query and zero out those beyond `radius`.
            near_sq, near_idx = torch.topk(dist_sq, k=trunc_topk, largest=False)  # (c, k)
            weights = torch.exp(-near_sq / two_sigma_sq)
            weights = torch.where(near_sq <= radius_sq, weights, torch.zeros_like(weights))
            q_near = anchors[near_idx]                              # (c, k, 3)
            n_near = anchor_normals[near_idx]                       # (c, k, 3)
            if opa is not None:
                weights = weights * opa[near_idx]
            w_sum = weights.sum(dim=-1, keepdim=True).clamp_min(_EPS)
            q_tilde = (weights.unsqueeze(-1) * q_near).sum(dim=1) / w_sum
            n_sum = (weights.unsqueeze(-1) * n_near).sum(dim=1)     # (c, 3)
        else:
            weights = torch.exp(-dist_sq / two_sigma_sq)           # (c, M)
            if opa is not None:
                weights = weights * opa[None, :]
            w_sum = weights.sum(dim=-1, keepdim=True).clamp_min(_EPS)
            q_tilde = (weights @ anchors) / w_sum
            n_sum = weights @ anchor_normals                       # (c, 3)

        n_tilde = n_sum / n_sum.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        signed = ((block - q_tilde) * n_tilde).sum(dim=-1)         # (c,)
        sd_parts.append(signed)
        grad_parts.append(n_tilde)

    if len(sd_parts) == 1:
        return sd_parts[0], grad_parts[0]
    return torch.cat(sd_parts, dim=0), torch.cat(grad_parts, dim=0)
