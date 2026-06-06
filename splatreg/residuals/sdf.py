"""Splat-to-splat signed-distance residual — splatreg's flagship, Gaussian-SDF residual.

The optimisation variable ``T`` maps the *source* splat into the *target* splat's frame. This
residual samples points from the source, pushes them through ``T``, and reads the target's
Gaussian-derived signed-distance field (:func:`~splatreg.geometry.gaussian_sdf.gaussian_sdf`)
at the transformed points. The signed distances *are* the residual: they vanish exactly when
the source points land on the target's surface, i.e. when the two splats are aligned.

No competitor packages this — registration here is driven by an implicit field derived straight
from the target Gaussians, with no mesh and no correspondences.

Convention: right-perturbation ``T_new = T @ exp(xi)``, ``xi = [tx, ty, tz, rx, ry, rz]``.
The analytic Jacobian below covers these six SE(3) columns; a Sim(3) (7-DOF) solve, whose 7th
tangent is a log-scale, must autodiff the residual (or extend the chain) for that extra column.

Jacobian
--------
``jacobian`` is **analytic**. The residual for the k-th sampled source point ``s_k`` is::

    p_k = T @ s_k                      (source point in target frame)
    r_k = sdf_target(p_k)              (signed distance, from gaussian_sdf)

Under the right-perturbation ``T @ exp(xi)`` the transformed point moves (to first order) by
``dp_k = R . (v + w x s_k)`` where ``R`` is ``T``'s rotation and ``xi = [v | w]``. The chain rule
then needs the SDF's TRUE spatial gradient ``g_k = ∇_p d(p_k)`` — which is **not** the proxy
surface normal ``n~`` that ``gaussian_sdf`` also returns: ``n~`` drops the first-order
``∂q~/∂p`` term of the kernel-weighted centroid (a numerical audit, ``tests/test_jacobians.py``,
measured ``max|Δ|≈10.8`` from using ``n~``). The exact ``g_k`` is computed in CLOSED FORM by
``gaussian_sdf_grad`` (non-truncated path — one fused pass, no autograd graph; the fast SE(3)
path) or by autodiff (truncated path). Chaining it through the pose Jacobian::

    d r_k / d v = g_k^T R                          (translation block, 1x3)
    d r_k / d w = -g_k^T R [s_k]_x = (R^T g_k) x s_k    (rotation block, 1x3)

stacked as ``J_k = [ g_k^T R | (R^T g_k) x s_k ]`` (shape ``(N, 6)``) — the same gradient-times-
pose-Jacobian chain the ICP residual uses (numerically verified there), now with the EXACT field
gradient rather than the dropped-curvature ``n~`` proxy.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from ..core.types import Gaussians
from ..geometry.gaussian_sdf import gaussian_sdf
from .base import Residual

__all__ = ["SDF"]


class SDF(Residual):
    """Align ``source`` Gaussians onto ``target``'s Gaussian-derived SDF surface.

    ``residual`` returns the signed distance at ``n_points`` sampled, ``T``-transformed source
    points (``-> 0`` at alignment); ``dim`` is the number of sampled points. ``requires`` is
    ``{"source_gaussians"}`` because ``source`` must be a :class:`~splatreg.core.types.Gaussians`.

    Args:
        sigma: Gaussian-SDF kernel bandwidth for the *target* field. **Required** (the field's
            single most important knob; no universal default). Same units as the splat means.
        n_points: number of source anchors sampled per evaluation (the residual length). If the
            source has fewer Gaussians, all of them are used.
        weight: scalar multiplier on the residual (sqrt-weight is applied by the solver).
        robust: optional robust kernel forwarded to the solver (unchanged here).
        target_normals: optional ``(M_target, 3)`` precomputed normals for the target anchors
            (e.g. from a mesh); otherwise estimated once per call by ``gaussian_sdf``.
        trunc_sigmas / use_opacity / knn / chunk_size: forwarded to ``gaussian_sdf`` (see there).
        seed: RNG seed for the deterministic source subsample (stable across iterations).
    """

    def __init__(
        self,
        sigma: float,
        n_points: int = 2048,
        weight: float = 1.0,
        robust: Optional[Any] = None,
        *,
        target_normals: Optional[torch.Tensor] = None,
        trunc_sigmas: Optional[float] = None,
        use_opacity: bool = False,
        knn: int = 50,
        chunk_size: int = 2048,
        seed: int = 0,
    ):
        super().__init__(weight=weight, robust=robust)
        if not (sigma > 0.0):
            raise ValueError(f"SDF residual: sigma must be > 0, got {sigma}.")
        self.sigma = float(sigma)
        self.n_points = int(n_points)
        self.target_normals = target_normals
        self.trunc_sigmas = trunc_sigmas
        self.use_opacity = bool(use_opacity)
        self.knn = int(knn)
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        # Per-source-identity cache of the sampled-index set, so the residual is evaluated on
        # the same points across every LM iteration of a registration (variable = T only).
        self._sample_cache_key: Optional[tuple] = None
        self._sample_idx: Optional[torch.Tensor] = None
        # Cache the TARGET anchor normals. They are constant across a registration (the target
        # never moves), but estimating them (k-NN PCA + a per-anchor SVD) every residual/Jacobian
        # call dominated runtime — and truncation does NOT touch it, which is why trunc gave no
        # speed-up. Compute once per target identity, reuse on every LM iteration.
        self._norm_cache_key: Optional[tuple] = None
        self._norm_cache: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ helpers

    def _source_points(self, source: Any) -> torch.Tensor:
        """Deterministically subsample ``n_points`` anchors (means) from the source splat."""
        if not isinstance(source, Gaussians):
            raise TypeError(
                "SDF residual requires `source` to be a Gaussians (splat-to-splat "
                f"registration); got {type(source).__name__}. See self.requires()."
            )
        means = source.means
        m = int(means.shape[0])
        if m == 0:
            raise ValueError("SDF residual: source splat has no Gaussians.")
        if self.n_points <= 0 or m <= self.n_points:
            return means

        key = (id(source), m, means.data_ptr(), self.n_points, self.seed)
        if self._sample_cache_key != key or self._sample_idx is None:
            gen = np.random.default_rng(self.seed)
            idx_np = gen.choice(m, size=self.n_points, replace=False)
            self._sample_idx = torch.from_numpy(np.sort(idx_np)).to(device=means.device, dtype=torch.long)
            self._sample_cache_key = key
        return means[self._sample_idx]

    def _transformed(self, T: torch.Tensor, source: Any):
        """Return ``(src_pts, p, R)``: sampled source points, ``T``-transformed points, ``T``'s R."""
        src_pts = self._source_points(source)  # (N, 3)
        R = T[:3, :3].to(dtype=src_pts.dtype)
        t = T[:3, 3].to(dtype=src_pts.dtype)
        p = src_pts @ R.T + t  # (N, 3)
        return src_pts, p, R

    def _resolve_normals(self, target: Gaussians):
        """Target anchor normals — explicit if given, else estimated ONCE and cached.

        The target is fixed across a registration, so its k-NN-PCA normals are constant; caching
        avoids re-running the per-anchor SVD on every residual/Jacobian call (the dominant SDF cost;
        truncation does not touch it). Keyed by target identity + storage pointer + count."""
        if self.target_normals is not None:
            return self.target_normals
        key = (id(target), target.means.data_ptr(), int(target.means.shape[0]))
        if self._norm_cache_key != key or self._norm_cache is None:
            from ..geometry.gaussian_sdf import estimate_anchor_normals

            self._norm_cache = estimate_anchor_normals(target.means, k=self.knn)
            self._norm_cache_key = key
        return self._norm_cache

    def _sdf(self, target: Gaussians, p: torch.Tensor):
        return gaussian_sdf(
            target,
            p,
            sigma=self.sigma,
            normals=self._resolve_normals(target),
            trunc_sigmas=self.trunc_sigmas,
            use_opacity=self.use_opacity,
            knn=self.knn,
            chunk_size=self.chunk_size,
        )

    # ------------------------------------------------------------------ contract

    def residual(self, T: torch.Tensor, target: Gaussians, source: Any) -> torch.Tensor:
        _, p, _ = self._transformed(T, source)
        sd, _ = self._sdf(target, p)
        return sd * self.weight

    def jacobian(self, T: torch.Tensor, target: Gaussians, source: Any) -> Optional[torch.Tensor]:
        src_pts, p, R = self._transformed(T, source)
        # EXACT field gradient g_k = ∇_p sdf(p_k), via autodiff through the field. The surface
        # normal n~ that gaussian_sdf also returns is only a first-order *proxy* for this
        # gradient — it drops the ∂q~/∂p term of the kernel-weighted centroid, which is itself
        # first-order (NOT curvature). Using n~ gave a materially wrong Jacobian (numerically
        # audited, tests/test_jacobians.py: max|Δ|≈10 vs numerical); autodiff is exact.
        # Non-truncated path: CLOSED-FORM gradient (one fused pass, no autograd graph + no second
        # forward — the fast SE(3) path). Truncated path: autodiff. Both are the EXACT ∇_p sdf.
        if self.trunc_sigmas is None:
            from ..geometry.gaussian_sdf import gaussian_sdf_grad

            _, grad = gaussian_sdf_grad(
                target,
                p,
                sigma=self.sigma,
                normals=self._resolve_normals(target),
                use_opacity=self.use_opacity,
                knn=self.knn,
                chunk_size=self.chunk_size,
            )
        else:
            pg = p.detach().requires_grad_(True)
            with torch.enable_grad():
                sd, _ = self._sdf(target, pg)
                grad = torch.autograd.grad(sd.sum(), pg, create_graph=False)[0]  # (N, 3) ∇_p sdf
            grad = grad.detach()
        # Chain the field gradient through the pose-Jacobian of p = T·s_k — the SAME chain the
        # ICP residual uses (numerically verified there). rt_n[k] = R^T g_k.
        rt_n = grad @ R  # (N, 3)  d r/dv = g_k^T R
        # d r/dw = -(R^T g_k) x s_k = cross(s_k, R^T g_k)   (right-perturbation rotation block)
        j_rot = torch.cross(src_pts, rt_n, dim=-1)  # (N, 3)
        jac = torch.cat([rt_n, j_rot], dim=-1)  # (N, 6)
        return jac * self.weight

    def dim(self) -> int:
        return self.n_points

    def requires(self) -> set:
        return {"source_gaussians"}
