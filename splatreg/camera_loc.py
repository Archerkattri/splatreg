"""Camera localization in a splat (v0.2): estimate a camera pose from a query image + a splat.

The dual of object-pose estimation. There the scene moves and the camera is fixed; **here the splat
is fixed in world and the camera pose ``T_WC`` is the unknown** — "where was this photo taken, given
this 3DGS scene?". This is the relocalisation / pose-refinement task iNeRF, NeRF-based localisers,
and photometric visual-SLAM front-ends solve, ported onto a Gaussian splat rendered by **gsplat**.

Implementation — differentiable rendering (the robust path)
-----------------------------------------------------------
:func:`localize_camera` optimizes the camera pose directly through **gsplat's own differentiable
rasteriser**. The pose lives as a right-perturbation tangent ``δ`` of the camera→world extrinsic
(``T_WC ← T_WC · exp(δ)``, splatreg's standard convention); each step renders the splat from the
current pose, takes the photometric loss against the query image, and back-props *through the render*
to ``δ``. The Jacobian is therefore exact-by-construction (gsplat's analytic render gradient) — no
hand-derived image Jacobian to get the inverse-compositional sign wrong. Verified to converge on a
synthetic textured scene (rotation 7°→1.7°, translation 132→37 mm from a cold-ish init; see
``tests/test_camera_loc.py``).

This reuses splatreg's :class:`~splatreg.core.types.Gaussians` / :class:`~splatreg.core.types.Frame`
contracts and Lie ``exp`` (``splatreg.core.lie.se3_exp``); the optimizer is a small Adam loop on the
6-vector tangent with periodic pose-folding (re-anchor ``T_WC`` and zero ``δ``) for numerical health.

Basin: like all direct image alignment, the convergence basin is a few degrees / a few percent of
depth, so ``init_T_WC`` should be a reasonable prior (a nearby keyframe pose or a coarse geometric
seed). It refines a prior; it is not a global relocaliser.

An experimental **analytic** residual (:class:`CameraPhotometric`) is also provided for users who want
the inverse-compositional residual in the LM stack, but the differentiable-render path above is the
recommended, validated one. gsplat is OPTIONAL — guarded with a clear install hint.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .core.lie import se3_exp
from .core.types import Frame, Gaussians, RegisterResult
from .residuals.base import Residual
from .residuals.photometric import (
    _GSPLAT_AVAILABLE,
    _GSPLAT_INSTALL_HINT,
    _image_gradients,
    _rgb_to_hwc,
)

__all__ = ["localize_camera", "CameraPhotometric"]


def _render_rgb_depth(
    splat: Gaussians, T_WC: torch.Tensor, K: torch.Tensor, width: int, height: int, sh_degree
):
    """Render ``splat`` from camera pose ``T_WC`` via gsplat → (rgb HW3, depth HW). Differentiable."""
    from gsplat import rasterization as _rast

    if splat.colors is None:
        raise ValueError("camera localization needs splat.colors (RGB or SH) for rendering")
    scales = splat.scales.exp() if splat.log_scales else splat.scales
    opac = splat.opacities
    opac = opac.squeeze(-1) if opac.dim() == 2 else opac
    T_CW = torch.linalg.inv(T_WC)  # world→camera viewmat
    render, _alpha, _meta = _rast(
        means=splat.means.to(torch.float32),
        quats=splat.quats.to(torch.float32),
        scales=scales.to(torch.float32),
        opacities=opac.to(torch.float32),
        colors=splat.colors.to(torch.float32),
        viewmats=T_CW.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode="RGB+ED",
    )
    out = render[0]  # (H, W, 4)
    return out[..., :3], out[..., 3]


def localize_camera(
    splat: Gaussians,
    frame: Frame,
    init_T_WC: torch.Tensor,
    *,
    iters: int = 150,
    lr: float = 1e-2,
    refold_every: int = 40,
    mask_to_rendered: bool = True,
    huber_k: Optional[float] = None,
    sh_degree: Optional[int] = None,
) -> RegisterResult:
    """Localize a query camera in a ``splat`` by differentiable-render pose optimization.

    Refines ``init_T_WC`` so the gsplat render of ``splat`` from that camera matches ``frame.rgb``,
    optimizing the camera→world pose through gsplat's differentiable rasteriser (exact render
    gradient). Returns a :class:`~splatreg.core.types.RegisterResult` whose ``T`` is the refined
    ``T_WC`` and whose ``info`` carries the loss history.

    Parameters
    ----------
    splat : the world-fixed Gaussian splat (must carry ``colors``).
    frame : the query observation — needs ``rgb`` and ``K`` (optional ``mask``).
    init_T_WC : ``(4, 4)`` initial camera→world prior. Direct image alignment has a limited basin
        (a few degrees / a few percent of depth); this refines a prior, it does not relocalise blind.
    iters : Adam steps on the pose tangent.
    lr : Adam learning rate on the 6-vector right-perturbation tangent.
    refold_every : every this many steps, fold the accumulated tangent into ``T_WC`` and reset it to
        zero (keeps ``exp`` near the origin where it is best-conditioned).
    mask_to_rendered : when ``True``, the photometric loss is restricted to pixels the splat actually
        renders (rendered depth > 0), so empty background does not dominate the loss.
    huber_k : optional Huber threshold on the per-pixel RGB residual (robust to occluders / outliers);
        ``None`` uses a plain L2 loss.
    sh_degree : SH degree if ``splat.colors`` are SH coefficients; ``None`` treats them as RGB.

    Returns
    -------
    :class:`~splatreg.core.types.RegisterResult` with the refined ``T_WC``; ``info['mode'] ==
    'camera_loc'``, ``info['loss']`` (final), ``info['loss_history']``.
    """
    if not _GSPLAT_AVAILABLE:
        raise ImportError(_GSPLAT_INSTALL_HINT)
    if not isinstance(init_T_WC, torch.Tensor) or init_T_WC.shape[-2:] != (4, 4):
        raise ValueError("localize_camera(): init_T_WC must be a (4, 4) tensor.")
    if not isinstance(splat, Gaussians) or splat.colors is None:
        raise ValueError("localize_camera(): splat must be a Gaussians with .colors for rendering.")
    if frame.rgb is None or frame.K is None:
        raise ValueError("localize_camera(): frame needs .rgb and .K.")

    device = init_T_WC.device
    T_anchor = init_T_WC.to(device=device, dtype=torch.float32).detach()
    K = frame.K.to(device=device, dtype=torch.float32)
    rgb_gt = _rgb_to_hwc(frame.rgb.to(device=device, dtype=torch.float32))
    H, W = rgb_gt.shape[0], rgb_gt.shape[1]
    mask = frame.mask.to(device) if frame.mask is not None else None
    if mask is not None and mask.dtype != torch.bool:
        mask = mask > 0.5

    delta = torch.zeros(6, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=float(lr))
    loss_history: list = []

    for it in range(int(iters)):
        opt.zero_grad(set_to_none=True)
        T_WC = T_anchor @ se3_exp(delta)
        rgb_pred, depth_pred = _render_rgb_depth(splat, T_WC, K, W, H, sh_degree)

        diff = rgb_pred - rgb_gt  # (H, W, 3)
        pix_w = torch.ones(H, W, device=device, dtype=torch.float32)
        if mask_to_rendered:
            pix_w = pix_w * (depth_pred.detach().abs() > 1e-6).to(torch.float32)
        if mask is not None:
            pix_w = pix_w * mask.to(torch.float32)
        per_pix = (diff * diff).sum(dim=-1)  # (H, W) squared RGB error
        if huber_k is not None:
            # Huber on the per-pixel L2 norm: quadratic within huber_k, linear beyond.
            n = per_pix.clamp_min(1e-12).sqrt()
            k = float(huber_k)
            per_pix = torch.where(n <= k, per_pix, 2 * k * n - k * k)
        denom = pix_w.sum().clamp_min(1.0)
        loss = (per_pix * pix_w).sum() / denom
        loss.backward()
        opt.step()
        loss_history.append(float(loss.detach()))

        if refold_every > 0 and (it + 1) % refold_every == 0:
            with torch.no_grad():
                T_anchor = (T_anchor @ se3_exp(delta)).detach()
                delta.data.zero_()

    with torch.no_grad():
        T_final = (T_anchor @ se3_exp(delta)).detach()

    return RegisterResult(
        T=T_final,
        scale=1.0,
        converged=len(loss_history) > 1 and loss_history[-1] <= loss_history[0],
        info={"mode": "camera_loc", "loss": loss_history[-1] if loss_history else float("nan"),
              "loss_history": loss_history, "n_iters": int(iters)},
    )


class CameraPhotometric(Residual):
    """EXPERIMENTAL analytic camera-pose photometric residual (inverse-compositional, gsplat render).

    The optimization variable ``T`` is the camera→world extrinsic ``T_WC``; the splat (``target``) is
    fixed in world. Provided for users who want a camera-pose term inside the LM residual stack
    (:func:`splatreg.solvers.lm.run_lm`). The geometry block ``∂X_c/∂δ = [−I | [X_c]_×]`` is
    verified against numerical differentiation, but the full inverse-compositional Jacobian (image
    gradients sampled on the *rendered* image) shares the narrow, sign-sensitive basin of direct
    image alignment and is **not** the validated path — prefer :func:`localize_camera` (differentiable
    render), which is exact-by-construction.

    Args mirror the kept knobs of the photometric residual: ``rgb_gt`` / ``K`` / ``mask`` /
    ``width`` / ``height`` / ``max_pixels`` / ``grad_threshold`` / ``huber_k`` / ``depth_min`` /
    ``depth_max`` / ``sh_degree`` / ``weight`` / ``robust``.
    """

    def __init__(
        self,
        rgb_gt: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        *,
        device: Optional[torch.device] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_pixels: int = 800,
        grad_threshold: float = 0.05,
        huber_k: float = 0.05,
        depth_min: float = 0.05,
        depth_max: float = 10.0,
        sh_degree: Optional[int] = None,
        weight: float = 1.0,
        robust: Optional[Any] = None,
    ):
        super().__init__(weight=weight, robust=robust)
        if not _GSPLAT_AVAILABLE:
            raise ImportError(_GSPLAT_INSTALL_HINT)
        self._device = device or (rgb_gt.device if rgb_gt is not None else torch.device("cpu"))
        self.rgb_gt = (
            _rgb_to_hwc(rgb_gt.to(self._device, dtype=torch.float32)) if rgb_gt is not None else None
        )
        self.K = K.to(self._device, dtype=torch.float32) if K is not None else None
        self.mask = mask
        self.width = int(width) if width is not None else None
        self.height = int(height) if height is not None else None
        self.max_pixels = int(max_pixels)
        self.grad_threshold = float(grad_threshold)
        self.huber_k = float(huber_k)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.sh_degree = sh_degree
        self._dim = 0

    def requires(self) -> set:
        return {"rgb", "K"}

    def dim(self) -> int:
        return self._dim

    def _resolve_obs(self, source: Any):
        rgb, K, mask = self.rgb_gt, self.K, self.mask
        if isinstance(source, Frame):
            if rgb is None and source.rgb is not None:
                rgb = _rgb_to_hwc(source.rgb.to(self._device, dtype=torch.float32))
            if K is None and source.K is not None:
                K = source.K.to(self._device, dtype=torch.float32)
            if mask is None:
                mask = source.mask
        if rgb is None:
            raise ValueError("CameraPhotometric needs rgb (pass rgb_gt or a Frame with .rgb)")
        if K is None:
            raise ValueError("CameraPhotometric needs K (pass K or a Frame with .K)")
        return rgb, K, mask

    def _select(self, target: Gaussians, T_WC: torch.Tensor, source: Any):
        rgb_gt, K, mask = self._resolve_obs(source)
        H, W = rgb_gt.shape[0], rgb_gt.shape[1]
        width = self.width or W
        height = self.height or H
        rgb_pred, depth_pred = _render_rgb_depth(target, T_WC, K, width, height, self.sh_degree)

        grad = _image_gradients(rgb_pred)
        grad_mag = grad.norm(dim=(-2, -1))
        depth_ok = (depth_pred.abs() > self.depth_min) & (depth_pred.abs() < self.depth_max)
        cand = grad_mag > self.grad_threshold
        if mask is not None:
            m = mask.to(self._device)
            m = m > 0.5 if m.dtype != torch.bool else m
            cand = cand & m
        cand = cand & depth_ok
        if not bool(cand.any()):
            return None
        flat_mag = torch.where(cand, grad_mag, torch.zeros_like(grad_mag)).reshape(-1)
        n_sel = min(self.max_pixels, int(cand.sum().item()))
        if n_sel <= 0:
            return None
        _, top_flat = flat_mag.topk(n_sel)
        v = (top_flat // width).long()
        u = (top_flat % width).long()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        return rgb_pred, depth_pred, grad, v, u, (fx, fy, cx, cy), rgb_gt

    def _huber(self, r: torch.Tensor) -> torch.Tensor:
        abs_r = r.abs()
        return torch.where(abs_r <= self.huber_k, torch.ones_like(r), self.huber_k / abs_r.clamp_min(1e-6))

    def residual(self, T: torch.Tensor, target: Gaussians, source: Any) -> torch.Tensor:
        T_WC = T.to(device=self._device, dtype=torch.float32)
        sel = self._select(target, T_WC, source)
        if sel is None:
            self._dim = 0
            return target.means.new_zeros(0)
        rgb_pred, _d, _g, v, u, _intr, rgb_gt = sel
        r = (rgb_pred[v, u] - rgb_gt[v, u]).reshape(-1)
        r = r * self._huber(r)
        self._dim = int(r.shape[0])
        return r

    def jacobian(self, T: torch.Tensor, target: Gaussians, source: Any) -> Optional[torch.Tensor]:
        T_WC = T.to(device=self._device, dtype=torch.float32)
        sel = self._select(target, T_WC, source)
        if sel is None:
            self._dim = 0
            return target.means.new_zeros(0, 6)
        rgb_pred, depth_pred, grad, v, u, (fx, fy, cx, cy), rgb_gt = sel
        r = (rgb_pred[v, u] - rgb_gt[v, u]).reshape(-1)
        huber_w = self._huber(r)

        z = depth_pred[v, u].abs().clamp_min(1e-6)
        x_cam = (u.float() - cx) * z / fx
        y_cam = (v.float() - cy) * z / fy
        inv_z2 = 1.0 / (z * z)
        zeros = torch.zeros_like(z)
        # ∂(u,v)/∂X_c (OpenCV pinhole)
        J_proj = torch.stack(
            [
                torch.stack([fx * z * inv_z2, zeros, -fx * x_cam * inv_z2], dim=1),
                torch.stack([zeros, fy * z * inv_z2, -fy * y_cam * inv_z2], dim=1),
            ],
            dim=1,
        )  # (N, 2, 3)
        # ∂X_c/∂δ = [ −I | [X_c]_× ] for the camera-pose right-perturbation (verified vs numerical).
        N = z.shape[0]
        X = torch.stack([x_cam, y_cam, z], dim=1)
        Xx, Xy, Xz = X[:, 0], X[:, 1], X[:, 2]
        zr = torch.zeros_like(Xx)
        X_skew = torch.stack(
            [
                torch.stack([zr, -Xz, Xy], dim=1),
                torch.stack([Xz, zr, -Xx], dim=1),
                torch.stack([-Xy, Xx, zr], dim=1),
            ],
            dim=1,
        )
        neg_I = -torch.eye(3, device=self._device, dtype=torch.float32).unsqueeze(0).expand(N, 3, 3)
        J_cam = torch.cat([neg_I, X_skew], dim=2)  # (N, 3, 6)

        grad_uv = grad[v, u]
        J_pix = J_proj @ J_cam
        J = (grad_uv.unsqueeze(-1) * J_pix.unsqueeze(1)).sum(dim=2).reshape(-1, 6)
        J = J * huber_w.unsqueeze(1)
        self._dim = int(J.shape[0])
        return J
