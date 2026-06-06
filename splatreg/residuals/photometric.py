"""Photometric residual — render ``target`` via gsplat and compare to an observed RGB frame.

This is the term that breaks rotational symmetry that depth/ICP cannot observe (rotation about
a symmetry axis is invisible to geometry but visible to texture). It ports the inverse-
compositional, analytically-sampled Jacobian of GaussianFeels' ``PhotometricResidual`` and
retargets it onto **gsplat** as the renderer (standard OpenCV pinhole convention), so splatreg
has no dependency on GaussianFeels' custom rasterizer.

Formulation
-----------
The ``target`` Gaussians are fixed in world; the camera extrinsic ``T_WC`` is fixed per frame
(given at construction); the optimization variable ``T`` is the object pose ``T_WO`` moved by a
right-perturbation ``T_WO ← T_WO · exp(δ)``, ``δ = [tx,ty,tz, rx,ry,rz]``. Render once per LM
iteration, sample Sobel image gradients at the *predicted* pixel grid (inverse-compositional),
and form the analytic per-pixel Jacobian:

    r(u)     = I_render(u) − I_gt(u)                       (per RGB channel)
    J_e(u)   = ∇I(u) · J_proj(u) · J_pose(u)
    J_proj   = (1/z²) [[fx·z, 0, −fx·x], [0, fy·z, −fy·y]]   (OpenCV pinhole)
    J_pose   = [ R_CW·R_WO | −R_CW·R_WO·[X_obj]× ]          (right-perturbation on T_WO)

Only high-gradient pixels inside the (optional) mask with valid rendered depth contribute, so
``dim()`` (= 3 × #selected pixels) is comparable to the geometric residuals.

gsplat is OPTIONAL: the import is guarded and a clear install hint is raised at construction.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from ..core.types import Frame, Gaussians
from .base import Residual

try:  # gsplat is an optional dependency (the [render] extra).
    from gsplat import rasterization as _gsplat_rasterization

    _GSPLAT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when gsplat is absent
    _gsplat_rasterization = None
    _GSPLAT_AVAILABLE = False

_GSPLAT_INSTALL_HINT = (
    "Photometric residual requires gsplat, which is not installed. "
    'Install the render extra:  pip install "splatreg[render]"'
)


# Sobel filters (kept on CPU; per-device/dtype copies cached so the hot loop never re-allocates).
_SOBEL_X_CPU = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
_SOBEL_Y_CPU = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
_SOBEL_CACHE: dict = {}


def _get_sobel_filters(device: torch.device, dtype: torch.dtype):
    key = (str(device), dtype)
    if key not in _SOBEL_CACHE:
        sx = _SOBEL_X_CPU.to(device, dtype=dtype).repeat(3, 1, 1, 1)
        sy = _SOBEL_Y_CPU.to(device, dtype=dtype).repeat(3, 1, 1, 1)
        _SOBEL_CACHE[key] = (sx, sy)
    return _SOBEL_CACHE[key]


def _image_gradients(image_hw3: torch.Tensor) -> torch.Tensor:
    """Sobel x/y gradient per channel of an (H, W, 3) image → (H, W, 3, 2)."""
    sx, sy = _get_sobel_filters(image_hw3.device, image_hw3.dtype)
    chw = image_hw3.permute(2, 0, 1).unsqueeze(0)
    grad_x = F.conv2d(F.pad(chw, (1, 1, 1, 1), mode="replicate"), sx, groups=3)
    grad_y = F.conv2d(F.pad(chw, (1, 1, 1, 1), mode="replicate"), sy, groups=3)
    grad = torch.stack([grad_x.squeeze(0), grad_y.squeeze(0)], dim=-1)  # (3, H, W, 2)
    return grad.permute(1, 2, 0, 3)  # (H, W, 3, 2)


def _rgb_to_hwc(rgb: torch.Tensor) -> torch.Tensor:
    """Normalize an RGB tensor from CHW / HWC / BCHW / BHWC to HWC."""
    if rgb.dim() == 4:
        if rgb.shape[0] != 1:
            raise ValueError(f"Photometric residual expects batch size 1, got {tuple(rgb.shape)}")
        rgb = rgb[0]
    if rgb.dim() != 3:
        raise ValueError(f"Photometric residual expects 3D RGB, got {tuple(rgb.shape)}")
    if rgb.shape[-1] == 3:
        return rgb
    if rgb.shape[0] == 3:
        return rgb.permute(1, 2, 0)
    raise ValueError(f"Photometric residual cannot infer RGB layout from {tuple(rgb.shape)}")


class Photometric(Residual):
    """RGB photometric residual + analytic SE(3) Jacobian via gsplat (inverse-compositional).

    The optimization variable ``T`` is the object pose ``T_WO``. The camera extrinsic ``T_WC``
    and the observed frame are fixed per construction. ``evaluate``-style use is via
    ``residual(T, target, source)`` / ``jacobian(T, target, source)`` where ``source`` is a
    :class:`Frame` (supplying ``rgb`` and ``K``, optional ``mask``) or these are passed at
    construction. One instance per camera.

    Args:
        T_WC: 4×4 camera→world extrinsic (fixed for this camera/frame). The optimized ``T`` is
            the object pose; the Gaussians are not moved — the residual lives in image space.
        rgb_gt: optional (H, W, 3) observed RGB in [0, 1]. If ``None`` it is read from the
            ``source`` Frame at call time.
        K: optional (3, 3) intrinsics. If ``None`` it is read from ``source.K``.
        mask: optional (H, W) bool object mask; off-object pixels are excluded.
        width / height: render resolution; default to the observed image size.
        max_pixels: cap on selected high-gradient pixels (residual dim = 3 × selected).
        grad_threshold: minimum per-pixel gradient magnitude to be selected.
        huber_k: Huber threshold on the intensity residual (0–1 range).
        depth_min / depth_max: rendered-depth gate for pixel selection (metres).
        sh_degree: SH degree if ``target.colors`` holds SH coefficients ((N, K, 3)); ``None``
            treats ``colors`` as plain (N, 3) RGB.
        weight, robust: forwarded to :class:`Residual`.
    """

    def __init__(
        self,
        T_WC: torch.Tensor,
        rgb_gt: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_pixels: int = 500,
        grad_threshold: float = 0.05,
        huber_k: float = 0.05,
        depth_min: float = 0.05,
        depth_max: float = 1.5,
        sh_degree: Optional[int] = None,
        weight: float = 1.0,
        robust: Optional[Any] = None,
    ):
        super().__init__(weight=weight, robust=robust)
        if not _GSPLAT_AVAILABLE:
            raise ImportError(_GSPLAT_INSTALL_HINT)
        if T_WC is None:
            raise ValueError("Photometric requires a 4x4 T_WC camera extrinsic")

        self._device = T_WC.device
        self.T_WC = T_WC.to(device=self._device, dtype=torch.float32)
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

        # T_CW = world→camera (gsplat viewmat). Static for this camera.
        self.T_CW = torch.linalg.inv(self.T_WC)
        self.R_CW = self.T_CW[:3, :3]
        self.t_CW = self.T_CW[:3, 3]

        self._dim = 0  # residual dim of the last call (= 3 × selected pixels)

    def requires(self) -> set:
        return {"rgb", "K"}

    def dim(self) -> int:
        return self._dim

    # ── observation resolution ───────────────────────────────────────────────────

    def _resolve_obs(self, source: Any):
        rgb = self.rgb_gt
        K = self.K
        mask = self.mask
        if isinstance(source, Frame):
            if rgb is None and source.rgb is not None:
                rgb = _rgb_to_hwc(source.rgb.to(self._device, dtype=torch.float32))
            if K is None and source.K is not None:
                K = source.K.to(self._device, dtype=torch.float32)
            if mask is None:
                mask = source.mask
        if rgb is None:
            raise ValueError("Photometric needs rgb (pass rgb_gt or a Frame with .rgb)")
        if K is None:
            raise ValueError("Photometric needs K (pass K or a Frame with .K)")
        return rgb, K, mask

    # ── rendering ────────────────────────────────────────────────────────────────

    def _render(self, target: Gaussians, K: torch.Tensor, width: int, height: int):
        """Render ``target`` through this camera. Returns (rgb_pred HW3, depth_pred HW)."""
        if target.colors is None:
            raise ValueError("Photometric requires target.colors (RGB or SH) for rendering")
        scales = target.scales.exp() if target.log_scales else target.scales
        opac = target.opacities
        opac = opac.squeeze(-1) if opac.dim() == 2 else opac
        viewmats = self.T_CW.unsqueeze(0)  # (1, 4, 4) world→cam
        Ks = K.unsqueeze(0)  # (1, 3, 3)
        render, _alpha, _meta = _gsplat_rasterization(
            means=target.means.to(torch.float32),
            quats=target.quats.to(torch.float32),
            scales=scales.to(torch.float32),
            opacities=opac.to(torch.float32),
            colors=target.colors.to(torch.float32),
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=self.sh_degree,
            render_mode="RGB+ED",
        )
        out = render[0]  # (H, W, 4)
        rgb_pred = out[..., :3]
        # Expected depth Σwᵢzᵢ / Σwᵢ — the per-pixel surface depth, robust to the
        # alpha-weighted accumulation that darkens edge pixels (exactly the high-gradient
        # pixels this residual selects). Back-projecting it lands on the rendered surface.
        depth_pred = out[..., 3]
        return rgb_pred, depth_pred

    def _select(self, target: Gaussians, source: Any):
        """Render, pick high-gradient masked pixels with valid depth.

        Returns ``(rgb_pred, depth_pred, grad, v, u, K, rgb_gt)`` or ``None`` if no pixel
        qualifies. Shared by :meth:`residual` and :meth:`jacobian` so both linearize the
        same pixel set at the current pose.
        """
        rgb_gt, K, mask = self._resolve_obs(source)
        H, W = rgb_gt.shape[0], rgb_gt.shape[1]
        width = self.width or W
        height = self.height or H
        rgb_pred, depth_pred = self._render(target, K, width, height)

        grad = _image_gradients(rgb_pred)  # (H, W, 3, 2)
        grad_mag = grad.norm(dim=(-2, -1))  # (H, W)
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
        n_total = int(cand.sum().item())
        n_sel = min(self.max_pixels, n_total)
        if n_sel <= 0:
            return None
        _, top_flat = flat_mag.topk(n_sel)
        v = (top_flat // width).long()
        u = (top_flat % width).long()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        return rgb_pred, depth_pred, grad, v, u, (fx, fy, cx, cy), rgb_gt

    # ── residual / jacobian ──────────────────────────────────────────────────────

    def residual(self, T: torch.Tensor, target: Gaussians, source: Any) -> torch.Tensor:
        sel = self._select(target, source)
        if sel is None:
            self._dim = 0
            return target.means.new_zeros(0)
        rgb_pred, _depth_pred, _grad, v, u, _intr, rgb_gt = sel
        r_rgb = (rgb_pred[v, u] - rgb_gt[v, u]).reshape(-1)  # (3N,)
        abs_r = r_rgb.abs()
        huber_w = torch.where(
            abs_r <= self.huber_k,
            torch.ones_like(r_rgb),
            self.huber_k / abs_r.clamp_min(1e-6),
        )
        r_rgb = r_rgb * huber_w
        self._dim = int(r_rgb.shape[0])
        return r_rgb

    def jacobian(self, T: torch.Tensor, target: Gaussians, source: Any) -> Optional[torch.Tensor]:
        T_WO = T.to(device=self._device, dtype=torch.float32)
        T_OW = torch.linalg.inv(T_WO)
        sel = self._select(target, source)
        if sel is None:
            self._dim = 0
            return target.means.new_zeros(0, 6)
        rgb_pred, depth_pred, grad, v, u, (fx, fy, cx, cy), rgb_gt = sel

        # Huber weights (recomputed to scale J consistently with residual()).
        r_rgb = (rgb_pred[v, u] - rgb_gt[v, u]).reshape(-1)
        abs_r = r_rgb.abs()
        huber_w = torch.where(
            abs_r <= self.huber_k,
            torch.ones_like(r_rgb),
            self.huber_k / abs_r.clamp_min(1e-6),
        )

        # Back-project predicted depth to camera-frame points (OpenCV pinhole, gsplat convention).
        z = depth_pred[v, u].abs().clamp_min(1e-6)  # (N,)
        x_cam = (u.float() - cx) * z / fx
        y_cam = (v.float() - cy) * z / fy
        # ∂(u,v)/∂p_cam = (1/z²) [[fx·z, 0, −fx·x], [0, fy·z, −fy·y]]
        inv_z2 = 1.0 / (z * z)
        zeros = torch.zeros_like(z)
        J_proj = torch.stack(
            [
                torch.stack([fx * z * inv_z2, zeros, -fx * x_cam * inv_z2], dim=1),
                torch.stack([zeros, fy * z * inv_z2, -fy * y_cam * inv_z2], dim=1),
            ],
            dim=1,
        )  # (N, 2, 3)

        # ∂p_cam/∂δ_WO  (right-perturbation T_WO ← T_WO · exp(δ)).
        p_cam = torch.stack([x_cam, y_cam, z], dim=1)  # (N, 3)
        p_world = (self.T_WC[:3, :3] @ p_cam.T).T + self.T_WC[:3, 3]
        X_obj = (T_OW[:3, :3] @ p_world.T).T + T_OW[:3, 3]
        R_CW_R_WO = self.R_CW @ T_WO[:3, :3]  # (3, 3)
        N = X_obj.shape[0]
        Xx, Xy, Xz = X_obj[:, 0], X_obj[:, 1], X_obj[:, 2]
        zr = torch.zeros_like(Xx)
        Xobj_skew = torch.stack(
            [
                torch.stack([zr, -Xz, Xy], dim=1),
                torch.stack([Xz, zr, -Xx], dim=1),
                torch.stack([-Xy, Xx, zr], dim=1),
            ],
            dim=1,
        )  # (N, 3, 3)
        R_b = R_CW_R_WO.unsqueeze(0).expand(N, 3, 3)
        J_trans = R_b  # ∂p_cam/∂v = R_CW·R_WO
        J_rot = -R_b @ Xobj_skew  # ∂p_cam/∂ω
        J_pose = torch.cat([J_trans, J_rot], dim=2)  # (N, 3, 6)

        grad_uv = grad[v, u]  # (N, 3, 2)
        J_pix = J_proj @ J_pose  # (N, 2, 6)
        J = (grad_uv.unsqueeze(-1) * J_pix.unsqueeze(1)).sum(dim=2)  # (N, 3, 6)
        J = J.reshape(-1, 6)  # (3N, 6)
        J = J * huber_w.unsqueeze(1)
        self._dim = int(J.shape[0])
        return J
