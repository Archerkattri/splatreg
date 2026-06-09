"""Photometric residuals — terms that compare *renders*, not geometry.

Two residuals live here:

* :class:`Photometric` — render ``target`` via gsplat and compare to an **observed RGB frame**
  (object tracking). Inverse-compositional analytic Jacobian.
* :class:`SplatPhotometric` — render the **source splat under T** and compare to renders of the
  **target splat** from the same synthetic camera ring: pure splat-to-splat photometric
  alignment, no real images required. This is the PhotoReg trick (arXiv 2410.05044 — geometric
  alignment leaves visible seams; a final photometric stage through the rasterizer fixes them)
  adapted to splat-vs-splat, used by ``register(..., refine="photometric")`` via
  :func:`refine_photometric`.

:class:`Photometric` is the term that breaks rotational symmetry that depth/ICP cannot observe
(rotation about a symmetry axis is invisible to geometry but visible to texture). It ports the
inverse-compositional, analytically-sampled Jacobian of GaussianFeels' ``PhotometricResidual``
and retargets it onto **gsplat** as the renderer (standard OpenCV pinhole convention), so
splatreg has no dependency on GaussianFeels' custom rasterizer.

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

import math
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from ..core.lie import se3_exp, sim3_exp
from ..core.types import Frame, Gaussians, RegisterResult
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


# ════════════════════════════════════════════════════════════════════════════════════════════════
# Splat-to-splat photometric refinement (PhotoReg-style, adapted: splat-vs-splat, no real images)
# ════════════════════════════════════════════════════════════════════════════════════════════════

# Camera-ring defaults. Stand-off = _RING_RADIUS_MULT x the target's bounding radius — the same
# "typical object-framing distance" the coarse camera sweep uses (camera_loc), and comfortably
# inside a 60 deg FOV (object angular radius atan(1/2.5) ~ 22 deg < 30 deg half-FOV).
_RING_RADIUS_MULT = 2.5
_RING_FOV_DEG = 60.0
# Two interleaved elevation rings so the views are not coplanar (a single equatorial ring leaves
# the translation along the ring axis weakly observed).
_RING_ELEVATIONS_DEG = (25.0, -20.0)
# D-SSIM window (uniform box, "lite" — no Gaussian window) and the standard SSIM constants.
_SSIM_C1 = 0.01**2
_SSIM_C2 = 0.03**2
# Finite-difference step defaults for the FD Jacobian: rotation/log-scale in radians/nats; the
# translation step is this fraction of the target bounding radius (scene-scale aware).
_FD_ROT_EPS = 1.0e-3
_FD_SCALE_EPS = 1.0e-3
_FD_TRANS_FRAC = 1.0e-3


def _look_at_T_WC(cam: torch.Tensor, center: torch.Tensor, device, dtype) -> torch.Tensor:
    """Camera->world pose at ``cam`` looking at ``center`` (OpenCV: +z forward, +y down)."""
    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    fwd = center - cam
    fwd = fwd / fwd.norm().clamp_min(1e-9)
    if abs(float(torch.dot(fwd, up))) >= 0.95:  # looking (anti)parallel to up: switch hint axis
        up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    right = torch.linalg.cross(up, fwd)
    right = right / right.norm().clamp_min(1e-9)
    down = torch.linalg.cross(fwd, right)
    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = torch.stack([right, down, fwd], dim=1)  # columns = camera axes in world
    T[:3, 3] = cam
    return T


def camera_ring(
    target: Gaussians,
    n_views: int = 8,
    *,
    radius: Optional[float] = None,
    radius_mult: float = _RING_RADIUS_MULT,
    width: int = 64,
    height: int = 64,
    fov_deg: float = _RING_FOV_DEG,
    elevations_deg: Sequence[float] = _RING_ELEVATIONS_DEG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A small ring of synthetic look-at cameras around ``target``, for splat-vs-splat photometrics.

    ``n_views`` poses sit on a sphere around the target centroid at ``radius`` (default
    ``radius_mult x`` the target's bounding radius — a sensible framing distance), azimuths evenly
    spaced, elevations cycling through ``elevations_deg`` so the views are not coplanar. Returns
    ``(T_WC, K)``: a ``(V, 4, 4)`` stack of camera->world poses and the shared ``(3, 3)`` pinhole
    intrinsics derived from ``fov_deg`` at ``width x height``.
    """
    if not isinstance(target, Gaussians) or len(target) == 0:
        raise ValueError("camera_ring() needs a non-empty Gaussians target.")
    if n_views < 1:
        raise ValueError(f"camera_ring() needs n_views >= 1, got {n_views}.")
    means = target.means
    device, dtype = means.device, means.dtype
    center = means.mean(dim=0)
    brad = float((means - center).norm(dim=1).max().item())
    r = float(radius) if radius is not None else radius_mult * max(brad, 1e-6)

    poses = []
    for i in range(int(n_views)):
        az = 2.0 * math.pi * i / int(n_views)
        el = math.radians(float(elevations_deg[i % len(elevations_deg)])) if elevations_deg else 0.0
        cam = center + r * torch.tensor(
            [math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)],
            device=device,
            dtype=dtype,
        )
        poses.append(_look_at_T_WC(cam, center, device, dtype))
    T_WC = torch.stack(poses, dim=0)

    f = 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)
    K = torch.tensor(
        [[f, 0.0, width * 0.5], [0.0, f, height * 0.5], [0.0, 0.0, 1.0]], device=device, dtype=dtype
    )
    return T_WC, K


def _quat_from_matrix_wxyz(R: torch.Tensor) -> torch.Tensor:
    """Rotation 3x3 -> unit quaternion (w, x, y, z). Branch-free (Shepperd-style), differentiable.

    (Sub)gradients flow through the sqrt/clamp and the magnitude argument of ``copysign``; the
    sign argument contributes none — fine for the small refinement perturbations this serves.
    """
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    t = m00 + m11 + m22
    w = torch.sqrt(torch.clamp(1.0 + t, min=1e-12)) * 0.5
    x = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-12)) * 0.5
    y = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-12)) * 0.5
    z = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-12)) * 0.5
    x = torch.copysign(x, R[2, 1] - R[1, 2])
    y = torch.copysign(y, R[0, 2] - R[2, 0])
    z = torch.copysign(z, R[1, 0] - R[0, 1])
    q = torch.stack([w, x, y, z])
    return q / q.norm().clamp_min(1e-12)


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of a single wxyz quaternion ``a`` (4,) onto a batch ``b`` (N, 4)."""
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    out = torch.stack([w, x, y, z], dim=-1)
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _transform_gaussians_diff(g: Gaussians, T: torch.Tensor) -> Gaussians:
    """SE(3)/Sim(3)-transform a splat's means/quats/scales, DIFFERENTIABLY w.r.t. ``T``.

    The pose-baking twin of ``api._apply_transform_to_gaussians``, but every op (including the
    Sim(3) scale ``s = det(sR)^(1/3)``) stays a torch tensor op so the residual can be autodiffed
    (or finite-differenced) through it. Opacities/colors pass through unchanged.

    NOTE on SH colors: view-dependent SH coefficients are NOT rotated here (the DC term is
    rotation-invariant). For the small refinement deltas this stage takes that approximation is
    second-order; for strongly view-dependent splats prefer RGB colors (``sh_degree=None``).
    """
    T = T.to(device=g.means.device, dtype=g.means.dtype)
    block = T[:3, :3]
    t = T[:3, 3]
    s = torch.det(block).abs().clamp_min(1e-18) ** (1.0 / 3.0)  # scalar tensor, differentiable
    R = block / s
    means = g.means @ block.transpose(-1, -2) + t
    quats = _quat_mul_wxyz(_quat_from_matrix_wxyz(R), g.quats)
    if g.log_scales:
        scales = g.scales + torch.log(s)
    else:
        scales = g.scales * s
    return Gaussians(
        means=means, quats=quats, scales=scales, opacities=g.opacities, colors=g.colors, log_scales=g.log_scales
    )


def _gsplat_render_views(
    splat: Gaussians,
    T_CW: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    sh_degree: Optional[int],
) -> torch.Tensor:
    """Default render callable: gsplat-rasterize ``splat`` from all views at once -> (V, H, W, 3).

    ``T_CW`` is the (V, 4, 4) world->camera viewmat stack; ``K`` the shared (3, 3) intrinsics.
    Black background (both sides of the comparison render with the same one, so it cancels).
    """
    if not _GSPLAT_AVAILABLE:  # pragma: no cover - callers check at construction
        raise ImportError(_GSPLAT_INSTALL_HINT)
    if splat.colors is None:
        raise ValueError("splat-to-splat photometrics need .colors (RGB or SH) on both splats")
    scales = splat.scales.exp() if splat.log_scales else splat.scales
    opac = splat.opacities
    opac = opac.squeeze(-1) if opac.dim() == 2 else opac
    V = int(T_CW.shape[0])
    render, _alpha, _meta = _gsplat_rasterization(
        means=splat.means.to(torch.float32),
        quats=splat.quats.to(torch.float32),
        scales=scales.to(torch.float32),
        opacities=opac.to(torch.float32),
        colors=splat.colors.to(torch.float32),
        viewmats=T_CW.to(torch.float32),
        Ks=K.to(torch.float32).unsqueeze(0).expand(V, 3, 3),
        width=int(width),
        height=int(height),
        sh_degree=sh_degree,
        render_mode="RGB",
    )
    return render[..., :3]  # (V, H, W, 3)


def _dssim_lite(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-pixel D-SSIM map ``(1 - SSIM)/2`` over a uniform 3x3 window ("lite": box, not Gaussian).

    ``pred`` / ``gt`` are (V, H, W, 3); returns the same shape. Differentiable; the box window
    keeps it cheap enough to sit inside an FD Jacobian loop.
    """
    p = pred.permute(0, 3, 1, 2)  # (V, 3, H, W)
    g = gt.permute(0, 3, 1, 2)
    pool = lambda x: F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
    mu_p, mu_g = pool(p), pool(g)
    var_p = pool(p * p) - mu_p * mu_p
    var_g = pool(g * g) - mu_g * mu_g
    cov = pool(p * g) - mu_p * mu_g
    ssim = ((2.0 * mu_p * mu_g + _SSIM_C1) * (2.0 * cov + _SSIM_C2)) / (
        (mu_p * mu_p + mu_g * mu_g + _SSIM_C1) * (var_p + var_g + _SSIM_C2)
    )
    return ((1.0 - ssim) * 0.5).permute(0, 2, 3, 1)


class SplatPhotometric(Residual):
    """Splat-to-splat photometric residual: render SOURCE under ``T`` vs render TARGET, same cameras.

    The PhotoReg insight (arXiv 2410.05044) adapted to need **no real images**: geometric
    registration leaves a visible seam (sub-voxel pose error + per-capture appearance), so a final
    photometric stage *through the rasterizer* aligns what actually gets rendered. Here both sides
    of the comparison are renders — the target splat from a small synthetic camera ring, and the
    source splat transformed by the optimization variable ``T`` from the same ring — so the
    residual is exactly "how different do the two splats LOOK", per pixel::

        r = vec( I(source | T, cam_v) - I(target | cam_v) )      v = 1..V   (Huber-scaled)
        [+ dssim_weight * vec( DSSIM_3x3(I_src, I_tgt) )]                    (optional)

    SE(3)- and Sim(3)-capable: ``T``'s scale is carried into the rendered splat (means, per-anchor
    scales), so a Sim(3) solve sees the silhouette grow/shrink.

    Robustness is IRLS, not residual-clipping: the residual rows stay the RAW difference (smooth,
    so FD and autodiff Jacobians agree) and ``huber_k`` becomes a Huber sqrt-weight ``robust``
    kernel the solver applies per-row. Folding the Huber into the residual itself (as the
    frame-based :class:`Photometric` does with its analytic Jacobian) would make saturated rows
    piecewise-constant — zero gradient under autodiff/FD — which is exactly wrong for this
    derivative-free residual.

    Jacobian (``jac_mode``):

    * ``"fd"`` (default) — central finite differences on the <=7-dim tangent (2 x dof render
      passes per LM iteration). Exact enough at image scale, renderer-agnostic, and the only mode
      that works with gsplat's rasterizer (a custom autograd.Function that ``torch.func`` cannot
      transform). The residual uses ALL pixels (no data-dependent selection) so the FD dimension
      is stable across perturbed evaluations.
    * ``"autodiff"`` — return ``None`` so the solver's ``jacrev`` fallback (row-chunked via
      ``jac_row_chunk``) differentiates through the render. Requires a pure-torch,
      ``torch.func``-compatible ``render_fn`` (e.g. a test mock); gsplat's CUDA rasterizer is not.

    Parameters
    ----------
    cameras : (V, 4, 4) camera->world poses (build with :func:`camera_ring`).
    K : (3, 3) shared pinhole intrinsics.
    width, height : render resolution (keep small — 64-128 px; this is a refinement signal).
    render_fn : optional render callable ``(splat, T_CW (V,4,4), K (3,3), width, height,
        sh_degree) -> (V, H, W, 3)``; ``None`` uses the gsplat rasterizer (raising a clear
        ImportError here, at call time, when gsplat is absent).
    sh_degree : SH degree when the splats' ``colors`` are SH coefficients; ``None`` = plain RGB.
    huber_k : Huber threshold on the per-channel intensity residual (0-1 range; ``<= 0`` disables
        the robust kernel). Ignored when an explicit ``robust`` callable is given.
    dssim_weight : weight of the appended D-SSIM-lite rows; ``0`` (default) disables them.
    jac_mode : ``"fd"`` or ``"autodiff"`` (see above).
    fd_rot_eps / fd_scale_eps : FD step for the rotation / log-scale channels (rad / nats).
    fd_trans_eps : FD step for translation; ``None`` auto-derives ``1e-3 x`` the target bounding
        radius at first use (scene-scale aware).
    weight, robust : forwarded to :class:`Residual`.
    """

    def __init__(
        self,
        cameras: torch.Tensor,
        K: torch.Tensor,
        *,
        width: int = 64,
        height: int = 64,
        render_fn: Optional[Callable] = None,
        sh_degree: Optional[int] = None,
        huber_k: float = 0.1,
        dssim_weight: float = 0.0,
        jac_mode: str = "fd",
        fd_rot_eps: float = _FD_ROT_EPS,
        fd_scale_eps: float = _FD_SCALE_EPS,
        fd_trans_eps: Optional[float] = None,
        weight: float = 1.0,
        robust: Optional[Any] = None,
    ):
        if robust is None and huber_k > 0.0:
            # IRLS Huber sqrt-weight: w = sqrt(min(1, k/|r|)) so the solver's weighted cost
            # 0.5*sum(w^2 r^2) transitions quadratic->linear at |r| = k, per row.
            k = float(huber_k)

            def robust(abs_r: torch.Tensor) -> torch.Tensor:
                return torch.where(abs_r <= k, torch.ones_like(abs_r), k / abs_r.clamp_min(1e-12)).sqrt()

        super().__init__(weight=weight, robust=robust)
        if render_fn is None and not _GSPLAT_AVAILABLE:
            raise ImportError(
                "Splat-to-splat photometric refinement requires gsplat (or an explicit render_fn). "
                + _GSPLAT_INSTALL_HINT
            )
        if jac_mode not in ("fd", "autodiff"):
            raise ValueError(f"jac_mode must be 'fd' or 'autodiff', got {jac_mode!r}")
        cameras = torch.as_tensor(cameras)
        if cameras.dim() != 3 or cameras.shape[-2:] != (4, 4):
            raise ValueError(f"cameras must be (V, 4, 4) camera->world poses, got {tuple(cameras.shape)}")
        self._device = cameras.device
        self.T_WC = cameras.to(dtype=torch.float32)
        self.T_CW = torch.linalg.inv(self.T_WC)  # (V, 4, 4) world->camera viewmats
        self.K = K.to(device=self._device, dtype=torch.float32)
        self.width = int(width)
        self.height = int(height)
        self.render_fn = render_fn if render_fn is not None else _gsplat_render_views
        self.sh_degree = sh_degree
        self.huber_k = float(huber_k)
        self.dssim_weight = float(dssim_weight)
        self.jac_mode = jac_mode
        self.fd_rot_eps = float(fd_rot_eps)
        self.fd_scale_eps = float(fd_scale_eps)
        self.fd_trans_eps = None if fd_trans_eps is None else float(fd_trans_eps)
        self._dim = 0
        # Target render cache: the target splat is fixed across an LM run, so its V renders are
        # computed once (keyed by the target object's id; a new target invalidates).
        self._tgt_cache: Optional[torch.Tensor] = None
        self._tgt_cache_key: Optional[int] = None

    def requires(self) -> set:
        return {"source_gaussians", "colors"}

    def dim(self) -> int:
        return self._dim

    # ── rendering ────────────────────────────────────────────────────────────────

    def _render(self, splat: Gaussians) -> torch.Tensor:
        return self.render_fn(splat, self.T_CW, self.K, self.width, self.height, self.sh_degree)

    def _target_images(self, target: Gaussians) -> torch.Tensor:
        if self._tgt_cache is None or self._tgt_cache_key != id(target):
            if not isinstance(target, Gaussians) or target.colors is None:
                raise ValueError("SplatPhotometric needs target to be a Gaussians with .colors")
            self._tgt_cache = self._render(target).detach()
            self._tgt_cache_key = id(target)
        return self._tgt_cache

    # ── residual / jacobian ──────────────────────────────────────────────────────

    def residual(self, T: torch.Tensor, target: Gaussians, source: Any) -> torch.Tensor:
        if not isinstance(source, Gaussians) or source.colors is None:
            raise ValueError("SplatPhotometric needs source to be a Gaussians with .colors")
        tgt = self._target_images(target)
        pred = self._render(_transform_gaussians_diff(source, T))

        # RAW difference rows — robustness is the IRLS `robust` kernel (solver-applied), see class
        # docstring. Keeping r smooth is what makes the FD and autodiff Jacobians consistent.
        r = (pred - tgt).reshape(-1)
        if self.dssim_weight > 0.0:
            r = torch.cat([r, self.dssim_weight * _dssim_lite(pred, tgt).reshape(-1)])
        r = r.to(dtype=T.dtype)
        self._dim = int(r.shape[0])
        return r

    def jacobian(
        self, T: torch.Tensor, target: Gaussians, source: Any, *, dof: int = 6
    ) -> Optional[torch.Tensor]:
        if self.jac_mode == "autodiff":
            return None  # solver autodiffs (jacrev, row-chunked) — needs a torch.func-able render_fn
        if dof not in (6, 7):
            raise ValueError(f"SplatPhotometric.jacobian: dof must be 6 or 7, got {dof}.")
        exp_fn = se3_exp if dof == 6 else sim3_exp
        eps = self._fd_eps(target, dof, T.device, T.dtype)
        cols = []
        for k in range(dof):
            d = torch.zeros(dof, device=T.device, dtype=T.dtype)
            d[k] = eps[k]
            r_plus = self.residual(T @ exp_fn(d), target, source)
            r_minus = self.residual(T @ exp_fn(-d), target, source)
            cols.append((r_plus - r_minus) / (2.0 * eps[k]))
        J = torch.stack(cols, dim=1)  # (R, dof)
        self._dim = int(J.shape[0])
        return J

    def _fd_eps(self, target: Gaussians, dof: int, device, dtype) -> torch.Tensor:
        """Per-channel central-difference steps: [trans x3 | rot x3 | (log_s)]."""
        if self.fd_trans_eps is None:
            means = target.means
            brad = float((means - means.mean(dim=0)).norm(dim=1).max().item())
            self.fd_trans_eps = max(_FD_TRANS_FRAC * brad, 1e-8)
        vals = [self.fd_trans_eps] * 3 + [self.fd_rot_eps] * 3
        if dof == 7:
            vals.append(self.fd_scale_eps)
        return torch.tensor(vals, device=device, dtype=dtype)


def refine_photometric(
    target: Gaussians,
    source: Gaussians,
    T0: torch.Tensor,
    *,
    transform: str = "se3",
    cameras: Optional[torch.Tensor] = None,
    K: Optional[torch.Tensor] = None,
    n_views: int = 8,
    radius: Optional[float] = None,
    radius_mult: float = _RING_RADIUS_MULT,
    width: int = 64,
    height: int = 64,
    fov_deg: float = _RING_FOV_DEG,
    render_fn: Optional[Callable] = None,
    sh_degree: Optional[int] = None,
    huber_k: float = 0.1,
    dssim_weight: float = 0.0,
    jac_mode: str = "fd",
    max_iters: int = 10,
    damping: float = 1e-4,
    max_trans_step: Optional[float] = None,
    max_rot_step: float = 0.05,
    convergence_tol: float = 1e-6,
    jac_row_chunk: int = 256,
) -> RegisterResult:
    """PhotoReg-style photometric REFINEMENT of an already-geometric pose ``T0`` (splat-vs-splat).

    Runs a short second LM stage whose only residual is :class:`SplatPhotometric`: the source
    splat is rendered under the current ``T`` from a small synthetic camera ring around the target
    (built by :func:`camera_ring` unless explicit ``cameras``/``K`` are given) and compared against
    renders of the target splat from the same cameras. No real images are required. This is the
    seam-fixing stage ``register(..., refine="photometric")`` calls; it can also be used directly.

    It is a *refiner*: direct image alignment has a basin of a few degrees / a few percent of
    depth, so ``T0`` must already be geometrically close (exactly what the geometric ``register``
    stage produces). Both ``"se3"`` and ``"sim3"`` transforms are supported; step clamps default
    conservative (``max_trans_step`` = 2% of the target bounding radius, ``max_rot_step`` ~ 2.9 deg)
    because the stage polishes, never re-bases.

    gsplat is required unless a custom ``render_fn`` is supplied — checked at residual
    construction (i.e. here, at call time) with a clear install hint, never at import.

    Returns a :class:`~splatreg.core.types.RegisterResult`; ``info`` carries the LM diagnostics
    plus ``stage="photometric"`` and ``n_views``.
    """
    from ..solvers.lm import run_lm  # local import: residuals must not pull solver code at import

    if not isinstance(target, Gaussians) or len(target) == 0 or target.colors is None:
        raise ValueError("refine_photometric() needs a non-empty target Gaussians with .colors")
    if not isinstance(source, Gaussians) or len(source) == 0 or source.colors is None:
        raise ValueError("refine_photometric() needs a non-empty source Gaussians with .colors")
    T0 = T0.to(device=target.means.device)  # render-side tensors live on the target's device

    if cameras is None:
        cameras, ring_K = camera_ring(
            target, n_views, radius=radius, radius_mult=radius_mult, width=width, height=height, fov_deg=fov_deg
        )
        if K is None:
            K = ring_K
    elif K is None:
        raise ValueError("refine_photometric(): explicit cameras need an explicit K")

    res = SplatPhotometric(
        cameras,
        K,
        width=width,
        height=height,
        render_fn=render_fn,
        sh_degree=sh_degree,
        huber_k=huber_k,
        dssim_weight=dssim_weight,
        jac_mode=jac_mode,
    )

    if max_trans_step is None:
        means = target.means
        brad = float((means - means.mean(dim=0)).norm(dim=1).max().item())
        max_trans_step = max(0.02 * brad, 1e-8)

    result = run_lm(
        T0,
        [res],
        target,
        source,
        transform=transform,
        n_iters=int(max_iters),
        damping=damping,
        max_trans_step=float(max_trans_step),
        max_rot_step=float(max_rot_step),
        convergence_tol=convergence_tol,
        jac_row_chunk=int(jac_row_chunk),
    )
    result.info["stage"] = "photometric"
    result.info["n_views"] = int(cameras.shape[0])
    return result
