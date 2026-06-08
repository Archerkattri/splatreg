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

import math
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

__all__ = ["localize_camera", "CameraPhotometric", "coarse_localize_camera"]


def _project_points(means: torch.Tensor, T_WC: torch.Tensor, K: torch.Tensor):
    """Project world points into pixel coords for camera pose ``T_WC``. Returns ``(uv, z, in_front)``.

    Pure pinhole projection (``T_CW = inv(T_WC)``, then ``K · X_c``) — no rasteriser, so it runs on
    CPU with no gsplat. ``uv`` is ``(M, 2)`` float pixel coords, ``z`` the camera-frame depth, and
    ``in_front`` a bool mask of points with ``z > 0``. This is the cheap scoring primitive the coarse
    pose sweep uses (project-and-count-overlap), not a photometric render.
    """
    T_CW = torch.linalg.inv(T_WC)
    R = T_CW[:3, :3]
    t = T_CW[:3, 3]
    Xc = means @ R.transpose(-1, -2) + t  # (M, 3) camera frame
    z = Xc[:, 2]
    in_front = z > 1e-6
    zc = z.clamp_min(1e-6)
    u = K[0, 0] * (Xc[:, 0] / zc) + K[0, 2]
    v = K[1, 1] * (Xc[:, 1] / zc) + K[1, 2]
    return torch.stack([u, v], dim=1), z, in_front


def _dilate(occ: torch.Tensor, r: int) -> torch.Tensor:
    """Binary dilation of a ``(grid, grid)`` mask by a ``±r`` square (fills sparse occupancy)."""
    if r <= 0:
        return occ
    x = occ.float().view(1, 1, *occ.shape)
    x = torch.nn.functional.max_pool2d(x, kernel_size=2 * r + 1, stride=1, padding=r)
    return x.view(*occ.shape) > 0.5


def _occupancy(
    uv: torch.Tensor, valid: torch.Tensor, W: int, H: int, grid: int, dilate: int = 1
) -> torch.Tensor:
    """Low-res ``(grid, grid)`` boolean occupancy of the in-image projected points (silhouette proxy).

    A point set splat is a *sparse* sampling, so a bare hit-bitmap leaves the silhouette interior
    full of holes and makes the IoU score both low and viewpoint-ambiguous. A small binary
    ``dilate`` closes those holes into a connected silhouette, which is what the coarse score should
    compare — it raises the score and, more importantly, sharpens the discrimination between
    viewpoints (the filled outline is far more viewpoint-specific than scattered hits).
    """
    occ = torch.zeros(grid * grid, dtype=torch.bool, device=uv.device)
    if valid.sum() == 0:
        return occ.view(grid, grid)
    uvv = uv[valid]
    inb = (uvv[:, 0] >= 0) & (uvv[:, 0] < W) & (uvv[:, 1] >= 0) & (uvv[:, 1] < H)
    uvv = uvv[inb]
    if uvv.shape[0] == 0:
        return occ.view(grid, grid)
    gx = (uvv[:, 0] / W * grid).long().clamp(0, grid - 1)
    gy = (uvv[:, 1] / H * grid).long().clamp(0, grid - 1)
    occ[gy * grid + gx] = True
    return _dilate(occ.view(grid, grid), dilate)


def _candidate_poses(center: torch.Tensor, radius: float, n_az: int, n_el: int, device, dtype):
    """A sphere of look-at camera→world poses around ``center`` at distance ``radius``.

    Samples ``n_az`` azimuths × ``n_el`` elevations; each camera sits on the sphere and looks at
    ``center`` (OpenCV convention: ``+z`` forward, ``+y`` down). This is the coarse viewpoint grid the
    sweep scores — wide enough to seed a localiser that has no prior at all.
    """
    poses = []
    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    els = torch.linspace(-1.0, 1.0, n_el) * math.pi / 3.0  # ±60° elevation
    azs = torch.linspace(0.0, 2 * math.pi * (1 - 1.0 / max(n_az, 1)), n_az)
    for el in els:
        for az in azs:
            ce = math.cos(float(el))
            cam = center + radius * torch.tensor(
                [ce * math.sin(float(az)), math.sin(float(el)), ce * math.cos(float(az))],
                device=device, dtype=dtype,
            )
            fwd = center - cam
            fwd = fwd / fwd.norm().clamp_min(1e-9)
            u = up if abs(float(torch.dot(fwd, up))) < 0.95 else torch.tensor(
                [0.0, 0.0, 1.0], device=device, dtype=dtype
            )
            right = torch.linalg.cross(u, fwd)
            right = right / right.norm().clamp_min(1e-9)
            down = torch.linalg.cross(fwd, right)
            Rwc = torch.stack([right, down, fwd], dim=1)  # columns = cam axes in world
            T = torch.eye(4, device=device, dtype=dtype)
            T[:3, :3] = Rwc
            T[:3, 3] = cam
            poses.append(T)
    return poses


def coarse_localize_camera(
    splat: Gaussians,
    frame: Frame,
    *,
    candidates: Optional[list] = None,
    n_az: int = 12,
    n_el: int = 5,
    radius: Optional[float] = None,
    grid: int = 24,
    dilate: int = 2,
    return_score: bool = False,
):
    """Coarse, prior-free camera-pose seed by a project-and-compare viewpoint sweep (CPU, no gsplat).

    :func:`localize_camera` refines a pose only within the narrow basin of direct image alignment, so
    it needs a decent prior. This provides that prior when none exists (a *wide-baseline* relocalise):
    it scores a sphere of candidate camera poses by how well the splat's **projected occupancy**
    overlaps the query frame's foreground silhouette (from ``frame.mask``, or a luminance threshold of
    ``frame.rgb``), and returns the best-scoring pose. Pure pinhole projection — it runs on CPU and
    needs no rasteriser — so it is a coarse *seed*, deliberately cheap, not a final pose. Feed its
    result as ``init_T_WC`` to :func:`localize_camera` for the fine refine.

    Parameters
    ----------
    splat : the world-fixed splat (only ``means`` are used here).
    frame : query observation; needs ``K`` and a foreground cue (``mask`` preferred, else ``rgb``).
    candidates : explicit list of ``(4, 4)`` ``T_WC`` to score; ``None`` auto-builds a look-at sphere
        (``n_az`` azimuths × ``n_el`` elevations) around the splat centroid at ``radius``.
    n_az / n_el / radius : the auto sphere's resolution and stand-off distance (``radius`` ``None``
        ⇒ ~2.5× the splat's bounding radius, a typical object-framing distance).
    grid : occupancy-bitmap resolution the IoU score is computed at (coarse on purpose).
    dilate : binary-dilation radius (in grid cells) applied to both the projected and the query
        occupancy before scoring — closes the holes a *sparse* point splat leaves so the IoU compares
        connected silhouettes (higher and far more viewpoint-discriminative). ``0`` disables it.
    return_score : also return the best IoU score (diagnostic).

    Returns
    -------
    The best ``(4, 4)`` ``T_WC`` (camera→world); with ``return_score=True`` a ``(T_WC, score)`` tuple.
    """
    if frame.K is None:
        raise ValueError("coarse_localize_camera(): frame needs .K.")
    means = splat.means
    device, dtype = means.device, means.dtype
    K = frame.K.to(device=device, dtype=dtype)

    # Query foreground occupancy (the target silhouette) at the same coarse grid.
    if frame.mask is not None:
        m = frame.mask.to(device)
        fg = m > 0.5 if m.dtype != torch.bool else m
    elif frame.rgb is not None:
        rgb = _rgb_to_hwc(frame.rgb.to(device=device, dtype=dtype))
        fg = rgb.mean(dim=-1) > 1e-3  # non-black = foreground (synthetic / masked captures)
    else:
        raise ValueError("coarse_localize_camera(): frame needs .mask or .rgb for a foreground cue.")
    H, W = fg.shape[-2], fg.shape[-1]
    # Downsample the foreground to the score grid.
    ys = (torch.arange(grid, device=device).float() + 0.5) / grid * H
    xs = (torch.arange(grid, device=device).float() + 0.5) / grid * W
    gy = ys.long().clamp(0, H - 1)
    gx = xs.long().clamp(0, W - 1)
    fg_grid = _dilate(fg[gy][:, gx], dilate)  # (grid, grid), dilated to match the projected occupancy

    center = means.mean(dim=0)
    if radius is None:
        brad = float((means - center).norm(dim=1).max())
        radius = 2.5 * max(brad, 1e-6)
    if candidates is None:
        candidates = _candidate_poses(center, float(radius), n_az, n_el, device, dtype)

    best_T, best_score = None, -1.0
    for T_WC in candidates:
        T_WC = T_WC.to(device=device, dtype=dtype)
        uv, z, in_front = _project_points(means, T_WC, K)
        occ = _occupancy(uv, in_front, W, H, grid, dilate=dilate)
        inter = (occ & fg_grid).sum().float()
        union = (occ | fg_grid).sum().float().clamp_min(1.0)
        score = float(inter / union)
        if score > best_score:
            best_score, best_T = score, T_WC
    if best_T is None:  # no candidates
        best_T = torch.eye(4, device=device, dtype=dtype)
    return (best_T, best_score) if return_score else best_T


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
    init_T_WC,
    *,
    iters: int = 150,
    lr: float = 1e-2,
    refold_every: int = 40,
    mask_to_rendered: bool = True,
    huber_k: Optional[float] = None,
    sh_degree: Optional[int] = None,
    coarse_kwargs: Optional[dict] = None,
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
    init_T_WC : ``(4, 4)`` initial camera→world prior, OR the string ``"coarse"`` to first run the
        prior-free :func:`coarse_localize_camera` viewpoint sweep and refine its seed. Direct image
        alignment has a limited basin (a few degrees / a few percent of depth), so without ``"coarse"``
        a *wide-baseline* query (no good prior) falls outside it — that is exactly what the coarse
        seed bridges. ``coarse_kwargs`` is forwarded to the sweep.
    iters : Adam steps on the pose tangent.
    lr : Adam learning rate on the 6-vector right-perturbation tangent.
    refold_every : every this many steps, fold the accumulated tangent into ``T_WC`` and reset it to
        zero (keeps ``exp`` near the origin where it is best-conditioned).
    mask_to_rendered : when ``True``, the photometric loss is restricted to pixels the splat actually
        renders (rendered depth > 0), so empty background does not dominate the loss.
    huber_k : optional Huber threshold on the per-pixel RGB residual (robust to occluders / outliers);
        ``None`` uses a plain L2 loss.
    sh_degree : SH degree if ``splat.colors`` are SH coefficients; ``None`` treats them as RGB.
    coarse_kwargs : forwarded to :func:`coarse_localize_camera` when ``init_T_WC == "coarse"``
        (e.g. ``n_az`` / ``n_el`` / ``grid`` / ``radius``).

    Returns
    -------
    :class:`~splatreg.core.types.RegisterResult` with the refined ``T_WC``; ``info['mode'] ==
    'camera_loc'``, ``info['loss']`` (final), ``info['loss_history']``.
    """
    if not _GSPLAT_AVAILABLE:
        raise ImportError(_GSPLAT_INSTALL_HINT)
    if not isinstance(splat, Gaussians) or splat.colors is None:
        raise ValueError("localize_camera(): splat must be a Gaussians with .colors for rendering.")
    if frame.rgb is None or frame.K is None:
        raise ValueError("localize_camera(): frame needs .rgb and .K.")
    if isinstance(init_T_WC, str):
        if init_T_WC != "coarse":
            raise ValueError("localize_camera(): init_T_WC string must be 'coarse'.")
        # Prior-free seed: the projection-only viewpoint sweep (no gsplat needed for the seed).
        init_T_WC = coarse_localize_camera(splat, frame, **(coarse_kwargs or {}))
    if not isinstance(init_T_WC, torch.Tensor) or init_T_WC.shape[-2:] != (4, 4):
        raise ValueError("localize_camera(): init_T_WC must be a (4, 4) tensor or 'coarse'.")

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
