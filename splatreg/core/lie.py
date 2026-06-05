"""SE(3) / Sim(3) Lie-algebra helpers — the matrix exp/log at the heart of the registration loop.

Convention: right-perturbation, ``T_new = T @ se3_exp(xi)`` with the tangent ordered
``xi = [tx, ty, tz, rx, ry, rz]`` (translation first, then rotation). Every Jacobian in
splatreg uses this same layout.

GPU-sync-free: no Python ``if`` branches on tensor *values* (only on shapes / dtypes), so the
whole file is ``torch.compile`` / ``vmap`` / ``jacrev`` friendly and never forces a device sync.

Sim(3) scale DoF
----------------
A 7-DoF Sim(3) step adds a single log-scale component to the tail of the tangent
(``xi = [tx,ty,tz, rx,ry,rz, rho]``) and maps it to a similarity transform
``T = [[s*R, t], [0, 1]]`` with ``s = exp(rho)``. Two layers coexist:

* :func:`se3_exp` / :func:`se3_log` are the rigid SE(3) fast path (unit scale). ``se3_exp``
  accepts a 6- or 7-vector but ignores any 7th slot; ``se3_log(dof=7)`` emits a log-scale tail
  recovered from the block determinant (zero for a pure SE(3) input).
* :func:`sim3_exp` / :func:`sim3_log` are the real similarity path: ``sim3_exp`` scales the
  Rodrigues rotation block by ``s = exp(rho)`` (the SE(3) ``V``-matrix still maps the
  translational tangent), and ``sim3_log`` recovers ``rho`` from the similarity scale (the
  cube-root of the 3x3 block determinant) before logging the de-scaled rotation.

Both pairs share the same right-perturbation convention and tangent ordering, so a solver can pick
the exp/log pair by transform without changing the matrix layout or the Jacobian column order.
"""
from __future__ import annotations

import torch

DTYPE = torch.float32


def skew(v: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric 3x3 from a 3-vector: ``skew(v) @ x == cross(v, x)``.

    Built with ``stack`` (not in-place writes) so it is differentiable and vmap-safe.
    """
    z = torch.zeros_like(v[0])
    return torch.stack([
        torch.stack([z, -v[2], v[1]]),
        torch.stack([v[2], z, -v[0]]),
        torch.stack([-v[1], v[0], z]),
    ])


def se3_exp(delta: torch.Tensor) -> torch.Tensor:
    """SE(3) (or rigid part of Sim(3)) exponential: tangent -> 4x4 matrix.

    Accepts a 6-vector ``[trans | rot]`` or, for forward-compatibility, a 7-vector
    ``[trans | rot | log_s]`` whose 7th component is currently ignored (unit scale). Uses the
    closed-form Rodrigues rotation and the matching left-Jacobian ``V`` for translation. No
    Python branches on tensor values.
    """
    dev, dtype = delta.device, delta.dtype
    v = delta[:3]
    w = delta[3:6]
    theta = w.norm().clamp_min(1e-12)
    axis = w / theta
    W = skew(axis)
    W2 = W @ W
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    I3 = torch.eye(3, device=dev, dtype=dtype)
    R = I3 + sin_t * W + (1.0 - cos_t) * W2
    # V maps the translational tangent into SE(3). With Omega = skew(w) = theta*W
    # (W the unit-axis skew), the standard V = I + (1-cos)/theta^2 Omega +
    # (theta-sin)/theta^3 Omega^2 simplifies via Omega = theta*W to
    # V = I + (1-cos)/theta * W + (theta-sin)/theta * W^2.
    c0 = (1.0 - cos_t) / theta      # (1 - cos) / theta
    c1 = (theta - sin_t) / theta    # (theta - sin) / theta
    V = I3 + c0 * W + c1 * W2
    T = torch.eye(4, device=dev, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def se3_log(T: torch.Tensor, dof: int = 6) -> torch.Tensor:
    """SE(3) logarithm: 4x4 -> tangent vector.

    Returns a 6-vector ``[trans | rot]`` for ``dof == 6``; for ``dof == 7`` (Sim(3)
    forward-compat) it appends a log-scale component recovered from ``det(R*S)`` (zero for a
    plain SE(3) input). No Python branches on tensor values for the rigid part.
    """
    dev, dtype = T.device, T.dtype
    R = T[:3, :3]
    t = T[:3, 3]
    trace = ((R[0, 0] + R[1, 1] + R[2, 2]) - 1.0) * 0.5
    trace_c = trace.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(trace_c)
    theta_s = theta.clamp_min(1e-12)
    sin_t = torch.sin(theta_s)
    cos_t = torch.cos(theta_s)
    w_axis = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * sin_t)
    w = theta_s * w_axis
    W = skew(w_axis)
    W2 = W @ W
    I3 = torch.eye(3, device=dev, dtype=dtype)
    # V_inv for V = I + (1-cos)/theta * W + (theta-sin)/theta^2 * W^2 (standard SE(3) log).
    # Near pi: (1+cos)/sin -> 0 via L'Hopital, so the W^2 coefficient -> 1, the correct
    # near-pi behaviour. sin_t is bounded away from 0 by the trace_c clamp above.
    sin_t_safe = sin_t.clamp_min(1e-12)
    V_inv = I3 - (theta_s * 0.5) * W + (1.0 - theta_s * (1.0 + cos_t) / (2.0 * sin_t_safe)) * W2
    v = V_inv @ t
    near_zero = theta < 1e-10
    w = torch.where(near_zero, torch.zeros_like(w), w)
    v = torch.where(near_zero, t, v)
    xi = torch.cat([v, w])
    if dof == 7:
        # Sim(3) forward-compat from the SE(3) path: recover log-scale from the rotation*scale
        # block. For a pure SE(3) input det(R) == 1 so log_s == 0. The dedicated similarity log is
        # :func:`sim3_log`; this branch only lets an SE(3)-shaped result carry a (here zero) 7th slot.
        log_s = torch.log(torch.linalg.det(T[:3, :3]).abs().clamp_min(1e-12)) / 3.0
        xi = torch.cat([xi, log_s.reshape(1)])
    return xi


def sim3_exp(delta: torch.Tensor) -> torch.Tensor:
    """Sim(3) exponential: 7-vector tangent ``[trans | rot | rho]`` -> 4x4 similarity matrix.

    Builds ``T = [[s*R, t], [0, 1]]`` where ``R`` is the closed-form Rodrigues rotation of the
    rotation tangent ``w``, ``s = exp(rho)`` is the similarity scale, and ``t = V @ v`` uses the
    same SE(3) left-Jacobian ``V`` as :func:`se3_exp` (the scale multiplies only the rotation
    block, not the translation map). Passing a 6-vector falls back to unit scale, so this stays a
    drop-in for :func:`se3_exp`. No Python branches on tensor values, so it is ``vmap`` / ``jacrev``
    safe (the autodiff Jacobian column for ``rho`` flows straight through ``s = exp(rho)``).
    """
    dev, dtype = delta.device, delta.dtype
    v = delta[:3]
    w = delta[3:6]
    rho = delta[6] if delta.shape[0] > 6 else torch.zeros((), device=dev, dtype=dtype)
    s = torch.exp(rho)
    theta = w.norm().clamp_min(1e-12)
    axis = w / theta
    W = skew(axis)
    W2 = W @ W
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    I3 = torch.eye(3, device=dev, dtype=dtype)
    R = I3 + sin_t * W + (1.0 - cos_t) * W2
    # Translation uses the SE(3) V-matrix (scale lives only in the s*R block), matching se3_exp.
    c0 = (1.0 - cos_t) / theta      # (1 - cos) / theta
    c1 = (theta - sin_t) / theta    # (theta - sin) / theta
    V = I3 + c0 * W + c1 * W2
    T = torch.eye(4, device=dev, dtype=dtype)
    T[:3, :3] = s * R
    T[:3, 3] = V @ v
    return T


def sim3_log(T: torch.Tensor) -> torch.Tensor:
    """Sim(3) logarithm: 4x4 similarity ``[[s*R, t], [0, 1]]`` -> 7-vector ``[trans | rot | rho]``.

    Recovers the similarity scale ``s`` as the cube-root of the 3x3 block determinant
    (``det(s*R) = s^3`` for ``R`` in SO(3)), sets ``rho = log s``, de-scales the block to a pure
    rotation ``R = (s*R) / s``, and runs the SE(3) log on ``[[R, t], [0, 1]]`` for the rigid
    tangent. Inverse of :func:`sim3_exp` to numerical precision. No Python branches on tensor values.
    """
    dev, dtype = T.device, T.dtype
    A = T[:3, :3]
    # det(s*R) = s^3 det(R) = s^3 (R in SO(3)); abs guards an fp32-drifted near-zero/negative det.
    s = torch.linalg.det(A).abs().clamp_min(1e-12) ** (1.0 / 3.0)
    rho = torch.log(s)
    T_rigid = torch.eye(4, device=dev, dtype=dtype)
    T_rigid[:3, :3] = A / s
    T_rigid[:3, 3] = T[:3, 3]
    xi6 = se3_log(T_rigid, dof=6)
    return torch.cat([xi6, rho.reshape(1)])


def se3_inv(T: torch.Tensor) -> torch.Tensor:
    """Inverse of an SE(3) matrix.

    The closed-form ``[R^T | -R^T t]`` is only exact when ``R`` is precisely in SO(3). In long
    trajectories the pose accumulates fp32 drift from repeated ``T = T @ se3_exp(delta)`` updates,
    so ``R^T R != I`` after a few hundred frames and the closed form degrades. The general
    LU-based inverse handles drift cleanly, so we use it unconditionally.
    """
    return torch.linalg.inv(T)


def so3_project(R: torch.Tensor) -> torch.Tensor:
    """Project a near-rotation 3x3 onto SO(3) via SVD: ``R ~= U @ V^T``.

    Use only at entry points where numerical drift may have made ``R`` non-orthogonal (e.g.
    composing several poses from a multi-hypothesis seed). Inside the LM loop :func:`se3_exp`
    already returns clean rotations, so projection is unnecessary there. Guards against a
    reflection (det == -1) by flipping the last column of ``U``. Supports a batched ``(..., 3, 3)``.
    """
    U, _, Vh = torch.linalg.svd(R)
    R_proj = U @ Vh
    det = torch.linalg.det(R_proj)
    if det.dim() == 0:
        if det < 0:
            U = U.clone()
            U[..., -1] *= -1.0
            R_proj = U @ Vh
    else:
        flip = (det < 0).to(U.dtype)
        if flip.any():
            U = U.clone()
            U[..., -1] *= (1.0 - 2.0 * flip).unsqueeze(-1)
            R_proj = U @ Vh
    return R_proj
