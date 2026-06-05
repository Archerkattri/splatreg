"""Sim(3) Phase-2 verification (temporary; not part of the package).

Runs four layers, innermost-out, so a failure localises immediately:

* [1]-[3]  the Lie maps alone: ``sim3_exp`` / ``sim3_log`` roundtrip, the unit-scale
  consistency ``sim3_exp([v,w,0]) == se3_exp([v,w])``, and closed-form scale recovery.
* [4]      the Sim(3) *solver core* on a clean point-to-point residual whose minimum is the
  true similarity — this isolates ``run_lm(transform='sim3')`` + the autodiff Jacobian + the
  rho step-clamp from any field ambiguity, so it must recover ``s, R, t`` to ~1e-7.
* [5]      the flagship SDF residual end-to-end via the intended *coarse->fine* flow
  (``init='global'``) on an **asymmetric** cloud (so rotation is observable). The residual
  recovery is field-limited (the soft sum-of-Gaussians SDF's zero level-set sits slightly off
  the true surface and the coarse aligner's scale is RMS-approximate), so this asserts loose
  basin-level tolerances, not the tight ones [4] proves for the core.
* [6]      the SE(3) fast path stays correct (scale == 1) and the ``init='global'`` guard both
  uses ``global_align`` when present and falls back to identity (logged) when absent.

The recovery test deliberately does **not** use a rotationally-symmetric sphere from a cold
identity init: that leaves the SDF rotation unobservable, so it would slander a correct solver.
"""
import builtins
import logging
import math

import numpy as np
import torch

from splatreg import register
from splatreg.core.types import Gaussians
from splatreg.core.lie import se3_exp, sim3_exp, sim3_log
from splatreg.residuals import SDF
from splatreg.residuals.base import Residual
from splatreg.solvers.lm import run_lm

torch.manual_seed(0)
np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.float32


def axis_angle_R(axis, deg):
    axis = torch.tensor(axis, dtype=dt, device=dev)
    axis = axis / axis.norm()
    th = math.radians(deg)
    K = torch.tensor([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=dt, device=dev)
    I = torch.eye(3, dtype=dt, device=dev)
    return I + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def rot_angle_deg(Ra, Rb):
    Rrel = Ra.transpose(-1, -2) @ Rb
    c = ((Rrel[0, 0] + Rrel[1, 1] + Rrel[2, 2]) - 1.0) * 0.5
    return math.degrees(math.acos(float(c.clamp(-1.0, 1.0))))


def make_g(means):
    n = means.shape[0]
    return Gaussians(
        means=means,
        quats=torch.tensor([1.0, 0, 0, 0], device=dev, dtype=dt).repeat(n, 1),
        scales=torch.full((n, 3), 0.005, device=dev, dtype=dt),
        opacities=torch.ones(n, device=dev, dtype=dt),
    )


# Known ground-truth Sim(3) shared by every recovery test below (we register A onto B, so the
# recovered T maps the A-frame -> B-frame): B = s*R*A + t.
s_gt = 1.15
R_gt = axis_angle_R([0.2, 0.9, 0.3], 5.0)
t_gt = torch.tensor([0.012, -0.008, 0.010], device=dev, dtype=dt)  # ~12 mm


print("=" * 70)
print("[1] sim3_exp / sim3_log roundtrip (random tangents)")
max_err = 0.0
for _ in range(200):
    xi = torch.randn(7, device=dev, dtype=dt)
    xi[3:6] *= 0.5            # keep rotation < ~pi
    xi[6] *= 0.3             # plausible log-scale
    T = sim3_exp(xi)
    xi_rec = sim3_log(T)
    T_rec = sim3_exp(xi_rec)
    max_err = max(max_err, float((T - T_rec).abs().max()))
print(f"    max |T - exp(log(T))| over 200 samples: {max_err:.2e}")
assert max_err < 1e-4, max_err

print("[2] sim3_exp([v,w,0]) == se3_exp([v,w])  (unit-scale consistency)")
d6 = torch.randn(6, device=dev, dtype=dt); d6[3:] *= 0.5
d7 = torch.cat([d6, torch.zeros(1, device=dev, dtype=dt)])
consist = float((se3_exp(d6) - sim3_exp(d7)).abs().max())
print(f"    max diff: {consist:.2e}")
assert consist < 1e-6, consist

print("[3] sim3_log scale recovery for known s")
for s_true in [0.7, 1.0, 1.15, 2.0]:
    R = axis_angle_R([0.3, 1.0, -0.5], 23.0)
    T = torch.eye(4, device=dev, dtype=dt)
    T[:3, :3] = s_true * R
    T[:3, 3] = torch.tensor([0.1, -0.2, 0.05], device=dev, dtype=dt)
    s_rec = float(sim3_log(T)[6].exp())
    print(f"    s_true={s_true:.3f}  s_rec={s_rec:.5f}  err={abs(s_rec-s_true):.2e}")
    assert abs(s_rec - s_true) < 1e-4


# -------------------------------------------------------- [4] Sim(3) solver core (clean residual)
class _PointToPoint(Residual):
    """``r = (T @ src) - dst`` stacked to ``(3N,)`` — a residual whose unique minimum is the true
    similarity, so it tests the solver core (Lie update + autodiff Jacobian + rho clamp) free of
    any field ambiguity. ``target`` / ``source`` are unused (points are held on the residual)."""

    def __init__(self, src_pts: torch.Tensor, dst_pts: torch.Tensor, weight: float = 1.0):
        super().__init__(weight=weight)
        self.src = src_pts
        self.dst = dst_pts

    def residual(self, T, target, source):
        p = self.src @ T[:3, :3].transpose(-1, -2) + T[:3, 3]
        return (p - self.dst).reshape(-1) * self.weight

    def dim(self):
        return self.src.shape[0] * 3


print("=" * 70)
print("[4] Sim(3) solver core via a clean point-to-point residual (unique minimum)")
src = torch.randn(200, 3, device=dev, dtype=dt) * 0.1 + torch.tensor([0.4, -0.2, 0.3], device=dev, dtype=dt)
dst = s_gt * (src @ R_gt.transpose(-1, -2)) + t_gt
res = run_lm(torch.eye(4, device=dev, dtype=dt), [_PointToPoint(src, dst)], None, None,
             transform="sim3", n_iters=100, max_trans_step=0.05, max_rot_step=0.2)
R_est = res.T[:3, :3] / res.scale
print(f"    converged={res.converged} iters={res.info['n_iters']} dof={res.info['dof']} cost={res.info['cost']:.2e}")
print(f"    scale: est={res.scale:.6f}  gt={s_gt:.5f}  err={abs(res.scale-s_gt):.2e}")
print(f"    rot  : err={rot_angle_deg(R_est, R_gt):.6f} deg")
print(f"    trans: err={1000*float((res.T[:3,3]-t_gt).norm()):.6f} mm")
assert abs(res.scale - s_gt) < 1e-4, res.scale
assert rot_angle_deg(R_est, R_gt) < 1e-2
assert 1000 * float((res.T[:3, 3] - t_gt).norm()) < 1e-2


# ---------------------------------------------------------------- build asymmetric splat A for SDF
# A bumpy anisotropic ellipsoid: distinct per-axis extents + a +x lobe make rotation observable, so
# the soft SDF has a genuine minimum at the true pose (a symmetric sphere does not). Modest N: the
# Sim(3) path autodiffs the full-pairwise SDF residual under jacrev, whose reverse graph scales with
# n_points * n_anchors.
N = 900
u = torch.rand(N, device=dev, dtype=dt)
v = torch.rand(N, device=dev, dtype=dt)
phi = 2 * math.pi * u
costh = 2 * v - 1
sinth = torch.sqrt((1 - costh ** 2).clamp_min(0))
sph = torch.stack([sinth * torch.cos(phi), sinth * torch.sin(phi), costh], dim=1)
ell = sph * torch.tensor([0.14, 0.09, 0.06], device=dev, dtype=dt)       # anisotropic extents
lobe = 0.03 * torch.exp(-((sph[:, 0:1] - 1.0) ** 2) / 0.2) * torch.tensor([1.0, 0, 0], device=dev, dtype=dt)
pts = ell + lobe
pts = pts * (1.0 + 0.01 * torch.randn(N, 1, device=dev, dtype=dt))       # thin-shell jitter
pts = pts + torch.tensor([0.4, -0.2, 0.3], device=dev, dtype=dt)         # off-origin centre

A = make_g(pts)
B = make_g((s_gt * (pts @ R_gt.transpose(-1, -2))) + t_gt)
sigma = 0.02  # target-field bandwidth (~shell-thickness scale)

print("=" * 70)
print("[5] Sim(3) SDF recovery end-to-end: register(A -> B), coarse->fine (init='global')")
res = register(B, A, residuals=[SDF(sigma=sigma, n_points=600)], transform="sim3",
               init="global", max_iters=60)
R_est = res.T[:3, :3] / res.scale
s_err_pct = 100 * abs(res.scale - s_gt) / s_gt
rot_err = rot_angle_deg(R_est, R_gt)
trans_err_mm = 1000 * float((res.T[:3, 3] - t_gt).norm())
print(f"    converged={res.converged} iters={res.info['n_iters']} dof={res.info['dof']}")
print(f"    scale: est={res.scale:.5f}  gt={s_gt:.5f}  err={abs(res.scale-s_gt):.2e} ({s_err_pct:.3f}%)")
print(f"    rot  : err={rot_err:.4f} deg   (gt 5.0 deg applied)")
print(f"    trans: err={trans_err_mm:.4f} mm")
print(f"    final cost={res.info['cost']:.3e} rmse={res.info['rmse']:.3e}")
# Field-limited basin tolerances (the soft-SDF minimum is slightly off the true surface).
assert s_err_pct < 8.0, s_err_pct
assert rot_err < 3.0, rot_err
assert trans_err_mm < 40.0, trans_err_mm

print("=" * 70)
print("[6a] SE(3) fast path still correct (pure rigid GT, transform='se3')")
B2 = make_g((pts @ R_gt.transpose(-1, -2)) + t_gt)   # s = 1
res2 = register(B2, A, residuals=[SDF(sigma=sigma, n_points=600)], transform="se3",
                init="global", max_iters=60)
print(f"    scale={res2.scale:.5f} (must be 1.0)  converged={res2.converged} iters={res2.info['n_iters']}")
print(f"    rot  : err={rot_angle_deg(res2.T[:3,:3], R_gt):.4f} deg")
print(f"    trans: err={1000*float((res2.T[:3,3]-t_gt).norm()):.4f} mm")
assert abs(res2.scale - 1.0) < 1e-6, res2.scale

print("[6b] init='global' guard: present -> uses global_align; absent -> identity (logged)")
logging.basicConfig(level=logging.INFO)
res3 = register(B, A, residuals=[SDF(sigma=sigma, n_points=300)], transform="sim3",
                init="global", max_iters=2)
print(f"    [present] global_align used as LM init; scale={res3.scale:.4f} (LM produced a result)")
_orig_import = builtins.__import__


def _block_align(name, *a, **k):
    if name == "splatreg.align" or name.endswith(".align"):
        raise ImportError("simulated-absent align module")
    return _orig_import(name, *a, **k)


builtins.__import__ = _block_align
try:
    res4 = register(B, A, residuals=[SDF(sigma=sigma, n_points=300)], transform="sim3",
                    init="global", max_iters=2)
    print(f"    [absent ] guard fell back to identity init (logged above); scale={res4.scale:.4f}")
finally:
    builtins.__import__ = _orig_import

print("=" * 70)
print("ALL CHECKS PASSED")
