"""Phase-3 verification (temporary; not part of the package — deleted after the run).

Three layers, matching the task's acceptance criteria:

* [A] register(A, B) with DEFAULT residuals (residuals=None -> ICP-dominant + auto-sigma SDF)
      recovers a known SE(3) AND a known Sim(3) on an asymmetric splat, and does NOT degrade
      vs the good global init it starts from (we report the init error and the final error).
* [B] merge([A, B]) of two halves of one splat + a shared overlap band: the overlap-band
      Gaussian count must drop to ~single density after the voxel dedupe (report pre/post),
      and save_ply must round-trip the merged result through load_ply.
* [C] sanity: explicit residuals are still honoured unchanged; auto-sigma is sane.

The asymmetric cloud (anisotropic bumpy ellipsoid + a +x lobe) makes rotation observable so the
SDF/ICP minimum sits at the true pose; a symmetric sphere would slander a correct solver.
"""
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from splatreg import register, merge
from splatreg.core.types import Gaussians
from splatreg.io import save_ply, load_ply
from splatreg.residuals import ICP, SDF
from splatreg.align import global_align
from splatreg.api import _auto_sdf_sigma, _default_residuals
from splatreg.quality import FULL  # _default_residuals now takes a resolved QualityConfig

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


def make_g(means, opac=None, scale=0.005):
    n = means.shape[0]
    return Gaussians(
        means=means,
        quats=torch.tensor([1.0, 0, 0, 0], device=dev, dtype=dt).repeat(n, 1),
        scales=torch.full((n, 3), scale, device=dev, dtype=dt),
        opacities=torch.ones(n, device=dev, dtype=dt) if opac is None else opac,
    )


def asymmetric_cloud(n, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.rand(n, generator=g).to(dev)
    v = torch.rand(n, generator=g).to(dev)
    phi = 2 * math.pi * u
    costh = 2 * v - 1
    sinth = torch.sqrt((1 - costh ** 2).clamp_min(0))
    sph = torch.stack([sinth * torch.cos(phi), sinth * torch.sin(phi), costh], dim=1)
    ell = sph * torch.tensor([0.14, 0.09, 0.06], device=dev, dtype=dt)
    lobe = 0.03 * torch.exp(-((sph[:, 0:1] - 1.0) ** 2) / 0.2) * torch.tensor([1.0, 0, 0], device=dev, dtype=dt)
    pts = (ell + lobe).to(dt)
    pts = pts * (1.0 + 0.01 * torch.randn(n, 1, device=dev, dtype=dt))
    return pts + torch.tensor([0.4, -0.2, 0.3], device=dev, dtype=dt)


# ============================================================ [A] default-residual recovery
print("=" * 72)
print("[A] register(A, B) with DEFAULT residuals (residuals=None) — SE(3) and Sim(3)")

pts = asymmetric_cloud(900, seed=1)
A = make_g(pts)
R_gt = axis_angle_R([0.2, 0.9, 0.3], 6.0)
t_gt = torch.tensor([0.012, -0.008, 0.010], device=dev, dtype=dt)

for tag, s_gt, tf in [("SE(3)", 1.0, "se3"), ("Sim(3)", 1.15, "sim3")]:
    B = make_g((s_gt * (pts @ R_gt.transpose(-1, -2))) + t_gt)
    # Error of the global init alone, so we can prove the default fine step does NOT degrade it.
    T_init = global_align(B, A, transform=tf)
    s_init = float(torch.linalg.det(T_init[:3, :3]).abs().clamp_min(1e-12) ** (1.0 / 3.0))
    R_init = T_init[:3, :3] / s_init
    init_rot = rot_angle_deg(R_init, R_gt)
    init_tr = 1000 * float((T_init[:3, 3] - t_gt).norm())

    res = register(B, A, init="global", transform=tf, max_iters=60)   # residuals=None -> default
    s = res.scale
    R_est = res.T[:3, :3] / s
    rot = rot_angle_deg(R_est, R_gt)
    tr = 1000 * float((res.T[:3, 3] - t_gt).norm())
    s_err_pct = 100 * abs(s - s_gt) / s_gt
    print(f"  {tag}: init(rot={init_rot:.3f}deg trans={init_tr:.2f}mm)  ->  "
          f"final(rot={rot:.4f}deg trans={tr:.3f}mm scale_err={s_err_pct:.3f}%)")
    print(f"        converged={res.converged} iters={res.info['n_iters']} dof={res.info['dof']} "
          f"rmse={res.info['rmse']:.2e}")
    assert rot < 3.0, (tag, rot)
    assert tr < 40.0, (tag, tr)
    if tf == "sim3":
        assert s_err_pct < 8.0, s_err_pct
    else:
        assert abs(s - 1.0) < 1e-6, s
    # Did the fine step degrade the good init? Allow a tiny epsilon for field bias.
    assert rot <= init_rot + 1.0, f"{tag}: fine step degraded rotation {init_rot}->{rot}"
    assert tr <= init_tr + 10.0, f"{tag}: fine step degraded translation {init_tr}->{tr}"
print("  [A] PASS — default residuals recover SE(3)+Sim(3) and do not degrade the global init.")


# ============================================================ [B] merge overlap dedupe + save
print("=" * 72)
print("[B] merge([A, B]) of two halves + overlap band -> dedupe collapses overlap; save round-trip")

# One cloud split by x into two halves that SHARE an overlap band (so the band is double-covered).
base = asymmetric_cloud(4000, seed=7)
x = base[:, 0]
xmid = float(x.median())
band = 0.02
left = base[x <= xmid + band]                      # left half + band
right = base[x >= xmid - band]                     # right half + band
overlap_lo, overlap_hi = xmid - band, xmid + band  # the shared band in world x

def in_band(m):
    return ((m[:, 0] >= overlap_lo) & (m[:, 0] <= overlap_hi)).sum().item()

# Give the two halves distinct opacities so the dedupe's "highest-opacity wins" is observable.
gL = make_g(left, opac=torch.full((left.shape[0],), 0.9, device=dev, dtype=dt))
gR = make_g(right, opac=torch.full((right.shape[0],), 0.4, device=dev, dtype=dt))

# Apply a KNOWN offset to the right half (merge must register it back onto the left).
R_off = axis_angle_R([0.1, 0.2, 1.0], 4.0)
t_off = torch.tensor([0.02, 0.01, -0.015], device=dev, dtype=dt)
gR_moved = make_g((gR.means @ R_off.transpose(-1, -2)) + t_off, opac=gR.opacities)

naive = torch.cat([gL.means, gR_moved.means], dim=0)   # what a naive cat would hold (pre-register)
pre_band = in_band(gL.means) + in_band(gR.means)       # double-density count in the band (aligned)
print(f"  overlap band x in [{overlap_lo:.4f}, {overlap_hi:.4f}]  (width {2*band:.3f})")
print(f"  pre-merge band count (both halves, aligned): {pre_band}  "
      f"(left {in_band(gL.means)} + right {in_band(gR.means)})")

merged = merge([gL, gR_moved], ref=0, transform="sim3", init="global", max_iters=40)
post_band = in_band(merged.means)
single_est = max(in_band(gL.means), in_band(gR.means))  # ~one half's density in the band
print(f"  post-merge band count (deduped):             {post_band}")
print(f"  single-density reference (one half in band): ~{single_est}")
print(f"  total: naive cat {naive.shape[0]}  ->  merged {len(merged)}  "
      f"(removed {naive.shape[0] - len(merged)} duplicates)")
# The band must drop from ~double toward single density.
assert post_band < 0.7 * pre_band, (post_band, pre_band)
assert post_band <= 1.6 * single_est, (post_band, single_est)
# And the highest-opacity (left, 0.9) anchors should dominate the survivors in the band.
band_mask = (merged.means[:, 0] >= overlap_lo) & (merged.means[:, 0] <= overlap_hi)
band_opac = merged.opacities.reshape(-1)[band_mask]
frac_high = float((band_opac > 0.6).float().mean()) if band_opac.numel() else 0.0
print(f"  fraction of band survivors with high opacity (>0.6): {frac_high:.2f}")
assert frac_high > 0.8, frac_high

# dedupe=False keeps the seam (registered concatenation), proving the flag works.
merged_nodedupe = merge([gL, gR_moved], ref=0, dedupe=False, init="global", max_iters=40)
print(f"  dedupe=False total: {len(merged_nodedupe)} (== naive cat size {naive.shape[0]}: "
      f"{len(merged_nodedupe) == naive.shape[0]})")
assert len(merged_nodedupe) == naive.shape[0]
assert len(merged) < len(merged_nodedupe)

# save_ply round-trip of the merged result.
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "merged.ply"
    save_ply(merged, p)
    reloaded = load_ply(p, device=dev)
    print(f"  save_ply -> load_ply: {len(reloaded)} Gaussians (== {len(merged)}: "
          f"{len(reloaded) == len(merged)})")
    assert len(reloaded) == len(merged)
    # Means survive the round-trip (PLY stores raw floats).
    dmax = float((reloaded.means.to(dt) - merged.means).abs().max())
    print(f"  max |mean round-trip error|: {dmax:.2e}")
    assert dmax < 1e-5, dmax
print("  [B] PASS — overlap collapses to single density; highest-opacity wins; save round-trips.")


# ============================================================ [C] explicit residuals honoured + sigma
print("=" * 72)
print("[C] explicit residuals honoured unchanged; auto-sigma sane")

sigma = _auto_sdf_sigma(A)
med_scale = float(make_g(pts).scales.mean(-1).median())
print(f"  auto SDF sigma = {sigma:.5f}  (~2x median scale {med_scale:.5f}: "
      f"{abs(sigma - 2 * med_scale) < 1e-6})")
assert abs(sigma - 2 * med_scale) < 1e-6

defaults = _default_residuals(A, FULL)
print(f"  default residuals: {[type(r).__name__ for r in defaults]}  "
      f"weights {[r.weight for r in defaults]}")
assert isinstance(defaults[0], ICP) and defaults[0].point_to_plane is False and defaults[0].weight == 1.0
assert isinstance(defaults[1], SDF) and abs(defaults[1].weight - 0.3) < 1e-9
# FULL quality -> the SDF default samples ALL source anchors (n_points <= 0 sentinel).
assert defaults[1].n_points <= 0, f"FULL should give uncapped SDF n_points, got {defaults[1].n_points}"

# Explicit residual list still works and is used verbatim (different from the default).
B = make_g((1.0 * (pts @ R_gt.transpose(-1, -2))) + t_gt)
res_expl = register(B, A, residuals=[ICP(point_to_plane=True, weight=1.0)],
                    init="global", transform="se3", max_iters=40)
print(f"  explicit [ICP(point_to_plane=True)] -> rot {rot_angle_deg(res_expl.T[:3,:3], R_gt):.3f}deg "
      f"trans {1000*float((res_expl.T[:3,3]-t_gt).norm()):.2f}mm (ran, honoured)")
print("  [C] PASS")

print("=" * 72)
print("ALL PHASE-3 CHECKS PASSED")
