# PLY interop

splatreg reads and writes the **standard 3D Gaussian Splatting PLY** — the layout INRIA's
reference implementation (`graphdeco`) defined and the whole ecosystem adopted:

```text
x y z   f_dc_0..2   f_rest_0..M   opacity   scale_0..2   rot_0..3
```

That makes the workflow framework-agnostic: train anywhere, register with splatreg, view
anywhere.

| Producer / consumer | Format | Works with splatreg |
|---|---|---|
| INRIA `gaussian-splatting` (`point_cloud.ply`) | standard PLY, SH degree 3 | yes — bit-for-bit round-trip |
| gsplat / Nerfstudio **splatfacto** (`ns-export gaussian-splat`) | standard PLY | yes — bit-for-bit round-trip |
| SuperSplat (import & export) | standard PLY (+ a *compressed* `.ply` variant) | yes — export **uncompressed** PLY from SuperSplat |
| PlayCanvas SplatTransform | standard PLY in/out | yes — pipe either way |
| antimatter15 `.splat`, `.ksplat`, `.spz` | packed binary variants | no — convert to PLY first (SuperSplat or SplatTransform do this) |

ASCII-PLY files need the optional permissive parser: `pip install plyfile` (3DGS exporters
write binary, so this rarely comes up).

## Raw parameters: what's actually inside the file

The standard PLY stores **raw (pre-activation)** values. `load_ply` / `save_ply` keep them
raw — what comes out is what went in, no silent re-encoding of geometry:

| PLY property | meaning | splatreg `Gaussians` field |
|---|---|---|
| `x y z` | centre | `means` |
| `opacity` | **pre-sigmoid logit** | `opacities` (raw) |
| `scale_0..2` | **log**-scales (pre-`exp`) | `scales`, with `log_scales=True` |
| `rot_0..3` | quaternion, **wxyz**, possibly un-normalised | `quats` |
| `f_dc_0..2` | SH degree-0 (DC) colour coefficient | `colors[:, 0, :]` |
| `f_rest_0..M` | higher-order SH coefficients, **channel-major** | `colors[:, 1:, :]` |

Two details every hand-rolled loader trips over, handled at splatreg's PLY boundary:

1. **SH coefficient order.** The PLY stores `f_rest` *channel-major* (all R coefficients,
   then all G, then all B); gsplat's internal tensors are *coefficient-major, channel-last*
   `(N, K, 3)`. `load_ply`/`save_ply` apply that transpose at the boundary, so a splat
   written by gsplat reloads bit-for-bit and vice-versa.
2. **Raw vs activated.** `opacity` is a logit and `scale_*` are log-scales. splatreg keeps
   them raw through registration and merge (geometry math uses the linear values internally
   via `log_scales`); `to_gsplat()` hands the rasteriser linear scales.

Colour conventions on the `Gaussians` side: a 3-D `colors` tensor `(N, K, 3)` is SH
(coefficient-major); a 2-D `(N, 3)` tensor is treated as **linear RGB** by `save_ply` and
encoded to a DC-only SH (`(rgb - 0.5) / C0`). If you build splats from your own tensors,
hand SH in as `(N, K, 3)` — including `K == 1` — to keep coefficients untouched.

!!! note "Fixed in v1.1: DC-only round-trip"
    `load_ply` of a DC-only file used to return the **raw SH-DC coefficients in the RGB
    slot**, so a following `save_ply` re-applied the RGB→DC encoding and colors drifted on
    every load→save cycle. DC-only loads now decode to true RGB `(N, 3)`, making
    load→save→load lossless (and full-SH files were and remain bit-exact). Regression-locked
    in [`tests/test_io_roundtrip_dc.py`](https://github.com/Archerkattri/splatreg/blob/main/tests/test_io_roundtrip_dc.py).

## What happens to a splat under a recovered transform

When `splatreg align` / `merge` bakes a recovered Sim(3) `T = [[s·R, t], [0, 1]]` into a
splat, each parameter needs its own, different update — this is where naive merges go wrong:

```text
means'  = s · (R @ means) + t        # the homogeneous point transform
quats'  = quat(R) ⊗ quats            # compose R onto each anchor's orientation
scales' = s · scales                 # in log space: log_scales + log s
SH'     = D(R) @ SH                  # real-SH Wigner-D rotation of the colour bands
```

What splatreg gets right (each one a classic naive-merge bug):

- **Covariance orientation.** Every Gaussian is an *anisotropic ellipsoid*; rotating only
  the means leaves every ellipsoid pointing the old way (visible as a "brushed" / streaky
  surface). splatreg composes `quat(R)` onto every anchor quaternion (Hamilton product,
  wxyz), with `R` first de-scaled out of the Sim(3) block so the quaternion stays unit.
- **Scale under Sim(3).** The similarity scales each anchor's extent: linear scales are
  multiplied by `s`; log-stored scales get `+ log s` — the `log_scales` flag is preserved
  either way, so the written PLY stays standard.
- **Raw opacity.** Logits pass through untouched — no double-sigmoid.
- **DC colour.** The degree-0 SH basis function is constant over directions, so the DC
  coefficient is **rotation-invariant**: carrying `f_dc` through unchanged is exactly
  correct, not an approximation.

### The spherical-harmonics rotation detail

Here is the subtle one. View-dependent colour is stored as SH coefficients **in world
space**. When you rotate the splat by `R`, the appearance field should rotate with it — and
for SH that means each degree-ℓ band of coefficients must be mixed by the corresponding
**Wigner rotation matrix** `D^ℓ(R)` (a 3×3 rotation of the degree-1 triple, a 5×5 for
degree 2, 7×7 for degree 3). Degree 0 (DC) is invariant; the higher bands are not.

Almost every merge pipeline skips this — including manual gizmo workflows in most editors —
because the coefficients still *look* plausible: the diffuse (DC) term dominates, and the
error only shows as view-dependent sheen/specular highlights that stay "stuck" in the old
world orientation while the geometry rotates away under them.

**What splatreg does (v1.2+):** every band is handled exactly. The DC band is invariant
(untouched); every higher-order `f_rest` band is multiplied by its real-basis Wigner-D block,
built for any degree by the Ivanic–Ruedenberg recurrence (*J. Phys. Chem. A* 100 (1996) 6342,
with the 1998 erratum corrections) in `splatreg.sh`. The blocks are produced directly in the
3DGS/gsplat sign convention (the `(-y, +z, -x)` degree-1 basis), so the rotated coefficients
drop straight back into a standard PLY. `apply_transform`, `merge`, and the `align` CLI all
route through this — view-dependent sheen now turns **with** the splat instead of staying
stuck in the old capture frame.

```python
from splatreg.sh import rotate_sh, sh_rotation_matrix

D = sh_rotation_matrix(R, n_coeffs=16)   # (16, 16) block-diagonal, degree 3
g.colors = rotate_sh(g.colors, R)        # (N, K, 3) SH stack, rotated; DC untouched
```

The math is locked by renderer-free tests against an independent hand-coded 3DGS SH basis
evaluator ([`tests/test_sh_rotation.py`](https://github.com/Archerkattri/splatreg/blob/main/tests/test_sh_rotation.py)):
evaluating the rotated coefficients at `d` equals evaluating the originals at `R⁻¹d` to
< 1e-5 over random rotations; the degree-1 block equals its signed-permutation closed form;
`D(R1·R2) = D(R1)·D(R2)`; identity maps to identity; and the rotated stack round-trips
through `save_ply`/`load_ply` exactly.

## Inspecting a file

`splatreg info` prints the layout it found — count, bounds, SH degree, raw-opacity range,
log/linear scale stats:

```text
$ splatreg info scan.ply
file      : scan.ply
gaussians : 103482
bounds min: [-1.2034, -0.8211, -0.4310]
bounds max: [ 1.1098,  0.7990,  1.2247]
extent    : [ 2.3132,  1.6201,  1.6557]
colors    : SH degree 3 (16 coefficients per channel)
opacity   : raw [-7.214, 12.331]  sigmoid mean 0.842
scales    : log-stored, linear median 0.00521  max 0.19883
```

## Nerfstudio recipe

No plugin needed — splatfacto's export is the standard PLY:

```bash
ns-export gaussian-splat --load-config outputs/.../config.yml --output-dir exports/a
ns-export gaussian-splat --load-config outputs/.../config.yml --output-dir exports/b
splatreg merge exports/a/splat.ply exports/b/splat.ply -o fused.ply
```

`fused.ply` opens directly in SuperSplat / the PlayCanvas viewer / any standard 3DGS viewer.
