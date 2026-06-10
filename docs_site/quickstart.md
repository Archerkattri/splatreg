# Quickstart

## Install

```bash
pip install splatreg            # torch + numpy only
```

Optional extras:

```bash
pip install "splatreg[render]"  # gsplat, camera localization + photometric residual
pip install "splatreg[mac]"     # networkx, the init="mac" maximal-clique seed
pip install "splatreg[pypose]"  # PyPose solver backend
pip install "splatreg[theseus]" # Theseus solver backend
```

Development install:

```bash
git clone https://github.com/Archerkattri/splatreg.git
cd splatreg
pip install -e ".[test]"
python -m pytest tests/ -q
```

## Register two splats

```python
from splatreg import register
from splatreg.io import load_ply

target = load_ply("scan_a.ply")   # the reference, stays fixed
source = load_ply("scan_b.ply")   # gets aligned onto the target

result = register(target, source, transform="sim3")   # or "se3" for rigid
```

`result` is a [`RegisterResult`][splatreg.core.types.RegisterResult]:

```python
result.T          # (4, 4) transform mapping source -> target ([[s*R, t], [0, 1]] for sim3)
result.scale      # recovered scale s (1.0 for se3)
result.converged  # solver convergence flag
result.info       # diagnostics: rmse, n_iters, cost, ambiguous/confidence (feature inits)
```

Builtin-LM solves also expose the pose uncertainty for pose-graph / loop-closure weighting:

```python
result.info["information"]   # undamped JᵀWJ at the optimum: (6,6) se3 / (7,7) sim3
result.info["covariance"]    # its scaled inverse; None when singular, never faked
```

The default `init="fast"` (FPFH + GPU-batched RANSAC seed, ~17 ms) suits objects and
full-overlap captures. For real metre-scale scans use `init="robust"` or `init="learned"`;
see [Init modes](init-modes.md).

If the shape under-constrains the pose (rotational symmetry, texture-carried detail), add the
opt-in photometric stage (geometric residuals can't see color, this stage can):

```python
result = register(target, source, transform="se3",
                  refine="photometric")    # needs `pip install "splatreg[render]"`
```

Measured: on a rotation-symmetric colored sphere the geometric solve *worsens* 6.0°→11.2°
while the photometric stage lands 2.2° (0.36° with the real gsplat rasterizer); on
dense-overlap scans it is neutral. [When & why](photometric.md).

## Align without merging

To keep the scans as separate files (just registered into one frame), bake the recovered
transform into the source with [`apply_transform`][splatreg.api.apply_transform]. Colour is
handled fully: DC is rotation-invariant, quats are composed, and the higher-order SH bands
are Wigner-rotated so view-dependent colour turns with the splat
([PLY interop](ply-interop.md)):

```python
from splatreg import apply_transform
from splatreg.io import save_ply

aligned = apply_transform(source, result.T, result.scale)
save_ply(aligned, "source_aligned.ply")
# target.ply stays untouched; the two files now sit registered in any viewer.
```

Same thing from the shell: `splatreg align target.ply source.ply -o source_aligned.ply`.

## Merge + dedupe

```python
from splatreg import merge
from splatreg.io import save_ply

fused = merge([target, source])     # registers everything onto splats[ref] (default 0),
save_ply(fused, "fused.ply")        # fuses, and dedupes the double-density overlap
```

`merge` is **not a naive cat**: each non-reference splat is registered (Sim(3) by default, so
scale differences between captures are absorbed), the transform is baked into its
means/quats/scales, and the overlap region is deduped (voxel-grid by default, `"knn"`
available) so the seam collapses to single density.

## Bring your own tensors (no PLY)

Any 3DGS framework works, pass gsplat-style tensors directly:

```python
from splatreg.io import from_gsplat, to_gsplat

g = from_gsplat(means, quats, scales, opacities, colors)   # wraps, no copy
out = gsplat.rasterization(viewmats=..., Ks=..., width=W, height=H, **to_gsplat(g))
```

## Object pose + camera localization

```python
from splatreg import estimate_object_pose, localize_camera, coarse_localize_camera

# 6-DoF pose of a known object splat in a new observation (ADD / ADD-S / AUC out of the box)
result = estimate_object_pose(model_splat, observation_splat)

# Refine a camera pose against a scene splat through gsplat's differentiable rasteriser
result = localize_camera(scene_splat, frame, init_T_WC=T_init)   # needs splatreg[render]

# Wide-baseline / prior-free coarse seed (CPU-only, no rasteriser)
T_coarse = coarse_localize_camera(scene_splat, frame)
```

## Real-time tracking

```python
from splatreg import Tracker

tracker = Tracker(target, residuals=[...])   # fixed target, warm-started across frames
for frame in stream:
    result = tracker.track(frame)            # ~17 ms/frame
```

## The Gaussian-SDF field, standalone

splatreg's flagship residual is a reusable primitive: a smooth signed-distance field derived
directly from the target Gaussians (no mesh, no marching cubes):

```python
from splatreg.geometry.gaussian_sdf import gaussian_sdf, gaussian_sdf_grad

sdf, normal = gaussian_sdf(target, query_points, sigma=0.02)
sdf, grad   = gaussian_sdf_grad(target, query_points, sigma=0.02)  # exact closed-form grad
```

## Know the edges

!!! warning "Honest limitations"
    - **Overlap ≤ 40% is genuinely ambiguous**, the disambiguating geometry is physically
      absent. splatreg flags these (`result.info["ambiguous"]` / `["confidence"]`) instead of
      silently wrong-posing. `merge` is designed for high-overlap captures.
    - **Scale is unobservable under thin overlap** (~20% shared geometry): the Sim(3) scale
      valley is flat, no algorithm can recover what the geometry doesn't carry.
    - **On rigid SE(3), plain ICP is far cheaper** and reaches the same success on easy cases;
      the SDF residual buys scale + implicit-field robustness at a real compute cost. Use
      `Tracker` for the warm-start real-time path.
