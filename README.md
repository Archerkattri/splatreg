<div align="center">

<img src="https://raw.githubusercontent.com/Archerkattri/splatreg/main/assets/banner.png" alt="splatreg" width="680">

# splatreg

### Register Gaussian splats — align & merge two 3DGS scans into one SE(3)/Sim(3) frame.

[![PyPI](https://img.shields.io/pypi/v/splatreg)](https://pypi.org/project/splatreg/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![gsplat](https://img.shields.io/badge/inverse%20of-gsplat-ee4c2c.svg)](https://github.com/nerfstudio-project/gsplat)

<img src="assets/registration_demo.png" alt="splatreg before/after registration" width="92%">

</div>

---

## Is this for you?

- **Two 3DGS scans of the same scene / object that need to be merged** — `register` + `merge` finds the rigid or similarity transform and fuses them into one deduped `.ply`, no manual gizmo needed.
- **Object pose estimation against a known splat** — `estimate_object_pose` recovers the SE(3) pose between a reference model splat and a new observation (ADD / ADD-S / AUC out of the box).
- **Camera localization inside a known splat** — `localize_camera` places a new camera into a scene splat without retraining; `coarse_localize_camera` seeds it prior-free from a silhouette sweep.

Works with any 3DGS framework — gsplat, Nerfstudio, INRIA, custom — as long as you can pass Gaussian means and covariances as PyTorch tensors. Pure PyTorch — no meshing, no CUDA extension, no point-cloud detour.

---

## Install

```bash
pip install splatreg
```

```bash
# editable / dev
git clone https://github.com/Archerkattri/splatreg.git
cd splatreg
pip install -e ".[test]"
```

## Quickstart

```python
from splatreg.api import register, merge

# Align `source` onto `target` (both are Gaussians objects: .means, .covs, .opacities tensors).
result = register(target, source, transform="sim3")   # init="fast" by default (~17 ms)
# Real metre-scale scans: init="robust" (FPFH+RANSAC) or init="learned" (GeoTransformer, best accuracy)
print(result.T)          # recovered 4×4 similarity [[s·R, t], [0, 1]] — maps source → target
print(result.scale)      # recovered scale s  (1.0 for transform="se3")
print(result.converged)  # solver convergence flag

# Merge + dedupe a list of splats into one fused splat
fused = merge([source, target], transform="sim3")
```

Object pose and camera localization:

```python
from splatreg import estimate_object_pose, localize_camera, coarse_localize_camera

# Object pose: recover T_SO between a model splat and an observation
result = estimate_object_pose(model_splat, observation_splat)

# Camera localization: refine camera pose through gsplat's differentiable rasteriser
result = localize_camera(scene_splat, frame, init_T_WC=T_init)
# Wide-baseline / prior-free: coarse seed from silhouette sweep (CPU-only, no rasteriser)
T_coarse = coarse_localize_camera(scene_splat, frame)
```

The Gaussian-SDF field standalone:

```python
from splatreg.geometry.gaussian_sdf import gaussian_sdf, gaussian_sdf_grad
sdf, normal = gaussian_sdf(target, query_points, sigma=0.02)       # signed distance + surface normal
sdf, grad   = gaussian_sdf_grad(target, query_points, sigma=0.02)  # signed distance + exact ∇_p d
```

---

## Results

| | **splatreg** | reference |
|---|---|---|
| **Real-splat merge** (real 103k-Gaussian capture) | Chamfer **10.3→2.0 mm (5.1×)** · overlap **0.03→0.67 (22×)** | naive concat |
| **vs splat competitors** (real splat, known GT Sim3) | **5.2°** (SE3) · recovers scale (Sim3) | splatalign 15.3° · GaussianSplattingRegistration 36.3° |
| **Sim(3) scale estimation** | ✅ native | ✗ none of these do it |
| **Object pose (YCB-CAD, 14 models × 4 poses)** | ADD-S AUC **0.995**, 100% < 2 cm | — |
| **Camera localization (real splat, known perturbation)** | median **5°/10 mm → 0.11°/1.35 mm**, 11/12 converged | — |
| **Official 3DMatch recall** (1279 pairs, Choi/Zeng protocol) | **91.5%** mean · 93.5% pooled | GeoTransformer ~92% · Open3D ~77% |
| **Official 3DLoMatch** (hard, 10–30% overlap) | 72.5% mean · **74.4%** pooled | GeoTransformer ~74% · Open3D ~20% |
| **Registration speed** | **~17 ms** (fast) · 104 ms (learned) | GeoTransformer ~50 ms · Open3D 142 ms |

splatreg is the **only library** that registers native Gaussian splats with SE(3)+**Sim(3)** behind a closed-form-Jacobian Gaussian-SDF. It beats both splat-specific tools outright (5.2° vs 15.3° / 36.3°) and matches GeoTransformer on official 3DMatch while adding the Sim(3) scale DoF they lack.

### Init modes — trade speed ↔ robustness

| `init=` | what | when |
|---|---|---|
| `"fast"` *(default)* | FPFH + GPU-batched RANSAC seed → closed-form LM | objects / full-overlap, **~17 ms** |
| `"robust"` | Open3D FPFH+RANSAC seed → splatreg refine + scale | real metre-scale scans |
| `"learned"` | pretrained GeoTransformer seed → splatreg refine + scale | best accuracy on real scans |
| `"global"` | blind super-Fibonacci SO(3) sweep | robust fallback, any rotation |

---

## How it works

**splatreg takes two splats and finds the rigid (SE(3)) or similarity (Sim(3), +scale) transform that aligns them** — then optionally merges + dedupes them into one. It is the missing *registration* half of the Gaussian-splatting toolchain — the splat-to-splat alignment SuperSplat / INRIA / geospatial users keep asking for, where today's tooling punts to a manual gizmo.

The pipeline is two stages:

```mermaid
flowchart LR
    A["splat A<br/>(target)"]:::s --> G
    B["splat B<br/>(source)"]:::s --> G
    G["<b>Global aligner</b><br/>super-Fibonacci SO(3) seeds<br/>+ batched trimmed ICP<br/><i>(or FPFH / learned)</i>"]:::g --> L
    L["<b>Levenberg–Marquardt</b><br/>multi-residual:<br/>ICP + Gaussian-SDF<br/>SE(3) / Sim(3)"]:::l --> T["T*  (4×4)<br/>+ merge / dedupe"]:::o
    classDef s fill:#e8f6f8,stroke:#17becf,color:#0b3d44;
    classDef g fill:#fff1ee,stroke:#ff6b5b,color:#5a1a12;
    classDef l fill:#eef7ee,stroke:#2e8b57,color:#143d22;
    classDef o fill:#f3eefc,stroke:#7d52c7,color:#2c1654;
```

1. **Global init** — a coarse pose from a dense super-Fibonacci rotation sweep + batched trimmed ICP (no local-minimum trap), with optional FPFH+RANSAC and learned (GeoTransformer) seeds for harder real scans.
2. **Refinement** — a from-scratch Levenberg–Marquardt core over ICP (point-to-point / point-to-plane) *and* splatreg's flagship **Gaussian-SDF** residual, solving the full SE(3) or Sim(3) tangent.

### The Gaussian-SDF residual

No competitor packages this. splatreg derives a smooth **signed-distance field directly from the target Gaussians** — no mesh, no marching cubes — and drives registration by it:

```
w_i(p) = exp(−‖p − q_i‖² / 2σ²)              # Gaussian kernel weight per anchor
q̃(p)   = Σ w_i q_i / Σ w_i                    # kernel-weighted centroid
ñ(p)   = Σ w_i n_i / ‖Σ w_i n_i‖              # kernel-weighted surface normal
d(p)   = (p − q̃(p)) · ñ(p)                    # signed distance — the residual
```

`d(p)` vanishes exactly when source points land on the target surface. It has a **closed-form, audited Jacobian** and is a reusable primitive: `gaussian_sdf(splat, points, sigma=...) → (sdf, normal)`.

---

## Validation

Every number is reproducible; full record in [`RESULTS.md`](RESULTS.md).

```bash
python -m pytest tests/ -q                        # 72 passing
python tests/test_jacobians.py                    # analytic vs numerical Jacobian audit
SPLATREG_DEVICE=cuda python examples/validate_recovery.py --device cuda   # 36/36 recovery
SPLATREG_DEVICE=cuda python benchmarks/robustness_bench.py --device cuda
python examples/merge_demo.py                     # real-splat merge demo
```

---

## Limitations

splatreg is honest about its edges (full detail in [`RESULTS.md`](RESULTS.md)):

- **Heavy overlap (≤ 40%) is genuinely ambiguous.** At keep ≤ 40% the rotation-disambiguating geometry is physically absent — even the true pose doesn't seat cleanly. The aligner flags these honestly (`result.info['ambiguous']` / `['confidence']`) and never silently wrong-poses. `merge` and `track` are designed for high-overlap captures.
- **Scale is unobservable under thin overlap.** Under ~20% shared geometry the Sim(3) scale residual valley is flat — the golden-section line-search tightens scale on its own objective but cannot recover what the geometry doesn't carry. `merge` is reliable for high-overlap captures.
- **Cost on rigid SE(3).** Plain ICP reaches the same SE(3) success and is far faster; the SDF residual buys scale + implicit-field robustness at a real compute cost. Use `track()` (~17 ms/frame) for the warm-start real-time path.

## License & layout

BSD 3-Clause — permissive, composes with the gsplat / Theseus / GTSAM ecosystem. `splatreg/` — library (`api`, `align`, `align_features`, `bundle`, `spatial_index`, `core/lie`, `geometry/gaussian_sdf`, `residuals/`, `solvers/lm`). `tests/` · `benchmarks/` · `examples/`. Full validation record: [`RESULTS.md`](RESULTS.md).
