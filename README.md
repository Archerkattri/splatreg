<div align="center">

<img src="https://raw.githubusercontent.com/Archerkattri/splatreg/main/assets/banner.png" alt="splatreg" width="680">

# splatreg

### Register Gaussian splats — align & merge two 3DGS scans into one SE(3)/Sim(3) frame.

[![PyPI](https://img.shields.io/pypi/v/splatreg)](https://pypi.org/project/splatreg/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20618389-1682D4.svg)](https://doi.org/10.5281/zenodo.20618389)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-archerkattri.github.io%2Fsplatreg-teal.svg)](https://archerkattri.github.io/splatreg/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Archerkattri/splatreg/blob/main/examples/splatreg_quickstart.ipynb)
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

## What's new in v1.2

- **Spherical harmonics rotate WITH the splat** — when a recovered transform is baked in (`apply_transform`, `merge`, the `align` CLI), the higher-order SH bands (`f_rest`) are now mixed by the real-basis **Wigner-D** matrix (Ivanic–Ruedenberg recurrence, 1996 + the 1998 erratum, built directly in the 3DGS sign convention — `splatreg.sh`). Every other splat tool we know of leaves the view-dependent lobes stuck in the old capture frame after a registration, so glossy highlights point the wrong way; splatreg is, to our knowledge, the only splat registrar that rotates view-dependent colour correctly. *Evidence (renderer-free math tests vs an independent hand-coded 3DGS basis evaluator, [`tests/test_sh_rotation.py`](tests/test_sh_rotation.py)):* rotated coefficients evaluated at `d` equal the originals at `R⁻¹d` to < 1e-5 over random rotations up to degree 3; degree-1 equals its signed-permutation closed form; `D(R₁R₂) = D(R₁)D(R₂)`; rotated stacks round-trip PLY exactly.
- **Photometric exposure compensation (default ON)** — independently-captured pairs disagree on exposure/white balance; the refine stage now alternates a bounded per-channel gain/bias fit on the rendered source (gain ∈ [0.5, 2.0]) with the pose LM. *Measured:* a ×1.3 + 0.05 source tint absorbs into the Sim(3) **scale** without it (scale err 0.10% → **3.99%**); with it the tinted pair recovers **0.47%** and the fitted gain lands at ≈ 1/1.3 — harmless on clean pairs (0.01%). [Details](https://archerkattri.github.io/splatreg/photometric/).
- **Coarse-to-fine render ladder** — `refine_kwargs=dict(ladder=(96, 160, 256))` breaks the fixed-resolution accuracy floor; each rung warm-starts the next. *Measured:* from a 6° offset a cold 96 px rung stalls at **5.61°**, the 32→64→96 ladder lands **2.55°** at equal per-stage budget.
- **Pose covariance for pose graphs** — builtin-LM results now expose `info["information"]` (the undamped `JᵀWJ` at the final accepted linearisation; 6×6 SE(3) / 7×7 Sim(3)) and `info["covariance"]` (`σ̂²(JᵀWJ)⁻¹`; `None` if singular — never faked). *Tested:* symmetry/SPD on well-constrained solves, 2× noise → looser covariance, singular → `None` ([`tests/test_pose_covariance.py`](tests/test_pose_covariance.py)).
- **`validate_recovery.py --fast`** — a CPU smoke preset of the recovery harness (same protocol/gates, smaller budget: 1 seed × the grid corners, 400 anchors, 30 iters). *Measured (CPU, `OMP_NUM_THREADS=2`):* **6/6 cells within gate in 41 s wall** — worst rot err 0.16°, worst scale err 0.14%.

## What's new in v1.1

- **`refine="photometric"`** — opt-in PhotoReg-style ([arXiv 2410.05044](https://arxiv.org/abs/2410.05044)) splat-to-splat photometric stage after the geometric solve, for the poses geometry can't see (symmetry / texture-only DoF) — no real images needed. *Measured:* on a rotation-symmetric colored sphere, geometric registration **worsens** 6.0°→11.2° while the photometric stage lands **2.2°** (real gsplat rasterizer: 5°/7 mm → **0.36°/0.5 mm** in ~1.1 s); on a dense-overlap real 103k-Gaussian pair it is neutral (+1.7 s) because geometry already pins the pose — so it ships opt-in. 21 tests + bench: [when & why](https://archerkattri.github.io/splatreg/photometric/) · [recorded runs](benchmarks/photometric_refine_results.md).
- **`splatreg` CLI** — `align` / `merge` / `info` from the shell, standard 3DGS PLY in/out (the SplatTransform-style workflow: 3DGS practitioners are CLI-first). *Measured:* the recorded `align` run takes a source from **154 mm Chamfer off the target to 0.05 mm** with no Python written. 10 end-to-end tests: [CLI guide](https://archerkattri.github.io/splatreg/cli/).
- **DC-only PLY round-trip fix** — `load_ply` used to return raw SH-DC values in the RGB slot, so a following `save_ply` double-encoded them and colors drifted every load→save cycle; DC-only loads now return true RGB and round-trip losslessly (full-SH round-trip stays bit-exact). Regression-locked in [`tests/test_io_roundtrip_dc.py`](tests/test_io_roundtrip_dc.py).

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

From the shell — `pip install` puts a `splatreg` command on your PATH (standard 3DGS PLY in/out,
so it composes with SuperSplat / gsplat / Nerfstudio exports; see the
[CLI guide](https://archerkattri.github.io/splatreg/cli/)):

```bash
splatreg align target.ply source.ply -o aligned.ply    # register + write the aligned source
splatreg merge a.ply b.ply -o fused.ply                # register + fuse + dedupe N splats
splatreg info x.ply                                    # count / bounds / SH degree / stats
```

In Python:

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
| `"mac"` | MAC maximal-clique consensus (Zhang et al. CVPR 2023) over the correspondences → weighted SVD → refine | outlier-heavy / multi-consensus correspondence sets |
| `"global"` | blind super-Fibonacci SO(3) sweep | robust fallback, any rotation |

`init="mac"` reimplements the MAC hypothesis generator (rigidity compatibility graph with the SC² second-order weighting → maximal cliques → weighted SVD per clique, with explicit correspondence/degree/clique-count/time caps) in pure torch + networkx (`pip install "splatreg[mac]"`). *Evidence ([`tests/test_mac.py`](tests/test_mac.py), CPU synthetic):* matches the fast-init RANSAC engine at 30/60/90 % random outliers (≤ 0.2° rot err); on a 90 %-contaminated set with a structured (reflection-consistent) decoy cluster the greedy-prefilter+RANSAC engine fails at ~78° while MAC stays < 0.2°; all-outlier sets return an honest `success=False` identity; 500 correspondences ≈ 0.1 s CPU. It also plugs into the learned path (`learned_feature_align(..., seed_selector="mac")` — MAC over GeoTransformer's correspondences, the combination the MAC paper reports lifting 3DLoMatch recall ~71→78 %). **Measured on the full official splits (GPU, same forward/voxel/refine — only the hypothesis stage differs): a wash, not a lift** — 3DLoMatch 72.1 % mean / 74.6 % pooled vs LGR's 72.5 % / 74.4 %, 3DMatch 91.7 % / 93.8 % vs 91.5 % / 93.5 % (every delta within ±4 pairs), at ~+50 % runtime; GeoTransformer's native-voxel correspondences are already consensus-dominated (median 600–800 MAC inliers), so the default stays `seed_selector="lgr"` and `"mac"` remains the tool for genuinely contaminated correspondence sets (see [RESULTS §5k](RESULTS.md)).

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
python -m pytest tests/ -q                        # 143 passing
python tests/test_jacobians.py                    # analytic vs numerical Jacobian audit
python examples/validate_recovery.py --fast       # CPU smoke: 6/6 recovery in ~41 s
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

## Documentation

Full docs at **<https://archerkattri.github.io/splatreg/>** — [quickstart](https://archerkattri.github.io/splatreg/quickstart/),
[CLI guide](https://archerkattri.github.io/splatreg/cli/), [init modes](https://archerkattri.github.io/splatreg/init-modes/),
[photometric refinement](https://archerkattri.github.io/splatreg/photometric/) (when & why, with the measured three-case table),
[PLY interop](https://archerkattri.github.io/splatreg/ply-interop/) (splatfacto/INRIA/SuperSplat round-trip + the
SH-under-rotation detail), [benchmarks](https://archerkattri.github.io/splatreg/benchmarks/), and the
[API reference](https://archerkattri.github.io/splatreg/api/). Or run the
[Colab quickstart](https://colab.research.google.com/github/Archerkattri/splatreg/blob/main/examples/splatreg_quickstart.ipynb)
(CPU-only, no assets needed).

## Citation

If splatreg is useful in your research, please cite it (see [`CITATION.cff`](CITATION.cff) — GitHub's
"Cite this repository" button gives BibTeX/APA):

```bibtex
@software{attri_splatreg,
  author  = {Attri, Krishi},
  title   = {splatreg: composable SE(3)/Sim(3) registration for 3D Gaussian Splatting},
  url     = {https://github.com/Archerkattri/splatreg},
  version = {1.2.0},
  year    = {2026}
}
```

## License & layout

BSD 3-Clause — permissive, composes with the gsplat / Theseus / GTSAM ecosystem. `splatreg/` — library (`api`, `align`, `align_features`, `bundle`, `spatial_index`, `core/lie`, `geometry/gaussian_sdf`, `residuals/`, `solvers/lm`, `cli`). `tests/` · `benchmarks/` · `examples/` · `docs_site/`. Full validation record: [`RESULTS.md`](RESULTS.md).
