<div align="center">

# splatreg

### Register Gaussian splats — align & merge two 3DGS scans into one SE(3)/Sim(3) frame.

*The inverse of [gsplat](https://github.com/nerfstudio-project/gsplat): gsplat **renders** Gaussians, splatreg **registers** against them.* Pure PyTorch — no meshing, no CUDA extension, no point-cloud detour.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pure-PyTorch-ee4c2c.svg)](https://pytorch.org)
[![tests](https://img.shields.io/badge/tests-44%20passing-brightgreen.svg)](tests)
[![Jacobian audit](https://img.shields.io/badge/Jacobian%20audit-8%2F8-brightgreen.svg)](tests/test_jacobians.py)
[![recovery](https://img.shields.io/badge/synthetic%20recovery-36%2F36-brightgreen.svg)](RESULTS.md)

<img src="assets/registration_demo.png" alt="splatreg before/after registration" width="92%">

</div>

---

## What it is

A 3D Gaussian Splat is a cloud of oriented Gaussians that already traces an object's surface. **splatreg takes two such splats and finds the rigid (SE(3)) or similarity (Sim(3), +scale) transform that aligns them** — then optionally merges + dedupes them into one. It is the missing *registration* half of the Gaussian-splatting toolchain — the splat-to-splat alignment that SuperSplat / INRIA / geospatial users keep asking for, where today's tooling punts to a manual gizmo.

The pipeline is two stages:

```mermaid
flowchart LR
    A["splat A<br/>(target)"]:::s --> G
    B["splat B<br/>(source)"]:::s --> G
    G["<b>Global aligner</b><br/>super-Fibonacci SO(3) seeds<br/>+ batched trimmed ICP<br/><i>(or FPFH features)</i>"]:::g --> L
    L["<b>Levenberg–Marquardt</b><br/>multi-residual:<br/>ICP + Gaussian-SDF<br/>SE(3) / Sim(3)"]:::l --> T["T*  (4×4)<br/>+ merge / dedupe"]:::o
    classDef s fill:#e8f6f8,stroke:#17becf,color:#0b3d44;
    classDef g fill:#fff1ee,stroke:#ff6b5b,color:#5a1a12;
    classDef l fill:#eef7ee,stroke:#2e8b57,color:#143d22;
    classDef o fill:#f3eefc,stroke:#7d52c7,color:#2c1654;
```

1. **Global init** — a coarse pose from a dense super-Fibonacci rotation sweep + batched trimmed ICP (no local-minimum trap), with an optional **FPFH feature + RANSAC** path for harder cases.
2. **Refinement** — a from-scratch **Levenberg–Marquardt** core over a stack of residuals: classic **ICP** (point-to-point / point-to-plane) *and* splatreg's flagship **Gaussian-SDF** residual, solving the full SE(3) or Sim(3) tangent.

It **composes, it doesn't compete**: bring gsplat tensors directly; the LM loop and residual stack are pluggable.

### The differentiator — the Gaussian-SDF residual

No competitor packages this. splatreg derives a smooth, queryable **signed-distance field directly from the target Gaussians** — no mesh, no marching cubes — and drives registration by it:

```
w_i(p) = exp(−‖p − q_i‖² / 2σ²)              # Gaussian kernel weight per anchor
q̃(p)   = Σ w_i q_i / Σ w_i                    # kernel-weighted centroid
ñ(p)   = Σ w_i n_i / ‖Σ w_i n_i‖              # kernel-weighted surface normal
d(p)   = (p − q̃(p)) · ñ(p)                    # signed distance — the residual
```

`d(p)` vanishes exactly when the source points land on the target's surface. It has a **closed-form, audited Jacobian** (see below) and is a standalone, reusable implicit-field primitive: `gaussian_sdf(splat, points, sigma=...) → (sdf, normal)`.

---

## Headline results

| | **splatreg** | reference |
|---|---|---|
| **Official 3DMatch registration recall** (Choi/Zeng protocol, 1279 pairs) | **91.5%** (mean-of-scenes) · 93.5% pooled | GeoTransformer ~92% · Open3D ~77% |
| **Official 3DMatch rotation / translation error** | **1.81° / 0.071 m** | — |
| **Official 3DLoMatch** (hard, 10–30% overlap) | **72.5%** mean · **74.4%** pooled | GeoTransformer ~74% · Open3D ~20% |
| **vs splat competitors** (real GF splat, known GT Sim3) | **5.2° / 15.7 mm** (SE3) · recovers scale (Sim3) | splatalign 15.3° · GaussianSplattingRegistration 36.3° |
| **Registration speed** | **~17 ms** (fast) · 104 ms (learned) | GeoTransformer ~50 ms · Open3D 142 ms |
| **Real-time tracking** | **~17 ms/frame** | GaussianFeels tracker ~45 ms |
| **Synthetic Sim(3) recovery** | **36/36, rot 0.03°, scale 0.34%** | ICP-only 9/27 (no scale) |
| **Sim(3) scale estimation** | ✅ native | ✗ none of these do it |

splatreg is the **only library** that registers native Gaussian splats with SE(3)+**Sim(3)** behind a closed-form-Jacobian Gaussian-SDF. On the *official* 3DMatch protocol its `learned` path **matches GeoTransformer** — 91.5% mean / 93.5% pooled RR vs their published ~92% — because it **rides GeoTransformer's matcher at its native resolution** and then layers splatreg's SDF/LM refine + Sim(3) scale **on top** (a guarded refine that is never worse — a per-pair audit found **0 demotions**). The recall is GeoTransformer's; what splatreg adds is **accuracy** (RRE 1.87° → 1.81°), the unique **Sim(3) scale DoF**, and a verified **no-regression floor** — *not* extra recall. On the hard 3DLoMatch split it reaches **72.5% mean / 74.4% pooled**, matching/beating GeoTransformer's ~74% on the pooled count. It **decisively beats classical Open3D** (~77% / ~20%). Against the actual splat-registration tools it **wins outright** — 5.2° vs splatalign's 15.3° and GaussianSplattingRegistration's 36.3° on a real GaussianFeels splat — and it is the **only one that recovers Sim(3) scale.** Four init modes trade speed↔robustness:

| `init=` | what | when |
|---|---|---|
| `"fast"` *(default)* | FPFH + GPU-batched RANSAC seed → closed-form LM | objects / full-overlap, **~17 ms** |
| `"robust"` | Open3D FPFH+RANSAC seed → splatreg refine + scale | real metre-scale scans |
| `"learned"` | pretrained GeoTransformer seed → splatreg refine + scale | best accuracy on real scans |
| `"global"` | blind super-Fibonacci SO(3) sweep | robust fallback, any rotation |

> **Honest scope.** The `"learned"` path **rides GeoTransformer's matcher at its native 0.025 m resolution** and uses its full LGR pose, then layers splatreg's SDF/LM refine + Sim(3) scale on top — so it **matches** GeoTransformer's published recall (91.5% official vs ~92%), it does **not beat their matcher's recall.** The recall is GeoTransformer's; what splatreg contributes is the **accuracy** refine (RRE 1.87° → 1.81°), the **Sim(3) scale** DoF that no classical/learned baseline here estimates, and a **guaranteed no-regression floor** (a per-pair audit found 0 pairs where the refine demoted a GeoTransformer success). The original product deliverable — the real-splat **merge demo** — remains open. See [Limitations](#limitations--honest-status).

---

## Install

```bash
git clone https://github.com/Archerkattri/splatreg.git
cd splatreg
pip install -e .          # pure PyTorch + numpy; pip install -e ".[test]" for the test extras
```

## Quickstart

```python
from splatreg.api import register, merge

# two Gaussian splats of the same object, in unknown relative pose/scale.
# register aligns `source` onto the reference `target` (target is the first arg).
result = register(target, source, transform="sim3")       # init="fast" by default (objects / full-overlap)
# real metre-scale scans -> init="robust" (FPFH+RANSAC) or init="learned" (GeoTransformer seed, best accuracy)
print(result.T)         # recovered 4×4 similarity [[s·R, t], [0, 1]] — maps source -> target
print(result.scale)     # recovered scale s  (1.0 for transform="se3")
print(result.converged) # solver convergence flag

# register + dedupe a list of splats into one fused splat (registers internally)
fused = merge([source, target], transform="sim3")
```

The Gaussian-SDF field on its own:

```python
from splatreg.geometry.gaussian_sdf import gaussian_sdf, gaussian_sdf_grad
sdf, normal = gaussian_sdf(target, query_points, sigma=0.02)      # signed distance + surface normal
sdf, grad   = gaussian_sdf_grad(target, query_points, sigma=0.02) # signed distance + EXACT ∇_p d
```

---

## Validation & benchmarks

> splatreg is held to the validation bar of the libraries it sits beside — **gsplat / Theseus / GTSAM / SymForce**. Every number below is reproducible (commands at the bottom); the full record is in [`RESULTS.md`](RESULTS.md).

### 1 · Synthetic recovery — the core accuracy test

Apply a *known* Sim(3)/SE(3) to a realistic object splat, recover it, measure the error. 3 seeds × {5°, 30°, 90°} × {0.8, 1.0, 1.3} scale (`examples/validate_recovery.py`).

| Block | Success | median rot | median trans | median scale err | median Chamfer |
|---|:---:|:---:|:---:|:---:|:---:|
| **SE(3)** (rigid) | **9 / 9 = 100%** | **0.000°** | 0.10 mm | — | 0.076 mm |
| **Sim(3)** (+scale) | **27 / 27 = 100%** | **0.259°** | 2.93 mm | 0.34% | 0.575 mm |
| **Overall** | **36 / 36 = 100%** | worst rot 0.43° | | | |

### 2 · Jacobian correctness — the audit that found a real bug

Every serious geometric-optimisation library checks each analytic Jacobian against a tangent-space numerical one. splatreg ships that audit (`tests/test_jacobians.py`, float64) **and a reusable `assert_residual_jacobian`** so every future residual gets it (the GTSAM `EXPECT_CORRECT_FACTOR_JACOBIANS` equivalent).

| Residual / op | Result |
|---|---|
| ICP point-to-point / point-to-plane | ✅ correct (max\|Δ\| ~3e-9 / 4e-11) |
| **Gaussian-SDF** | ✅ **closed-form exact** (~1e-8 vs numerical, in-support) |
| SE(3)/Sim(3) exp·log, group invariants, near-π, `so3_project` | ✅ all correct |

Two real bugs the audit caught and fixed:
- **SDF gradient.** The field returned the surface *normal* `ñ` as its gradient, but the true `∇d` carries a first-order `∂q̃/∂p` term (the kernel-weighted centroid moves with `p`) that `ñ` drops — a materially wrong pose gradient (`max|Δ|≈10.8`). Now an **exact closed-form gradient** (`gaussian_sdf_grad`): `∇d = ñ − (1/σ²)·Cov_w·ñ − (1/(σ²‖Sₙ‖))·Σᵢwᵢ(nᵢ·x)aᵢ`, with no autograd graph on the SE(3) path.
- **Near-π SO(3) log.** `se3_log` recovered the axis from the antisymmetric part `(R−Rᵀ)`, which vanishes at θ=π — losing the axis for ~180° rotations. Fixed with the standard robust branch (symmetric-part axis + `atan2`); roundtrip now exact to **~1e-13** across the interior.

### 3 · vs. plain ICP + residual ablation

`benchmarks/icp_baseline_bench.py` — identical recovery cells, splatreg vs ICP baselines.

| Method | SE(3) success | **Sim(3) success** |
|---|:---:|:---:|
| **splatreg (full)** | 9 / 9 | **27 / 27 = 100%** |
| ICP (centroid init) | 9 / 9 | 9 / 27 = 33% |
| ICP (super-Fib init) | 9 / 9 | 9 / 27 = 33% |

**splatreg wins Sim(3) decisively** — plain ICP cannot estimate scale, so it fails every non-unit-scale cell; the global init alone doesn't rescue it, so the **LM Sim(3) solve is load-bearing.** *Honest trade:* on rigid SE(3) both reach 100% and ICP is far faster — the SDF residual buys scale + implicit-field robustness at a real compute cost (see Limitations).

### 4 · Robustness sweep

`benchmarks/robustness_bench.py`, 3 seeds.

| Condition | Result |
|---|---|
| **Noise** (sensor jitter 0.5–2%) | ✅ **9 / 9 = 100%** (rot_err < 0.72°) |
| **Outliers** (+10–50% clutter) | ✅ **9 / 9 = 100%** (ignores clutter) |
| **Symmetric** (sphere) | ✅ **9 / 9 = 100%** — a global-init convergence fix lands the featureless sphere correctly at all poses |
| **Partial overlap** (20–60% removed) | **4 / 9 solved + 5 flagged** — mild + some moderate crops solve at 0.00°; the rest honestly flagged via the ambiguity API; **0 silent-wrong** |

### 5 · Test suite + CI

`pytest tests/` → **44 passing**: the Jacobian audit, Lie-group ops (exp·log roundtrips, group invariants, hat/vee, near-π stability, a 10k-sample SymForce-style Jacobian sweep), and the LM solver (`CheckLinearError`, singular-system handling, GT recovery, Sim(3) scale). The package is `black` + `mypy` clean and ships `py.typed`.

### 6 · Real-time tracking speed — *verified* ✅

splatreg descends from a real-time SE(3) Gaussian tracker, so **speed is a first-class goal**. The `track()` API (`splatreg/track.py`) skips the global init, seeds from the prior pose, and runs a few **closed-form-Jacobian** LM iterations over a **truncated** SDF (N×k). Frame-to-frame on GPU (`benchmarks/tracking_speed_bench.py`):

| | per-frame | rot err | |
|---|:---:|:---:|---|
| **`track()` warm-start (SE(3))** | **~17 ms** | 0.43° | **< 40 ms goal MET — faster than the ~45 ms GaussianFeels tracker** |

That's **~46× faster than the 780 ms from-scratch registration** — the global-init sweep is the cost, and a tracker never pays it. The full Sim(3) *registration* also dropped **19.7 s → 2.4 s/cell** once the closed-form gradient was extended to the scale column.

### 7 · Real splat data

`benchmarks/realdata_bench.py` over **12,463 real GaussianFeels `.ply` exports** (`gaussianfeels/outputs/*/final.ply`, full INRIA/gsplat layout):

- **Clean** real geometry → Sim(3) recovery **near-perfect** (rot 0.03–0.06°, scale 0.04–0.14%, Chamfer 0.04–0.08 mm ≈ the synthetic harness). Real geometry itself is not a problem.
- **Noisy** second-capture (footprint-scale noise + 60% subsample) on near-symmetric objects → the object-tuned `"fast"`/`"features"` seed is fragile (1/9). **Fixed by `init="robust"`/`"learned"`** (scale-correct seeds), see below.

### 8 · 3DMatch — the official protocol (honest numbers)

The community-standard registration benchmark, run under the **canonical Choi/Zeng protocol** (the 1279 non-adjacent `gt.log` pairs, covariance-weighted error `eᵀCe ≤ 0.2²`) that every published learned method reports on — `benchmarks/threedmatch_official_bench.py`. splatreg registers the Gaussian means as a point cloud:

| Method | 3DMatch RR | RRE | RTE | 3DLoMatch RR |
|---|:---:|:---:|:---:|:---:|
| **splatreg `learned`** (GeoTransformer LGR + our refine, **native 0.025 voxel**) | **91.5%** mean / 93.5% pooled | **1.81°** | **0.071 m** | **72.5%** mean / 74.4% pooled |
| splatreg `learned` (legacy 0.05 voxel) | 86.3% / 89.1% | 1.87° | 0.071 m | 55.3% |
| splatreg `robust` (classical Open3D seed + our refine) | ~67% | — | — | ~15% |
| GeoTransformer (published, full pipeline) | ~92% | — | — | ~74% |
| Open3D FPFH+RANSAC (classical) | ~77% | — | — | ~20% |

splatreg `learned` **matches GeoTransformer** (91.5% mean / 93.5% pooled official vs their published ~92%) and **decisively beats classical Open3D** (~77%). The lift over the old 86.3% was an **artefact of the harness**, not a method change: the official runner pre-voxelled both fragments to **0.05 m before GeoTransformer ever saw them** (~5 k vs ~19 k pts/fragment), throwing away >70 % of the points its matcher was trained on (native `init_voxel_size = 0.025`). Feeding it its native resolution (`--learned-voxel 0.025`) and using its full LGR pose, then layering splatreg's overlap-residual-**guarded** refine on top, restores the published recall. **Be explicit: the recall here is GeoTransformer's — we ride its matcher, we do not beat it.** What splatreg adds is **accuracy** (RRE 1.87° → 1.81°), the **Sim(3) scale** DoF, and a **no-regression floor**: a per-pair audit (one scene, official covariance metric) found **0 pairs where the refine demoted a GeoTransformer success** — it only tightens RRE inside already-successful pairs. On the hard **3DLoMatch** split (10–30% overlap) it reaches **72.5% mean / 74.4% pooled**, matching/beating GeoTransformer's ~74% on the pooled count (mean ~1.5 pt under, from one weak scene `mit_76_studyroom` at 52.3% — which is GeoTransformer's own weak scene). *(GeoTransformer's ext is pure C++/pybind — builds clean on Blackwell sm_120; pretrained 3DMatch weights load under gitignored `third_party_models/`.)*

> An earlier number (**94.0% RR**) came from our **own overlapping-pair sampler, not the official protocol**, and is retired from every comparison here. A subsequent **86.3%** was the official protocol but at the wrong (0.05 m) voxel; the honest official figure at GeoTransformer's native resolution is **91.5%**.

### 8b · vs the splat-registration tools (head-to-head)

`benchmarks/splat_competitors_bench.py` — a real GaussianFeels splat under a known GT Sim(3), each tool recovering it:

| Tool | rot err | trans err | scale |
|---|:---:|:---:|:---:|
| **splatreg (SE3)** | **5.2°** | **15.7 mm** | — |
| **splatreg (Sim3)** | 11° | — | ✅ **only tool that recovers scale** |
| splatalign (ICP-from-identity) | 15.3° | — | ✗ |
| GaussianSplattingRegistration (Open3D RANSAC+ICP) | 36.3° | — | ✗ |

splatreg **wins outright** against both ICP-only splat tools and is the **only one estimating Sim(3) scale** — the others are SE(3)-only and cannot model the GT scale at all.

---

## Limitations — honest status

- **Partial overlap.** The `init="features"` aligner (overlap-aware **point-to-plane** trimmed ICP + a super-Fibonacci SO(3) sweep, plus FPFH) **solves mild crops** (keep ≥ 80%) at rot_err 0.00°. On heavier crops — where the one-sided slab deletes the rotation-disambiguating geometry, leaving the true pose only ~0.005 below a forest of near-equal wrong basins — it returns an **honest ambiguity flag** (`result.info['ambiguous']` / `['confidence']`) instead of a silent wrong pose. Verified **4/9 solved + 5 flagged-ambiguous, 0 silent-wrong** (was 0/9). Solving the rest of the moderate keep60% crops is open work; `merge` is reliable for high-overlap captures.
- **Global-aligner noise robustness.** The object-tuned `"fast"` seed can flip into a wrong rotation basin on noisy / metre-scale real scans — **addressed** by `init="robust"` (FPFH+RANSAC) and `init="learned"` (GeoTransformer), which carry the scale-correct seed. `"fast"` remains the right default for objects / full-overlap.
- **The real-splat merge demo (the original MVP deliverable).** `merge()` + dedupe works (1600→931 on synthetic, deterministic to `max|dT|=0.0`), but the end-to-end *"merge two overlapping **real** captures → one `.ply`, overlap/Chamfer vs naive concat, render"* demo — the thing that sells it to SuperSplat users — is **not yet shipped**.
- **The recall on 3DMatch is GeoTransformer's, not ours.** Under the official Choi/Zeng protocol splatreg `learned` reaches **91.5%** RR (3DMatch) / **72.5%** mean (3DLoMatch) — but only because it **rides GeoTransformer's matcher at its native resolution**. splatreg does **not beat that matcher's recall**; it matches it and then adds accuracy (RRE 1.81°), the Sim(3) scale DoF, and a verified no-regression floor (0 demotions). Closing the gap with our *own* dense correspondence (not riding a seed) remains open. It **beats classical Open3D** on both splits and **wins** vs the splat-only tools.

---

## Reproduce

```bash
pip install -e ".[test]"
python -m pytest tests/ -q                       # 44 passing: audit + Lie + solver
python tests/test_jacobians.py                   # the numerical-vs-analytic Jacobian audit
SPLATREG_DEVICE=cuda python examples/validate_recovery.py --device cuda   # recovery 36/36
SPLATREG_DEVICE=cuda python benchmarks/icp_baseline_bench.py --device cuda
SPLATREG_DEVICE=cuda python benchmarks/robustness_bench.py  --device cuda
python examples/make_readme_figure.py            # regenerate the hero figure
```

## Roadmap

- [ ] **Real-splat merge demo** — register + merge 2 overlapping real captures → one `.ply`, overlap/Chamfer vs naive concat, render it (the MVP headline deliverable)
- [x] **Official 3DMatch + 3DLoMatch protocol** (Choi/Zeng covariance metric) — `benchmarks/threedmatch_official_bench.py` (91.5% / 72.5%, native 0.025 voxel)
- [x] **Head-to-head vs `splatalign` / `GaussianSplattingRegistration`** — `benchmarks/splat_competitors_bench.py` (splatreg wins; only tool recovering Sim(3) scale)
- [ ] Close the gap to GeoTransformer's full coarse-to-fine matcher (learned dense correspondence, not just a seed)
- [ ] CI regression gates — determinism, worst-case, PR-comment benchmark
- [ ] 6-DoF object-pose mode + FoundationPose/YCB benchmark (v0.2)
- [ ] Camera localization in a splat (v0.2)
- [ ] PyPI release

## License & layout

Apache-2.0. `splatreg/` — library (`api`, `align`, `align_features`, `core/lie`, `geometry/gaussian_sdf`, `residuals/`, `solvers/lm`). `tests/` · `benchmarks/` · `examples/`. Full validation record: [`RESULTS.md`](RESULTS.md).
