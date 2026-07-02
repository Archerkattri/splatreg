<div align="center">

<img src="https://raw.githubusercontent.com/Archerkattri/splatreg/main/assets/banner.png" alt="splatreg" width="680">

# splatreg

### Register Gaussian splats: align and merge 3DGS scans into one SE(3)/Sim(3) frame.

[![PyPI](https://img.shields.io/pypi/v/splatreg)](https://pypi.org/project/splatreg/)
[![DOI](https://zenodo.org/badge/1260804203.svg)](https://zenodo.org/badge/latestdoi/1260804203)
[![Paper](https://img.shields.io/badge/engrXiv-10.31224%2F7313-009E73.svg)](https://doi.org/10.31224/7313)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-archerkattri.github.io%2Fsplatreg-teal.svg)](https://archerkattri.github.io/splatreg/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Archerkattri/splatreg/blob/main/examples/splatreg_quickstart.ipynb)
[![gsplat](https://img.shields.io/badge/inverse%20of-gsplat-ee4c2c.svg)](https://github.com/nerfstudio-project/gsplat)

<img src="assets/registration_demo.png" alt="splatreg before/after registration" width="92%">

</div>

---

gsplat renders your Gaussians; **splatreg registers them**. Two 3DGS scans of the same scene
go in, one SE(3) or Sim(3) transform comes out, and (optionally) one fused, deduped splat.
Pure PyTorch, no meshing, no CUDA extension, no point-cloud detour; works with anything that
speaks the standard 3DGS PLY (gsplat, Nerfstudio, INRIA, SuperSplat) or hands over tensors.

What you get that no other splat registrar ships (each claim traced in
[Results](#results) and [`RESULTS.md`](RESULTS.md)):

- **Provably correct SH rotation.** When a recovered transform is baked in, the higher-order
  spherical-harmonic bands (`f_rest`) are mixed by the real-basis Wigner-D matrix, so glossy
  highlights turn *with* the splat instead of staying stuck in the old capture frame.
  Test-locked against an independent basis evaluator to **~2.4e-15** in float64
  ([`tests/test_sh_rotation.py`](tests/test_sh_rotation.py)).
- **Align WITHOUT merging.** `apply_transform()` (and `splatreg align`) bakes the recovered
  pose into the source and writes it as its own PLY: both scans stay separate files, now in
  one frame, ready for any viewer or editor.
- **Photometric refinement** with per-pair **exposure compensation** and a **coarse-to-fine
  render ladder**, for the poses geometry cannot see (symmetry, texture-only DoF):
  5°/7 mm down to **0.36°/0.5 mm** on the real rasterizer.
- **Pose covariance** on every builtin-LM solve (`info["information"]` /
  `info["covariance"]`), so the result plugs straight into a pose graph with an honest
  weight, `None` when singular, never faked.
- **MAC maximal-clique seed** (`init="mac"`, Zhang et al. CVPR 2023) for contaminated
  correspondence sets, with the honest measured verdict: a wash on official
  3DMatch/3DLoMatch, a decisive win in the structured-decoy regime (78° failure vs <0.2°).
- **Sim(3) scale recovery**, which none of the competing splat tools attempt at all.

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

## 30-second quickstart

The 3-line merge (Python):

```python
from splatreg.api import merge
from splatreg.io import load_ply, save_ply

fused = merge([load_ply("a.ply"), load_ply("b.ply")])   # register + fuse + dedupe
save_ply(fused, "fused.ply")                            # opens in SuperSplat / any viewer
```

Align without merging (both scans stay separate files, registered into one frame):

```python
from splatreg.api import register, apply_transform
from splatreg.io import load_ply, save_ply

target, source = load_ply("a.ply"), load_ply("b.ply")
result = register(target, source, transform="sim3")     # init="fast" by default (~17 ms)
save_ply(apply_transform(source, result.T, result.scale), "b_aligned.ply")
# a.ply untouched; a.ply + b_aligned.ply now line up in any viewer.

result.T          # recovered 4x4 similarity [[s*R, t], [0, 1]], maps source -> target
result.scale      # recovered scale s (1.0 for transform="se3")
result.converged  # solver convergence flag
result.info       # diagnostics incl. pose information/covariance, ambiguity flag
```

Or entirely from the shell (standard 3DGS PLY in/out, composes with SuperSplat / gsplat /
Nerfstudio exports; see the [CLI guide](https://archerkattri.github.io/splatreg/cli/)):

```bash
splatreg align target.ply source.ply -o aligned.ply    # register + write the aligned source
splatreg merge a.ply b.ply -o fused.ply                # register + fuse + dedupe N splats
splatreg info x.ply                                    # count / bounds / SH degree / stats
```

Object pose and camera localization ride on the same core:

```python
from splatreg import estimate_object_pose, localize_camera, coarse_localize_camera

result = estimate_object_pose(model_splat, observation_splat)    # ADD / ADD-S / AUC built in
result = localize_camera(scene_splat, frame, init_T_WC=T_init)   # needs splatreg[render]
T_coarse = coarse_localize_camera(scene_splat, frame)            # prior-free CPU seed
```

## Capability matrix

Honest comparison against the tools people actually use for this job. The accuracy row is
measured head-to-head on a real splat with known ground truth
([`RESULTS.md` §5c](RESULTS.md)); editor columns reflect their design (manual transforms,
not registration).

| | **splatreg** | splatalign | GaussianSplattingRegistration | SuperSplat / SplatTransform |
|---|---|---|---|---|
| Automatic splat-to-splat registration | yes (6 init modes) | ICP from identity | Open3D RANSAC+ICP | no (manual gizmo / user-given transform) |
| Measured rotation error, real splat + GT | **5.2°** | 15.3° | 36.3° | n/a |
| Sim(3) scale recovery | **yes, native** | no (SE(3) only) | no (SE(3) only) | manual |
| SH (`f_rest`) rotated with the splat | **yes, test-locked** | no | no | not in any splat registrar we know of |
| Merge + overlap dedupe | yes | no | no dedupe | concat only |
| Photometric refine (exposure comp + ladder) | yes | no | no | no |
| Pose covariance for pose graphs | yes | no | no | n/a |
| Honest ambiguity flag (never silent-wrong) | yes | no | no | n/a |
| Pure PyTorch library + CLI | yes | script | GUI | editor / CLI |

## Results

Every number is measured and reproducible; the provenance column points at the full record.

| Benchmark | splatreg | reference | provenance |
|---|---|---|---|
| Real-splat merge (103k Gaussians) | Chamfer **10.3 → 2.0 mm (5.1×)**, overlap 0.03 → 0.67 (22×) | naive concat | [`RESULTS.md` §5d](RESULTS.md), `examples/merge_demo.py` |
| Photometric refine (real rasterizer) | 5°/7 mm → **0.36°/0.5 mm** (~1.1 s) | geometric stage alone worsens the symmetric case 6.0°→11.2° | [`benchmarks/photometric_refine_results.md`](benchmarks/photometric_refine_results.md) |
| Official 3DMatch recall (1279 pairs, Choi/Zeng protocol) | **91.5%** mean, 93.5% pooled | GeoTransformer ~92%, Open3D ~77% | [`RESULTS.md` §5b](RESULTS.md) |
| Official 3DLoMatch (hard, 10-30% overlap) | 72.5% mean, **74.4%** pooled | GeoTransformer ~74%, Open3D ~20% | [`RESULTS.md` §5b](RESULTS.md) |
| vs splat competitors (real splat, known GT Sim3) | **5.2°** (SE3), recovers scale (Sim3) | splatalign 15.3°, GS-Registration 36.3° | [`RESULTS.md` §5c](RESULTS.md) |
| Object pose (canonical YCB CAD, 14 models × 4 poses) | ADD-S AUC **0.995**, 100% < 2 cm | n/a | [`RESULTS.md` §5f-ycb](RESULTS.md) |
| Camera localization (real splat, known perturbation) | median 5°/10 mm → **0.11°/1.35 mm** | n/a | [`RESULTS.md` §5g](RESULTS.md) |
| Known-transform recovery | **36/36 = 100%** (GPU full grid); 6/6 CPU smoke in 41 s | n/a | [`RESULTS.md` §1, §5j](RESULTS.md) |
| Registration speed | **~17 ms** (fast init), 104 ms (learned) | GeoTransformer ~50 ms, Open3D 142 ms | [`RESULTS.md` §5e](RESULTS.md) |
| SH rotation correctness | rotated-coeff evaluation error **~2.4e-15** (float64) | n/a | [`tests/test_sh_rotation.py`](tests/test_sh_rotation.py), [`RESULTS.md` §5j](RESULTS.md) |
| Exposure compensation | tinted-pair scale error 3.99% → **0.47%** (clean: 0.01%, harmless) | no-compensation baseline | [`RESULTS.md` §5j](RESULTS.md) |
| Pose covariance | SPD when well-constrained, scales with noise, `None` when singular | n/a | [`tests/test_pose_covariance.py`](tests/test_pose_covariance.py) |

### Init modes: trade speed for robustness

| `init=` | what | when |
|---|---|---|
| `"fast"` *(default)* | FPFH + GPU-batched RANSAC seed → closed-form LM | objects / full-overlap, **~17 ms** |
| `"robust"` | Open3D FPFH+RANSAC seed → splatreg refine + scale | real metre-scale scans |
| `"learned"` | pretrained GeoTransformer seed → splatreg refine + scale | best accuracy on real scans |
| `"bufferx"` | pretrained **BUFFER-X** zero-shot seed (ICCV 2025) → splatreg refine + scale | cross-sensor / cross-scale scans with **no per-dataset training** |
| `"mac"` | MAC maximal-clique consensus (Zhang et al. CVPR 2023) → weighted SVD → refine | outlier-heavy / multi-consensus correspondence sets |
| `"global"` | blind super-Fibonacci SO(3) sweep | robust fallback, any rotation |

Two options refine the *seed* rather than the pose: `init="learned"` accepts `seed_gate=True`
(off by default), a Decision-PCR-style confidence check (arXiv 2507.14965) that scores the learned
seed (mutual-NN inlier ratio + SC² spatial consistency) and reseeds a low-confidence hypothesis from
the classical `"robust"` path *before* LM refinement, instead of blindly refining a bad seed. And
`init="bufferx"` swaps GeoTransformer for **BUFFER-X** ("Towards Zero-Shot Point Cloud Registration
in Diverse Scenes", ICCV 2025) — a single generalist model that registers across sensors and scales
with no per-dataset training. Both learned backends are optional and lazily loaded; when their
weights / CUDA extensions are absent they fall back to `"robust"` with a logged note (BUFFER-X setup:
[`splatreg/third_party_models/README-BUFFERX.md`](splatreg/third_party_models/README-BUFFERX.md)).

**2026 positioning.** Per-dataset-trained backbones like **PSReg** and **DiffusionPCR** now top the
3DMatch leaderboard (95%+ registration recall), above the ~91.5% GeoTransformer seed splatreg wraps.
splatreg deliberately keeps a *zero-shot* learned option (BUFFER-X) rather than chasing that number:
a splat registrar should not require training a per-scene/per-sensor model to align two captures, so
the value is a generalist seed + splatreg's provable SH rotation, honest pose covariance, Sim(3)
scale, and overlap-aware refine on top — not the last recall point on one benchmark. Drop in a
higher-recall correspondence model as the seed the day it ships a permissive, zero-shot checkpoint.

**The BUFFER-X seed, built and validated.** The zero-shot seed is built and run on **real
3DMatch**, with both seeds pushed through the *identical* splatreg refine so the comparison
isolates the seed rather than the pipeline. On the **official `gt.log` pair set** (6/8 scenes
scored so far, n=1250; a pair counts as recalled at RRE < 15° and RTE < 0.3 m) the BUFFER-X seed
reaches **0.974 recall** (median RRE 1.46°) against **0.670** (1.94°) for the classical robust
FPFH seed — and the gap widens on the harder non-adjacent pairs (0.973 vs 0.612, n=998). In the
low-overlap 3DLoMatch regime (overlap 0.10–0.30, n=400, from an earlier GT-derived run) it holds
**0.752** (3.23°) against **0.092** (107.9°) — an **8× recall** lift where classical FPFH
collapses to ~random. BUFFER-X wins every scene in both regimes.

<div align="center">
<img src="https://raw.githubusercontent.com/Archerkattri/splatreg/main/assets/bufferx_recall.png" alt="BUFFER-X zero-shot seed vs classical FPFH seed: registration recall on 3DMatch and the low-overlap 3DLoMatch regime" width="82%">
</div>

<sub>*3DMatch bars are the official `gt.log` pair set (6/8 scenes, n=1250); the low-overlap bars are
a 50/scene GT-derived run (n=400). Both seeds share the identical lighter `feature_align` refine,
so these isolate the seed rather than report full-pipeline absolute numbers; the remaining scenes
and the official 3DLoMatch runs are in progress.*</sub>

**Registration, happening.** One real low-overlap pair, watched end to end: the source fragment
starts unaligned, the classical FPFH+RANSAC seed slews it into the *wrong* basin (151.5° off), then
the BUFFER-X seed + splatreg refine rotates it onto the target and locks on at 2.0°.

<div align="center">
<img src="https://raw.githubusercontent.com/Archerkattri/splatreg/main/assets/registration_lowoverlap.gif" alt="Three-phase animation of a real low-overlap 3DMatch pair: unaligned, wrong classical FPFH+RANSAC basin (RRE 151.5°), then BUFFER-X seed + splatreg refine locking onto the target (RRE 2.0°)" width="62%">
</div>

<sub>*Real 3DMatch pair `7-scenes-redkitchen` 35→46, GT overlap **0.10**. Both transforms are the
actual `robust_feature_align` (classical, RRE **151.5°** ✗) and `bufferx_feature_align` (BUFFER-X,
RRE **2.0°** ✓) library outputs — the animation interpolates between the real estimates, nothing is
hand-posed. At 10 % overlap the source only shares a corner with the target, so a correct lock
overlaps just that corner. Regenerate: [`examples/make_lowoverlap_gif.py`](examples/make_lowoverlap_gif.py).*</sub>

Weights come from Hugging Face `Hyungtae-Lim/BUFFER-X`; a native build on a modern stack
(CUDA 12.8 / sm_120 / torch 2.11 / numpy 2.x) is nontrivial, with the full sudo-free recipe in
[`docs/BUFFERX_BUILD_MODERN_CUDA.md`](docs/BUFFERX_BUILD_MODERN_CUDA.md). Note the checkpoints are
full-model state_dicts: loading them into the `.Desc`/`.Pose` submodules silently loads nothing
(random weights → garbage seeds), fixed in `c54d8c9`.

**The MAC verdict, stated honestly.** `init="mac"` reimplements the MAC hypothesis generator
(SC²-weighted rigidity graph → maximal cliques → weighted SVD per clique, with explicit
caps) in pure torch + networkx (`pip install "splatreg[mac]"`). On synthetic contaminated
sets ([`tests/test_mac.py`](tests/test_mac.py)) it matches the RANSAC engine at 30/60/90%
random outliers and decisively wins the structured-decoy regime (RANSAC fails at ~78°, MAC
stays <0.2°). Measured on the **full official splits** (same forward/voxel/refine, only the
hypothesis stage differs) it is **a wash, not a lift**: 3DLoMatch 72.1/74.6 vs LGR's
72.5/74.4, 3DMatch 91.7/93.8 vs 91.5/93.5, every delta within ±4 pairs, at ~+50% runtime.
GeoTransformer's native-voxel correspondences are already consensus-dominated, so the
default stays `seed_selector="lgr"`; `"mac"` is the tool for genuinely contaminated
correspondence sets ([`RESULTS.md` §5k](RESULTS.md)).

## How it works

splatreg takes two splats and finds the rigid (SE(3)) or similarity (Sim(3), +scale)
transform that aligns them, then optionally merges and dedupes them into one. It is the
missing *registration* half of the Gaussian-splatting toolchain (the splat-to-splat
alignment SuperSplat / INRIA / geospatial users keep asking for, where today's tooling
punts to a manual gizmo).

```mermaid
flowchart LR
    A["splat A<br/>(target)"]:::s --> G
    B["splat B<br/>(source)"]:::s --> G
    G["<b>Global aligner</b><br/>super-Fibonacci SO(3) seeds<br/>+ batched trimmed ICP<br/><i>(or FPFH / learned / MAC)</i>"]:::g --> L
    L["<b>Levenberg-Marquardt</b><br/>multi-residual:<br/>ICP + Gaussian-SDF<br/>SE(3) / Sim(3)"]:::l --> T["T*  (4×4)<br/>+ merge / dedupe"]:::o
    classDef s fill:#e8f6f8,stroke:#17becf,color:#0b3d44;
    classDef g fill:#fff1ee,stroke:#ff6b5b,color:#5a1a12;
    classDef l fill:#eef7ee,stroke:#2e8b57,color:#143d22;
    classDef o fill:#f3eefc,stroke:#7d52c7,color:#2c1654;
```

1. **Global init**: a coarse pose from a dense super-Fibonacci rotation sweep + batched
   trimmed ICP (no local-minimum trap), with FPFH+RANSAC, learned (GeoTransformer), and MAC
   maximal-clique seeds for harder real scans.
2. **Refinement**: a from-scratch Levenberg-Marquardt core over ICP (point-to-point /
   point-to-plane) *and* splatreg's flagship **Gaussian-SDF** residual, solving the full
   SE(3) or Sim(3) tangent, with the pose information/covariance exposed at the optimum.

### The Gaussian-SDF residual

No competitor packages this. splatreg derives a smooth **signed-distance field directly from
the target Gaussians** (no mesh, no marching cubes) and drives registration by it:

```
w_i(p) = exp(−‖p − q_i‖² / 2σ²)              # Gaussian kernel weight per anchor
q̃(p)   = Σ w_i q_i / Σ w_i                    # kernel-weighted centroid
ñ(p)   = Σ w_i n_i / ‖Σ w_i n_i‖              # kernel-weighted surface normal
d(p)   = (p − q̃(p)) · ñ(p)                    # signed distance, the residual
```

`d(p)` vanishes exactly when source points land on the target surface. It has a
**closed-form, audited Jacobian** and is a reusable primitive:

```python
from splatreg.geometry.gaussian_sdf import gaussian_sdf, gaussian_sdf_grad
sdf, normal = gaussian_sdf(target, query_points, sigma=0.02)       # signed distance + normal
sdf, grad   = gaussian_sdf_grad(target, query_points, sigma=0.02)  # + exact ∇_p d
```

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

## Limitations

splatreg is honest about its edges (full detail in [`RESULTS.md`](RESULTS.md)):

- **Heavy overlap loss (keep ≤ 40%) is genuinely ambiguous.** The rotation-disambiguating
  geometry is physically absent; even the true pose does not seat cleanly. The aligner flags
  these honestly (`result.info['ambiguous']` / `['confidence']`) and never silently
  wrong-poses. `merge` and `track` are designed for high-overlap captures.
- **Scale is unobservable under thin overlap.** Under ~20% shared geometry the Sim(3) scale
  residual valley is flat; the line-search tightens scale on its own objective but cannot
  recover what the geometry does not carry.
- **Cost on rigid SE(3).** Plain ICP reaches the same SE(3) success and is far faster; the
  SDF residual buys scale + implicit-field robustness at a real compute cost. Use `track()`
  (~17 ms/frame) for the warm-start real-time path.

## Documentation

Full docs at **<https://archerkattri.github.io/splatreg/>**:
[quickstart](https://archerkattri.github.io/splatreg/quickstart/),
[CLI guide](https://archerkattri.github.io/splatreg/cli/),
[init modes](https://archerkattri.github.io/splatreg/init-modes/) (incl. the MAC verdict),
[photometric refinement](https://archerkattri.github.io/splatreg/photometric/) (when and why,
with the measured three-case table),
[PLY interop](https://archerkattri.github.io/splatreg/ply-interop/) (splatfacto/INRIA/SuperSplat
round-trip + the SH-under-rotation detail),
[benchmarks](https://archerkattri.github.io/splatreg/benchmarks/), and the
[API reference](https://archerkattri.github.io/splatreg/api/). Or run the
[Colab quickstart](https://colab.research.google.com/github/Archerkattri/splatreg/blob/main/examples/splatreg_quickstart.ipynb)
(CPU-only, no assets needed).

## Citation

If splatreg is useful in your research, please cite it (see [`CITATION.cff`](CITATION.cff);
GitHub's "Cite this repository" button gives BibTeX/APA). The DOI is the Zenodo concept DOI
and always resolves to the latest archived release:

```bibtex
@software{attri_splatreg,
  author  = {Attri, Krishi},
  title   = {splatreg: composable SE(3)/Sim(3) registration for 3D Gaussian Splatting},
  url     = {https://github.com/Archerkattri/splatreg},
  doi     = {10.5281/zenodo.20618389},
  version = {1.3.0},
  year    = {2026}
}
```

**Paper (preprint).** engrXiv, [doi:10.31224/7313](https://doi.org/10.31224/7313):

```bibtex
@article{attri2026splatreg,
  author  = {Attri, Krishi},
  title   = {Registering Gaussian Splats Without the Point-Cloud Detour: Accuracy,
             Representation Semantics, and a Negative Result on Hypothesis-Stage Transfer},
  journal = {engrXiv},
  doi     = {10.31224/7313},
  year    = {2026}
}
```

## License & layout

BSD 3-Clause: permissive, composes with the gsplat / Theseus / GTSAM ecosystem.
`splatreg/` is the library (`api`, `align`, `align_features`, `mac`, `sh`, `bundle`,
`spatial_index`, `core/lie`, `geometry/gaussian_sdf`, `residuals/`, `solvers/lm`, `cli`),
plus `tests/`, `benchmarks/`, `examples/`, `docs_site/`. Full validation record:
[`RESULTS.md`](RESULTS.md).
