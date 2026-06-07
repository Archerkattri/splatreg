# splatreg — validation results

**What splatreg is:** an open, pure-PyTorch library for **registering Gaussian splats** —
align/merge two splats into one SE(3)/Sim(3) frame via a global aligner + multi-residual
Levenberg-Marquardt (the inverse of gsplat: gsplat *renders* Gaussians, splatreg *registers*
against them).

**What this file is:** every claim, with the measured number and the command to reproduce
it — and the honest limitations. Validation is held to the bar of the libraries splatreg
sits beside (gsplat / Theseus / GTSAM / SymForce).

_Last validated 2026-06-07, single box, CUDA._

---

## 1. Synthetic recovery (known-transform, the core accuracy test)

Apply a known Sim(3)/SE(3) to a realistic object splat, recover it, measure error.
`examples/validate_recovery.py` — 3 seeds × {5°,30°,90°} × {0.8,1.0,1.3 scale}.

| Block | Success | median rot | median trans | median scale | median Chamfer |
|---|---|---|---|---|---|
| **SE(3)** (rigid) | **9/9 = 100%** | **0.000°** | 0.10 mm | — | 0.076 mm |
| **Sim(3)** (+scale) | **27/27 = 100%** | **0.259°** | 2.93 mm | 0.344% | 0.575 mm |
| **Overall** | **36/36 = 100%** | worst rot 0.43° | | | |

Run on GPU, peak 8.5 GiB. (These are *post* the SDF-Jacobian fix below — which left
recovery at 36/36 and improved SE(3) convergence: seed-1 cells now solve in 1 iteration.)

## 2. Jacobian correctness — the audit that found a real bug

The discipline every serious geometric-optimisation library enforces: check each analytic
Jacobian against a tangent-space numerical one. `tests/test_jacobians.py` (float64).

| Residual / op | Result |
|---|---|
| ICP point-to-point | ✅ correct, max\|Δ\| ~3e-9 |
| ICP point-to-plane | ✅ correct, max\|Δ\| ~4e-11 |
| **SDF** | ❌→✅ **found wrong (max\|Δ\|=10.8), fixed** — now an exact **closed-form** gradient, ~1e-8 |
| SE(3)/Sim(3) exp·log, invariants, near-π, `so3_project`, LM solver | ✅ all correct (`tests/test_lie.py` + `test_solver.py`) |

**The SDF bug (found + fixed):** the Gaussian-SDF returned the surface *normal* `n~` as its
gradient, but the true gradient of `d(p)=(p−q~(p))·n~(p)` includes a **first-order** `∂q~/∂p`
term (the kernel-weighted centroid moves with `p`) that `n~` drops. A docstring had wrongly
called this "exact to first order." Fixed `residuals/sdf.py` to use the exact **closed-form**
field gradient (`gaussian_sdf_grad` — no autograd graph on the SE(3) path); re-audited (8/8,
~1e-8) and re-validated recovery (still 36/36). A second audit-adjacent fix: the **near-π SO(3)
log** lost the rotation axis at θ=π (the antisymmetric part vanishes); now robust (symmetric-part
axis + atan2), roundtrip exact to ~1e-13. A wrong registration gradient silently yields a wrong
pose, so these are the highest-value catches the audit could make.

## 3. vs. plain ICP + residual ablation

`benchmarks/icp_baseline_bench.py` — same recovery cells, splatreg vs ICP baselines.

| Method | SE(3) success | **Sim(3) success** |
|---|---|---|
| **splatreg (full)** | 9/9 | **27/27 = 100%** |
| ICP (centroid init) | 9/9 | **9/27 = 33%** |
| ICP (super-Fib init) | 9/9 | 9/27 = 33% |

**splatreg wins Sim(3) decisively** — plain ICP cannot estimate scale, so it fails every
non-unit-scale cell (23–25% scale error); the super-Fib init alone does not rescue it, so the
**LM Sim(3) autodiff is the load-bearing component.** Honestly: on SE(3) both reach 100% and
**ICP is ~1000× faster** (0.03 s vs 33 s) — these synthetic cells have a clean centroid offset
that favours centroid-ICP. Ablation: ICP-only is best for rigid SE(3); the SDF residual's value
is scale + implicit-field robustness, and it costs ~80× in SE(3) (see limitations).

## 4. Robustness (`benchmarks/robustness_bench.py`, 3 seeds)

| Condition | Result |
|---|---|
| **Noise** (sensor jitter 0.5–2%) | **9/9 = 100%** (rot_err < 0.72°) |
| **Outliers** (+10–50% clutter) | **9/9 = 100%** (ignores clutter, rot_err ≈ 0°) |
| Symmetric (sphere) | **9/9 = 100%** (a global-init convergence fix lands the featureless sphere correctly at all poses) |
| **Partial overlap** (20–60% removed) | **4/9 solved + 5 flagged-ambiguous** (0 silent-wrong) — mild + some moderate crops solve at 0.00°; the rest honestly flagged |

## 5. Test suite (library-bar rigor)

`pytest tests/` → **44 passing** (Jacobian audit + Lie ops + LM solver). `tests/conftest.py` (deterministic
seed fixture), `splatreg/testing.py` (a shippable `assert_residual_jacobian` so every future
residual gets the numerical audit — the GTSAM `EXPECT_CORRECT_FACTOR_JACOBIANS` equivalent).
`black` + `mypy` are clean and `splatreg/py.typed` ships.

## 5b. Official 3DMatch / 3DLoMatch (canonical Choi/Zeng protocol)

`benchmarks/threedmatch_official_bench.py` — the **canonical** protocol every published learned
method reports on: the 1279 non-adjacent `gt.log` pairs (8 test scenes), covariance-weighted
transform error `eᵀ C e ≤ 0.2²` from `gt.info`. (This **supersedes** the earlier
`threedmatch_bench.py`, whose 94.0% RR came from splatreg's *own overlapping-pair sampler* — NOT
the official protocol. That 94% is retired from all comparisons.)

| Method | 3DMatch RR | RRE | RTE | 3DLoMatch RR |
|---|---|---|---|---|
| **splatreg `learned`** (GeoTransformer LGR + our refine, **native 0.025 voxel**) | **91.5%** mean-of-scenes / 93.5% pooled | **1.81°** | **0.071 m** | **72.5%** mean / **74.4%** pooled |
| splatreg `learned` (legacy 0.05 voxel) | 86.3% / 89.1% | 1.87° | 0.071 m | 55.3% |
| splatreg `robust` (classical Open3D seed) | ~67.1% | — | — | ~15% |
| GeoTransformer (published, full coarse-to-fine) | ~92% | — | — | ~74% |
| Open3D FPFH+RANSAC (classical) | ~77% | — | — | ~20% |

**Honest reading:** the gap to GeoTransformer was an *artefact of the harness*, not the method.
The official runner pre-voxelled both fragments to **0.05 m before GeoTransformer ever saw them**
(~5 k vs ~19 k pts/fragment), throwing away >70 % of the points the learned matcher was trained on
(its native `init_voxel_size = 0.025`). Feeding GeoTransformer its native resolution and using its
full LGR pose (`estimated_transform`) as our starting pose — then layering splatreg's
overlap-residual-**guarded** ICP / Sim(3) refine on top (accepted only when it does not worsen the
overlap residual, so it never degrades the learned pose) — lifts 3DMatch **86.3 % → 91.5 %** (now
matching GeoTransformer's published ~92 %) and 3DLoMatch **55.3 % → 72.5 % mean / 74.4 % pooled**
(matching/beating GeoTransformer's published ~74 % on the pooled count). A per-pair audit (one scene,
official covariance metric) found **0 pairs where our refine demoted a GeoTransformer success**; the
refine only tightens RRE within already-successful pairs. The only change required was in the
*benchmark* (run `learned` at the GeoTransformer-native voxel via `--learned-voxel 0.025`, default);
the `learned_feature_align` path already used the full LGR pose, not a coarse seed.

## 5c. vs the splat-registration tools (real GaussianFeels splat, known GT Sim3)

`benchmarks/splat_competitors_bench.py` — a real GF splat under a known GT Sim(3); each tool recovers it.

| Tool | rot err | trans err | scale |
|---|---|---|---|
| **splatreg (SE3)** | **5.2°** | **15.7 mm** | — |
| **splatreg (Sim3)** | 11° | — | ✅ **only tool that recovers scale** |
| splatalign (ICP-from-identity) | 15.3° | — | ✗ SE(3)-only |
| GaussianSplattingRegistration (Open3D RANSAC+ICP) | 36.3° | — | ✗ SE(3)-only |

splatreg **wins outright** vs both ICP-only splat tools and is the **only** one estimating Sim(3)
scale (the others are SE(3)-only and cannot model the GT scale at all).

**Determinism note:** the `robust`/Sim(3) path seeds Open3D's RANSAC
(`o3d.utility.random.seed`, default 42) in `align_features._open3d_fpfh_ransac_seed` — without it
the draw is non-deterministic (one run hit 117° where clean reruns sit ~11°). Same input now →
bit-identical transform across runs.

## 6. Honest limitations (no overstating)

- **Partial overlap (4/9 solved + 5 flagged, 0 silent-wrong).** The `init="features"` aligner —
  an overlap-aware **point-to-plane** trimmed ICP (target→source, so the partial slab slides to
  its true tangential position) driven by a super-Fibonacci SO(3) sweep, plus FPFH — now **solves
  the mild crops** (keep ≥ 80%) plus keep60-seed0 at rot_err 0.00°. The remaining heavier crops are *inherently-ambiguous*:
  the one-sided slab deletes the disambiguating geometry, leaving the true pose only ~0.005 below
  a forest of near-equal wrong basins. There the aligner returns an **honest ambiguity flag**
  (`result.info['ambiguous']` / `['confidence']`) rather than a silent wrong pose — verified 0
  silent-wrong. Reliably solving the moderate keep60% crops is open work (a heavy standalone
  config recovered one but was not reproducible through the library, so it was not shipped).
  **`merge` is reliable for high-overlap captures.**
- **Speed — DONE (the headline).** Warm-start `track()` runs **~17 ms/frame** (< 40 ms goal, faster
  than the ~45 ms GaussianFeels tracker; `benchmarks/tracking_speed_bench.py`, rot 0.43°), via
  skip-global-init + closed-form-Jacobian LM + SDF truncation (`trunc_sigmas`, N×k). The full Sim(3)
  *registration* also dropped **19.7 → 2.4 s/cell** (closed-form gradient extended to the scale
  column). The 780 ms from-scratch SE(3) registration is global-init-dominated — a tracker never
  pays it, so it is irrelevant to the real-time goal.
- **Real splat data (`benchmarks/realdata_bench.py`, 12,463 real `.ply` exports).** CLEAN real
  geometry → Sim(3) recovery near-perfect (rot 0.03–0.06°, scale 0.04–0.14%, Chamfer 0.04–0.08 mm
  ≈ synthetic). NOISY second-capture (footprint-scale noise + 60% subsample) on near-symmetric
  objects → **global-aligner fragility** (1/9; flips into ~180° basins — NOT a Sim(3) bug, SE(3)
  fails identically). This blind-search-under-noise robustness is the **main open item**;
  `init="robust"`/`"learned"` (scale-correct seeds) address it for real scans (see §5b).

## 7. Reproduce

```bash
cd third_party/splatreg
PYTHONPATH=. python tests/test_jacobians.py     # the Jacobian audit
PYTHONPATH=. python tests/test_lie.py           # Lie-group ops
PYTHONPATH=. pytest tests/ -q                   # the suite
SPLATREG_DEVICE=cuda PYTHONPATH=. python examples/validate_recovery.py --device cuda
SPLATREG_DEVICE=cuda PYTHONPATH=. python benchmarks/robustness_bench.py --device cuda
SPLATREG_DEVICE=cuda PYTHONPATH=. python benchmarks/icp_baseline_bench.py --device cuda
CUDA_VISIBLE_DEVICES=1 SPLATREG_DEVICE=cuda python benchmarks/threedmatch_official_bench.py --split 3DMatch --init learned
CUDA_VISIBLE_DEVICES=1 SPLATREG_DEVICE=cuda python benchmarks/splat_competitors_bench.py
```
