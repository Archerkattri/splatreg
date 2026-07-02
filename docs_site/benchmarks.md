# Benchmarks

Every number below is measured, reproducible, and recorded with its command in
[`RESULTS.md`](https://github.com/Archerkattri/splatreg/blob/main/RESULTS.md), including
the honest limitations. Validation is held to the bar of the libraries splatreg sits beside
(gsplat / Theseus / GTSAM / SymForce). Core numbers validated 2026-06-07 (single box, CUDA);
the v1.2 additions (SH rotation, exposure compensation, render ladder, pose covariance) and
the v1.3 MAC verdict validated 2026-06-10.

## Headline

| | **splatreg** | reference |
|---|---|---|
| **Real-splat merge** (real 103k-Gaussian capture) | Chamfer **10.3 → 2.0 mm (5.1×)** · overlap **0.03 → 0.67 (22×)** | naive concat |
| **vs splat competitors** (real splat, known GT Sim3) | **5.2°** (SE3) · recovers scale (Sim3) | splatalign 15.3° · GaussianSplattingRegistration 36.3° |
| **Sim(3) scale estimation** | native | none of these do it |
| **Object pose** (YCB-CAD, 14 models × 4 poses) | ADD-S AUC **0.995**, 100% < 2 cm | n/a |
| **Camera localization** (real splat, known perturbation) | median **5°/10 mm → 0.11°/1.35 mm**, 11/12 converged | n/a |
| **Official 3DMatch recall** (1279 pairs, Choi/Zeng protocol) | **91.5%** mean · 93.5% pooled | GeoTransformer ~92% · Open3D ~77% |
| **Official 3DLoMatch** (hard, 10–30% overlap) | 72.5% mean · **74.4%** pooled | GeoTransformer ~74% · Open3D ~20% |
| **Registration speed** | **~17 ms** (fast) · 104 ms (learned) | GeoTransformer ~50 ms · Open3D 142 ms |

<figure class="sr-figure">
  <img src="https://raw.githubusercontent.com/Archerkattri/splatreg/main/assets/merge_fusion.gif" alt="Three-stage animation of merging two real overlapping 3DMatch scans: misaligned, registered by SE(3), then fused with the overlap deduped">
  <figcaption>The <code>merge</code> pipeline on <strong>two real overlapping 3DMatch scans</strong> (<code>7-scenes-redkitchen</code>): register (SE(3), <strong>0.58° / 17 mm</strong> vs the 3DMatch ground truth; seam gap 101 → 18 mm, overlap 0.27 → 0.82), then fuse + voxel-dedupe the double-covered seam (<strong>38,059 → 23,502</strong> Gaussians). Measured this run. Regenerate: <code>examples/make_merge_fusion_gif.py</code>.</figcaption>
</figure>

## Synthetic recovery (known-transform)

`examples/validate_recovery.py`: apply a known Sim(3)/SE(3), recover it.
3 seeds × {5°, 30°, 90°} × {0.8, 1.0, 1.3 scale}:

| Block | Success | median rot | median trans | median scale | median Chamfer |
|---|---|---|---|---|---|
| **SE(3)** (rigid) | **9/9 = 100%** | **0.000°** | 0.10 mm | n/a | 0.076 mm |
| **Sim(3)** (+scale) | **27/27 = 100%** | **0.259°** | 2.93 mm | 0.344% | 0.575 mm |

## Jacobian audit

Every analytic Jacobian is checked against a tangent-space numerical one
(`tests/test_jacobians.py`, float64), the GTSAM `EXPECT_CORRECT_FACTOR_JACOBIANS`
discipline. The audit **found and fixed a real bug**: the Gaussian-SDF gradient had dropped
the first-order `∂q̃/∂p` term; it is now an exact closed-form field gradient (max
|analytic − numerical| ≈ 1e-8). ICP point-to-point ~3e-9, point-to-plane ~4e-11; SE(3)/Sim(3)
`exp`/`log` round-trips exact to ~1e-13 including the near-π branch.

## vs plain ICP (residual ablation)

| Method | SE(3) success | **Sim(3) success** |
|---|---|---|
| **splatreg (full)** | 9/9 | **27/27 = 100%** |
| ICP (centroid init) | 9/9 | 9/27 = 33% |
| ICP (super-Fib init) | 9/9 | 9/27 = 33% |

Plain ICP cannot estimate scale: it fails every non-unit-scale cell. Honest flip side: on
easy rigid SE(3), ICP is ~1000× faster; the SDF residual's value is scale + implicit-field
robustness.

## Robustness sweep

| Condition | Result |
|---|---|
| Noise (sensor jitter 0.5–2%) | **9/9**, rot < 0.72° |
| Outliers (+10–50% clutter) | **9/9**, rot ≈ 0° |
| Symmetric object (sphere) | **9/9** |
| Partial overlap (20–60% removed) | 4/9 solved + 5 flagged ambiguous, **0 silent-wrong** |

## Official 3DMatch / 3DLoMatch

Canonical Choi/Zeng protocol (1279 non-adjacent `gt.log` pairs, covariance-weighted error):

| Method | 3DMatch RR | RRE | RTE | 3DLoMatch RR |
|---|---|---|---|---|
| **splatreg `learned`** (GeoTransformer seed + guarded refine) | **91.5%** / 93.5% pooled | 1.81° | 0.071 m | 72.5% / **74.4%** pooled |
| splatreg `learned`, `seed_selector="mac"` (MAC cliques, same forward/refine) | 91.7% / 93.8% | 1.83° | 0.071 m | 72.1% / 74.6% pooled |
| splatreg `robust` (classical Open3D seed) | ~67.1% | n/a | n/a | ~15% |
| GeoTransformer (published) | ~92% | n/a | n/a | ~74% |
| Open3D FPFH+RANSAC | ~77% | n/a | n/a | ~20% |

The refine is *guarded* (accepted only when it does not worsen the overlap residual): a
per-pair audit found **0 pairs** where it demoted a GeoTransformer success.

The `seed_selector="mac"` row is the measured answer to "does the MAC paper's ~71→78 %
3DLoMatch lift transfer?": **no, a wash** (every delta within ±4 pairs, ~+50 % runtime), because
at native voxel GeoTransformer's correspondences are already consensus-dominated (median
600–800 MAC inliers) and the guarded refine absorbs seed-level differences. `lgr` stays the
default; details in `RESULTS.md` §5k.

## BUFFER-X zero-shot seed vs classical seed (real 3DMatch)

`init="bufferx"` swaps the learned seed for **BUFFER-X** (ICCV 2025), a single zero-shot model
that registers across sensors and scales with no per-dataset training. The BUFFER-X seed and the
classical robust FPFH seed are pushed through the *identical* splatreg refine, so the numbers
below isolate the seed rather than the pipeline. Recall counts a pair as recalled at
**RRE < 15° and RTE < 0.3 m**.

| Regime | BUFFER-X seed | classical robust seed | pair set |
|---|---|---|---|
| **3DMatch** (n=1619) | **0.962** · median RRE 1.46° | 0.630 · 2.12° | complete official `gt.log`, 8/8 scenes |
| **3DLoMatch** (n=1781) | **0.777** · 2.77° | 0.122 · 103.4° | complete official `gt.log` |
| 3DLoMatch regime, earlier GT-derived run (n=400) | 0.752 · 3.23° | 0.092 · 107.9° | 50/scene, `.info.txt`-derived |

<figure class="sr-figure" markdown="span">
  <img src="https://raw.githubusercontent.com/Archerkattri/splatreg/main/assets/bufferx_recall.png" alt="BUFFER-X zero-shot seed vs classical FPFH seed: registration recall on 3DMatch and the low-overlap regime">
  <figcaption>Zero-shot BUFFER-X seed vs the classical FPFH seed, identical splatreg refine. Final numbers are the complete official <code>gt.log</code> pair sets (3DMatch 8/8 scenes; official 3DLoMatch). Both seeds share the lighter <code>feature_align</code> refine — a fair head-to-head that isolates the seed rather than reporting full-pipeline absolute numbers. BUFFER-X wins every scene on both splits.</figcaption>
</figure>

## Object pose (ADD / ADD-S)

YCB `google_16k` CAD models, 14 objects × 4 poses, BOP symmetry convention:

| Observation | ADD-S AUC (0–10 cm) | median ADD-S | ADD-S < 2 cm |
|---|---|---|---|
| full view | **0.995** | 0.32 mm | 100% |
| 40% occluded | **0.995** | 0.13 mm | 100% |

Most objects recover to 0.02–0.6 mm ADD at ~0.1° rotation. The ADD/ADD-S gap is the standard
symmetry story (cans/spheres have unobservable spin); `sugar_box` is the one honest failure
(a real 180° geometric flip that only texture can break).

## Photometric refinement (new in v1.1)

The opt-in `refine="photometric"` stage, measured on three regimes (full table, scoping and
PhotoReg positioning: [Photometric refinement](photometric.md); recorded runs:
[`benchmarks/photometric_refine_results.md`](https://github.com/Archerkattri/splatreg/blob/main/benchmarks/photometric_refine_results.md)):

| Case | Geometric register | + photometric refine |
|---|---|---|
| Rotation-symmetric colored sphere (mock renderer, CPU) | 6.0° → **11.2°** (worse) | **2.2°** |
| Real gsplat rasterizer (CUDA), from 5°/7 mm | n/a | **0.36°/0.5 mm** in ~1.1 s |
| Dense-overlap real 103k pair, injected 2°/1.24 mm seam | **0.239°/0.26 mm** in 56 s | +1.7 s, neutral |

Decisive when geometry under-constrains the pose (symmetry / texture-only DoF); neutral when
dense overlap already pins it; floor set by render resolution (~0.3°). Hence opt-in.

<figure class="sr-figure">
  <img src="https://raw.githubusercontent.com/Archerkattri/splatreg/main/assets/photometric_refine.gif" alt="Photometric refinement converging: a colour splat knocked 9 degrees out of alignment locks onto the target through the gsplat rasterizer, with rotation and translation error ticking down to zero">
  <figcaption>A colour splat knocked <strong>9° / 151 mm</strong> out of alignment, polished by the splat-vs-splat photometric LM through the <strong>gsplat</strong> rasterizer down to <strong>0.04° / 0.04 mm</strong> — a real per-iteration trajectory (LM damping raised so the steps are visible). Regenerate: <code>examples/make_photometric_refine_gif.py</code>.</figcaption>
</figure>

## SH rotation, exposure compensation, ladder, covariance (v1.2)

Each addition ships with its measured evidence (full detail: `RESULTS.md` §5j):

| Addition | Evidence |
|---|---|
| SH (`f_rest`) Wigner rotation | rotated coefficients evaluated at `d` equal the originals at `R⁻¹d`, measured **~2.4e-15** in float64 (gate < 1e-5); `D(R₁R₂) = D(R₁)D(R₂)` exact; PLY round-trip exact ([`tests/test_sh_rotation.py`](https://github.com/Archerkattri/splatreg/blob/main/tests/test_sh_rotation.py)) |
| Exposure compensation (default ON) | a ×1.3 + 0.05 source tint absorbs into the Sim(3) scale without it (0.10% → **3.99%** scale error); with it: **0.47%**, fitted gain ≈ 1/1.3; clean pair 0.01% (harmless) |
| Coarse-to-fine render ladder | from a 6° offset a cold 96 px rung stalls at **5.61°**; the 32→64→96 ladder lands **2.55°** at equal per-stage budget |
| Pose information / covariance | SPD on well-constrained solves, 2× noise → looser covariance, singular → `None` ([`tests/test_pose_covariance.py`](https://github.com/Archerkattri/splatreg/blob/main/tests/test_pose_covariance.py)) |
| `validate_recovery.py --fast` | CPU smoke preset: **6/6 cells within gate in ~41 s** (worst rot err 0.16°, worst scale err 0.14%) |

<figure class="sr-figure">
  <img src="https://raw.githubusercontent.com/Archerkattri/splatreg/main/assets/sh_rotation.png" alt="A view-dependent-coloured Gaussian sphere rotated 90 degrees three ways and rendered by gsplat: naive rotation (wrong colour), splatreg Wigner-D (correct), and an independent ground truth">
  <figcaption>The SH-rotation row, rendered through <strong>gsplat</strong>: a view-dependent-coloured splat rotated 90° — the <strong>naive</strong> rotation (SH left in the old frame) is <strong>13–15 dB</strong> off an independent ground truth, while the real-basis <strong>Wigner-D</strong> render is <em>pixel-identical</em> to it; coefficient round-trip to <strong>~2e-16</strong> in float64. Regenerate: <code>examples/make_sh_rotation_figure.py</code>.</figcaption>
</figure>

## Speed

| Path | splatreg | reference |
|---|---|---|
| `register(init="fast")` | **~17 ms** | n/a |
| `register(init="learned")` | ~104 ms | GeoTransformer ~50 ms · Open3D 142 ms |
| `Tracker.track()` warm start | **~17 ms/frame** | n/a |
| Full Sim(3) cold registration | 2.4 s/cell | n/a |

## Honest limitations

- **Overlap ≤ 40% is genuinely ambiguous**: flagged (`info["ambiguous"]`), never silently
  wrong. `merge` is designed for high-overlap captures.
- **Scale under thin overlap (~20%)**: the Sim(3) scale valley is flat; no algorithm can
  recover what the geometry doesn't carry.
- **Rigid SE(3) cost**: plain ICP reaches the same easy-case success far faster; use
  `Tracker` for real time.

Full detail, including the failure analyses: [`RESULTS.md`](https://github.com/Archerkattri/splatreg/blob/main/RESULTS.md).

## Reproduce

```bash
git clone https://github.com/Archerkattri/splatreg.git && cd splatreg
pip install -e ".[test]"
python -m pytest tests/ -q                          # the suite (incl. Jacobian audit)
SPLATREG_DEVICE=cuda python examples/validate_recovery.py --device cuda
SPLATREG_DEVICE=cuda python benchmarks/robustness_bench.py --device cuda
SPLATREG_DEVICE=cuda python benchmarks/icp_baseline_bench.py --device cuda
SPLATREG_DEVICE=cuda python benchmarks/threedmatch_official_bench.py --split 3DMatch --init learned
```
