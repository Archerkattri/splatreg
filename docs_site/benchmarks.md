# Benchmarks

Every number below is measured, reproducible, and recorded with its command in
[`RESULTS.md`](https://github.com/Archerkattri/splatreg/blob/main/RESULTS.md) — including
the honest limitations. Validation is held to the bar of the libraries splatreg sits beside
(gsplat / Theseus / GTSAM / SymForce). Last validated 2026-06-07, single box, CUDA.

## Headline

| | **splatreg** | reference |
|---|---|---|
| **Real-splat merge** (real 103k-Gaussian capture) | Chamfer **10.3 → 2.0 mm (5.1×)** · overlap **0.03 → 0.67 (22×)** | naive concat |
| **vs splat competitors** (real splat, known GT Sim3) | **5.2°** (SE3) · recovers scale (Sim3) | splatalign 15.3° · GaussianSplattingRegistration 36.3° |
| **Sim(3) scale estimation** | native | none of these do it |
| **Object pose** (YCB-CAD, 14 models × 4 poses) | ADD-S AUC **0.995**, 100% < 2 cm | — |
| **Camera localization** (real splat, known perturbation) | median **5°/10 mm → 0.11°/1.35 mm**, 11/12 converged | — |
| **Official 3DMatch recall** (1279 pairs, Choi/Zeng protocol) | **91.5%** mean · 93.5% pooled | GeoTransformer ~92% · Open3D ~77% |
| **Official 3DLoMatch** (hard, 10–30% overlap) | 72.5% mean · **74.4%** pooled | GeoTransformer ~74% · Open3D ~20% |
| **Registration speed** | **~17 ms** (fast) · 104 ms (learned) | GeoTransformer ~50 ms · Open3D 142 ms |

## Synthetic recovery (known-transform)

`examples/validate_recovery.py` — apply a known Sim(3)/SE(3), recover it.
3 seeds × {5°, 30°, 90°} × {0.8, 1.0, 1.3 scale}:

| Block | Success | median rot | median trans | median scale | median Chamfer |
|---|---|---|---|---|---|
| **SE(3)** (rigid) | **9/9 = 100%** | **0.000°** | 0.10 mm | — | 0.076 mm |
| **Sim(3)** (+scale) | **27/27 = 100%** | **0.259°** | 2.93 mm | 0.344% | 0.575 mm |

## Jacobian audit

Every analytic Jacobian is checked against a tangent-space numerical one
(`tests/test_jacobians.py`, float64) — the GTSAM `EXPECT_CORRECT_FACTOR_JACOBIANS`
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

Plain ICP cannot estimate scale — it fails every non-unit-scale cell. Honest flip side: on
easy rigid SE(3), ICP is ~1000× faster; the SDF residual's value is scale + implicit-field
robustness.

## Robustness sweep

| Condition | Result |
|---|---|
| Noise (sensor jitter 0.5–2%) | **9/9**, rot < 0.72° |
| Outliers (+10–50% clutter) | **9/9**, rot ≈ 0° |
| Symmetric object (sphere) | **9/9** |
| Partial overlap (20–60% removed) | 4/9 solved + 5 flagged ambiguous — **0 silent-wrong** |

## Official 3DMatch / 3DLoMatch

Canonical Choi/Zeng protocol (1279 non-adjacent `gt.log` pairs, covariance-weighted error):

| Method | 3DMatch RR | RRE | RTE | 3DLoMatch RR |
|---|---|---|---|---|
| **splatreg `learned`** (GeoTransformer seed + guarded refine) | **91.5%** / 93.5% pooled | 1.81° | 0.071 m | 72.5% / **74.4%** pooled |
| splatreg `robust` (classical Open3D seed) | ~67.1% | — | — | ~15% |
| GeoTransformer (published) | ~92% | — | — | ~74% |
| Open3D FPFH+RANSAC | ~77% | — | — | ~20% |

The refine is *guarded* (accepted only when it does not worsen the overlap residual): a
per-pair audit found **0 pairs** where it demoted a GeoTransformer success.

## Object pose (ADD / ADD-S)

YCB `google_16k` CAD models, 14 objects × 4 poses, BOP symmetry convention:

| Observation | ADD-S AUC (0–10 cm) | median ADD-S | ADD-S < 2 cm |
|---|---|---|---|
| full view | **0.995** | 0.32 mm | 100% |
| 40% occluded | **0.995** | 0.13 mm | 100% |

Most objects recover to 0.02–0.6 mm ADD at ~0.1° rotation. The ADD/ADD-S gap is the standard
symmetry story (cans/spheres have unobservable spin); `sugar_box` is the one honest failure
(a real 180° geometric flip that only texture can break).

## Speed

| Path | splatreg | reference |
|---|---|---|
| `register(init="fast")` | **~17 ms** | — |
| `register(init="learned")` | ~104 ms | GeoTransformer ~50 ms · Open3D 142 ms |
| `Tracker.track()` warm start | **~17 ms/frame** | — |
| Full Sim(3) cold registration | 2.4 s/cell | — |

## Honest limitations

- **Overlap ≤ 40% is genuinely ambiguous** — flagged (`info["ambiguous"]`), never silently
  wrong. `merge` is designed for high-overlap captures.
- **Scale under thin overlap (~20%)** — the Sim(3) scale valley is flat; no algorithm can
  recover what the geometry doesn't carry.
- **Rigid SE(3) cost** — plain ICP reaches the same easy-case success far faster; use
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
