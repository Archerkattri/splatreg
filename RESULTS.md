# splatreg — validation results

**What splatreg is:** an open, pure-PyTorch library for **registering Gaussian splats** —
align/merge two splats into one SE(3)/Sim(3) frame via a global aligner + multi-residual
Levenberg-Marquardt (the inverse of gsplat: gsplat *renders* Gaussians, splatreg *registers*
against them).

**What this file is:** every claim, with the measured number and the command to reproduce
it — and the honest limitations. Validation is held to the bar of the libraries splatreg
sits beside (gsplat / Theseus / GTSAM / SymForce); see `docs/04_validation_roadmap.md`.

_Last validated 2026-06-06, single box, CUDA._

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
| Symmetric (sphere) | **9/9 = 100%** with `init="features"` (8/9 with global init) |
| **Partial overlap** (20–60% removed) | **0/9** — see limitations (a real + partly *inherent* gap) |

## 5. Test suite + CI (library-bar rigor)

`pytest tests/` → **30 passing** (Jacobian audit + Lie ops + LM solver). `tests/conftest.py` (deterministic
seed fixture), `splatreg/testing.py` (a shippable `assert_residual_jacobian` so every future
residual gets the numerical audit — the GTSAM `EXPECT_CORRECT_FACTOR_JACOBIANS` equivalent),
`.github/workflows/test.yml` (CI across Python 3.10–3.12). Roadmap to full parity:
`docs/04_validation_roadmap.md`.

## 6. Honest limitations (no overstating)

- **Partial overlap (0/9).** A genuine, known-hard problem (the territory of feature-based
  methods like TEASER++/Predator). Investigated in `docs/03`: gating the fine ICP helps only
  marginally (2/9, fragile), and the random-direction crop conflates *fixable* partial (overlap
  keeps the disambiguating feature) with *inherently-ambiguous* partial (the crop deletes the
  feature → unrecoverable by **any** method). The credible fix is a feature-based robust aligner
  (FPFH + TEASER/RANSAC) + honest "ambiguous" reporting — scoped as the next major feature, not
  a quick patch. **`merge` is reliable for high-overlap captures; large partial overlap is WIP.**
- **SE(3) speed.** Dominated by the SDF field evaluation. The **closed-form gradient is now
  landed** (no autograd graph + no second forward on the SE(3) path — the correctness half of the
  speed work); the wall-time benchmark vs the GaussianFeels tracker + SDF truncation
  (`trunc_sigmas`, N×k instead of N×M) are the GPU follow-up.
- **Real-scan data.** Validation so far is synthetic-known-transform + robustness corruptions;
  a real-scan benchmark (GaussReg's ScanNet-GSReg protocol — RRE/RTE/RSE/success/time) is the
  next external anchor (`docs/04` P4).

## 7. Reproduce

```bash
cd third_party/splatreg
PYTHONPATH=. python tests/test_jacobians.py     # the Jacobian audit
PYTHONPATH=. python tests/test_lie.py           # Lie-group ops
PYTHONPATH=. pytest tests/ -q                   # the suite
SPLATREG_DEVICE=cuda PYTHONPATH=. python examples/validate_recovery.py --device cuda
SPLATREG_DEVICE=cuda PYTHONPATH=. python benchmarks/robustness_bench.py --device cuda
SPLATREG_DEVICE=cuda PYTHONPATH=. python benchmarks/icp_baseline_bench.py --device cuda
```
