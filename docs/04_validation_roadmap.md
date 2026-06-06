# splatreg — validation roadmap

The bar set by the libraries splatreg sits beside — **gsplat**, **Theseus**, **GTSAM
4.3**, **SymForce** — distilled from a direct study of their repos + papers, mapped to
splatreg's state, and prioritised. Goal: be as rigorous as these, or more.

## The bar (what all four enforce)

1. **Numerical-vs-analytic Jacobian check for every residual and every Lie/manifold op.**
   The single most important discipline. GTSAM's `numericalDerivative` +
   `EXPECT_CORRECT_FACTOR_JACOBIANS` (tangent-space central difference, δ=1e-5, tol 1e-5);
   SymForce's 10,000-random-sample manifold-native check at tol `10·√δ`; Theseus's
   `autograd.functional.jacobian` vs analytic at `atol=5e-7`; gsplat's dual-path
   CUDA-vs-PyTorch backward. *No hand-derived Jacobian ships unaudited.*
2. **Lie-group op tests:** exp/log + retract/local roundtrips, group invariants
   (`compose(T,inv(T))=I`, `between(A,B)=inv(A)·B`), hat/vee roundtrip, and **near-zero /
   near-π** stability (GTSAM sweeps 17 magnitudes; Theseus tests `π−1e-11`).
3. **Solver correctness:** recover a known GT to `atol≈1e-6`; convergence to a known
   solution; SymForce's **CheckLinearError** (the linearised cost predicted by J matches
   the actual post-step cost); singular-system handling (warn + FAIL status).
4. **pytest suite + deterministic seed fixture (`conftest.py`) + CI** (Black + mypy +
   pytest, multi-platform; CUDA-skipped fallback when no GPU runner).
5. **Benchmarks with published numbers:** recovery + robustness + vs-baselines, AND a
   **real-data** benchmark. For splat registration the SOTA protocol is **GaussReg**
   (ECCV 2024): ScanNet-GSReg / Objaverse, metrics **RRE/RTE/RSE/success/wall-time** vs
   HLoc + ICP. GTSAM/Theseus validate on SLAM datasets (Victoria Park, sphere2500, BAL).
6. **CI regression gates:** determinism to ~1e-10 (Theseus `test_pgo_benchmark`), worst-case
   gates, and a **PR-comment benchmark comparison** (GTSAM `timeSFMBAL` posts head-vs-base
   deltas on every PR).

## splatreg status

| Practice | Status | Where |
|---|---|---|
| Numerical Jacobian audit (ICP + SDF) | ✅ **DONE — found + fixed a real SDF bug** | `tests/test_jacobians.py` |
| Lie exp/log roundtrips | ✅ done (SE3 + Sim3, 1000 samples) | `tests/test_jacobians.py` |
| Lie group invariants / retract / hat-vee / near-π | ❌ TODO | — |
| Solver GT-recovery | ✅ via harness (36/36) | `examples/validate_recovery.py` |
| CheckLinearError / convergence-rate / singular | ❌ TODO | — |
| Synthetic recovery benchmark | ✅ done | `examples/validate_recovery.py` |
| Robustness sweep | ✅ done (25/36; partial-overlap + symmetric gaps found) | `benchmarks/robustness_bench.py`, `docs/03` |
| Competitor (vs ICP) + residual ablation | ✅ done | `benchmarks/icp_baseline_bench.py` |
| **Real-data benchmark (GaussReg / Redwood)** | ❌ TODO | — |
| pytest infra + `conftest.py` (seed fixture) | ⏳ partial (1 test file) | — |
| CI (Black/mypy/pytest) | ❌ TODO | — |
| CI regression gates + PR benchmark comparison | ❌ TODO | — |
| Determinism (1e-10) / committed real-data fixture | ❌ TODO | — |
| mypy + `py.typed` | ❌ TODO | — |

## Prioritised plan

- **P0 — numerical Jacobian audit. ✅ DONE.** Built `tests/test_jacobians.py`; it found the
  SDF Jacobian used the surface normal (a first-order proxy) instead of the true field
  gradient (max|Δ|≈10.8) → fixed to the exact autodiff gradient (now 1e-7). ICP Jacobians
  verified correct. This is exactly the win the study predicted.
- **P1 — formalise the pytest suite.** `tests/conftest.py` (seed + device fixtures, an
  `assert_success_rate` helper); a reusable `splatreg/testing.py::assert_residual_jacobian`
  (the `EXPECT_CORRECT_FACTOR_JACOBIANS` equivalent, run for every residual class); promote
  `validate_recovery` / `robustness_bench` / `_verify_sim3` to `tests/test_solver.py`,
  `tests/test_robustness.py`, `tests/test_lie.py`.
- **P2 — complete the manifold + solver tests.** Group invariants, retract/local, hat/vee,
  near-π stability; the SymForce 10k-random-sample Jacobian sweep; batch-consistency;
  CheckLinearError in the LM loop; singular-system handling.
- **P3 — CI.** `.github/workflows/test.yml`: Black + mypy + `pytest tests/` (CUDA-skip
  pattern); pre-commit (black/flake8/mypy/isort).
- **P4 — real-data benchmark.** Adopt the GaussReg protocol: a real-scan splat-pair loader
  (ScanNet-GSReg fragments or Redwood object scans), metrics RRE/RTE/RSE/success/wall-time
  vs ICP + (stretch) TEASER++; publish the table. This is the external anchor splatreg lacks.
- **P5 — CI regression gates.** Determinism-to-1e-10 regression test; worst-case gate
  (fail if any cell's rot_err > 3× the success threshold); a `timeSFMBAL`-style PR-comment
  benchmark comparison.
- **P6 — robustness fixes (from `docs/03`).** Gate + robustify the default ICP (auto
  `max_correspondence_dist` + a robust kernel), overlap-aware global init, target→source
  matching — to close the partial-overlap 0/9; degenerate-PCA skip + stability tie-break for
  the symmetric 2/9. Re-run the robustness sweep to confirm.
- **P7 — speed.** The vs-ICP benchmark showed the (now exact, autodiff) SDF adds ~80× cost
  in SE(3) for negligible benefit; a **closed-form** exact SDF gradient (vs autodiff) is the
  fast+correct version, and the SE(3) path should be able to skip the Sim(3)-autodiff route.

## References (sources of the bar)
- gsplat — [arXiv:2409.06765](https://arxiv.org/abs/2409.06765), [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)
- Theseus — [NeurIPS 2022, arXiv:2207.09442](https://arxiv.org/abs/2207.09442), [facebookresearch/theseus](https://github.com/facebookresearch/theseus)
- GTSAM 4.3 — [borglab/gtsam](https://github.com/borglab/gtsam) (`numericalDerivative.h`, `factorTesting.h`, `testLie.h`)
- SymForce — [RSS 2022, arXiv:2204.07889](https://arxiv.org/abs/2204.07889), [symforce-org/symforce](https://github.com/symforce-org/symforce)
- GaussReg (real-data benchmark protocol) — [ECCV 2024, arXiv:2407.05254](https://arxiv.org/abs/2407.05254)
