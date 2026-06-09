# Photometric refinement — recorded results

Reference runs for `refine="photometric"` (the PhotoReg-style splat-to-splat stage,
`splatreg/residuals/photometric.py`). Narrative + when/why guidance:
[docs — Photometric refinement](https://archerkattri.github.io/splatreg/photometric/).

## 1. Symmetry case (mock renderer, CPU)

Scenario from `tests/test_photometric_refine.py::test_register_refine_beats_geometric_on_symmetric_splat`:
two resamplings (500 Gaussians each) of the same sphere — geometry rotation-symmetric,
colors azimuth-painted — so the true aligning transform is identity and rotation is carried
**only by color**. Re-measured on CPU (`OMP_NUM_THREADS=2`, pure-torch mock renderer,
6 views × 32 px):

| stage | rot err (deg) | trans err (mm) |
|---|---|---|
| injected offset | 6.00 | 8.94 |
| geometric `register` (ICP+SDF) | **11.21** ← worse | 1.56 |
| `refine="photometric"` | **2.20** | 1.01 |

The geometric stage fixes translation but walks the rotation *away* from the truth — every
rotation about the sphere axis scores identically to ICP/SDF. The photometric stage sees the
azimuth paint and recovers it. (The residual 2.2° is the mock renderer's 32-px resolution
floor; the real rasterizer case below lands at 0.36°.)

## 2. Real gsplat rasterizer (CUDA)

`tests/test_photometric_refine.py::test_gsplat_refine_reduces_pose_error` —
2,000-Gaussian color sphere, 6 views × 64 px, 10 LM iterations:

| stage | rot err (deg) | trans err (mm) | time |
|---|---|---|---|
| injected offset | 5.0 | ~7 | — |
| `refine_photometric` | **0.36** | **0.5** | ~1.1 s |

The Sim(3) variant (`test_gsplat_refine_sim3`) additionally pulls a 5% injected scale error
back to 0.9997 via silhouette size.

## 3. Dense-overlap real capture (GPU reference run)

`benchmarks/photometric_refine_bench.py`, defaults — the real 102,944-Gaussian capture used
by the merge headline (extent 201 mm, SH degree 3), random-split into two disjoint halves,
source half moved by a known 2° / 1.24 mm (0.5% of extent) seam error; geometric stage on
8k-anchor subsamples, photometric on the full halves, 6 views × 96 px:

| stage | rot err (deg) | trans err (mm) | time |
|---|---|---|---|
| injected seam error | 2.000 | 1.24 | — |
| geometric `register` | **0.239** | **0.26** | 56 s |
| + photometric refine | ≈ unchanged (**neutral**) | ≈ unchanged | +1.7 s |

Honest scoping, recorded as measured: with dense overlapping geometry the geometric stage
already lands *below* the photometric stage's render-resolution floor (~0.3° at this ring),
so the photometric stage neither helps nor hurts here. It is decisive only when geometry
under-constrains the pose (case 1/2) — which is why it ships opt-in.

## Reproduce

```bash
python -m pytest tests/test_photometric_refine.py -q     # 21 tests (CPU; gsplat cases auto-skip without CUDA)
python benchmarks/photometric_refine_bench.py            # case 3 (needs GPU + the capture PLY; --ply to point elsewhere)
```
