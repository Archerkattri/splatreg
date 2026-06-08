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

## 5d. Real-splat merge (`examples/merge_demo.py`, real 103k-Gaussian capture)

Two overlapping captures of a real 103k-Gaussian splat fused into one deduped `.ply` (registered
Sim(3) + voxel/KNN overlap dedupe), vs a naive `torch.cat`:

| Metric | naive cat | splatreg merge |
|---|---|---|
| Chamfer to GT | 10.3 mm | **2.0 mm (5.1× closer)** |
| overlap (IoU-style) | 0.03 | **0.67 (22× more)** |
| overlap duplicates removed | — | ~9k |

Verified on GPU across two independent runs.

## 5e. Registration / tracking speed (`benchmarks/tracking_speed_bench.py`, `speed_bench.py`)

| Path | splatreg | reference |
|---|---|---|
| `register(init="fast")` (objects / full overlap) | **~17 ms** | — |
| `register(init="learned")` (GeoTransformer seed + refine) | ~104 ms | GeoTransformer ~50 ms · Open3D 142 ms |
| `track()` warm-start (real-time) | **~17 ms/frame** (rot 0.43°) | GaussianFeels tracker ~45 ms |
| full Sim(3) cold registration | 2.4 s/cell (was 19.7 s) | — |

**Determinism note:** the `robust`/Sim(3) path seeds Open3D's RANSAC
(`o3d.utility.random.seed`, default 42) in `align_features._open3d_fpfh_ransac_seed` — without it
the draw is non-deterministic (one run hit 117° where clean reruns sit ~11°). Same input now →
bit-identical transform across runs.

## 5f. 6-DoF object-pose mode (v0.2, FoundationPose/YCB-style ADD/ADD-S)

`benchmarks/object_pose_bench.py` — estimate a known object splat's 6-DoF pose `T_SO` from an
observation, scored with the standard **ADD / ADD-S / AUC** metrics (`splatreg.add_metric` /
`adds_metric` / `add_auc`; unit-tested in `tests/test_object_pose.py`). The estimator reuses the
`register`/`track` core; `ObjectPoseEstimator` warm-starts across frames (FoundationPose regime).

| Observation | ADD-S AUC (0–10 cm) | median ADD-S | ADD < 2 cm | median rot |
|---|---|---|---|---|
| full view (keep 1.0) | **0.999** | 0.10 mm | 100% | 0.04° |
| 60% occluded (keep 0.6) | **0.985** | 1.43 mm | 100% | 0.54° |

**Honest scope:** numbers are on a **synthetic proxy** (procedural object splat), NOT real
YCB-Video / FoundationPose data — that needs a real loader (`iter_dataset()` is the documented swap
point; the metric + estimate path is dataset-agnostic and already validated). ADD-S handles symmetry
correctly (a 180° sphere flip → ADD-S ≈ 0 while ADD is large; tested). The partial-view limit is the
same as `register` — heavy occlusion can leave the pose ambiguous, surfaced via `info['ambiguous']`.

## 5g. Camera localization in a splat (v0.2)

`localize_camera(splat, frame, init_T_WC)` (`tests/test_camera_loc.py`) — refine a query camera pose
against a world-fixed splat by optimizing the SE(3) right-perturbation tangent **through gsplat's
differentiable rasteriser** (exact render gradient, correct by construction — no hand-derived
inverse-compositional Jacobian to mis-sign). On a synthetic textured scene:

| init error | recovered |
|---|---|
| rot 7.1° / trans 132 mm | **rot 1.7° / trans 37 mm** |
| rot 3.5° / trans 66 mm | rot 1.4° / trans 32 mm |
| rot 1.4° / trans 26 mm | rot 0.6° / trans 14 mm |

**Honest scope:** this **refines a pose prior** within the direct-alignment basin (a few degrees / a
few % of depth) — it is not yet a *fine* global relocaliser, and is evaluated on a synthetic
scene only. An experimental analytic residual (`CameraPhotometric`, geometry block verified vs
numerical) is also shipped, but the differentiable-render path is the validated one. *Note:* the
pre-existing object-pose `Photometric` residual's inverse-compositional analytic Jacobian was found
to have the same narrow/sign-sensitive basin on these synthetic scenes — which is why the v0.2
camera path uses differentiable rendering instead.

### v0.3 hardening — wide-baseline coarse seed (`tests/test_camera_loc_coarse.py`)

The refine basin above means a *wide-baseline* query (no good prior) is unrecoverable by refinement
alone. `coarse_localize_camera(splat, frame)` supplies the missing seed by a **projection-only
silhouette-overlap viewpoint sweep**: it scores a sphere of look-at candidate poses by the IoU of the
splat's projected (dilated) occupancy against the query's foreground silhouette (`frame.mask`, or a
luminance cue from `frame.rgb`). It is **pure pinhole projection** — no rasteriser — so the seed runs
on **CPU with no gsplat/CUDA**; `localize_camera(..., init_T_WC="coarse")` chains it into the
differentiable-render refine. On a chiral-shape (letter-`F`) wide-baseline case:

| | rotation error | silhouette IoU |
|---|---|---|
| Wide-baseline prior (back view) | 180° (unrecoverable by refine) | — |
| **Coarse seed (prior-free sweep)** | **~22.5°** (≈ one grid step, in the refine basin) | **0.58** |

The rgb-foreground cue reproduces the mask cue **exactly** (same seed chosen). *Honest scope:* the
seed is **grid-resolution** (the azimuth step bounds its accuracy — it gets you *into* the refine
basin, not to the final pose) and the silhouette score is viewpoint-discriminative only for an
asymmetric object (a symmetric blob stays ambiguous — a fundamental limit of any silhouette cue).
Synthetic only.

## 5h. Multi-splat joint / bundle registration (v0.3)

`bundle_register(splats, ref=0, pairs="auto")` (`tests/test_bundle.py`) — register `N` overlapping
splats *jointly* into one loop-consistent frame. It builds a relative-pose constraint `T_ij` per
overlapping pair (each via the existing `register`), then optimises all `N` **absolute** poses over a
**pose graph** — Gauss-Newton in the SE(3)/Sim(3) tangent (edge residual
`e_ij = log((T_i T_ij)^{-1} T_j)`, Jacobians autodiffed through `core/lie`'s `exp`/`log`), with the
reference pose pinned to fix the gauge. The sequential merge-to-ref chains the edges, so on a *loop*
it dumps all accumulated drift onto the loop-closure edge; the joint optimum spreads it.

Validated on a synthetic **ring** of 5 noisy captures (the same object placed at known poses around a
loop, with per-capture footprint-scale jitter so the pairwise solves carry real error):

| | max pairwise inconsistency | mean |
|---|---|---|
| Sequential chain (merge-style) | 3.7e-2 | 7.5e-3 |
| **Joint bundle** | **7.5e-3** | 7.5e-3 |
| | **~5× lower max** | (≈ equal) |

The headline is the **~5× drop in the worst-edge inconsistency** — the loop closes. The *mean* edge
error is essentially unchanged, which is the honest, correct behaviour: the joint solve is a
least-squares optimum that *redistributes* the same total measurement error off the worst (loop-
closure) edge and across the graph; it does not reduce the total residual (the pairwise measurements
themselves carry the noise floor). `fuse=True` returns one merged splat baked from the jointly
optimised poses. *Scope:* synthetic loop only.

### v0.3 hardening — robust outlier-edge rejection (`tests/test_bundle.py`)

A wrong pairwise `register` result (a bad edge) must not corrupt the global poses. The pose-graph
solve (`solve_pose_graph`, the core of `bundle_register`) is now an **IRLS** with a per-edge
**Huber/Cauchy** robust kernel (`robust="huber"` default, `robust=None` recovers plain least
squares) plus a **graduated-non-convexity (GNC)** schedule. GNC is the load-bearing part: plain IRLS
is initialisation-dependent — if the seed chain already *satisfies* a bad edge it has a tiny residual
at iteration 0 and IRLS keeps it while down-weighting an innocent neighbour — so the solve starts
near-convex (a large robust scale, every edge ~unit weight) and **anneals the scale down** so the true
outlier surfaces as the largest residual and is rejected. The robust scale auto-adapts each iteration
via a MAD estimate (`1.4826·median ‖e_ij‖`, itself outlier-proof).

Validated by injecting one gross blunder (~40° rotation + 15 cm) on a single edge of a **redundant**
graph (all-pairs, so the outlier has a consistent majority to disagree with):

| | recovered-pose error vs clean solution | bad-edge weight |
|---|---|---|
| Un-gated least squares | 7.9e-2 | (full weight — corrupts the solve) |
| **Robust (Huber + GNC)** | **1.3e-3** (~**60× better**) | **≈ 0.01** (rejected) |

Good edges keep substantial weight; the rejected edge is reported in `info.rejected_edges` /
`info.edge_weights`. **Honest topological limit:** a *bare ring* (every node degree 2) has **no**
redundancy — one bad edge's error spreads perfectly evenly over the loop and is mathematically
indistinguishable from the others at the least-squares optimum, so **no** robust kernel can localise
it there (verified; this is why the test uses a redundant graph — the realistic loop-closure case).

## 5i. Scene-scale spatial index (v0.3)

`SpatialIndex` / `build_index(splat)` (`tests/test_spatial_index.py`) — a voxel-hash grid over the
Gaussian means supporting **exact** `knn` / `radius` / `region` queries, so the SDF / dedupe / merge
query path scales past the brute-force `cdist`. Wired as an **opt-in acceleration** (brute force stays
the default/fallback): `gaussian_sdf(..., index=idx)` serves the truncated-SDF support, and
`knn_dedupe(..., use_index=True)` serves the cross-splat overlap dedupe.

* **Correctness:** `knn` / `radius` / `region` return identical sets to a brute-force scan (the grid
  only prunes which anchors are distance-tested). The index-accelerated SDF matches the brute path to
  float32 tolerance (sdf max-Δ ≈ 4e-6).
* **Speedup (cross-splat radius dedupe, the O(N²) case, CPU, 2 threads):**

  | anchors N | brute `cdist` | index | speedup |
  |---|---|---|---|
  | 48,000 | 12.1 s | 2.6 s | **4.6×** |
  | ~115,000 | ~48 s | ~4 s | **~12×** |

  Survivor set identical up to a negligible float-boundary fraction (~8e-5 — a couple of
  exactly-on-the-radius duplicate pairs round across the two distance kernels; honest FP ties, not a
  logic difference).

### v0.3 hardening — vectorised loop-free batch queries (`tests/test_spatial_index.py`)

The original `knn` / `radius` loop over queries in Python, so on a moderate cloud with many queries
the per-query overhead dominated. `SpatialIndex.radius_batch` / `knn_batch` enumerate **all** queries'
candidates in **one vectorised pass** — every query's `±ring` neighbour cells built as a tensor, each
cell's anchor run located by a single `searchsorted` into the sorted unique cell keys (no Python dict
walk), the runs expanded into flat `(query, anchor)` candidate pairs via `repeat_interleave`, then a
single batched distance test (radius) or padded-scatter `topk` (knn). The result is **exact** — vs
brute force **and** vs the looped path — only the candidate pruning is shared.

| query op | looped | batch | speedup |
|---|---|---|---|
| radius (4000 queries, 8k anchors, CPU/2-thread) | 0.24 s | 0.016 s | **~15×** |
| knn k=8 (same) | 0.73 s | 0.12 s | **~6×** |

This is the regime warm-start tracking hits (many small queries per frame); the scene-scale O(N²)
dedupe win above is unchanged.

*Scope:* on a *small* cloud the heavily-vectorised brute `cdist` can still win outright; the index
(looped or batch) is the scene-scale / many-query path.

## 6. Honest limitations (no overstating)

- **Partial overlap (6/9 solved + 3 flagged, 0 silent-wrong).** The `init="features"` aligner —
  an overlap-aware **point-to-plane** trimmed ICP (target→source, so the partial slab slides to
  its true tangential position) driven by a super-Fibonacci SO(3) sweep, plus FPFH — now **solves
  ALL keep ≥ 60% crops at rot_err 0.00°** (previously only keep ≥ 80% + keep60-seed0; the moderate
  keep60-seed1/2 used to flip into ~175° mirror basins). Two changes fixed them, both verified on
  the robustness sweep: (1) the basin sweep keeps a **deeper candidate pool** (`topk` 40 → 200) and
  refines longer (`refine_iters` 60 → 150) — at keep60 the true basin's coarse seed ranks ~80–160
  in the cheap prefilter, so a shallow pool dropped it before the precise refine; and (2) the
  refined seeds are ranked by a **symmetric overlap residual** (target→source *and* source→target),
  which penalises the mirror flip the one-directional residual is blind to (true pose sym ≈ 0.000 vs
  flip ≈ 0.014). Cost: the deeper sweep is ~22 s/cell — registration path only, never the real-time
  tracker. The remaining heavy crops (keep ≤ 40%) are *genuinely ambiguous*: there even the true
  pose no longer seats (symmetric residual ≈ 0.003 against a forest of ≈ 0.017 wrong basins), so the
  aligner returns an **honest ambiguity flag** (`result.info['ambiguous']` / `['confidence']`)
  rather than a silent wrong pose — verified 0 silent-wrong. **`merge` is reliable for high-overlap
  captures.**
- **Sim(3) scale under low overlap (line-search, still loose at ~20%).** A dedicated golden-section
  **scale line-search** (`_scale_line_search`) refines the Sim(3) scale DoF after the pose solve,
  minimising the **symmetric** overlap residual — the one-directional fit is scale-blind (shrinking
  the source toward the overlap keeps every target point near *some* source point, so it never
  penalises a too-small scale; the source→target term is what gives scale a real minimum). It
  improves scale on its own objective without regressing the robustness sweep (still 36/36), but a
  thin shared band leaves a wide scale valley, so under ~20% overlap the recovered scale can still
  drift. Note: the real-100k-PLY merge demo's FPFH seed is non-deterministic run-to-run, so its
  scale number is not a stable before/after — the line-search is reported on its objective, not a
  single demo figure.
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
