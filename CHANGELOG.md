# Changelog

All notable changes to splatreg. Every claim below is backed by a recorded run or a test;
the full evidence trail lives in [`RESULTS.md`](RESULTS.md).

## Unreleased

### Added
- `init="bufferx"`: a zero-shot learned seed via **BUFFER-X** (ICCV 2025, "Towards Zero-Shot
  Point Cloud Registration in Diverse Scenes", MIT-SPARK/BUFFER-X) — a single generalist model
  that registers across sensors/scales with no per-dataset training — refined by the same
  overlap-aware ICP (+ Sim(3) scale) as `"learned"`/`"robust"`. Optional and lazily loaded
  (mirrors the GeoTransformer backend); falls back to `"robust"` with a logged note when its
  built CUDA extensions / Hugging Face weights are absent. Setup:
  `splatreg/third_party_models/README-BUFFERX.md`; full modern-stack build recipe
  (CUDA 12.8 / sm_120 / torch 2.11 / numpy 2.x) in `docs/BUFFERX_BUILD_MODERN_CUDA.md`. Added to
  the `register`/`splatreg align --init` choices.
- `register(init="learned", seed_gate=True)`: an opt-in (default off) Decision-PCR-style
  (arXiv 2507.14965) confidence gate that scores the learned seed (mutual-NN inlier ratio + SC²
  spatial consistency, reusing the `mac` rigidity machinery) and rejects/reseeds a low-confidence
  hypothesis from the classical `"robust"` path *before* LM refinement, instead of blindly refining
  the top seed. Scores surface in `result.info["seed_gate"]`. Tests:
  `tests/test_bufferx_seedgate.py` (fallback path + gate accepts good seed / rejects planted decoy).

### Fixed

- BUFFER-X weight loading: the pretrained checkpoints are **full-model** state_dicts (keys
  prefixed `Desc.`/`Pose.`), so loading them into the `.Desc`/`.Pose` submodules under
  `strict=False` matched nothing and silently ran on random weights (garbage seeds). Now loaded
  into the whole model, so `init="bufferx"` produces real seeds (commit `c54d8c9`).

### Changed

- Declared minimum dependency floors in `pyproject.toml` (`torch>=2.1`, `numpy>=1.24`) instead of
  unpinned `torch`/`numpy`, so a fresh install resolves an interpreter with the tensor APIs the
  package actually uses.
- `tests/test_cli.py::test_console_script_registered` now **skips** (was a hard failure) when the
  `splatreg` console-script entry point is not installed in the environment, with a note to run
  `pip install -e .`; the assertion still runs and is meaningful once the package is installed.

### Verified

- `init="bufferx"` built and run on **real 3DMatch**, both seeds pushed through the *identical*
  splatreg refine so the comparison isolates the seed. On the **official `gt.log` pair set**
  (6/8 scenes done so far, n=1250; recall = RRE < 15° and RTE < 0.3 m): BUFFER-X seed recall
  **0.974** (median RRE 1.46°) vs the classical robust FPFH seed **0.670** (1.94°); the win holds
  on the harder non-adjacent pairs (0.973 vs 0.612, n=998). The remaining two 3DMatch scenes and
  the official 3DLoMatch runs are in progress in the research project.
- Earlier GT-derived run (pairs derived from the fragments' `.info.txt` poses, 50/scene, all 8
  scenes): high-overlap (overlap ≥ 0.3, n=371) BUFFER-X 0.965 (median RRE 1.70°) vs 0.569 (3.04°);
  low-overlap 3DLoMatch regime (overlap 0.10–0.30, n=400) 0.752 (3.23°) vs 0.092 (107.9°) — an 8×
  recall lift where classical FPFH collapses to ~random. BUFFER-X wins all 8 scenes in both regimes.
- Caveat: both seeds share the lighter `feature_align` refine — a fair head-to-head that isolates
  the seed, but not the full-pipeline absolute numbers.

### Removed
- The ScanNet-GSReg (GaussReg ECCV'24) benchmark harness and all references to it.
  The dataset is not readily available, so the real-data validation anchor is the
  controlled-capture harness (`realdata_bench.py` / `bundle_real_bench.py`) instead.

## 1.3.3 (2026-06-25)

### Fixed

- Large-splat alignment no longer builds a dense source-by-target ICP distance matrix. The
  default ICP residual now honors the resolved `quality` source sample cap and evaluates nearest
  neighbors in query chunks, so `--quality low`, numeric quality values, and `quality="auto"` bound
  the ICP memory path as intended.
- The default SDF residual now receives target normals derived from the Gaussian scale/orientation
  data instead of estimating target normals with a dense target-by-target kNN pass. This removes a
  second full-target memory path for million-Gaussian PLYs.

### Verified

- Reproduced on a public 3,177,554-Gaussian PLY from `Voxel51/gaussian_splatting`. With
  `quality=0.05`, the largest observed `torch.cdist` call was `64 x 3,177,554` instead of an
  all-pairs `3,177,554 x 3,177,554` allocation.
- A controlled visual test on the same PLY recovered a known 5 degree / `[2.0, -1.2, 0.5]`
  synthetic offset with `0.000734` scene-unit translation error.

## 1.3.2 (2026-06-11)

### Fixed
- Sim(3) autodiff memory: `T` is now detached inside the `jacrev` closure of the LM
  driver, preventing the autograd graph from accumulating across iterations
  (regression-gated by `tests/test_sim3_autodiff_jacobian_detaches_T`).
- `merge_demo` synthetic mode restructured so A is the full object and B is a
  Sim(3)-transformed copy (previously A and B were different partial crops, which made
  ICP impossible); the correct `transform`/`init` are now threaded through. Chamfer on
  the demo improved 18.97 -> 0.28 mm.

## 1.3.1 (2026-06-10)

### Fixed

- SDF residual weight is now applied exactly once. `SDF.residual` / `SDF.jacobian` pre-multiplied
  by `self.weight` while the solver also folds in `sqrt(weight)`, so the effective least-squares
  objective scaled as `weight**3` (every other residual lets the solver own the weight). The default
  `register` stack `[ICP(weight=1.0), SDF(weight=0.3)]` therefore ran the SDF term at an effective
  weight of `0.3**3 ≈ 0.027` instead of `0.3`. Fixed so the SDF contribution scales linearly with
  its weight (`tests/test_weighting_and_fusion_fixes.py::test_sdf_weight_applied_once`). The feature
  (`init="learned"`) 3DMatch / 3DLoMatch headline numbers are NOT affected (that path never uses the
  SDF residual); the synthetic-recovery smoke stays at 100% of the gate, with slightly relaxed
  precision now that the SDF term contributes at its intended weight.
- `merge()` and bundle fusion now normalise every piece to the reference's `log_scales` convention
  before concatenating scales. Previously a mix of linear-scale and log-scale splats was concatenated
  raw and labelled with the reference flag, silently mis-exponentiating the odd-one-out
  (`tests/test_weighting_and_fusion_fixes.py::test_merge_preserves_scales_across_log_convention`).
- `SpatialIndex.knn()` on an index built from zero points now returns empty `(Q, 0)` results,
  matching `radius()`'s contract, instead of crashing in `topk`
  (`tests/test_weighting_and_fusion_fixes.py::test_empty_index_knn_returns_empty`).

## 1.3.0 (2026-06-10)

### Added

- `init="mac"`: MAC maximal-clique correspondence seed (Zhang et al., CVPR 2023),
  reimplemented in pure torch + networkx (`pip install "splatreg[mac]"`). SC²-weighted
  rigidity compatibility graph, Bron-Kerbosch maximal cliques as consensus hypotheses,
  weighted SVD per clique, plus a Sim(3) extension (median pairwise-distance-ratio scale).
  On synthetic contaminated sets it matches the RANSAC engine at 30/60/90% random outliers
  and wins the structured-decoy regime decisively (RANSAC ~78° failure, MAC <0.2°);
  all-outlier sets return an honest `success=False` identity (`tests/test_mac.py`).
- `seed_selector="mac"` inside `init="learned"`: MAC over GeoTransformer's correspondences.
- The measured MAC verdict on the full official splits (same forward/voxel/refine, only the
  hypothesis stage differs): a wash, not a lift. 3DLoMatch 72.1% mean / 74.6% pooled (MAC)
  vs 72.5% / 74.4% (LGR); 3DMatch 91.7% / 93.8% vs 91.5% / 93.5%; every delta within
  ±4 pairs at ~+50% runtime. `seed_selector="lgr"` stays the default; `"mac"` remains the
  tool for genuinely contaminated correspondence sets (`RESULTS.md` §5k).

### Changed

- Production sweep: dead code removed (unused private helpers in `align.py` /
  `align_features.py`, leftover unused locals), ruff clean across the package, tests, and
  benchmarks; the CLI and the Colab quickstart now use the public `apply_transform` instead
  of the private `_apply_transform_to_gaussians`.
- README restructured as a storefront: capability matrix vs the alternative tools, results
  table with per-row provenance, the 30-second quickstart with both the merge and the
  align-without-merge workflows.
- Docs site refreshed (MAC verdict, SH rotation, pose covariance, align-without-merge) and
  typography normalised across all public text.

## 1.2.0 (2026-06-10)

### Added

- Spherical harmonics rotate WITH the splat: `splatreg.sh` builds real-basis Wigner-D
  matrices for any degree (Ivanic-Ruedenberg recurrence, 1996 + the 1998 erratum) directly
  in the 3DGS sign convention; wired into `apply_transform`, `merge`, and the `align` CLI so
  the view-dependent lobes (`f_rest`) turn with a recovered transform. Math locked
  renderer-free against an independent hand-coded 3DGS basis evaluator: rotated coefficients
  evaluated at `d` equal the originals at `R⁻¹d` to ~2.4e-15 in float64
  (`tests/test_sh_rotation.py`).
- Public `apply_transform()`: the align-without-merging workflow. Register two scans, bake
  the recovered SE(3)/Sim(3) into the source, save each scan as its own PLY.
- Photometric exposure compensation (default ON): bounded per-channel gain/bias fit on the
  rendered source, alternated with the pose LM. A ×1.3 + 0.05 source tint absorbs into the
  Sim(3) scale without it (scale error 0.10% → 3.99%); with it the tinted pair recovers
  0.47% and a clean pair is unaffected (0.01%).
- Coarse-to-fine render ladder (`refine_kwargs=dict(ladder=(96, 160, 256))`): each rung
  warm-starts the next. From a 6° offset a cold 96 px rung stalls at 5.61°; the 32→64→96
  ladder lands 2.55° at equal per-stage budget.
- Pose covariance for pose graphs: builtin-LM results expose `info["information"]` (the
  undamped JᵀWJ at the final accepted linearisation; 6×6 SE(3) / 7×7 Sim(3)) and
  `info["covariance"]` (its scaled inverse; `None` if singular, never faked)
  (`tests/test_pose_covariance.py`).
- `validate_recovery.py --fast`: a CPU smoke preset of the recovery harness (same protocol
  and gates, smaller budget). Recorded: 6/6 cells within gate in ~41 s on a 2-thread CPU.
- Zenodo DOI: concept DOI 10.5281/zenodo.20618389 in `CITATION.cff` and the README badge.

## 1.1.0 (2026-06-09)

### Added

- `splatreg` CLI: `align` / `merge` / `info` from the shell, standard 3DGS PLY in/out. The
  recorded `align` run takes a source from 154 mm Chamfer off the target to 0.05 mm with no
  Python written.
- `refine="photometric"`: opt-in PhotoReg-style splat-to-splat photometric stage after the
  geometric solve, for the poses geometry cannot see (symmetry / texture-only DoF), no real
  images needed. On a rotation-symmetric colored sphere the geometric solve worsens
  6.0°→11.2° while the photometric stage lands 2.2° (real gsplat rasterizer: 5°/7 mm →
  0.36°/0.5 mm in ~1.1 s); neutral on dense-overlap pairs, hence opt-in.
- Docs site at <https://archerkattri.github.io/splatreg/> (mkdocs-material + mkdocstrings).

### Fixed

- DC-only PLY round-trip: `load_ply` used to return raw SH-DC values in the RGB slot, so a
  following `save_ply` double-encoded them and colors drifted every load→save cycle.
  DC-only loads now return true RGB and round-trip losslessly; full-SH round-trip stays
  bit-exact (`tests/test_io_roundtrip_dc.py`).

## 1.0.x (2026-06-07 / 2026-06-08)

- 1.0.0: first public release. SE(3)/Sim(3) registration behind the closed-form-Jacobian
  Gaussian-SDF residual, super-Fibonacci global init, `init="fast"/"robust"/"learned"`,
  official 3DMatch/3DLoMatch at GeoTransformer-class recall (91.5% mean / 93.5% pooled),
  real-splat merge demo (Chamfer 10.3 → 2.0 mm vs naive concat), BSD-3-Clause, PyPI.
- 1.0.1: README banner on PyPI; version sync.
- 1.0.2: v0.2/v0.3 features shipped to PyPI (object-pose mode with ADD/ADD-S/AUC, camera
  localization, multi-splat bundle registration, scene-scale spatial index).
- 1.0.3: relative paths in benchmarks/examples; discoverability README pass.
