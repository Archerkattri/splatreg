# Changelog

All notable changes to splatreg. Every claim below is backed by a recorded run or a test;
the full evidence trail lives in [`RESULTS.md`](RESULTS.md).

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
