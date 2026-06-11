# Changelog — splatreg

All notable changes, per version. Auto-generated from git tags by
`third_party/launch_materials/gen_changelogs.sh`; do not edit by hand.

## Unreleased

- docs: add per-version CHANGELOG (d0d1dfc)
- Remove ScanNet-GSReg benchmark and references (dataset not available) (95a4459)

## v1.3.2 — 2026-06-11

- chore(release): v1.3.2 (8c0a9f0)
- fix(merge_demo): restructure synthetic mode + correct transform/init selection (9835d54)
- test(sim3): regression-gate the autodiff memory fix (T detached in jacrev closure) (c421b6a)
- fix(sim3): detach T in autodiff Jacobian closure to prevent graph accumulation (bf07848)
- fix(doi): update Zenodo DOI to correct concept record (a1255b4)
- Correct MAC published gains to the primary source (+3.7/+3.9 points, CVPR 2023 Table 3) (79f81f8)
- bench: add official ScanNet-GSReg (GaussReg ECCV'24) 82-scene Sim(3) protocol harness (d6bae9b)
- Fix MAC author attribution (CVPR 2023: Zhang, Yang, Zhang and Zhang) (eb56315)
- chore: gitignore paper/ (local-only drafts) (3a44b97)

## v1.3.1 — 2026-06-10

- v1.3.1 (a79b26d)
- fix: apply SDF residual weight once, normalise log_scales on fusion, guard empty-index knn (5ab6483)

## v1.3.0 — 2026-06-10

- production sweep: dead-code removal, ruff-clean, public apply_transform in CLI/notebook, docstring typography (859ab95)
- MAC-seed verdict measured: official 3DMatch/3DLoMatch, both selectors — a wash, lgr stays default (20e10a2)
- init='mac': MAC maximal-clique correspondence seed (Zhang et al. CVPR 2023) (240c5d9)

## v1.2.0 — 2026-06-10

- v1.2: results-backed docs for the SH/exposure/ladder/covariance/--fast additions (27ddeb5)
- validate_recovery.py --fast: <2 min CPU smoke preset of the recovery harness (b1570be)
- expose pose information/covariance on RegisterResult.info (pose-graph use) (65540d2)
- photometric refine: per-pair exposure compensation + coarse-to-fine render ladder (6c9fa4e)
- SH Wigner rotation: f_rest now rotates with the splat (Ivanic-Ruedenberg) (4aacc2d)
- api: public apply_transform() — align scans without merging (917bc80)
- DOI badge: static shields.io (Zenodo badge endpoint 302s through GitHub's proxy) (4c4d8eb)
- CITATION.cff: real Zenodo concept DOI (61f9d85)
- Add Zenodo DOI (CITATION.cff + README badge) (865ddf0)

## v1.1.0 — 2026-06-09

- v1.1.0: results-backed docs for the new additions (bdb5445)
- CLI, docs site, photometric refinement, DC PLY round-trip fix (53a0ade)
- chore: use relative paths in benchmarks/examples; update test count (f771727)

## v1.0.3 — 2026-06-08

- chore: bump version to 1.0.3 (0892a43)
- docs: restructure README for discoverability — hook first, install/quickstart before deep tech (9a29e0b)
- docs: drop roadmap, trim limitations to genuine immovable edges only (87c81ad)
- docs: restructure roadmap — condense shipped items, remove non-workable todos, clean scope statement (071ecac)

## v1.0.2 — 2026-06-08

- release: v1.0.2 — ship v0.2/v0.3 features (object-pose, camera-loc, bundle, spatial-index) to PyPI; sync __version__ 0.0.1->1.0.2 (1bbe724)
- docs(v0.2): README — YCB-CAD object-pose ADD-S AUC 0.995, corrected official-protocol gap (b06b880)
- bench(v0.2): YCB-CAD object-pose — ADD-S AUC 0.995 on the canonical google_16k models (7f47c30)
- bench(v0.2/v0.3): real-data numbers on GaussianFeels splats (2c169bb)
- harden(v0.2/v0.3): robust bundle edges + vectorised batch index queries + wide-baseline camera seed (99aee5d)
- Delete .github/workflows directory (5d60ce3)
- feat(v0.3): multi-splat bundle registration + scene-scale spatial index (38eaf60)
- feat(v0.2): 6-DoF object-pose mode (ADD/ADD-S) + camera localization in a splat (0dc6d9b)
- chore: gitignore internal docs/ + sync RESULTS with README (merge + speed tables) (d965e30)
- chore: remove unused square logo (banner is the only displayed image) (e0369df)
- docs: badge row — add PyPI version + real links, drop vanity badges (3cf58ce)

## v1.0.1 — 2026-06-07

- release 1.0.1: show banner on PyPI (absolute-URL README header) + version bump (0f46557)
- docs: landscape banner in README header + transparent (bg-removed) square logo for PyPI/icon (9afffa7)
- docs: add logo to README header + social banner asset (5b8c5f4)
- docs: roadmap v1.0 shipped (BSD, public, PyPI: pip install splatreg) + icon brief (a645868)
- splatreg: BSD-3-Clause relicense + partial-overlap fix (4/9->6/9) + scale line-search (c7f1c63)
- docs: slim README to a clean product page (258->170) (a97a0fd)
- splatreg 1.0.0 — backends, official 3DMatch/3DLoMatch (matches GeoTransformer), CI, seeded RANSAC (fbaa3e3)
- feat(splatreg): real-splat MERGE DEMO — the MVP deliverable (4.9x Chamfer, 21.9x overlap vs naive concat) (6d6c00c)
- docs(README): thorough update — 3DMatch SOTA-beat, init modes (fast/robust/learned), real speed, honest open items (4fcfc26)
- feat(splatreg): init="learned" (GeoTransformer) — beats learned SOTA on 3DMatch (94% RR vs GeoTr 92.5%) (6cfd206)
- feat(splatreg): init="robust" — competitive on real 3DMatch (ties Open3D RR, ~2.5x better RRE/RTE) (a847f65)
- perf(splatreg): sub-30ms full registration + honest first 3DMatch benchmark (d0f2597)
- feat(splatreg): fast feature-init by default — from-scratch registration 4x faster, 36/36 preserved (d599df3)
- feat(splatreg): real-time warm-start tracking (<40ms) + Sim(3) closed-form speed + robustness + real-data (285360f)
- chore: remove orphaned scratch experiments; fix last dangling docs/ comment-ref (near-pi test docstring now reflects the landed fix) (7425ac4)
- feat(partial-overlap): overlap-aware point-to-plane feature aligner + honest ambiguity detection (930a2c2)
- fix(docs): correct symmetric robustness 9/9 -> 7/9 (re-verified by running the bench; prior figure was an unverified agent report) (eaa583a)
- style: black the docstring + example edits (30f1249)
- chore(repo): drop internal docs/ + .github CI/pre-commit; move hero figure to assets/; fix all references (540cbee)
- fix(ci): drop unused scipy from the install (broke CI: torch index has no scipy, and it's never imported); remove debug verify scripts (1df36ea)
- docs: refresh RESULTS.md to current state (closed-form gradient, near-pi log, 30 tests, symmetric 9/9) (9b7896c)
- docs: flagship README + before/after hero figure + Jacobian docstring accuracy (606d1fc)
- style: apply black (line-length 110) across the package (d23ddee)
- validation suite + closed-form SDF gradient + robust near-pi log + CI (71ac123)
- splatreg v0.1.0: composable SE(3)/Sim(3) Gaussian-splat registration (2c4ce50)

