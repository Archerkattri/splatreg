# splatreg — failure-mode root-cause analysis & fix brainstorm

From the 3-seed robustness sweep (`benchmarks/robustness_bench.py`, 2026-06-06):

| Condition | Result | |
|---|---|---|
| Noise (sensor jitter 0.5–2%R) | 9/9 (100%) | robust |
| Outliers (+10–50% clutter) | 9/9 (100%) | robust |
| Symmetric (sphere, no lobe) | 7/9 (78%) | **2 failures** |
| Partial overlap (20–60% removed) | 0/9 (0%) | **total failure** |

**Why this matters beyond the benchmark:** the v0.1 headline verb is `merge` — fusing two
splat *captures*. Two real captures of the same object **see different parts of it** — i.e.
they are *inherently partially overlapping*. So the partial-overlap 0/9 is not academic; it
means the headline feature is fragile on exactly the realistic input it exists to serve. This
is the highest-priority fix.

---

## Failure Mode A — PARTIAL OVERLAP (0/9)

**Symptom:** with 20% of the moved cloud removed, rot_err is already 9–21°; at 60% removed it
is 120–147° (near-random). `register(B_partial, A_full)` aligns the *full* reference `A`
(source) onto the *partial* `B` (target).

### Root cause: every stage silently assumes FULL overlap

**Global aligner (`align.py`)**
- **RC-P1 — centroid centering.** `_batched_trimmed_icp` centers on `pc = src.mean(0)` /
  `gc = tgt.mean(0)`, and the final `_umeyama` centers on the means. A partial cloud's centroid
  is **shifted toward the kept region**, so centering aligns a shifted centroid to a true one →
  a systematic offset that corrupts the correspondences (and hence the rotation), not just the
  translation.
- **RC-P2 — scale by RMS radius.** Sim(3) scale is `rg/rp`, the ratio of RMS radii about the
  centroids. A partial cloud's RMS radius ≠ the full cloud's → wrong scale seed.
- **RC-P3 — symmetric-Chamfer scoring.** Seeds are scored by `0.5·(d_xy + d_yx)`. The
  source→target direction penalizes `A`'s points sitting over `B`'s *missing* region, so the
  score is minimized by **shrinking/shifting `A` off the true pose** to avoid the empty region.
- **RC-P4 — fixed shallow trim.** `_ICP_TRIM_KEEP = 0.85` rejects only ~15% — fine for outliers,
  useless for 20–60% non-overlap.

**Fine LM residuals (`api.py` default set, `icp.py`, `sdf.py`)**
- **RC-P5 — the default ICP is un-gated and non-robust.** `_default_residuals` uses
  `ICP(point_to_plane=False, weight=1.0)` with **`max_correspondence_dist = 0` (keep ALL)** and
  **`robust = None`**. So every source point — including those over `B`'s missing region — adds a
  plain least-squares residual matched to a *wrong edge* point. This is the single biggest
  fragility.
- **RC-P6 — SDF reads garbage off-support.** For `A` points outside `B_partial`'s Gaussian
  support, the Gaussian-SDF returns large/near-zero-gradient values → spurious residuals.
- **RC-P7 — matching direction is source→target only.** `A_full → B_partial` NN means `A`'s
  extra points must match *something*. The partial-robust direction is target→source
  (`B_partial ⊂ transformed A_full`, so every observed `B` point has a correct match). The code
  never matches `B → A`.

### Why noise & outliers DON'T fail (confirms the root cause)
The perturbed cloud is always the **target** `B`. Outliers ADD target points: `A`'s points still
find their correct nearest `B` neighbour; the extra target points are simply never matched → no
harm. Noise jitters `B`'s points: averages out. Only *removing* target points breaks the
source→target NN. The fragility is precisely **missing target support**, not corruption — exactly
the full-overlap assumption.

### What the field does (literature)
- **Trimmed ICP (TrICP)** estimates the *overlap rate* and trims to it (vs our fixed 0.85), but
  is initialization-sensitive.
- **Overlap prediction** (Predator/STORM/DBDNet, 2021–2024): predict the overlapping region and
  route only those points into the estimator — "weakening the impact of non-overlapping regions."
- **Robust global registration** (TEASER++, FGR, RANSAC): feature correspondences (FPFH/learned)
  + graph-theoretic max-clique outlier pruning; TEASER++ is robust to >99% outliers *and* partial
  overlap, decoupling scale/rotation/translation, and works correspondence-free.

### Fixes (brainstorm, prioritized)
- **F-P1 [HIGH impact / LOW risk] — gate + robustify the default ICP.** Auto-set
  `max_correspondence_dist` to ~k·(median NN spacing of the target) so source points over the
  missing region are *dropped*, and attach a default robust kernel (Huber/Tukey/Welsch) so wrong
  correspondences are down-weighted. Fixes RC-P5, likely recovers most of mild–moderate partial.
- **F-P2 [HIGH / MED] — bidirectional / target→source correspondences.** Add the target→source
  match (or use the symmetric min) so partial overlap is covered from the dense side. Fixes RC-P7.
- **F-P3 [HIGH / MED] — overlap-aware global init.** Iterate the trimmed ICP with an *estimated*
  overlap rate (TrICP-style), and recompute centroid + scale on the **inlier/overlap subset**
  only, not the full clouds. Fixes RC-P1/P2/P4.
- **F-P5 [MED] — asymmetric scoring.** Score global-init seeds by the target→source Chamfer only,
  so the missing source region cannot penalize the true pose. Fixes RC-P3.
- **F-P4 [HIGH / HIGH effort] — feature + TEASER/RANSAC global init.** Replace the centroid-ICP
  sweep with FPFH/learned-feature mutual-NN correspondences + a max-clique robust estimator. The
  SOTA partial-overlap path; a v2 lift but the principled fix.
- **F-P6 [LOW] — expose the overlap/trim rate** as a `register(..., overlap=...)` knob so callers
  with known partial captures can set it.

**Recommended:** F-P1 + F-P3 first (small, low-risk, addresses the dominant cause), then F-P2/F-P5.

---

## Failure Mode B — SYMMETRIC OBJECT (2/9)

**Symptom:** on the lobe-less sphere, 2 cells (seed 2, 30° & 90°) land at rot_err ≈ 180° AND
**Chamfer ≈ 6.9 mm** — i.e. not a benign rotation-ambiguity (a sphere maps onto itself under any
rotation, Chamfer ≈ 0), but a genuine mis-alignment with the **translation/scale off**. The
successes took 0.9 s (global init already perfect); the failures ran the full 60 LM iters and
still missed — a borderline init the fine LM could not rescue.

### Root cause
- **RC-S1 — degenerate PCA seeds.** `_pca_seed_rotations` builds seeds from the clouds' principal
  axes. A sphere's covariance is isotropic → SVD eigenvalues are ~equal → the principal axes are
  **arbitrary/unstable**, so the PCA seeds (and their sign-flips) are noise that can seed a bad
  basin.
- **RC-S2 — score ties + first-index tie-break.** For a rotationally-symmetric cloud all SO(3)
  seeds score ~equally; `argmin(scores)` is a near-tie and the deterministic "first index" can
  pick a degenerate seed.
- **RC-S3 — ambiguous correspondences → degenerate Umeyama.** NN correspondences on a symmetric
  cloud are ill-defined (many equidistant matches); the closed-form Umeyama on them can converge
  to a wrong translation/scale (the 6.9 mm), and float32 compounds it.
- **RC-S4 — the metric is rotation-blind by design (correct), but the failure is real:** Chamfer
  6.9 mm proves it's a translation/scale miss, not a metric artifact.

### What the field does
Symmetry causes pose ambiguity; standard handling is to **estimate the full set of equivalent
poses** (multiple hypotheses, probabilistic / Bingham), use a **symmetry-aware rotation
representation**, or detect symmetry and report rotation as ambiguous. Degenerate-PCA on
symmetric shapes is a known pitfall.

### Fixes (brainstorm)
- **F-S1 [MED] — skip degenerate PCA seeds.** If the PCA eigenvalue spread is below a threshold
  (isotropic), drop the PCA seeds and rely on a denser super-Fibonacci grid. Fixes RC-S1.
- **F-S2 [MED] — stability-aware tie-break.** Among seeds within ε of the best score, pick the one
  whose Umeyama is most stable (scale closest to 1, lowest covariance condition number) instead of
  the first. Fixes RC-S2.
- **F-S4 [LOW] — float64 + more iters in the global ICP** for the winner refine; cheap, may
  stabilize the degenerate Umeyama (RC-S3).
- **F-S3 [MED] — multiple-hypothesis output.** Return the top-K basins for the fine LM to
  disambiguate (the field's standard symmetry fix), or detect symmetry and flag rotation as
  ambiguous (honest) while still nailing translation/scale.
- **F-S5 [LOW] — verify the winner by translation/scale sanity** (e.g. recovered scale within a
  band, centroid alignment) and fall back to the next seed if it fails.

**Recommended:** F-S1 + F-S2 + F-S4 (cheap, targeted) harden the global init for symmetric shapes.

---

## Cross-cutting takeaway
The default residual set — a **bare, un-gated, non-robust point-to-point ICP** — is the common
fragility. A *gated + robust* default ICP (F-P1) plus an *overlap-aware* global init (F-P3) would
address the dominant partial-overlap failure and harden the symmetric case, with small, low-risk
changes; the feature/TEASER path (F-P4) and multiple-hypothesis symmetry handling (F-S3) are the
principled v2 upgrades.

## Update — partial-overlap fix experiments (2026-06-06)

Three experiments tested the fixes above on the partial cells (SE3, rot 30°):

1. **Gate the fine ICP** (`max_correspondence_dist = k × median NN spacing`,
   `benchmarks/partial_fix_experiment.py`): **2/9** at best (k=3×), only the *mildest*
   crops (keep80%), and fragile (k=5/10× → 0/9). Gating drops missing-region
   correspondences but cannot rescue a bad global init.
2. **Standalone overlap-aware target→source aligner** (`partial_fix_experiment2.py`,
   n_rot 64 and 256): **0/9** — and *worse* than the default on keep80% s1 (20° vs the
   default+gate's 1.6°). The hand-rolled B→A ICP under-performed the existing batched
   aligner; the target→source *idea* remains untested inside the good machinery (it would
   mean changing `align.py`'s batched trimmed-ICP scoring, not a separate weaker re-impl).

**Refined diagnosis — the 0/9 is two distinct things:**
- **FIXABLE partial** (overlap retains the rotation-disambiguating feature): recoverable
  with a better global init + gated/robust fine LM (the default already gets keep80% s1
  into a basin the gated LM finishes to 1.6°).
- **INHERENTLY-AMBIGUOUS partial** (the crop removed the feature, e.g. the +x lobe): the
  remainder is a near-symmetric ellipsoid section → rotation unrecoverable by *any* method
  (TEASER/Predator included). The random-direction crop hits both regimes, so the raw 0/9
  conflates "splatreg is weak here" with "this is impossible."

**The real fix (prioritised — a significant feature, not a quick patch):**
1. **Feature-based robust global aligner** (FPFH/learned descriptors → mutual-NN matches →
   TEASER++/RANSAC max-clique), replacing the centroid-ICP sweep for the *fixable* regime —
   the SOTA partial-overlap path, what actually beats the alternatives.
2. **Honest uncertainty** — detect a feature-poor / near-symmetric overlap (low aligner
   score-margin / degenerate covariance) and report the pose AMBIGUOUS instead of a
   confident wrong answer (GaussReg's success-rate honesty).
3. **A fair partial benchmark** — crop so the overlap keeps features (or report
   recoverable-vs-ambiguous separately) so the number measures the fixable regime.
4. Gate + robust-kernel the default fine ICP — a small, real help for the fixable regime.

Shipping the marginal/fragile gate alone would overstate the fix; the credible fix is (1)+(2),
scoped as the next major splatreg feature.

## Update 2 — feature-based aligner + symmetric fix (2026-06-06)

Implemented `splatreg/align_features.py` (FPFH-lite positional descriptors + mutual-NN +
RANSAC) and wired as `init="features"` in `register()`.  Also added a centroid-based
stability tie-break in `global_align` (`align.py`) and kept the PCA degenerate-spread
detection as documentation (attempting to skip PCA seeds for isotropic clouds was tested and
made symmetric WORSE: 8/9→3/9, because at N=800 the Fibonacci grid is too coarse to
compensate without PCA seed coverage).

### Measured results (CPU, N=800, 3 seeds)

| Condition | init=global (before) | init=global (after) | init=features |
|---|---|---|---|
| NOISE (sensor jitter) | 9/9 (100%) | 9/9 (100%) | — |
| PARTIAL (occlusion, 20–60% removed) | **0/9 (0%)** | **0/9 (0%)** | **0/9 (0%)** |
| OUTLIERS (clutter) | 9/9 (100%) | 9/9 (100%) | — |
| SYMMETRIC (sphere, no lobe) | 8/9 (78%)¹ | 8/9 (89%)² | **9/9 (100%)** |

¹ Docs/03 originally said 78%; the re-run at N=800 (before/after) gives 8/9 = 89%.
² After changes: centroid tie-break in `align.py`; PCA skip was reverted (made it worse).

### Partial overlap: why features did NOT help

The FPFH-lite descriptors at N=800 / k=12 are **not discriminative enough** on the test
object (anisotropic ellipsoid shell with a single +x lobe):

- **Match quality:** MNN in feature space gives ~150–200 matches per cell, but mean GT error
  is ~12.5cm (≈ object radius).  Only 1–8 of those are correct (≤15mm).  RANSAC success
  probability with 3-point samples at 1–4% inlier rate is ~(0.03)^3 ≈ 0.003% per draw;
  512 iterations gives near-zero expected success.
- **Root cause — descriptor non-discrimination:** The ellipsoid surface has nearly uniform
  curvature/spacing everywhere (all points look similar under local PCA).  Full FPFH (which
  uses oriented normals + geodesic fan of point-pair features) has the same problem on smooth
  surfaces; the oriented-normal variant encodes orientation which changes with position on
  an anisotropic shell, but our positional-only variant cannot.  Real FPFH libraries (Open3D)
  require denser, well-normals-estimated point clouds to be discriminative.
- **Lobe analysis:** At keep80%, 2/3 seeds (0 and 2) have the crop direction remove the +x
  lobe entirely — the only distinctive feature.  For those seeds the problem is
  **inherently ambiguous** regardless of descriptor method.  Seed 1 (lobe kept) also fails
  because the ellipsoid shell around the lobe is ambiguous and produces too many false matches
  to get a 3-point clean RANSAC sample.

**Conclusion:** the partial-overlap 0/9 is not fixable by FPFH-lite descriptors alone at
this point count.  What would be needed: (a) proper oriented normals (requires 5–10× denser
sampling to estimate reliably), (b) learned descriptors (FCGF/SpinNet) trained on similar
geometry, or (c) the fair benchmark fix — crop so the distinctive region is guaranteed kept
(report `recoverable-vs-ambiguous` separately).  The implementation is correct and delivered
as specified; the descriptor quality is the honest ceiling at this geometry/density.

### Symmetric: honest characterisation of the fix

- `init="features"` gives **9/9** (vs 8/9 for `init="global"`).  The 1-failure case
  (90deg seed 2, Chamfer ≈ 8.9mm) is fixed because the RANSAC on the (near-featureless)
  sphere finds a correct translation/scale hypothesis that seeds the fine LM in the right
  basin.
- The centroid tie-break in `global_align` has no measurable effect on the symmetric benchmark
  (ties are rare with the current `_SCORE_EPS`).
- The PCA isotropy skip (RC-S1 fix, threshold 2.0 for sphere ~1.05 vs asymmetric ~2.5) was
  implemented, tested, and **reverted** — it reliably degraded 8/9→3/9 because the Fibonacci
  grid at 256 seeds is too coarse to compensate for losing the PCA candidates.  The
  documentation + helper `_pca_eigenvalue_spread` / `_PCA_ISOTROPY_THRESH` are kept in
  `align.py` for future use if the grid density is increased (512+ seeds).

## References
- Trimmed/partial ICP & overlap prediction: [Representative Overlapping Points (arXiv:2107.02583)](https://arxiv.org/pdf/2107.02583) · [STORM](https://www.researchgate.net/publication/358376938_STORM_Structure-based_Overlap_Matching_for_Partial_Point_Cloud_Registration) · [DBDNet (arXiv:2310.11733)](https://arxiv.org/html/2310.11733) · [Confidence-under-global-context (arXiv:2509.24275)](https://arxiv.org/pdf/2509.24275)
- Robust global registration: [TEASER (arXiv:2001.07715)](https://arxiv.org/abs/2001.07715) · [ROBIN (arXiv:2011.03659)](https://arxiv.org/pdf/2011.03659) · [Heuristics-guided parameter search (arXiv:2404.06155)](https://arxiv.org/pdf/2404.06155)
- Symmetry / pose ambiguity: [Symmetry-sensitive pose (IJCV 2026)](https://link.springer.com/article/10.1007/s11263-026-02770-x) · [Deep Bingham Networks (arXiv:2012.11002)](https://arxiv.org/pdf/2012.11002) · [On Object Symmetries and 6D Pose](https://www.researchgate.net/publication/336954789_On_Object_Symmetries_and_6D_Pose_Estimation_from_Images)
