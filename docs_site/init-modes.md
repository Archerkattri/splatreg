# Init modes

Registration is two stages: a **coarse initializer** finds the right basin, then the
Levenberg–Marquardt core polishes the pose within it. `register(..., init=...)` (and
`splatreg align --init ...`) selects the initializer: the single most important knob in the
library. The trade is speed ↔ robustness:

| `init=` | what | when | cost |
|---|---|---|---|
| `"fast"` *(default)* | FPFH descriptors + GPU-batched 3-point RANSAC seed → LM polish | objects / full-overlap captures | **~17 ms** |
| `"robust"` | Open3D FPFH+RANSAC seed (scale-correct, auto-voxelled) → overlap-aware refine | real metre-scale scans | ~100 ms+ |
| `"learned"` | pretrained GeoTransformer seed → the same overlap-aware refine | best accuracy on real scans (91.5% official 3DMatch) | ~104 ms |
| `"bufferx"` | pretrained BUFFER-X zero-shot seed (ICCV 2025) → the same overlap-aware refine | cross-sensor / cross-scale scans, **no per-dataset training** | GPU (falls back to `"robust"` on CPU) |
| `"global"` | blind super-Fibonacci SO(3) sweep + batched trimmed ICP | unknown, possibly huge rotation; no features needed | ~0.8–1.4 s |
| `"mac"` | MAC maximal-clique consensus over the FPFH correspondences (Zhang et al. CVPR 2023) → weighted SVD per clique → overlap-aware refine | outlier-heavy / multi-consensus correspondence sets | ~0.03–0.3 s (CPU, scales with clique count) |
| `"features"` | complete partial-overlap registrar (FPFH → clique-filtered RANSAC → overlap-aware point-to-plane refine + basin-sweep fallback) | the two captures see *different parts* of the object | seconds (deep sweep) |

You can also pass an explicit 4×4 tensor as `init` (e.g. a pose prior from odometry), or
`None`, which resolves to `"fast"`.

## Picking one

- **Merging two captures of an object / small scene** → keep the default `"fast"`.
  It handles full rotations and is two orders of magnitude faster than the blind sweep.
- **Real indoor/outdoor scans (metre scale, sensor noise)** → `"robust"`, or `"learned"`
  for the best accuracy (GeoTransformer-class recall; needs its weights available).
- **Captures from different sensors / at different scales, and you don't want to train a
  per-dataset model** → `"bufferx"` (zero-shot; falls back to `"robust"` when its weights are
  absent). Add `seed_gate=True` to `"learned"` to reject/reseed a low-confidence learned seed.
- **You have no idea how the splats are oriented and they look feature-poor** (smooth,
  near-symmetric geometry where descriptors are non-discriminative) → `"global"`.
- **Partial overlap** (one capture saw the left side, the other the right) → `"features"`.
- **Outlier-heavy or multi-consensus correspondences** (repetitive structure, symmetric
  decoys, a contaminated learned matcher) → `"mac"` (below).

## `init="mac"`: maximal-clique hypothesis generation

`"mac"` reimplements **MAC** (*3D Registration with Maximal Cliques*, Zhang, Sun, Wang & Guo,
CVPR 2023) in pure torch + networkx, replacing RANSAC minimal samples as the hypothesis
generator:

1. a **rigidity compatibility graph** over the correspondences (edge iff
   `| ‖p_i−p_j‖ − ‖q_i−q_j‖ | < γ`), edge weights re-scored by the **second-order SC²
   measure** (`w₂ = s ⊙ (S·S)`, an edge is only as strong as the compatible neighbourhood the
   two correspondences *share*, which zeroes chance-compatible outlier pairs);
2. **all maximal cliques** of that graph (Bron–Kerbosch with pivoting), each one a consensus
   hypothesis, including secondary consensus sets a greedy prefilter or a lucky-draw RANSAC
   never isolates. Worst-case blowup is capped: ≤ 1000 correspondences, per-node degree cap
   (top-48 edges by SC² weight), clique-count cap + wall-clock budget on the lazy enumeration,
   and node-guided selection down to ≤ 64 hypotheses;
3. a **weighted SVD** (Kabsch, SC² weights) per clique, scored by inlier count over all
   correspondences; the winner is refit on its full consensus set, then polished by the same
   overlap-aware ICP the `"robust"`/`"learned"` registrars use.

Sim(3): MAC's rigidity constraint is SE(3)-only, so the scale is estimated **first** (median
of correspondence pairwise-distance ratios), the source de-scaled, SE(3) MAC run, and a
residual scale refit on the consensus inliers.

Measured on synthetic contaminated correspondence sets (CPU, `tests/test_mac.py`): at 30/60/90 %
random outliers MAC matches the fast-init RANSAC engine (rot err ≤ 0.2°); on a 90 %-contaminated
set with a *structured* decoy cluster (reflection-consistent, it out-degrees the true inliers)
the greedy-prefilter+RANSAC engine fails at ~78° while MAC stays **< 0.2°**; an all-outlier set
returns an honest `info["success"]=False` identity. 500 correspondences run in ~0.1 s on a
2-thread CPU (budget-tested < 5 s).

Inside `init="learned"`, `seed_selector="mac"`
(`learned_feature_align(..., seed_selector="mac")`) runs MAC over **GeoTransformer's learned
correspondences** instead of the model's own LGR estimator: the exact combination the MAC
paper reports lifting GeoTransformer's 3DLoMatch registration recall **~71 % → ~78 %**.

!!! note "3DLoMatch verdict: measured, a wash; `lgr` stays the default"
    Measured on the **full official splits** (GPU, single shared forward, native 0.025 voxel,
    same residual-gated refine, only the hypothesis stage differs): 3DLoMatch
    **72.1 % mean / 74.6 % pooled** (MAC) vs **72.5 % / 74.4 %** (LGR); 3DMatch **91.7 % /
    93.8 %** vs **91.5 % / 93.5 %**. Every delta is within ±4 pairs, *not* the paper's
    +6–7 pp, at ~+50 % runtime. MAC genuinely engaged on 100 % of pairs (median ~600–800
    consensus inliers): at native voxel the learned correspondences are already
    consensus-dominated, so the multi-consensus regime MAC wins (the synthetic decoy above)
    does not occur, and the guarded refine absorbs seed-level differences. Details in
    `RESULTS.md` §5k. Needs networkx: `pip install "splatreg[mac]"`.

## `init="bufferx"`: zero-shot seed (BUFFER-X, ICCV 2025)

`"bufferx"` swaps GeoTransformer for **BUFFER-X** — *Towards Zero-Shot Point Cloud Registration in
Diverse Scenes* (ICCV 2025, [MIT-SPARK/BUFFER-X](https://github.com/MIT-SPARK/BUFFER-X)) — as the
coarse seed, then runs the *same* overlap-aware refine (+ Sim(3) scale) as `"learned"`/`"robust"`.
The point is generality: BUFFER-X is a single **zero-shot** model that registers across sensors and
scales with **no per-dataset training**, which is why it is splatreg's learned seed of choice — a
splat registrar should not need a per-scene/per-sensor trained model to align two captures.

BUFFER-X is optional and lazily loaded (its CUDA neighbour/subsampling extensions + Hugging Face
checkpoints are not shipped). When absent — always the case on a CPU box — `"bufferx"` transparently
falls back to the classical `"robust"` seed with a logged note. Setup (clone + build + weights) is in
[`splatreg/third_party_models/README-BUFFERX.md`](https://github.com/Archerkattri/splatreg).

### 2026 positioning

Per-dataset-trained backbones now lead 3DMatch: **PSReg** and **DiffusionPCR** report **95 %+**
registration recall, above the ~91.5 % GeoTransformer seed splatreg wraps. splatreg does *not* chase
that number — it keeps a zero-shot learned option (BUFFER-X) instead. The value is a generalist seed
plus splatreg's provable SH rotation, honest pose covariance, Sim(3) scale, and overlap-aware refine
on top, not the last recall point on one benchmark; a higher-recall correspondence model can be
dropped in as the seed the day it ships a permissive zero-shot checkpoint.

## `seed_gate=True`: Decision-PCR-style seed confidence (opt-in)

`register(init="learned", seed_gate=True)` (default **off**) adds a lightweight, training-free
stand-in for **Decision PCR**'s learned confidence head (arXiv 2507.14965). Before LM refinement it
scores the candidate learned seed with two cheap signals over mutual-NN correspondences — the
**inlier ratio** (fraction landing within tolerance after the seed transform) and an **SC² spatial
consistency** term (the same rigidity-graph machinery `init="mac"` uses) — and *rejects and reseeds*
a low-confidence hypothesis from the classical `"robust"` path (keeping whichever scores higher)
instead of blindly refining a bad seed. The scores land in `result.info["seed_gate"]`. On synthetic
known-transform pairs the gate never rejects a correct seed and rejects a planted decoy
([`tests/test_bufferx_seedgate.py`](https://github.com/Archerkattri/splatreg)); full retraining of
the classification head is out of scope.

## Partial overlap: the honest contract

`init="features"`, `"robust"`, `"learned"`, `"bufferx"`, and `"mac"` are *complete registrars*, not just seeds.
The default residual set assumes **full overlap** (its ICP would drag a good partial-overlap
pose off-target), so with the default residuals these modes return their own
registration **directly** and skip the LM. Pass an explicit overlap-safe `residuals=[...]`
if you want the LM to run on top of the feature init.

They also return honest diagnostics:

```python
result = register(target, source, init="features")
result.info["ambiguous"]    # True when the overlap does NOT constrain the pose
result.info["confidence"]   # 0..1 grade
result.info["feature"]      # the full per-stage diagnostic dict
```

!!! warning "Ambiguity is flagged, not hidden"
    When a crop removes the rotation-disambiguating geometry, the pose is *genuinely
    unrecoverable*, even the true pose doesn't seat cleanly. splatreg returns its best
    feasible guess **flagged** `ambiguous=True` rather than a silently wrong pose
    (verified: 0 silent-wrong across the robustness sweep). At overlap ≤ 40% expect flags;
    `merge` and `Tracker` are designed for high-overlap captures.

## Fallback chain

All string inits are guarded: `"learned"` and `"bufferx"` fall back to `"robust"`, `"fast"` falls
back to `"global"`, `"mac"` raises a clear `ImportError` with the install hint when networkx is
missing (an explicit opt-in extra, like the solver backends) and returns an honestly-flagged
identity when the correspondences carry no consensus, and everything else falls back to
identity (with a logged note) when an optional dependency or pretrained weight is unavailable. Your call never hard-fails because a model
file is missing.

## Quality policy

Orthogonal to `init`, `quality=` bounds the work the refinement does:

- `"full"` *(default)*, nothing capped, every source anchor.
- `"balanced"` / `"low"`, bounded sample counts + tighter autodiff chunks.
- a float `0..1`, interpolates the caps.
- `"auto"`, detect free GPU/CPU memory and pick the largest sizes that *fit*.

The Sim(3) autodiff Jacobian is always row-chunked, so peak memory stays bounded with no
quality loss. See [`resolve_quality`][splatreg.quality.resolve_quality].
