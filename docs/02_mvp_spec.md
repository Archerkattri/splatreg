# splatreg — v0.1 MVP Spec (registration-first)

> **Synthesis** of [`00_research_synthesis.md`](00_research_synthesis.md) (verdict · seam · seed assets · risks) and
> [`01_step0_findings.md`](01_step0_findings.md) (demand · API · competitor). **Date:** 2026-06-05.

## The one-sentence MVP

**Given two Gaussian splats, estimate the Sim(3) that aligns them and export a merged `.ply`** — the splat-to-splat registration that SuperSplat (9k★), INRIA `#990`, and Cesium users demand and *no tool provides*.

```python
import splatreg
T = splatreg.register(splat_a, splat_b)          # -> Sim(3) (R, t, scale)
merged = splatreg.merge([splat_a, splat_b])      # aligned, exports a single .ply
```

## Why this wedge (where 00 and 01 converge)

- **00 (supply):** the library slot is empty — every pose-vs-splat system is a monolithic research pipeline; the one differentiator nobody packages is a **Gaussian-derived SDF residual** in a composable SE(3) LM; the seed already has it and is gsplat-native (Apache-clean).
- **01 (demand):** the single most-requested, least-served use case — across *three* independent communities — is **splat-to-splat registration**, where the incumbent literally punts with "use manual gizmos." It's also less crowded than object-pose (which faces FoundationPose/6DOPE-GS) and camera-reloc (30+ papers).
- **Result:** registration-first is the rare wedge that is simultaneously the biggest unmet *demand* and the cleanest expression of our *differentiator* — and the lowest-dependency MVP (no dataset download, no GT-pose stream; a user brings their own two splats).

## v0.1 scope (hold the line — NARROW)

**IN:** pairwise **Sim(3)** registration (rotation + translation + **scale**) of two Gaussian splats; geometry-first (Gaussian-SDF + ICP residuals) with optional photometric (via gsplat); coarse global init → fine LM refine; export the transform + a merged `.ply`. Standalone `pip install`, composes with gsplat/PyPose.

**OUT (deferred, by milestone):** camera localization in a splat (v0.2); object-6DoF-pose tracking + the YCBInEOAT paper benchmark (v0.2); multi-splat joint/bundle registration & scene-scale spatial index (v0.3); SLAM / loop-closure (**never** — that's GTSAM's job; staying a *library* is what preserves the differentiation).

## The engine — and how much already exists

The registration solve is **coarse global init → fine multi-residual LM**, and *both halves already exist in the user's code*:

1. **Coarse global init** — reuse the **A/B-bench metric-side global aligner** (`super-Fib SO(3) candidate sweep + GPU-batched ICP`, the `project_ab_bench_fscore_alignment` work). It already solves "find the rough rotation between two point sets" robustly. This is the registration basin-finder, ready-made.
2. **Fine refine** — the seed's `se3_lm.py` multi-residual LM: sample points from splat B, transform by `T`, score against splat A's **Gaussian-derived SDF** (`signed_distance_via_gaussian_density`) + ICP point-to-plane + optional photometric. This is *exactly* what the GaussianFeels tracker does (register observed points against a frozen object's SDF) — splat-A's SDF replaces the frozen object map.
3. **The one real new piece → Sim(3) scale DoF.** `se3_lm` is SE(3) (6-DoF); splat-to-splat needs the 7th DoF (scale — the Cesium/geospatial demand is explicitly about scale). Add the scale Jacobian column (`d r / d log s`) to the LM, or wrap SE(3) + a separate scale line-search for v0.1.

So v0.1 is mostly **carve + glue two existing components + add one DoF**, not greenfield.

## The API (registration-first, mirrors `01` design)

```python
# one-shot functional
T = splatreg.register(
    splat_a, splat_b,                       # gsplat tensors / .ply (means, quats, scales, opacities, SH)
    residuals=[splatreg.residuals.SDF(1.0), splatreg.residuals.ICP(0.5)],  # + Photometric(gsplat) optional
    init="global",                          # "global" (super-Fib+ICP) | "identity" | a Sim3
    transform="sim3",                       # "sim3" (default) | "se3"
    backend="builtin",                      # | "pypose" | "theseus" | "gtsam"
)
T.matrix; T.scale; T.info["rmse"], T.info["overlap"], T.info["n_iters"]

# merge helper
merged = splatreg.merge([splat_a, splat_b], ref=0)   # registers all to ref, concatenates, dedupes overlap
merged.save("merged.ply")
```

`Residual` ABC + pluggable `backend=` exactly as `01` (Theseus-style plugin behind a PyPose-style facade; pure-Python by delegating CUDA to gsplat).

## MVP deliverable + how it's validated

**Deliverable:** `register()` + `merge()` working end-to-end, the **"merge two overlapping real captures into one `.ply`" demo** (the thing SuperSplat users beg for), Apache-2.0, pip-installable, README + 2 examples.

**Validation (two tiers — reuse the user's existing habit of real-GT random-transform recovery):**
1. **Synthetic recovery (the rigorous core):** take one real splat, apply a *known* random Sim(3) → recover it with `register`; report rotation error (deg), translation error, **scale error**, and post-align Chamfer/overlap. This isolates the solver and is the credible benchmark (no GT-pose dataset needed). Target: recover ≤1° / ≤1% scale on overlapping inputs.
2. **Real pairs (the demo):** 2–3 overlapping real captures (or two halves of one scan); qualitative merged render + overlap/Chamfer improvement vs. naive `cat`. This is the screenshot that sells it.

> Note on competitors: there is **no standard splat-registration benchmark** — so we *define* the synthetic-recovery protocol (rotation/translation/scale error + Chamfer) and compare against the only baselines that exist: naive concatenation (the SuperSplat status quo) and `GaussianSplattingRegistration`/`splatalign` (ICP-only). Beating ICP-only with the SDF residual is the headline.

## Milestones (~2–3 focused weeks to a usable v0.1)

| phase | work |
|---|---|
| **Carve** (3–4 d) | Extract `se3_lm` + the Gaussian-SDF query + the residual ABC into the `splatreg` package; decouple from GaussianFeels/tactile; define the gsplat/`.ply` input contract. |
| **Sim(3)** (2–3 d) | Add the scale DoF to the LM; the global-init wrapper (port the A/B-bench super-Fib + batched-ICP aligner). |
| **`register`/`merge`** (3–4 d) | Wire the coarse→fine pipeline behind the API; the merge+dedupe helper; PLY I/O. |
| **Validate** (3–4 d) | Synthetic Sim(3) recovery harness + metrics; 2–3 real overlapping pairs; the merge demo. |
| **Package** (2–3 d) | pyproject (Apache, pure-Python, `[render]`/`[pypose]`/`[theseus]` extras), `py.typed`, docs (quickstart + custom-residual + pose-convention page), CI (gradcheck + synthetic-recovery test). |

## Positioning + launch

*"SuperSplat renders and edits your splats; **splatreg registers** them. `pip install splatreg`; composes with gsplat (bring your renderer), PyPose/Theseus/GTSAM (bring your solver)."* Launch surfaces: answer the live `supersplat#53` / `graphdeco#990` / Cesium threads with a working `register()`; post in the gsplat discussions (where the pose/viewmat demand cluster lives).

## Risks (carried from 00/01)

- **`jashshah999/gtsam-splatfactors`** is circling the same GTSAM-factor idea — but it's SLAM-framed and hobbyist; splatreg's *registration* framing + the SDF residual + the SuperSplat-merge demo is a distinct, demand-led wedge. **Move now**; consider collaboration.
- **Scope creep** into SLAM/localization dilutes the wedge → keep v0.1 to pairwise registration; defer the rest by milestone.
- **Demand is fragmented** → lead every piece of comms with the *concrete* splat-merge demo, not the abstract "pose-registration library."
