# splatreg — Step-0 Findings (consolidated)

> **Date:** 2026-06-05 · Three parallel research agents (demand check · MVP benchmark · API design) + name resolution.
> Read **alongside** [`00_research_synthesis.md`](00_research_synthesis.md) to produce the registration-first MVP spec ([`02_mvp_spec.md`](02_mvp_spec.md)).

## Name: `gsfit` → **`splatreg`**
`gsfit` is **taken on PyPI** (tokamak-energy's Grad-Shafranov plasma fit, v0.0.5) → `pip install gsfit` unavailable. **`splatreg` is free on PyPI + GitHub (confirmed 2026-06-05)** and reads as "the registration library for splats" — matching the chosen wedge.

---

## 1. Demand — verdict: **MODERATE** (real but specialist + fragmented; researcher-led)

The demand is genuine, recurring, and **cross-community**, but it is *not* a groundswell — it scatters across differently-worded sub-problems rather than one loud "I need a pose-registration library." (Caveat: **Reddit was blocked** in the scan env → r/GaussianSplatting etc. unverified; re-check before trusting Reddit-specific counts.)

**#1 most-demanded, least-served use case → splat-to-splat registration (align/merge two splats into one Sim(3) frame).** The *only* use case independently demanded across **three** communities:
- [`graphdeco-inria/gaussian-splatting#990`](https://github.com/graphdeco-inria/gaussian-splatting/issues/990) — "merge multiple Gaussian models," open since 2024-09, still asked in 2025; only answers offered are naive `cat`, not registration.
- [`playcanvas/supersplat#53`](https://github.com/playcanvas/supersplat/issues/53) — maintainer: *"you can't combine splats right now."* **SuperSplat (9k★, the dominant editor) ships only manual gizmos**, no auto-registration ([#382](https://github.com/playcanvas/supersplat/issues/382), [#846](https://github.com/playcanvas/supersplat/issues/846)).
- [Cesium georeferencing thread](https://community.cesium.com/t/gaussian-splatting-georeferenced/43113) — geospatial Sim(3) (incl. **scale**) placement; current tools "can't scale+rotate+place."

**#2 → camera relocalization in a prebuilt splat.** Richer *research* pull (GS-CPR 146★, gsplatloc 143★, STDLoc 77★) but more crowded with published methods. In gsplat issues the framing is **"viewmat/pose gradient," not "localization"** (37 "pose" / 45 "viewmat" hits; "localization" = 0) — a naming cue. Key thread: [`gsplat#449`](https://github.com/nerfstudio-project/gsplat/issues/449) (unanswered SE(3)-manifold question), [`gsplat#180`](https://github.com/nerfstudio-project/gsplat/issues/180) (25 comments).

**User segments, by observed pull:** (1) 3DGS/SLAM researchers — strongest *measurable* (multi-star repos, dense paper stream, the gsplat pose cluster); (2) robotics/manipulation — strong, fast-growing, best fit for the "composes with GTSAM/Theseus/PyPose" pitch; (3) geospatial/digital-twin — concrete, underserved (needs Sim(3)+scale); (4) AR/VR relocalization — mostly academic so far.

**Competitor / "someone's building it" signals:**
- 🔴 [`jashshah999/gtsam-splatfactors`](https://github.com/jashshah999/gtsam-splatfactors) — "Gaussian Splatting meets Factor-Graph SLAM (iSAM2 + differentiable rendering factors)," created 2026-04-29, 11★, MIT. **≈ exactly splatreg's GTSAM-composition thesis** — but a solo hobbyist (7 followers), early. Watch / possibly collaborate.
- `terminusfilms/splatalign` (multi-scale ICP of temporal captures, 11★); `erikszasz/GaussianSplattingRegistration` (29★); academic: SIREN, RegGS.
- **Net:** field is nascent + unconsolidated — many tiny single-purpose repos, no dominant open-source pose-registration library. The opportunity *and* the risk (demand proven-but-fragmented → segment focus > raw capability).

---

## 2. MVP benchmark (object-pose — the research-credible *second* mode)

> Registration is the launch wedge (§ `02_mvp_spec`). Object-pose tracking is the **paper-grade benchmark** to add second.

- **Dataset → YCBInEOAT** (5 YCB objects, 9 videos / 7449 RGB-D frames, per-frame GT 6-DoF, standard YCB meshes, single 22 GB archive, MIT). The dataset 6DOPE-GS / BundleSDF / FoundationPose all report on → table slots into the literature.
- **Get the object splat:** Path A (honest) — train an object-centric gsplat from YCB-Video masked frames in canonical pose; Path B (fast fallback) — `mesh2splat` the YCB `.obj`.
- **Lead metric → ADD-S AUC (0–0.1 m, PoseCNN convention)** + tracking FPS. ADD AUC secondary.
- **Competitors (pick 2):** **6DOPE-GS** (the GS peer, RGB-D, YCBInEOAT, 93.79 ADD-S @ **3.5 Hz**) and **FoundationPose** (real-time ceiling, 96.42 ADD-S @ **~32 Hz**). BundleSDF (93.77 @ ~0.5 Hz) as a reference row.
- **Target → ADD-S AUC ≥ 90 @ ≥ 30 FPS** — i.e. *~10× faster than 6DOPE-GS, BundleSDF-class accuracy, FoundationPose-class speed,* via a single geometry-first SE(3)-LM solve with the **Gaussian-SDF residual** (the differentiator none of the GS competitors use). ~8–12 days to stand up.

---

## 3. API design (grounded in gsplat / PyPose / Theseus / kornia / nerfstudio)

**Central recommendation:** **Theseus-style `Residual` ABC** (the public, subclassable plugin point — the whole "bring your own residuals" value prop) **behind a PyPose-style 5-line `register()` / `Tracker` facade.** Specialize the optimization variable to a **single SE(3)/Sim(3)** (not a general factor graph), keeping Theseus's *analytic-or-autodiff Jacobian* duality.

- **Two entry layers:** functional one-shot `register(gaussians, observation, init_T, residuals=[...], backend=...)` (gsplat-style, returns result + info dict) + stateful `Tracker` (PyPose-style, warm-starts). Bundle observation channels into a typed `Frame` (do **not** copy gsplat's 40-positional-arg signature at the user level).
- **`Residual` ABC:** `residual()` (required) + optional analytic `jacobian()` (else autodiff via functorch) + `dim()` + `requires()`. Built-ins: `Photometric` (via gsplat), `Depth`, `SDF` (splatreg-native), `ICP`, `Prior`.
- **Two pluggability axes (keep separate):** `solver=` swaps the numerical step inside the builtin loop (PyPose-style LM/GN/strategy/kernel); `backend=` hands the whole linearized problem to an external engine — `builtin` / `pypose` / `theseus` (also gives **differentiable** registration) / `gtsam` (requires analytic Jacobians — gate on it). Adapter contract: every backend consumes one `LinearizedProblem(J, r, weight, robust)` → returns an `SE3Update`; a new backend is ~50 lines and never touches the residual plugins.
- **Packaging — the biggest ergonomic lever: stay PURE-PYTHON by delegating CUDA to gsplat.** Ships one universal wheel, sidesteps gsplat's per-(torch,cuda) wheel matrix → truly "installs like gsplat." Apache-2.0 + SPDX headers + `py.typed`; extras `[render]`(gsplat) `[pypose]` `[theseus]` `[gtsam]` `[dev]` `[docs]` — backends opt-in so core stays light. Docs: a prominent **pose-convention page** (`T_wc` vs `T_cw`, left/right perturbation — half of all registration bug reports) + an examples gallery (5-line quickstart, custom Residual, swap backend, differentiable registration). CI: `gradcheck`/finite-diff Jacobian tests + a synthetic recovery test.
- **Flags:** the `theseus` backend is the best-evidenced external path; `pypose`/`gtsam` adapter glue needs a source spike before promising. Multi-pose `optim_vars=` escape hatch = out of scope for v1.

**Primary sources:** [gsplat](https://github.com/nerfstudio-project/gsplat) · [PyPose](https://github.com/pypose/pypose) · [Theseus `CostFunction`](https://github.com/facebookresearch/theseus/blob/main/theseus/core/cost_function.py) · [kornia](https://github.com/kornia/kornia) · [nerfstudio pyproject](https://github.com/nerfstudio-project/nerfstudio/blob/main/pyproject.toml).
