# splatreg — Research Synthesis & Build Plan

> **Status:** pre-build research synthesis · **Date:** 2026-06-05
> **One-liner:** *the inverse of gsplat — a composable, geometry-first SE(3) pose-registration library for 3D Gaussian Splatting.*
> *"gsplat renders your Gaussians; splatreg registers against them."*

This document is the founding analysis: a landscape scan (5 parallel research agents, web-sourced, 2024–2026) plus the worthiness verdict, where-it-fits story, risk register, and a concrete build plan. Source inventories are preserved in the appendices.

---

## TL;DR Verdict — **BUILD, but narrow-then-build.** The narrowing is the whole game.

Five independent research streams converged on the same shape: **the *method* (tracking a pose against a splat) is saturated, but the *library* slot is genuinely empty**, and the seed (the GaussianFeels VT-SLAM tracker) is unusually well-positioned to fill it.

- **Don't** pitch "a new way to track splats" — that loses to ~100 GS-SLAM systems, FoundationPose/BundleSDF (objects), and ~13 localization repos.
- **Do** ship a `pip install`, Apache-2.0, **composable multi-residual SE(3)-LM registration library** that is gsplat-native and backend-pluggable (PyPose/Theseus/GTSAM).
- **Anchor on objects first** (frozen-map, real-time, rigid-object tracking — the seed already does this and the lane is open), add camera localization second, scene-scale third.
- **The defensible differentiator nobody else packages:** a **Gaussian-derived SDF residual** composed with ICP + photometric + priors under one robust LM. (3DGS-LM proved LM-on-3DGS for *mapping*; **LM-on-SDF-for-pose is unclaimed.**)

---

## 1. The precise gap — crowded vs. open

### Crowded (never claim novelty here)
- **"Track a camera against a splat"** — ~100 GS-SLAM systems, 40+ added in 2025 (MonoGS, SplaTAM, …). Settled in 2024.
- **"6-DoF pose / relocalization against a splat"** — ~13 repos (iComMa, 6DGS, GSplatLoc, GS-CPR, GS-Pose, POGS, …).
- **Render-inversion (photometric) pose** and **feature-matching + PnP pose** — both taken.

### Open (the wedge — real and currently empty)
1. **No reusable library.** Every system above is a monolithic `run_slam.py` research pipeline welded to its own mapper/dataloader. There is no `pip install …; register(splat, obs)`. The only true libraries (gsplat, nerfstudio) scope pose to **training-time** refinement (RGB-photometric, nerfstudio-coupled, SE(3)-manifold handling unconfirmed — nerfstudio issue #449). **Inference-time "register against a frozen splat" is not packaged anywhere.**
2. **Geometry-first is thin; SDF-for-pose is empty.** The geometric trackers (GS-ICP 107 FPS, RTG-SLAM, G2S-ICP, S³LAM) are all **ICP-on-depth**. Nobody does **SDF-on-Gaussians for pose.** 3DGS-LM (ICCV'25) proves LM-on-3DGS works — but only for *map* fitting.
3. **Frozen-map, real-time, rigid-OBJECT tracker** — essentially unoccupied. The hot object work (6DOPE-GS @3.5 Hz, GTR, GSGTrack) is *joint pose+reconstruction* (slower, different problem, mostly no released code). A frozen-map dense solver credibly **beats them on speed**.

**The unoccupied combination:** standalone pip library + geometric-SDF/ICP-first + SE(3) LM + real-time rigid-object tracking on a frozen splat + composes with gsplat/GTSAM/Theseus. Each ingredient has prior art; *the integration + library framing* is the white space.

---

## 2. Where it fits — the seam (the honest "composes-with" story)

gsplat is **forward** (render); splatreg is the **inverse** (register). Clean, non-redundant layering:

```
   gsplat            →  differentiable rasterizer + photometric pose gradient (∂loss/∂[R|t])   [DEPEND ON — don't rebuild]
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │  splatreg            →  composable multi-residual SE(3)-LM registration loop                  │  [OWN — nobody ships this]
 │                      residuals: SDF(Gaussian-derived) + ICP + photometric(via gsplat)      │
 │                      + depth + priors(temporal/centroid) + tactile(optional plugin)        │
 │                      robust kernels · analytic Jacobians · GPU-native · real-time          │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
   PyPose / Theseus / GTSAM  →  Lie-group optimizer / differentiable / factor-graph BA  [COMPOSE — pluggable backend]
```

gsplat hands you a raw photometric gradient on a 4×4 matrix and **nothing above it** — no SE(3) parameterization, no solver, no robust kernels, no priors, no geometric residual. That void *is* the product. Accurate framings: **"PyPose-for-splats"** / **"gsplat renders, splatreg registers."**

**Backend guidance:**
- **PyPose** (Apache-2.0, torch-native, *alive*, >160k downloads in 2025) — default optimizer substrate; build *on* it, don't reinvent Lie/LM.
- **Theseus** (MIT) — the differentiable end-to-end option, **but dormant (last release Sep-2024)**; interop target, not a hard runtime dep.
- **GTSAM** (BSD) — graph-level backend for multi-frame BA / loop-closure when growing past single-pose.

---

## 3. Asset audit — the seed's starting position is unusually strong

- **License path is CLEAN.** The seed already renders through **gsplat (Apache-2.0)** — `production/gaussian/render/rasterizer.py:25`, `gsplat>=1.5.3`. **No nvdiffrast** (NVIDIA non-commercial, viral) and **no INRIA research-only rasterizer** in the production path. The single legal trap every other GS project hits is already dodged → **Apache-2.0 release is unobstructed.**
  - *Guardrail:* keep the SDF/geometry path off SuGaR/2DGS non-commercial code. The seed already rolls its **own** Gaussian-SDF (`signed_distance_via_gaussian_density`, the surface-distance loss kernel), so this is satisfied.
- **The seed already does the hard, empty part:** a composable multi-residual SE(3) LM (`production/gaussian/core/pose/se3_lm.py`) + a Gaussian-derived SDF residual + real-time GPU engineering + robustness machinery (adaptive priors, residual caps, multi-hyp seeding, symmetry). That's exactly the piece nobody packages. This is *extract-and-generalize*, not greenfield.

---

## 4. Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **Incumbent commoditization** — gsplat/nerfstudio (or NVIDIA) ships a thin inference-pose-fit utility; core looks like a wrapper. | **High** | Depth they won't match: the SDF/geometric residual nobody offers, composable residuals as a first-class API, robust analytic-Jacobian LM, object+scene, real-time. First-mover. **Move now** (no announced roadmap). |
| 2 | **Scope creep into full SLAM** → re-enters crowded MonoGS/SplaTAM arena. | Med-High | Stay a **library** (pose/registration). Defer graph/BA/loop-closure to GTSAM. |
| 3 | **"Your SDF is just another reinvention"** (SuGaR/2DGS/DiGS each have one). | Med | Novelty = packaging + pose application + generality, framed honestly — not "new primitive." |
| 4 | **Demand evidence thin** — search collapsed to arXiv; no counted backlog of practitioner pleas (inferred from research volume + gsplat issue signals). | Med | **1-day manual demand check** (r/GaussianSplatting, gsplat/nerfstudio Discords + issues) **before heavy investment**. This is the gate. |
| 5 | **Scene scale forces engineering** — dense full-map geometric registration is not real-time at millions of Gaussians; needs spatial cull. | Med | Geometric tracking is real-time-proven at object/room scale (GS-ICP 107 FPS — Gaussian covariances make G-ICP correspondences ~free, no KD-tree). Defer scene scale to a later milestone gated on a voxel-hash/LoD spatial index. |

---

## 5. The plan

**Positioning:** the inference-time pose/registration tier the GS ecosystem lacks. Installs like gsplat (pip, Apache-2.0), composes with gsplat (consumes its gradient) and PyPose/Theseus/GTSAM (pluggable backend). **Library, not a SLAM system.**

**Name:** **`splatreg`** (mirrors `gsplat`; reads as its inverse — "fit poses to your Gaussians"). Alternate: `splatreg`. Avoid anything with "Loc" (GSplatLoc collision ×2).

**Core primitive it owns:** a composable multi-residual SE(3)-LM registration loop where residuals are **plugins** — `SDF`, `ICP`, `Photometric(gsplat)`, `Depth`, `Prior`, `Tactile(optional)` — with robust kernels, analytic Jacobians, GPU-native, real-time.

### Milestones
- **v0.1 (MVP — narrow & proven):** object 6-DoF tracking against a frozen object splat, vision-only (= seed minus tactile/feelsight). API: `t = SplatTracker(gaussians); pose = t.track(rgb, depth, K, init)`. Residuals: SDF + ICP + photometric. Benchmark vs **GS-Pose / 6DGS / POGS** (real-time + accuracy + "5-line import"). Apache-2.0, docs, 2 examples.
- **v0.2 (second mode — cleanest demand slot):** camera localization against a scene splat (AR/VR "localize in my splat") + the **"drop-in pose refiner for gsplat training"** demo (better poses → better reconstruction). Benchmark vs **GSplatLoc / GS-CPR** (Replica / 7-Scenes).
- **v0.3 (scale + backends):** spatial index (voxel-hash/LoD) for scene-scale geometric tracking; pluggable PyPose/Theseus/GTSAM backends; **differentiable mode** (learn residual weights end-to-end) — the research frontier where Theseus becomes infrastructure you build *on*.
- **Paper (parallel):** *"A composable, geometry-first registration library for Gaussian splats"* — the library + the SDF-LM-for-pose contribution + honest benchmarks. The gsplat/PyPose precedent shows well-packaged GS infra earns outsized citations.

### Immediate next steps (this week)
1. **Verify demand** (1 day) — manual scan of r/GaussianSplatting, gsplat/nerfstudio Discords + issues. **Gate.**
2. **Carve the seed** — extract `se3_lm` + the residual stack from GaussianFeels into a clean standalone repo skeleton; decouple tactile/feelsight; define the **residual plugin interface** + the **SDF-from-Gaussians query** as a reusable primitive.
3. **MVP object-tracking demo** on a public dataset (YCB-V or Toys4K/feelsight) + the first benchmark row vs GS-Pose.

**Bottom line:** a real, defensible, currently-empty niche — *if* shipped as a composable geometry-first library ("splatreg: the inverse of gsplat"), anchored on objects first, leaning on the SDF/LM residual nobody else has, and never pitched as "a new way to track splats." The window is open because the incumbents deliberately looked the other way — so the clock matters.

---

## Appendix A — Competitive landscape inventory (primary-sourced, 2024–2026)

### A.1 Camera-pose / GS-SLAM (CROWDED — don't compete on method)
| System | Year/Venue | Pose method | Lib/Pipe | License | Notes |
|---|---|---|---|---|---|
| MonoGS (Gaussian Splatting SLAM) | CVPR'24 | photometric (analytic SE3 Jac) | Pipeline | custom non-commercial | ~3 FPS; mono/RGB-D/stereo |
| SplaTAM | CVPR'24 | photometric (silhouette+RGB-D) | Pipeline | BSD-3 | RGB-D only |
| Photo-SLAM | CVPR'24 | geometric (ORB+factor graph) | Pipeline | **GPL-3.0** | 3DGS map only |
| Gaussian-SLAM | 3DV-era | photometric | Pipeline | MIT | sub-maps |
| RTG-SLAM | SIGGRAPH'24 | **geometric** point-to-plane ICP | Pipeline | research | on rendered depth+normals |
| GS-ICP-SLAM | ECCV'24 | **geometric** Generalized-ICP | Pipeline | MIT | **~100–107 FPS**; shared covariances |
| Splat-SLAM | NeurIPS'24 | DROID dense-BA + deformable GS | Pipeline | research | mono RGB |
| MASt3R-SLAM | CVPR'25 | learned pointmap matching | Pipeline | research | ~15 FPS; not a splat tracker |
| G2S-ICP SLAM | 2025 | **geometric** GICP + 2D-Gaussian | Pipeline | research | ~30 FPS |
| S³LAM | ICRA'26 | **geometric** surfel | Pipeline | research | real-time |
| GigaSLAM | SIGGRAPH Asia'25 | hierarchical sparse-voxel cull | Pipeline | research | **city-scale**; tracks on LoD subset |

### A.2 Camera relocalization against a splat
| System | Year | Method | License/★ | Notes |
|---|---|---|---|---|
| iComMa | Dec'23 | render-inv + keypoint hybrid | ~132★ | direct lineage to photometric residual |
| 6DGS | ECCV'24 | feat-match / ray-inversion, **init-free** | ~179★ | ~15 FPS; no iterative AbS |
| GSplatLoc | Dec'24 | **geometric depth** residual | ~93★, maintained 2026-06 | closest scene-level depth analog |
| GS-CPR | ICLR'25 | MASt3R matching + PnP/RANSAC | ~146★, maintained | test-time refiner; no pose grads |
| SplatLoc | TVCG'24 | learned 3DGS descriptors + PnP | — | AR-oriented |
| SplatPose | Mar'25 | feat-match + render, single-RGB | — | — |
| iNeRF | IROS'21 | render-inv | ~212★ stale | NeRF-era ancestor |

### A.3 Object 6-DoF pose against a splat (THE RELEVANT LANE — thinner, younger)
| System | Year | Method | Frozen-map? | License/★ | Notes |
|---|---|---|---|---|---|
| GS-Pose / GSPose | Mar'24 | hybrid retrieve + render-compare | 1-shot+track | **MIT, 142★** | most mature object repo |
| POGS | ICRA'25 | **geometric** depth+features | track | MIT, 50★, **nerfstudio-locked** | needs `ns-train`; not standalone |
| 6DOPE-GS | ICCV'25 | render-inv, joint pose+recon | no (builds map) | **no public code** | ~3.5 Hz RGB-D |
| GTR | May'25 | hybrid geom+appearance | no (recon) | no code (Toyota/TRI) | symmetry focus |
| GSGTrack | Dec'24 | render-inv silhouette | no (recon) | no code | RGB-only |
| GS2Pose | Nov'24 | PoseNet + Lie-algebra refiner | 1-shot+refine | — | RGB-D |

### A.4 Non-GS object-pose incumbents (the comparison users/reviewers will make)
- **FoundationPose** (NVlabs, CVPR'24) — zero-shot model-based+model-free object pose. The de-facto robotics answer today.
- **BundleSDF** (~10 Hz neural-SDF tracking) — **closest conceptual analog to the seed** (SE(3) optimization against a learned SDF).

### A.5 Key geometry/infra
- **3DGS-LM** (ICCV'25) — Levenberg-Marquardt for **map** fitting (+30% faster training). **Proves LM-on-3DGS works; pose application unclaimed.**

---

## Appendix B — Reference libraries (where splatreg slots in)
| Library | License | Alive? | Does for pose | Does NOT for pose |
|---|---|---|---|---|
| **gsplat** | Apache-2.0 | ✅ ~5.1k★ | photometric gradient ∂loss/∂`viewmats[R\|t]` | no SE(3) param, no solver, no robust kernels, no priors, **no geometric/SDF residual**; pose-opt out of scope |
| nerfstudio | Apache-2.0 | ✅ | `CameraOptimizer` SO3xR3/SE3, first-order Adam | **training-only**; no 2nd-order LM, no SDF residual, monolithic |
| **PyPose** | Apache-2.0 | ✅ >160k dl/yr | torch Lie groups + GN/LM (the substrate) | no splat/photometric/SDF awareness → **"PyPose-for-splats" = the gap** |
| Theseus | MIT | ⚠️ **dormant (Sep'24)** | differentiable NLS, batched, GPU, Lie | no splat cost; dense per-pixel autodiff slow; **sustainability bet** |
| GTSAM | BSD | ✅ | world-class **sparse** SE(3) (GN/LM/iSAM2) | not GPU/autodiff-native; painful dense per-pixel; **graph-level backend** |
| SymForce | Apache-2.0 | ✅ | symbolic → fast analytic Jacobian codegen | CPU/codegen-oriented; no splat hook → Jacobian-gen dep only |
| LieTorch | BSD-ish | ⚠️ low | SE3/SO3 tangent backprop primitive | source-build-only; **PyPI `lietorch` is an UNRELATED package (trap)** |
| jaxlie / jaxls | MIT-ish | ✅ | JAX LM/GN on manifold | JAX ecosystem orthogonal to torch/gsplat → breaks gsplat composition |
| kornia | Apache-2.0 | ✅ | SE3 Lie, PnP, RANSAC building blocks | no dense multi-residual LM, no splat integration |

---

## Appendix C — SDF-from-Gaussians camps (fragmented; no reusable primitive)
- **Baked regularizer:** SuGaR (non-commercial) — SDF is a training loss, not a query function.
- **Render-then-TSDF/Poisson:** 2DGS (non-commercial), RaDe-GS — reusable artifact is depth/normal; mesh via external TSDF.
- **Learned SDF MLP:** DiGS (Sep'25), GSDF, GSurf, GaussianRoom — auxiliary network, not analytic.
- **Lone arbitrary-3D-point evaluator:** GOF (Gaussian Opacity Fields) — tile-based opacity field, but view-projection-bound, not packaged standalone.
- → The seed's Gaussian-derived SDF is one more instance; **novelty = packaging + pose application + generality**, not the SDF idea.

---

## Appendix D — Confidence & caveats (verify before locking claims)
- **HIGH confidence:** library/method existence, licenses (gsplat Apache-2.0, SplaTAM BSD-3, Photo-SLAM GPL-3, GS-ICP/GS-Pose/Gaussian-SLAM MIT, nvdiffrast non-commercial), the "no reusable pose library" finding, gsplat scoping pose to training-only, GS-ICP 107 FPS geometric, 3DGS-LM = map-fitting only.
- **MEDIUM / flagged:** exact GitHub stars (drift; treat as ±), exact last-commit dates, large-scale FPS for G2S-ICP/GigaSLAM (PDFs opaque). gsplat `viewmats` differentiability confirmed by paper+nerfstudio but lightly API-documented — **read current `rasterization()` source before depending**.
- **WEAK / unverified:** direct practitioner demand (search collapsed to arXiv) → **manual demand check required**. "No released repo couples SDF-from-Gaussians + ICP + photometric for *pose*" is strong negative evidence but absence-of-evidence → **re-survey arXiv 2026 H1 before any novelty claim**.
- **Re-survey cadence:** object-pose-on-3DGS is moving fast (mostly 2024-H2→2025). Re-check within ~3 months.

---

*Generated from a 5-agent parallel landscape scan. Next: demand check → carve the standalone `splatreg` skeleton from the GaussianFeels seed.*
