<div class="sr-hero">
  <p class="sr-hero__eyebrow">the inverse of gsplat · SE(3) / Sim(3)</p>
  <h1 class="sr-hero__mark">splatreg<span class="caret">▍</span></h1>
  <p class="sr-hero__tagline">Register Gaussian splats: align and merge 3DGS scans into one frame.</p>
  <p class="sr-hero__pitch">gsplat <em>renders</em> your Gaussians; splatreg <em>registers</em> them. Two 3DGS scans of the same scene go in, one <code>SE(3)</code> or <code>Sim(3)</code> transform comes out, and optionally one fused, deduped splat. Pure PyTorch — no meshing, no CUDA extension, no point-cloud detour.</p>
  <div class="sr-install"><span class="sr-prompt">$</span> pip install splatreg</div>
  <div class="sr-cta">
    <a class="md-button md-button--primary" href="quickstart/">Quickstart</a>
    <a class="md-button" href="init-modes/">Init modes</a>
    <a class="md-button" href="https://github.com/Archerkattri/splatreg">GitHub</a>
    <a class="md-button" href="https://doi.org/10.31224/7313">Paper (engrXiv)</a>
  </div>
  <div class="sr-stats">
    <div class="sr-stat"><div class="sr-stat__value">0.974</div><div class="sr-stat__label">BUFFER-X zero-shot seed recall, official 3DMatch (6/8 scenes)</div></div>
    <div class="sr-stat"><div class="sr-stat__value">5.2°</div><div class="sr-stat__label">rotation error on a real splat, vs 15.3° / 36.3° for splat tools</div></div>
    <div class="sr-stat"><div class="sr-stat__value">native</div><div class="sr-stat__label">Sim(3) scale recovery — no other splat registrar does it</div></div>
    <div class="sr-stat"><div class="sr-stat__value">~2.4e-15</div><div class="sr-stat__label">SH-under-rotation error (float64), test-locked</div></div>
  </div>
</div>

splatreg is the missing *registration* half of the Gaussian-splatting toolchain: the
splat-to-splat alignment that SuperSplat / INRIA / geospatial users keep asking for, where
today's tooling punts to a manual gizmo. It works with anything that speaks the standard 3DGS
PLY (gsplat, Nerfstudio/splatfacto, INRIA, SuperSplat) or hands over means/covariance tensors.

## Headline result — a zero-shot seed that holds up on real data

<figure class="sr-figure">
  <img src="https://raw.githubusercontent.com/Archerkattri/splatreg/main/assets/bufferx_recall.png" alt="BUFFER-X zero-shot seed vs classical FPFH seed: registration recall on 3DMatch and the low-overlap 3DLoMatch regime">
  <figcaption>Registration recall for the zero-shot <strong>BUFFER-X</strong> seed (ICCV 2025) against the classical robust FPFH seed, both pushed through the <em>identical</em> splatreg refine so the bars isolate the seed. 3DMatch is the official <code>gt.log</code> pair set (6/8 scenes, n=1250); the low-overlap bars are a 50/scene GT-derived run (n=400). Both seeds share the lighter <code>feature_align</code> refine, so these isolate the seed rather than report full-pipeline absolute numbers; the remaining scenes and the official 3DLoMatch runs are in progress. See <a href="init-modes/">Init modes</a>.</figcaption>
</figure>

## What you get that no other splat registrar ships

<div class="grid cards" markdown>

-   __Provably correct SH rotation__

    Higher-order spherical-harmonic bands (`f_rest`) are mixed by the real-basis Wigner-D
    matrix, so glossy highlights turn *with* the splat. Test-locked to **~2.4e-15** in float64.

-   __Align without merging__

    `apply_transform()` / `splatreg align` bakes the recovered pose into the source and writes
    it as its own PLY — both scans stay separate files, now in one frame.

-   __Sim(3) scale recovery__

    Native scale estimation, which none of the competing splat tools attempt at all — plus
    photometric refinement (exposure compensation + coarse-to-fine ladder) for the poses
    geometry cannot see.

-   __Honest diagnostics__

    Pose covariance on every builtin-LM solve for pose-graph weighting (`None` when singular,
    never faked), and ambiguous overlaps are *flagged* — never silently wrong-posed.

</div>

## 30 seconds, end to end

```bash
pip install splatreg
splatreg align scan_a.ply scan_b.ply -o b_aligned.ply     # register + write aligned PLY
splatreg merge scan_a.ply scan_b.ply -o fused.ply          # register + fuse + dedupe
```

or in Python:

```python
from splatreg import register, merge, apply_transform
from splatreg.io import load_ply, save_ply

a = load_ply("scan_a.ply")          # target (stays fixed)
b = load_ply("scan_b.ply")          # source (gets aligned)

result = register(a, b, transform="sim3")   # init="fast" by default, ~17 ms
print(result.T)        # 4x4 similarity [[s*R, t], [0, 1]], maps source -> target
print(result.scale)    # recovered scale (1.0 for transform="se3")

fused = merge([a, b])               # register + concat + dedupe the overlap
save_ply(fused, "fused.ply")        # opens in SuperSplat / any 3DGS viewer

# or keep the scans separate, just registered into one frame:
save_ply(apply_transform(b, result.T, result.scale), "b_aligned.ply")
```

## Capability matrix

Honest comparison against the tools people actually use for this job. The accuracy row is
measured head-to-head on a real splat with known ground truth; editor columns reflect their
design (manual transforms, not registration).

<div class="sr-matrix" markdown>

| | **splatreg** | splatalign | GaussianSplattingRegistration | SuperSplat / SplatTransform |
|---|---|---|---|---|
| Automatic splat-to-splat registration | yes (6 init modes) | ICP from identity | Open3D RANSAC+ICP | no (manual gizmo) |
| Measured rotation error, real splat + GT | **5.2°** | 15.3° | 36.3° | n/a |
| Sim(3) scale recovery | **yes, native** | no (SE(3) only) | no (SE(3) only) | manual |
| SH (`f_rest`) rotated with the splat | **yes, test-locked** | no | no | no |
| Merge + overlap dedupe | yes | no | no dedupe | concat only |
| Photometric refine (exposure comp + ladder) | yes | no | no | no |
| Pose covariance for pose graphs | yes | no | no | n/a |
| Honest ambiguity flag (never silent-wrong) | yes | no | no | n/a |
| Zero-shot learned seed (BUFFER-X) | yes | no | no | no |

</div>

## How it works

```mermaid
flowchart LR
    A["splat A<br/>(target)"]:::s --> G
    B["splat B<br/>(source)"]:::s --> G
    G["<b>Global aligner</b><br/>super-Fibonacci SO(3) seeds<br/>+ batched trimmed ICP<br/><i>(or FPFH / learned / BUFFER-X / MAC)</i>"]:::g --> L
    L["<b>Levenberg-Marquardt</b><br/>multi-residual:<br/>ICP + Gaussian-SDF<br/>SE(3) / Sim(3)"]:::l --> T["T*  (4×4)<br/>+ merge / dedupe"]:::o
    classDef s fill:#e8f6f8,stroke:#17becf,color:#0b3d44;
    classDef g fill:#fff1ee,stroke:#ff6b5b,color:#5a1a12;
    classDef l fill:#eef7ee,stroke:#2e8b57,color:#143d22;
    classDef o fill:#f3eefc,stroke:#7d52c7,color:#2c1654;
```

1. **Global init**: a coarse pose from a dense super-Fibonacci rotation sweep + batched
   trimmed ICP (no local-minimum trap), with FPFH+RANSAC (`init="robust"`), learned
   GeoTransformer (`init="learned"`), zero-shot BUFFER-X (`init="bufferx"`), and MAC
   maximal-clique (`init="mac"`) seeds for harder real scans. See [Init modes](init-modes.md).
2. **Refinement**: a from-scratch Levenberg-Marquardt core over ICP (point-to-point /
   point-to-plane) *and* splatreg's flagship **Gaussian-SDF** residual (a smooth signed
   distance field derived directly from the target Gaussians, with a closed-form, audited
   Jacobian), solving the full SE(3) or Sim(3) tangent and exposing the pose
   information/covariance at the optimum.

## More headline numbers

| | **splatreg** | reference |
|---|---|---|
| Real-splat merge (103k Gaussians) | Chamfer **10.3 → 2.0 mm (5.1×)**, overlap **0.03 → 0.67 (22×)** | naive concat |
| Official 3DMatch recall (`learned` seed) | **91.5%** mean, 93.5% pooled | GeoTransformer ~92%, Open3D ~77% |
| Official 3DLoMatch (hard, 10–30% overlap) | 72.5% mean, **74.4%** pooled | GeoTransformer ~74%, Open3D ~20% |
| Photometric refine (real rasterizer) | 5°/7 mm → **0.36°/0.5 mm** | geometric alone worsens the symmetric case |
| Registration speed | **~17 ms** (fast) | Open3D 142 ms |

Full record with reproduce commands: [Benchmarks](benchmarks.md).

<div class="sr-limits" markdown>

<p class="sr-limits__kicker">Honest edges — the repo's signature</p>

## Limitations

splatreg states where it stops working, in the docs and in the diagnostics:

- **Heavy overlap loss (keep ≤ 40%) is genuinely ambiguous.** The rotation-disambiguating
  geometry is physically absent; even the true pose does not seat cleanly. The aligner flags
  these (`result.info["ambiguous"]` / `["confidence"]`) and never silently wrong-poses.
  `merge` and `track` are built for high-overlap captures.
- **Scale is unobservable under thin overlap.** Under ~20% shared geometry the Sim(3) scale
  valley is flat; no algorithm recovers what the geometry does not carry.
- **Cost on rigid SE(3).** Plain ICP reaches the same SE(3) success far faster; the SDF
  residual buys scale + implicit-field robustness at a real compute cost. Use `track()`
  (~17 ms/frame) for the warm-start real-time path.

Full detail, including the failure analyses, is in
[`RESULTS.md`](https://github.com/Archerkattri/splatreg/blob/main/RESULTS.md).

</div>

## Where next

- [Quickstart](quickstart.md): install + the core workflows in Python.
- [CLI guide](cli.md): `splatreg align / merge / info` from the shell.
- [Init modes](init-modes.md): speed vs robustness — `fast`, `robust`, `learned`, the
  zero-shot **`bufferx`** seed, `mac`, `global` — with the honest measured 3DMatch/3DLoMatch
  verdicts.
- [Photometric refinement](photometric.md): the opt-in stage for poses geometry can't see
  (symmetry / texture-only DoF), with the measured when-and-why table, per-pair **exposure
  compensation** (default ON), and the **coarse-to-fine render ladder**.
- [PLY interop](ply-interop.md): splatfacto / INRIA / SuperSplat round-trip, and what happens
  to spherical harmonics under a recovered rotation (higher-order SH bands are **Wigner-rotated
  with the splat**; Ivanic-Ruedenberg, test-locked math).
- [Benchmarks](benchmarks.md): every number with its reproduce command.
- [API reference](api.md): every public function, autodoc'd.
