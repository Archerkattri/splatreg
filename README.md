<h1 align="center">splatreg</h1>

<p align="center">
  <b>Composable geometry-first SE(3)/Sim(3) registration for 3D Gaussian Splatting.</b><br>
  <i>gsplat renders your Gaussians; splatreg registers against them.</i>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9%2B-blue.svg">
  <img alt="status: alpha" src="https://img.shields.io/badge/status-alpha-orange.svg">
</p>

---

**splatreg is the inverse of gsplat.** gsplat takes Gaussians and a camera and produces an
image; splatreg takes two Gaussians (or a splat and an observation) and produces the **Sim(3)
transform that aligns them** — rotation, translation, **and scale**. It is the splat-to-splat
registration that [SuperSplat](https://github.com/playcanvas/supersplat),
[INRIA `graphdeco #990`](https://github.com/graphdeco-inria/gaussian-splatting/issues/990), and
Cesium / geospatial users keep asking for, where today's tooling punts with "use the manual
gizmo."

It **composes, it does not compete**:

- **[gsplat](https://github.com/nerfstudio-project/gsplat)** — bring your renderer. splatreg
  consumes gsplat tensors directly and delegates the photometric residual to gsplat's CUDA
  rasteriser. It does not reimplement rendering.
- **[PyPose](https://pypose.org/) / [Theseus](https://github.com/facebookresearch/theseus) /
  [GTSAM](https://gtsam.org/)** — bring your solver. The Levenberg–Marquardt loop is pluggable
  via `backend=`; the built-in solver is pure-Python and dependency-light.
- **It is a library, not a SLAM system.** Pairwise registration and merge are the wedge;
  loop-closure and global bundle adjustment stay where they belong (GTSAM). Keeping splatreg a
  composable *library* is what keeps it distinct.

The differentiator is the **Gaussian-derived SDF residual**: instead of treating a splat as a
bag of points (ICP-only, like the existing `GaussianSplattingRegistration` / `splatalign`
tools), splatreg scores the source splat against the *density field* the target's Gaussians
define — the same trick the [GaussianFeels](https://github.com/KrishiAttriSNU/Gaussianfeels)
tracker uses to register observations against a frozen object map.

## Install

```bash
pip install splatreg
```

Core install is light — just `torch` and `numpy`. Optional extras pull in the heavier,
swappable pieces:

```bash
pip install "splatreg[render]"    # gsplat: the Photometric residual + splatreg.render
pip install "splatreg[pypose]"    # PyPose solver backend
pip install "splatreg[theseus]"   # Theseus differentiable-solver backend
pip install "splatreg[all]"       # gsplat + pypose + theseus
```

PLY I/O works out of the box with the built-in binary-PLY codec (zero extra deps). Installing
the optional `plyfile` package additionally enables ASCII / non-standard PLY headers.

## Quickstart

Register two overlapping Gaussian splats and merge them into a single `.ply`:

```python
import splatreg

a = splatreg.io.load_ply("scan_a.ply")           # standard 3DGS .ply
b = splatreg.io.load_ply("scan_b.ply")
result = splatreg.register(a, b)                  # Sim(3): rotation, translation, scale
merged = splatreg.merge([a, b])                   # aligns all to ref, concatenates, dedupes
splatreg.io.save_ply(merged, "merged.ply")        # the single .ply SuperSplat users want
```

`splatreg.register` returns a `RegisterResult` with the 4×4 transform and diagnostics:

```python
result.T                                          # (4, 4) transform aligning source -> target
result.scale                                      # the recovered Sim(3) scale factor
result.converged                                  # bool
result.info["rmse"], result.info["overlap"], result.info["n_iters"]
```

Everything is configurable — choose the residual stack, the initialisation, SE(3) vs Sim(3),
and the solver backend:

```python
result = splatreg.register(
    a, b,                                                   # target, source
    residuals=[splatreg.residuals.SDF(1.0),                # Gaussian-SDF (the differentiator)
               splatreg.residuals.ICP(0.5)],               # + point-to-plane ICP
    init="global",                                          # "global" coarse init | "identity" | a Sim3
    transform="sim3",                                       # "sim3" (default) | "se3"
    backend="builtin",                                      # | "pypose" | "theseus" | "gtsam"
    quality="full",                                         # "full" (default) | "balanced" | "low" | "auto" | 0..1
)
```

Add a custom residual by subclassing `splatreg.Residual` (a single SE(3)/Sim(3) cost — reads
like math, not factor-graph boilerplate); add `splatreg.residuals.Photometric` to fold in
gsplat-rendered appearance.

### gsplat interop

A splatreg `Gaussians` round-trips with the gsplat rasteriser, so you can register and then
render with no glue code:

```python
from gsplat import rasterization
g = splatreg.io.from_gsplat(means, quats, scales, opacities, colors)   # wrap gsplat tensors
render, alpha, _ = rasterization(viewmats=viewmats, Ks=Ks,
                                 width=W, height=H, **splatreg.io.to_gsplat(g))
```

### Quality & hardware adaptivity

The Sim(3) refine autodiffs its Gaussian-SDF residual, whose reverse-mode Jacobian is the
memory hot-spot. `register` / `merge` / `Tracker` take a single `quality=` knob so the same code
runs full-fidelity on a big GPU **and** (at reduced fidelity) on a small GPU or a CPU-only laptop
without OOM. **Full quality is the default — it is never silently lowered.**

```python
splatreg.register(a, b)                       # quality="full" — all source anchors, full fidelity
splatreg.register(a, b, quality="balanced")   # bounded source sample (faster, smaller)
splatreg.register(a, b, quality="low")        # smallest sample
splatreg.register(a, b, quality=0.5)          # 0..1 scale (1.0 == full)
splatreg.register(a, b, quality="auto")       # detect free GPU/CPU memory, pick the largest that FITS
```

Two kinds of knob sit behind it, and they are kept separate on purpose:

- **Accuracy** (`n_points` — the source-anchor sample size, plus `knn` / iteration count) is what
  `"full"` / `"balanced"` / `"low"` / a `0..1` scale actually trade. `"full"` keeps it maximal
  (every source anchor); `"auto"` keeps it maximal too unless the detected memory can't hold it.
- **Peak memory** (the autodiff row-chunk + the SDF forward block) is *numerically lossless* — a
  chunked Jacobian is bit-for-bit the unchunked one — so it is **always** auto-fitted to the
  available memory, even in `"full"` mode. That is what lets full quality run on a 2 GB GPU at
  well under 1 GB peak with identical results to a 32 GB card.

`"auto"` reads `torch.cuda.mem_get_info` on CUDA (or `psutil` / `os.sysconf` RAM on CPU) at call
time and sizes the work to fit. For total control, build a `splatreg.QualityConfig` yourself and
pass it as `quality=` (it is then used verbatim — you own the memory budget). The resolved policy
is recorded in `result.info["quality"]`.

## Status

**Alpha — registration-first v0.1.** Pairwise Sim(3) splat-to-splat registration + merge. Camera
localization, object-6DoF tracking, and multi-splat bundle registration are deferred by
milestone. See [`docs/`](docs/) for the MVP spec, the pose-convention reference, and the
synthetic Sim(3)-recovery validation protocol.

## Acknowledgements & Citation

splatreg carves out and generalises the registration core of
[**GaussianFeels**](https://github.com/KrishiAttriSNU/Gaussianfeels): the composable SE(3)
Levenberg–Marquardt solver and the Gaussian-derived SDF residual
(`signed_distance_via_gaussian_density`), extended here with the Sim(3) scale degree of freedom.
The PLY and tensor contracts are [**gsplat**](https://github.com/nerfstudio-project/gsplat)-native.

If you use splatreg in academic work, please cite:

```bibtex
@software{attri2026splatreg,
  author  = {Krishi Attri},
  title   = {splatreg: Composable SE(3)/Sim(3) registration for 3D Gaussian Splatting},
  year    = {2026},
  url     = {https://github.com/KrishiAttriSNU/splatreg},
  note    = {Built on the GaussianFeels SE(3)-LM and Gaussian-SDF; gsplat-native}
}
```

and the works splatreg builds on — [gsplat](https://github.com/nerfstudio-project/gsplat) and
the original 3D Gaussian Splatting
([Kerbl et al., SIGGRAPH 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)).

## License

[Apache License 2.0](LICENSE) — Copyright 2026 Krishi Attri.
