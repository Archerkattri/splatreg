# Photometric refinement: when and why

`register(..., refine="photometric")` runs an opt-in second stage **after** the geometric
solve: it renders both splats from a shared synthetic camera ring and minimises the
image-space difference over the SE(3)/Sim(3) tangent. No real images are needed: both
"views" are renders of the splats themselves.

```python
from splatreg import register

result = register(target, source, transform="se3",
                  refine="photometric")          # needs `pip install "splatreg[render]"`
print(result.info["refine"])                     # stage diagnostics: n_iters, cost history
```

(Any differentiable renderer can be substituted via `refine_kwargs=dict(render_fn=...)`,
the test suite runs the whole stage on CPU with a pure-torch mock renderer.)

## Why geometry alone can fail

Geometric residuals (ICP, Gaussian-SDF) score **shape**. Any degree of freedom that does not
change the shape is *invisible* to them: rotation about a symmetry axis, sliding along an
extrusion, anything carried only by **texture**. On such inputs a geometric solver doesn't
just plateau; it can confidently walk *away* from the truth, because every pose along the
symmetric orbit scores the same and noise picks the winner.

This is exactly the failure mode **PhotoReg** ([arXiv 2410.05044](https://arxiv.org/abs/2410.05044))
identified for 3DGS registration and fixed with photometric refinement against captured
training images. splatreg's stage is the **splat-to-splat variant**: both sides are rendered
from the same synthetic camera ring, so it works on bare `.ply` pairs with no access to the
original captures.

## Measured: three cases, three honest answers

| Case | Initial error | Geometric register | + photometric refine |
|---|---|---|---|
| **Rotation-symmetric colored sphere** (mock renderer, CPU; geometry symmetric, colors azimuth-painted) | 6.0° / 8.9 mm | **11.2°** / 1.6 mm, rotation gets *worse* | **2.2°** / 1.0 mm |
| **Real gsplat rasterizer** (CUDA, 2k Gaussians, same sphere) | 5° / 7 mm | n/a | **0.36° / 0.5 mm** in ~1.1 s |
| **Dense-overlap real capture** (102,944-Gaussian splat, disjoint random halves, injected 2° / 1.24 mm seam) | 2° / 1.24 mm | **0.239° / 0.26 mm** in 56 s | +1.7 s, **neutral**, geometry already pins the pose |

Sources: rows 1–2 are `tests/test_photometric_refine.py` (27 tests; row 1 re-measured on
CPU at the time of writing); row 3 is
[`benchmarks/photometric_refine_bench.py`](https://github.com/Archerkattri/splatreg/blob/main/benchmarks/photometric_refine_bench.py)
on the same real capture as the merge headline, GPU reference run recorded in
[`benchmarks/photometric_refine_results.md`](https://github.com/Archerkattri/splatreg/blob/main/benchmarks/photometric_refine_results.md).

## When to reach for it

**Photometric refinement is decisive when geometry under-constrains the pose, symmetric or
low-relief shapes whose remaining DoF live only in texture, and neutral when dense
overlapping geometry already constrains it; its accuracy floor is set by render resolution
(≈0.3° at the default small rings).** That is why it ships **opt-in**: on well-constrained
scans it costs render time and buys nothing (case 3), while on the symmetric case it is the
difference between a confidently wrong pose and a correct one (case 1).

Rules of thumb:

- **Use it** when the object/scene has rotational symmetry, repeated geometry, or flat
  texture-carried detail (labels, murals, painted props), anywhere two poses look the same
  to a depth sensor but different to a camera.
- **Skip it** when the overlap is dense and geometrically distinctive, the geometric stage
  already lands well under the photometric stage's render-resolution floor (case 3:
  geometric 0.239° vs a ≈0.3° floor).
- It refines, it does not rescue: start it from the geometric result (that is what
  `refine="photometric"` does automatically), not from scratch.

## Exposure compensation (v1.2, default ON)

Independently-captured splat pairs disagree on exposure / white balance, and without a model
for it that *global colour offset leaks into the pose*, the LM trades real alignment against
tint. The stage therefore fits a bounded per-pair appearance model (per-channel `gain`/`bias`
on the rendered source), **alternated** with the pose: fitted closed-form at the start of every
LM stage, held fixed within it (so the pose Jacobian stays clean), then refitted once more with
a short polish round. Bounds (`gain ∈ [0.5, 2.0]`, `|bias| ≤ 0.2`) stop the model from
absorbing real misalignment; the fit lands in `info["refine"]["exposure"]`.

*Measured* (mock renderer, Sim(3), source tinted ×1.3 + 0.05): the tint absorbs into the
**scale** DoF without compensation (scale error clean **0.10%** → tinted **3.99%**), and the
compensated run recovers **0.47%** with the fitted gain landing at ≈ 1/1.3 per channel; on a
clean pair the default-ON model is harmless (0.01%). Both directions asserted in
[`tests/test_photometric_refine.py`](https://github.com/Archerkattri/splatreg/blob/main/tests/test_photometric_refine.py).

## Coarse-to-fine ladder (v1.2)

A single render size trades basin width against the accuracy floor (≈0.3° at 96 px). Pass
`refine_kwargs=dict(ladder=(96, 160, 256))` to run the stage once per rung (square renders at
each size, each rung warm-starting the next, intrinsics rescaled per rung), so the coarse rung
supplies the basin and the fine rung the floor. Per-rung diagnostics land in
`info["refine"]["ladder"]`.

*Measured* (mock renderer, 6° initial offset, equal per-stage iteration budget): a cold single
96 px rung stalls at **5.61°**; the 32→64→96 ladder lands **2.55°**.

## Knobs

All forwarded through `refine_kwargs`:

| kwarg | default | meaning |
|---|---|---|
| `n_views` | 8 | cameras on the synthetic ring |
| `width` / `height` | 128 | render resolution, the accuracy floor |
| `ladder` | `None` | coarse-to-fine render sizes, e.g. `(96, 160, 256)`; overrides `width`/`height` |
| `exposure` | `True` | per-pair gain/bias appearance model, alternated with the pose |
| `gain_bounds` / `bias_bound` | `(0.5, 2.0)` / `0.2` | exposure-model clamps |
| `max_iters` | from `quality` policy (`full`=10, `balanced`=8, `low`=5) | LM iterations per stage |
| `dssim_weight` | 0.0 | append D-SSIM rows to the RGB residual |
| `jac_mode` | `"fd"` | `"fd"` finite differences or `"autodiff"` (row-chunked `jacrev`) |
| `render_fn` | gsplat | any differentiable `(splat, T_CW, K, W, H, sh_degree) -> images` |

`merge(..., refine="photometric")` applies the stage to each pairwise registration.

## Reproduce

```bash
python -m pytest tests/test_photometric_refine.py -q          # 27 tests, CPU mock renderer
python benchmarks/photometric_refine_bench.py                 # real 103k-Gaussian pair (GPU)
```
