# CLI guide

`pip install splatreg` puts a `splatreg` command on your PATH. It speaks the standard 3DGS
PLY (INRIA / gsplat / Nerfstudio-splatfacto exports; SuperSplat reads and writes it), so the
shell workflow is: **export PLYs → `splatreg align` or `merge` → open the result in your
viewer**. No Python required.

```text
$ splatreg --help
usage: splatreg [-h] [--version] {align,merge,info} ...

positional arguments:
  {align,merge,info}
    align             register source.ply onto target.ply and write the aligned source
    merge             register N splats into one frame, fuse, dedupe, write one .ply
    info              print what is inside a 3DGS .ply
```

All heavy subcommands accept `--quality` (`full` default, `balanced`, `low`, `auto`, or a
0–1 float), `--device` (default: CUDA when available, else CPU), and `--max-iters`.

---

## `splatreg align` — register two splats

```bash
splatreg align target.ply source.ply -o aligned.ply \
    [--transform se3|sim3] [--init fast|robust|learned|global|features] \
    [--quality Q] [--device DEV] [--max-iters N]
```

Registers **source onto target**, prints the recovered transform, and writes the source splat
with the transform baked in — open `target.ply` + `aligned.ply` together in SuperSplat and
they line up.

```text
$ splatreg align target.ply source.ply -o aligned.ply
target: target.ply (1500 Gaussians)
source: source.ply (1500 Gaussians)
registering on cpu (transform=se3, init=fast, quality=full)
T (4x4, maps source -> target):
  [ 0.913782   0.104365  -0.392568  -0.056074]
  [-0.054636   0.989227   0.135812   0.049781]
  [ 0.402513  -0.102654   0.909640  -0.064587]
  [ 0.000000   0.000000   0.000000   1.000000]
scale     : 1.000000
rmse      : 0.000286919
iterations: 20
converged : False
time      : 2.71 s
wrote aligned.ply (1500 Gaussians, source aligned into the target frame)
```

(That run took the source from **154 mm** Chamfer off the target to **0.05 mm**.)

- `--transform sim3` additionally recovers a **scale** factor (captures taken at different
  scales — splatreg is the only splat registrar that does this).
- `--init` picks the coarse initializer; the default `fast` suits objects / full-overlap.
  Real metre-scale scans: `robust` or `learned`. Unknown large rotation: `global`.
  Partial overlap: `features`. See [Init modes](init-modes.md).
- If the overlap doesn't constrain the pose, the CLI prints an explicit
  **`WARNING: pose flagged AMBIGUOUS`** to stderr rather than handing you a silently wrong
  transform.

## `splatreg merge` — N splats → one fused splat

```bash
splatreg merge a.ply b.ply [c.ply ...] -o fused.ply \
    [--ref 0] [--transform sim3|se3] [--init global|fast|robust|learned|features] \
    [--no-dedupe] [--dedupe-method voxel|knn] [--voxel EDGE]
```

Registers every splat onto the reference (`--ref`, default the first), bakes the transforms
in, concatenates, then **dedupes the double-density overlap** (voxel-grid by default; `knn`
also removes boundary-straddling duplicates). Defaults are merge-tuned: `--transform sim3`
absorbs scale differences between captures, `--init global` survives large inter-capture
offsets.

```text
$ splatreg merge target.ply source.ply -o fused.ply
loaded target.ply (1500 Gaussians)
loaded source.ply (1500 Gaussians)
merging 2 splats on cpu (ref=0 -> target.ply, transform=sim3, init=global, dedupe=True)
fused 3000 -> 1981 Gaussians in 126.12 s
wrote fused.ply
```

!!! tip "Speed"
    `--init global` is the robust blind sweep and is the slow path, especially on CPU. When
    your captures are already roughly aligned (small offset), `--init fast` is orders of
    magnitude quicker. On a CUDA machine the default device is the GPU and everything is much
    faster.

## `splatreg info` — inspect a 3DGS PLY

```bash
$ splatreg info fused.ply
file      : fused.ply
gaussians : 1981
bounds min: [0.2512, -0.2964, 0.2368]
bounds max: [0.5784, -0.1046, 0.3631]
extent    : [0.3272, 0.1918, 0.1263]
colors    : SH degree 0 (DC only)
opacity   : raw [1.000, 1.000]  sigmoid mean 0.731
scales    : log-stored, linear median 0.00400  max 0.00400
```

Useful sanity checks before a merge: do the two files have comparable extents (a 10× extent
mismatch means you want `--transform sim3`), what SH degree they carry, and whether opacities
look like raw logits (they should — see [PLY interop](ply-interop.md)).

---

## Recipe: merge two SuperSplat scenes

SuperSplat has no automatic merge — align with splatreg, finish in SuperSplat:

```bash
pip install splatreg
splatreg align sceneA.ply sceneB.ply -o sceneB_aligned.ply   # auto SE(3); add --transform sim3 for scale
splatreg merge sceneA.ply sceneB.ply -o fused.ply            # or let splatreg fuse + dedupe too
```

Then open `fused.ply` (or `sceneA.ply` + `sceneB_aligned.ply`) in SuperSplat and continue
editing there.
