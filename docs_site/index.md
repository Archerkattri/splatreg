# splatreg

**Register Gaussian splats — align & merge two 3DGS scans into one SE(3)/Sim(3) frame.**

gsplat *renders* your Gaussians; splatreg *registers* against them. It is the missing
registration half of the Gaussian-splatting toolchain: the splat-to-splat alignment that
SuperSplat / INRIA / geospatial users keep asking for, where today's tooling punts to a
manual gizmo.

- **Pure PyTorch** — no meshing, no CUDA extension, no point-cloud detour.
- **SE(3) and Sim(3)** — the only splat registrar that recovers **scale**.
- **Framework-agnostic** — gsplat, Nerfstudio/splatfacto, INRIA, SuperSplat, custom; anything
  that speaks the standard 3DGS PLY or hands over means/covariance tensors.
- **Honest diagnostics** — ambiguous overlaps are *flagged* (`info["ambiguous"]`,
  `info["confidence"]`), never silently wrong-posed.

## 30 seconds, end to end

```bash
pip install splatreg
splatreg align scan_a.ply scan_b.ply -o b_aligned.ply     # register + write aligned PLY
splatreg merge scan_a.ply scan_b.ply -o fused.ply          # register + fuse + dedupe
```

or in Python:

```python
from splatreg import register, merge
from splatreg.io import load_ply, save_ply

a = load_ply("scan_a.ply")          # target (stays fixed)
b = load_ply("scan_b.ply")          # source (gets aligned)

result = register(a, b, transform="sim3")   # init="fast" by default, ~17 ms
print(result.T)        # 4x4 similarity [[s*R, t], [0, 1]], maps source -> target
print(result.scale)    # recovered scale (1.0 for transform="se3")

fused = merge([a, b])               # register + concat + dedupe the overlap
save_ply(fused, "fused.ply")        # opens in SuperSplat / any 3DGS viewer
```

## How it works

```mermaid
flowchart LR
    A["splat A<br/>(target)"]:::s --> G
    B["splat B<br/>(source)"]:::s --> G
    G["<b>Global aligner</b><br/>super-Fibonacci SO(3) seeds<br/>+ batched trimmed ICP<br/><i>(or FPFH / learned)</i>"]:::g --> L
    L["<b>Levenberg–Marquardt</b><br/>multi-residual:<br/>ICP + Gaussian-SDF<br/>SE(3) / Sim(3)"]:::l --> T["T*  (4×4)<br/>+ merge / dedupe"]:::o
    classDef s fill:#e8f6f8,stroke:#17becf,color:#0b3d44;
    classDef g fill:#fff1ee,stroke:#ff6b5b,color:#5a1a12;
    classDef l fill:#eef7ee,stroke:#2e8b57,color:#143d22;
    classDef o fill:#f3eefc,stroke:#7d52c7,color:#2c1654;
```

1. **Global init** — a coarse pose from a dense super-Fibonacci rotation sweep + batched
   trimmed ICP (no local-minimum trap), with optional FPFH+RANSAC (`init="robust"`) and
   learned GeoTransformer (`init="learned"`) seeds for harder real scans.
   See [Init modes](init-modes.md).
2. **Refinement** — a from-scratch Levenberg–Marquardt core over ICP (point-to-point /
   point-to-plane) *and* splatreg's flagship **Gaussian-SDF** residual — a smooth signed
   distance field derived directly from the target Gaussians, with a closed-form, audited
   Jacobian — solving the full SE(3) or Sim(3) tangent.

## Headline numbers

| | **splatreg** | reference |
|---|---|---|
| Real-splat merge (103k Gaussians) | Chamfer **10.3 → 2.0 mm (5.1×)**, overlap **0.03 → 0.67 (22×)** | naive concat |
| vs splat competitors (known GT Sim3) | **5.2°** | splatalign 15.3°, GS-Registration 36.3° |
| Sim(3) scale estimation | **native** | none of these do it |
| Official 3DMatch recall | **91.5%** mean | GeoTransformer ~92%, Open3D ~77% |
| Registration speed | **~17 ms** (fast) | Open3D 142 ms |

Full record with reproduce commands: [Benchmarks](benchmarks.md).

## Where next

- [Quickstart](quickstart.md) — install + the core workflows in Python.
- [CLI guide](cli.md) — `splatreg align / merge / info` from the shell.
- [Photometric refinement](photometric.md) — *new in v1.1*: the opt-in stage for poses
  geometry can't see (symmetry / texture-only DoF), with the measured when-and-why table.
- [PLY interop](ply-interop.md) — splatfacto / INRIA / SuperSplat round-trip, and what
  happens to spherical harmonics under a recovered rotation.
- [API reference](api.md) — every public function, autodoc'd.
