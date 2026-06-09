# Init modes

Registration is two stages: a **coarse initializer** finds the right basin, then the
Levenberg–Marquardt core polishes the pose within it. `register(..., init=...)` (and
`splatreg align --init ...`) selects the initializer — the single most important knob in the
library. The trade is speed ↔ robustness:

| `init=` | what | when | cost |
|---|---|---|---|
| `"fast"` *(default)* | FPFH descriptors + GPU-batched 3-point RANSAC seed → LM polish | objects / full-overlap captures | **~17 ms** |
| `"robust"` | Open3D FPFH+RANSAC seed (scale-correct, auto-voxelled) → overlap-aware refine | real metre-scale scans | ~100 ms+ |
| `"learned"` | pretrained GeoTransformer seed → the same overlap-aware refine | best accuracy on real scans (91.5% official 3DMatch) | ~104 ms |
| `"global"` | blind super-Fibonacci SO(3) sweep + batched trimmed ICP | unknown, possibly huge rotation; no features needed | ~0.8–1.4 s |
| `"features"` | complete partial-overlap registrar (FPFH → clique-filtered RANSAC → overlap-aware point-to-plane refine + basin-sweep fallback) | the two captures see *different parts* of the object | seconds (deep sweep) |

You can also pass an explicit 4×4 tensor as `init` (e.g. a pose prior from odometry), or
`None` — which resolves to `"fast"`.

## Picking one

- **Merging two captures of an object / small scene** → keep the default `"fast"`.
  It handles full rotations and is two orders of magnitude faster than the blind sweep.
- **Real indoor/outdoor scans (metre scale, sensor noise)** → `"robust"`, or `"learned"`
  for the best accuracy (GeoTransformer-class recall; needs its weights available).
- **You have no idea how the splats are oriented and they look feature-poor** (smooth,
  near-symmetric geometry where descriptors are non-discriminative) → `"global"`.
- **Partial overlap** (one capture saw the left side, the other the right) → `"features"`.

## Partial overlap: the honest contract

`init="features"`, `"robust"`, and `"learned"` are *complete registrars*, not just seeds.
The default residual set assumes **full overlap** — its ICP would drag a good partial-overlap
pose off-target — so with the default residuals these three modes return their own
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
    unrecoverable* — even the true pose doesn't seat cleanly. splatreg returns its best
    feasible guess **flagged** `ambiguous=True` rather than a silently wrong pose
    (verified: 0 silent-wrong across the robustness sweep). At overlap ≤ 40% expect flags;
    `merge` and `Tracker` are designed for high-overlap captures.

## Fallback chain

All string inits are guarded: `"learned"` falls back to `"robust"`, `"fast"` falls back to
`"global"`, and everything falls back to identity (with a logged note) when an optional
dependency or pretrained weight is unavailable. Your call never hard-fails because a model
file is missing.

## Quality policy

Orthogonal to `init`, `quality=` bounds the work the refinement does:

- `"full"` *(default)* — nothing capped, every source anchor.
- `"balanced"` / `"low"` — bounded sample counts + tighter autodiff chunks.
- a float `0..1` — interpolates the caps.
- `"auto"` — detect free GPU/CPU memory and pick the largest sizes that *fit*.

The Sim(3) autodiff Jacobian is always row-chunked, so peak memory stays bounded with no
quality loss. See [`resolve_quality`][splatreg.quality.resolve_quality].
