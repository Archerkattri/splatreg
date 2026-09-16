# 3DMatch red-kitchen CPU recovery audit

Date: 2026-09-15  
Hardware: Windows CPU, PyTorch CPU, Open3D 0.19.0  
Dataset: official [3DMatch scene-fragment release](https://3dmatch.cs.princeton.edu/),
`7-scenes-redkitchen` plus its official evaluation `gt.log`.

## Protocol correction

The first bounded pilot evaluated the official transform in the wrong endpoint
direction. In `gt.log`, a header `a b n` is followed by the transform that maps
fragment `b` into fragment `a`; the old script treated `a` as the source. On the
first ten log entries this mistake reduced median ground-truth overlap from
`0.849` to `0.436` and raised median nearest-neighbour discrepancy from
`0.108 m` to `0.479 m`.

`benchmarks/threedmatch_bench.py` now names the endpoints explicitly, stores
per-pair results, reports Wilson intervals, supports deterministic dev/test
offsets and has a regression test for the official convention. The earlier
`0/10` numbers are invalid and must not be compared as model performance.

## Data identity

The archives remain outside the repository at
`C:/Users/krishi/Documents/research/datasets/3dmatch/redkitchen/`:

- 60 fragment PLYs; archive SHA-256
  `7CB9A1C9236E6833E910692B1D3F572B970C3FC3493E7641C28F1A45841FA51C`.
- Official evaluation archive; SHA-256
  `FF3EAA243025A0CDF6DD1CA5364A726ACF7C08B36444E49C685E1F014BC4F16E`.

Pairs are shuffled once with seed `0`. The first three pairs are the development
slice; the next seven are held out. Both methods receive the same downsampled,
deterministically capped point clouds. Recall uses the fixed correspondence-RMSE
threshold `< 0.2 m` and a `0.1 m` ground-truth correspondence band.

## Failure localization

The corrected endpoint ordering does not rescue the deliberately constrained
`0.2 m / 256-point / fast / no-basin-sweep` configuration: splatreg and Open3D
both remain `0/10` (95% Wilson interval `0.0–27.8%`). Every splatreg result is
flagged ambiguous with zero confidence. Starting the generic full-overlap LM at
ground truth reaches only `1/10`; it changes rotation by a median `13.69°`.
This identifies two failures rather than one: the coarse fast path has too few
features at that resolution, and the default full-overlap LM is inappropriate
for these partial-overlap room fragments.

The machine-readable diagnostic is
`3dmatch_redkitchen_cpu_corrected_2026-09-15.json`. Oracle initialization is a
diagnostic only and is not a deployable baseline.

## Frozen recovery configuration

Only the three-pair development slice was used to choose the recovery path. The
frozen configuration uses the existing `init="robust"` path at the standard
`0.05 m` voxel scale, a deterministic 4096-point cap and its task-matched
Open3D FPFH/RANSAC seed plus point-to-plane ICP refinement. No result from the
seven-pair slice was used to change the settings.

```powershell
$common = @(
  '--data-root', 'C:/Users/krishi/Documents/research/datasets/3dmatch/redkitchen/extracted/7-scenes-redkitchen',
  '--gt-log', 'C:/Users/krishi/Documents/research/datasets/3dmatch/redkitchen/evaluation/7-scenes-redkitchen-evaluation/gt.log',
  '--device', 'cpu', '--init', 'robust', '--voxel', '0.05', '--max-points', '4096',
  '--quality', 'low', '--max-iters', '3', '--min-overlap', '0.3'
)
python -u benchmarks/threedmatch_bench.py @common --n-pairs 3 --pair-offset 0 `
  --output-json benchmarks/3dmatch_redkitchen_dev_robust_2026-09-15.json
python -u benchmarks/threedmatch_bench.py @common --n-pairs 7 --pair-offset 3 `
  --output-json benchmarks/3dmatch_redkitchen_test_robust_2026-09-15.json
```

## Results

| Slice | Method | RR | 95% Wilson interval | median RRE | median RTE | median time |
|---|---|---:|---:|---:|---:|---:|
| development, n=3 | splatreg robust | 3/3 (100.0%) | 43.9–100.0% | 1.25° | 0.065 m | 637.6 ms |
| development, n=3 | Open3D FPFH/RANSAC | 2/3 (66.7%) | 20.8–93.9% | 7.57° | 0.246 m | 528.1 ms |
| held-out, n=7 | splatreg robust | 5/7 (71.4%) | 35.9–91.8% | 1.69° | 0.053 m | 584.7 ms |
| held-out, n=7 | Open3D FPFH/RANSAC | 5/7 (71.4%) | 35.9–91.8% | 5.19° | 0.152 m | 549.8 ms |

The held-out recall is tied. Splatreg's point-to-plane refinement reduces median
rotation and translation error among the same bounded inputs, at a 6.3% median
latency cost. The descriptive ten-pair aggregate is `8/10` for splatreg and
`7/10` for Open3D, but it mixes development and held-out pairs and is not the
headline estimate.

## Claim boundary

This closes the failed-pilot diagnosis and establishes positive held-out recall
with matched controls and uncertainty on one CPU scene. It is not a new full
3DMatch result, a 3DLoMatch result, or a learned-SOTA claim. The repository's
separately recorded complete official-split results remain the relevant
large-scale evidence; this audit neither reruns nor broadens them.
