# 3DMatch red-kitchen CPU benchmark

Date: 2026-09-15  
Hardware: Windows CPU, PyTorch CPU, Open3D 0.19.0  
Dataset: official [3DMatch scene-fragment release](https://3dmatch.cs.princeton.edu/),
`7-scenes-redkitchen` plus its official evaluation `gt.log`.

## Reproducibility

The archives are kept outside the repository at
`C:/Users/krishi/Documents/research/datasets/3dmatch/redkitchen/`:

- 60 real fragment PLYs; SHA-256
  `7CB9A1C9236E6833E910692B1D3F572B970C3FC3493E7641C28F1A45841FA51C`.
- Official evaluation archive; SHA-256
  `FF3EAA243025A0CDF6DD1CA5364A726ACF7C08B36444E49C685E1F014BC4F16E`.

The benchmark now accepts the released flat archive directly with `--gt-log`.
The recorded command was:

```powershell
$env:PYTHONPATH='.'
python -u benchmarks/threedmatch_bench.py `
  --data-root 'C:/Users/krishi/Documents/research/datasets/3dmatch/redkitchen/extracted/7-scenes-redkitchen' `
  --gt-log 'C:/Users/krishi/Documents/research/datasets/3dmatch/redkitchen/evaluation/7-scenes-redkitchen-evaluation/gt.log' `
  --n-pairs 10 --device cpu --init fast --voxel 0.2 --max-points 256 `
  --quality low --max-iters 3 --no-basin-sweep --min-overlap 0.3
```

`--no-basin-sweep` and the point/iteration caps are explicit bounded-CPU
diagnostic settings. Production `register(..., init="fast")` keeps the basin
recoverer enabled by default.

## Result

Ten deterministic, GT-overlapping pairs from one real scene were evaluated;
median overlap was 0.45.

| Method | RR, RMSE < 0.2 m | median RRE | median RTE | median time |
|---|---:|---:|---:|---:|
| splatreg FPFH path | 0/10 (0.0%) | 37.21° | 1.373 m | 88.2 ms |
| Open3D FPFH + RANSAC | 0/10 (0.0%) | 64.35° | 2.623 m | 119.4 ms |

## Interpretation and boundary

Splatreg is better than this matched classical baseline on median pose error
and latency in this bounded slice, but neither method reaches the registration
recall threshold. This is not a full 3DMatch leaderboard result: it covers one
scene, ten sampled pairs, a 256-point cap, low quality, three LM iterations,
and no global basin recovery. Learned SOTA, full-quality CPU, GPU, and the
eight-scene official gate remain unmeasured.
