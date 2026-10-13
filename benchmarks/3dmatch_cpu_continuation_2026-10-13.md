# CPU recovery continuation — 2026-10-13

The recovery runner now records fragment-file hashes, float32 registration-input
hashes, the ground-truth source-to-target matrix, canonical source hashes and
runtime versions. Its Open3D coarse control resets its RNG to 42 independently
of splatreg's robust initialization. `--seed` continues to control pair selection.

`--matched-controls` exposes the robust path's exact shared Open3D coarse seed
and raw point-to-plane refinement using the same float32 geometry, automatically
derived voxel scale, seed 42 and 30 refinement iterations. These are stage
diagnostics, not independent competing algorithms. The robust method additionally
retains its coarse pose when the overlap residual worsens. The older `open3d`
row remains a coarse-only control at the explicitly supplied voxel scale.

The current checkout has no real fragment data at either the runner's default
`data/3dmatch/3dmatch/test` or the historical
`C:/Users/krishi/Documents/research/datasets/3dmatch/redkitchen` path. No fresh
real-scene recall, timing or oracle-initialized refinement result was obtained.
The September corrected/dev/held-out JSON files are preserved historical
evidence. The historical held-out result is 5/7 for both methods, not the invalid
original direction-reversed 0/10 comparison. One scene and seven held-out pairs
cannot establish a full multi-scene or 3DLoMatch result.

Focused reproduction (Python 3.12, torch 2.13.0+cpu, Open3D 0.19.0):

```powershell
$env:OMP_NUM_THREADS = '2'
python -m pytest -q tests/test_threedmatch_bench.py tests/test_benchmark_provenance.py -p no:dash
```

For a future permitted real-input rerun, reuse the configuration and pair slices
in `3dmatch_redkitchen_cpu_2026-09-15.md`, add `--matched-controls`, and choose a
new `--output-json` path. Do not overwrite historical artifacts or tune on the
seven held-out pairs. A changed implementation is a new evaluation, not a
retroactive change to the historical reported numbers.
