# ScanNet-GSReg benchmark (community-run)

`benchmarks/scannet_gsreg_bench.py` implements the **GaussReg (ECCV 2024)**
splat-to-splat registration protocol on the 82-scene ScanNet-GSReg test split.
It lives on this `scannet-bench` branch (not `main`) because the dataset is not
readily downloadable, so we cannot run it ourselves and keep `main` free of a
dataset dependency we cannot satisfy.

**If you already have the GaussReg ScanNet-GSReg data, we would love a result.**

## Data layout expected

```
<root>/ScanNet-GSReg/
  test/<scene>/{A,B}/output/point_cloud/iteration_10000/point_cloud.ply
  test/test_transformations.npz        # per-scene ref/src Sim(3)
```
This is exactly GaussReg's released test layout.

## Run

```bash
pip install splatreg          # or: pip install -e . from this checkout
CUDA_VISIBLE_DEVICES=0 SPLATREG_DEVICE=cuda python benchmarks/scannet_gsreg_bench.py \
    --data /path/to/ScanNet-GSReg \
    --init learned --transform sim3 --refine photometric
```

The harness mirrors GaussReg's `compute_registration_error_w_scale` 1:1 and
reports **RRE / RTE / RSE / success-rate / wall-time** so the numbers drop
straight into their Table comparison.

## Share results

Please open an issue (or comment on the pinned ScanNet-GSReg discussion) with the
printed summary table and your GPU. We will add confirmed numbers to `RESULTS.md`
with attribution.
