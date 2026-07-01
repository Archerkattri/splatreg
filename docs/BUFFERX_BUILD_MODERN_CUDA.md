# Building BUFFER-X on a modern stack (CUDA 12.8 / sm_120 / torch 2.11 / numpy 2.x)

BUFFER-X (the `init="bufferx"` zero-shot seed) needs native extensions written for
an older stack. This is the exact, sudo-free recipe that made it build + run on an
RTX 5090 (sm_120), CUDA 12.8, torch 2.11, numpy 2.4 — verified 2026-07-01.

**Location.** The repo must live at `splatreg/third_party_models/BUFFER-X`
(sibling of the `splatreg/` package, i.e. repo-root `third_party_models/`), which
is where `_bufferx_paths()` looks. Cloning it one level deeper silently disables
the path (falls back to `robust`).

**Weights.** `python third_party_models/BUFFER-X/scripts/download_pretrained_models.py
--source hf --repo-id Hyungtae-Lim/BUFFER-X` → `snapshot/threedmatch/{Desc,Pose}/best.pth`.
NOTE: both are **full-model** state_dicts (keys prefixed `Desc.`/`Pose.`); load them
into the whole model, not `model.Desc`/`model.Pose` (loading a submodule matches
nothing under `strict=False` → random weights → garbage seeds). `align_features.py`
does this correctly.

**Python deps (additive; do not change torch/numpy):**
```
pip install easydict tbb tbb-devel        # einops, kornia usually already present
```
- `tbb`/`tbb-devel` provide `tbb/tbb.h` under `<venv>/include` (no `sudo apt install libtbb-dev` needed).

**Eigen (header-only, no sudo):** `git clone --depth 1 https://gitlab.com/libeigen/eigen.git`
and add it to the include path when building the KPConv neighbours wrapper.

**Pure-torch shims** for two CUDA deps that don't build cleanly / are overkill
(drop tiny modules on the path — see `docs/bufferx_shims/`):
- `knn_cuda.KNN` → `torch.cdist` + `topk` (BUFFER-X only uses `KNN(k=1)`).
- `torch_batch_svd.svd` → `torch.linalg.svd` (batched natively; return `(U,S,V)`).

**pointnet2_ops (CUDA, real build):** it hardcodes ancient archs
(`setup.py`: `os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;..."`), which nvcc
12.8 rejects (`unsupported gpu architecture 'compute_37'`). Patch that line to
`"12.0"` and `pip install --no-build-isolation .`.

**KPConv cpp_wrappers (C++ build, numpy-2.x port):** with
`CPLUS_INCLUDE_PATH=<venv>/include:<eigen>` and `LIBRARY_PATH=<venv>/lib`, run
`compile_wrappers.sh`. The `wrapper.cpp` files use numpy-1.x C-API — patch for
numpy 2.x: `NPY_IN_ARRAY` → `NPY_ARRAY_IN_ARRAY`, and cast the `*_array` /
`res_*_obj` `PyObject*` to `PyArrayObject*` where `PyArray_NDIM/DIM/DATA` are used.

After all of the above, `_bufferx_paths()` resolves, `_load_bufferx()` returns the
model, and `init="bufferx"` produces real seeds (validated: 0.75 recall / 1.9°
median on real 3DMatch high-overlap pairs, matching the classical seed; the
advantage is in the low-overlap regime — see the 3dgs-registration project's
`experiments/bufferx_recall.py`).
