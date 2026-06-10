"""PLY and gsplat interop for :class:`splatreg.core.types.Gaussians`.

This module is the I/O boundary of splatreg. It does two jobs:

1. **PLY round-trip**, :func:`load_ply` / :func:`save_ply` read and write the *standard*
   3D Gaussian Splatting PLY (the INRIA ``graphdeco`` / gsplat layout):

       x y z  f_dc_0..2  f_rest_0..M  opacity  scale_0..2  rot_0..3

   These files store **raw** parameters: ``opacity`` is the pre-sigmoid logit,
   ``scale_*`` are pre-exp log-scales, ``rot_*`` is a (usually un-normalised) ``wxyz``
   quaternion, and the colour is spherical-harmonics (DC = ``f_dc``, higher orders =
   ``f_rest``). Loaded ``Gaussians`` therefore carry ``log_scales=True`` and SH colours.

2. **gsplat bridge**, :func:`from_gsplat` / :func:`to_gsplat` convert to and from the tensor
   bundle gsplat's rasteriser consumes (``means, quats, scales, opacities, colors``), so a
   splatreg ``Gaussians`` drops straight into ``gsplat.rasterization(**to_gsplat(g))``.

Spherical-harmonics layout
--------------------------
``Gaussians.colors`` for the SH case is ``(N, K, 3)``, *coefficient-major, channel-last*;
a 2-D ``(N, 3)`` value is always RGB (DC-only files are converted to RGB on load)
(``K`` SH coefficients, each an RGB triple), matching gsplat's internal ``sh0``/``shN``
tensors. The standard PLY stores SH **channel-major** (all coefficients of R, then G, then B),
i.e. gsplat's ``features.transpose(0, 2, 1).reshape(N, -1)``. The conversions below apply that
transpose at the PLY boundary so a splat written by gsplat reloads bit-for-bit and vice-versa.

Dependency note
---------------
Reading/writing uses a small **hand-rolled binary-PLY codec** (numpy structured arrays) so the
package keeps its zero-extra-dependency promise (numpy + torch only, already required). If the
optional :mod:`plyfile` package is installed it is used for parsing instead (more permissive with
exotic/ASCII headers); writing always uses the built-in fast binary path. No new pyproject
dependency is required; ``plyfile`` is an *optional* robustness upgrade only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from .core.types import Gaussians

__all__ = ["load_ply", "save_ply", "from_gsplat", "to_gsplat"]

# Zeroth-order real SH basis constant: DC = (RGB - 0.5) / C0  ⇔  RGB = DC * C0 + 0.5.
_SH_C0 = 0.28209479177387814

_PathLike = Union[str, Path]


# --------------------------------------------------------------------------------------
# Spherical-harmonics <-> RGB helpers
# --------------------------------------------------------------------------------------
def sh_dc_to_rgb(dc: torch.Tensor) -> torch.Tensor:
    """Decode the DC SH coefficient ``(..., 3)`` to linear RGB in (roughly) ``[0, 1]``."""
    return dc * _SH_C0 + 0.5


def rgb_to_sh_dc(rgb: torch.Tensor) -> torch.Tensor:
    """Encode linear RGB ``(..., 3)`` into the DC SH coefficient."""
    return (rgb - 0.5) / _SH_C0


# --------------------------------------------------------------------------------------
# Low-level binary-PLY reader/writer (numpy structured arrays, no third-party dep)
# --------------------------------------------------------------------------------------
def _parse_ply_header(f) -> tuple[str, int, list[tuple[str, str]]]:
    """Parse a PLY header from an open binary file positioned at byte 0.

    Returns ``(fmt, count, fields)`` where ``fmt`` is one of
    ``{"binary_little_endian", "binary_big_endian", "ascii"}``, ``count`` is the vertex count,
    and ``fields`` is the ordered ``[(name, ply_type), ...]`` of the (single) ``vertex`` element.
    Only the ``vertex`` element is parsed; any trailing elements are ignored.
    """
    magic = f.readline().strip()
    if magic != b"ply":
        raise ValueError("Not a PLY file (missing 'ply' magic line).")

    fmt: Optional[str] = None
    count: Optional[int] = None
    fields: list[tuple[str, str]] = []
    in_vertex = False
    seen_other_element = False

    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected end of file while reading PLY header.")
        tokens = line.split()
        if not tokens:
            continue
        key = tokens[0]
        if key == b"format":
            fmt = tokens[1].decode("ascii")
        elif key == b"comment":
            continue
        elif key == b"element":
            name = tokens[1]
            if name == b"vertex":
                in_vertex = True
                seen_other_element = False
                count = int(tokens[2])
            else:
                in_vertex = False
                seen_other_element = True
        elif key == b"property":
            if seen_other_element:
                continue  # property of a non-vertex element; ignore
            if not in_vertex:
                continue
            if tokens[1] == b"list":
                raise ValueError("List properties are not supported on the vertex element.")
            ply_type = tokens[1].decode("ascii")
            pname = tokens[2].decode("ascii")
            fields.append((pname, ply_type))
        elif key == b"end_header":
            break

    if fmt is None or count is None:
        raise ValueError("Malformed PLY header (missing format or vertex element).")
    return fmt, count, fields


# PLY scalar type name -> numpy base dtype char.
_PLY_TO_NP = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _read_ply_vertex(path: _PathLike) -> dict[str, np.ndarray]:
    """Read the vertex element of a PLY into ``{property_name: (N,) float64 array}``.

    Uses :mod:`plyfile` when available (handles ASCII and unusual headers); otherwise falls back
    to a built-in binary-PLY reader. Every column is returned as float64 for uniform downstream
    indexing, callers cast to the precision they need.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PLY file not found: {path}")

    try:
        from plyfile import PlyData  # optional, more permissive parser
    except ImportError:
        PlyData = None

    if PlyData is not None:
        ply = PlyData.read(str(path))
        vertex = ply["vertex"]
        return {p.name: np.asarray(vertex[p.name], dtype=np.float64) for p in vertex.properties}

    # Built-in path: parse header, then read the binary block as a structured array.
    with open(path, "rb") as f:
        fmt, count, fields = _parse_ply_header(f)
        if fmt == "ascii":
            raise ValueError(
                "ASCII PLY parsing requires the optional 'plyfile' package "
                "(`pip install plyfile`). 3DGS exporters write binary PLY."
            )
        byteorder = "<" if fmt == "binary_little_endian" else ">"
        np_fields = [(name, byteorder + _PLY_TO_NP[t]) for name, t in fields]
        dtype = np.dtype(np_fields)
        data = np.frombuffer(f.read(count * dtype.itemsize), dtype=dtype, count=count)
    return {name: data[name].astype(np.float64) for name, _ in fields}


def _write_ply_vertex(path: _PathLike, columns: "list[tuple[str, np.ndarray]]") -> None:
    """Write an ordered list of ``(name, (N,) float32 array)`` as a binary-LE vertex-only PLY."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(columns[0][1])
    dtype = np.dtype([(name, "<f4") for name, _ in columns])
    arr = np.empty(n, dtype=dtype)
    for name, col in columns:
        arr[name] = np.ascontiguousarray(col, dtype=np.float32)

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    header += "".join(f"property float {name}\n" for name, _ in columns)
    header += "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        arr.tofile(f)


# --------------------------------------------------------------------------------------
# Public PLY API
# --------------------------------------------------------------------------------------
def load_ply(
    path: _PathLike,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> Gaussians:
    """Load a standard 3D Gaussian Splatting ``.ply`` into a :class:`Gaussians`.

    Recognises the canonical INRIA/gsplat layout (``x y z``, ``f_dc_0..2``, ``f_rest_*``,
    ``opacity``, ``scale_0..2``, ``rot_0..3``). Stored values are raw, the returned
    ``Gaussians`` has ``log_scales=True``, raw (pre-sigmoid) ``opacities``, ``wxyz`` ``quats``,
    and ``colors`` as SH coefficients shaped ``(N, K, 3)`` (or RGB ``(N, 3)`` if only DC is present,
    i.e. ``K == 1``, kept 2-D for convenience).

    Args:
        path: Path to the ``.ply`` file.
        device: Target device for the tensors (default: CPU).
        dtype: Floating dtype for the tensors (default: ``float32``).

    Returns:
        Gaussians: with ``log_scales=True`` and SH colours.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the required 3DGS properties are missing.
    """
    cols = _read_ply_vertex(path)

    required = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    missing = [k for k in required if k not in cols]
    if missing:
        raise ValueError(
            f"PLY is missing required 3DGS properties {missing}; "
            f"found {sorted(cols)}. Is this a 3D Gaussian Splatting export?"
        )

    means = np.stack([cols["x"], cols["y"], cols["z"]], axis=1)
    scales = np.stack([cols["scale_0"], cols["scale_1"], cols["scale_2"]], axis=1)
    quats = np.stack([cols["rot_0"], cols["rot_1"], cols["rot_2"], cols["rot_3"]], axis=1)
    opacities = cols["opacity"]
    n = means.shape[0]

    # --- Colour / SH ---------------------------------------------------------------
    # DC coefficients (f_dc_0/1/2). Required for a valid 3DGS export; if absent we fall back
    # to a neutral grey so geometry-only PLYs still load.
    if all(k in cols for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        dc = np.stack([cols["f_dc_0"], cols["f_dc_1"], cols["f_dc_2"]], axis=1)  # (N, 3)
    else:
        dc = np.zeros((n, 3), dtype=np.float64)  # SH DC 0 -> mid-grey after decode

    rest_keys = sorted(
        (k for k in cols if k.startswith("f_rest_")),
        key=lambda s: int(s.split("_")[-1]),
    )
    if rest_keys:
        # PLY stores f_rest channel-major: [R coeff0..R coeffM, G coeff0.., B coeff0..].
        n_rest = len(rest_keys)
        if n_rest % 3 != 0:
            raise ValueError(f"f_rest count {n_rest} is not divisible by 3 (RGB channels).")
        k_rest = n_rest // 3  # higher-order coefficients per channel
        rest = np.stack([cols[k] for k in rest_keys], axis=1)  # (N, 3*k_rest) chan-major
        rest = rest.reshape(n, 3, k_rest).transpose(0, 2, 1)  # (N, k_rest, 3) coeff-major
        dc_k = dc.reshape(n, 1, 3)  # (N, 1, 3)
        colors_np = np.concatenate([dc_k, rest], axis=1)  # (N, K, 3), K=1+k_rest
    else:
        # DC-only file: convert the SH-DC coefficients to RGB so the returned
        # 2-D ``colors`` honours the repo-wide "(N, 3) = RGB" convention that
        # ``save_ply`` (and the render paths) assume. Returning raw DC here
        # made a load->save round-trip double-encode DC-only files.
        colors_np = sh_dc_to_rgb(torch.from_numpy(dc)).numpy()  # (N, 3) RGB

    t_kw = dict(device=device, dtype=dtype)
    return Gaussians(
        means=torch.as_tensor(means, **t_kw),
        quats=torch.as_tensor(quats, **t_kw),
        scales=torch.as_tensor(scales, **t_kw),
        opacities=torch.as_tensor(opacities, **t_kw),
        colors=torch.as_tensor(colors_np, **t_kw),
        log_scales=True,
    )


def save_ply(gaussians: Gaussians, path: _PathLike) -> None:
    """Write a :class:`Gaussians` to a standard 3D Gaussian Splatting binary ``.ply``.

    The output is the canonical INRIA/gsplat layout consumable by SuperSplat, the antimatter15
    viewer, gsplat, nerfstudio, etc. Parameters are stored **raw**: ``scale_*`` are log-scales
    (the input is log-transformed if ``log_scales`` is False), ``opacity`` is the stored logit,
    and the colour is written as SH (``f_dc`` + ``f_rest``). RGB colours are encoded to a DC-only
    SH; an ``(N, K, 3)`` SH input is written with full ``f_rest``.

    Args:
        gaussians: The splat to serialise.
        path: Destination ``.ply`` path (parent dirs are created).
    """
    g = gaussians
    n = len(g)

    means = g.means.detach().cpu().numpy()
    quats = g.quats.detach().cpu().numpy()

    # Scales -> log-scales for the PLY (3DGS stores pre-exp values).
    scales = g.scales.detach().cpu().numpy()
    if not g.log_scales:
        scales = np.log(np.clip(scales, 1e-12, None))

    opac = g.opacities.detach().cpu().numpy().reshape(n)

    columns: list[tuple[str, np.ndarray]] = [
        ("x", means[:, 0]),
        ("y", means[:, 1]),
        ("z", means[:, 2]),
    ]

    # --- Colour / SH ---------------------------------------------------------------
    if g.colors is None:
        dc = np.zeros((n, 3), dtype=np.float32)  # neutral grey
        rest_flat = None
    else:
        colors = g.colors.detach().cpu().numpy()
        if colors.ndim == 2:  # (N, 3) RGB -> DC-only SH
            dc = rgb_to_sh_dc(torch.from_numpy(colors)).numpy()
            rest_flat = None
        elif colors.ndim == 3:  # (N, K, 3) SH coeffs, coeff-major
            dc = colors[:, 0, :]  # (N, 3)
            if colors.shape[1] > 1:
                rest = colors[:, 1:, :]  # (N, k_rest, 3) coeff-major
                # -> channel-major flat for the PLY: [R coeffs, G coeffs, B coeffs]
                rest_flat = rest.transpose(0, 2, 1).reshape(n, -1)
            else:
                rest_flat = None
        else:
            raise ValueError(f"Gaussians.colors must be (N,3) or (N,K,3); got {colors.shape}.")

    columns += [("f_dc_0", dc[:, 0]), ("f_dc_1", dc[:, 1]), ("f_dc_2", dc[:, 2])]
    if rest_flat is not None:
        for i in range(rest_flat.shape[1]):
            columns.append((f"f_rest_{i}", rest_flat[:, i]))

    columns.append(("opacity", opac))
    columns += [("scale_0", scales[:, 0]), ("scale_1", scales[:, 1]), ("scale_2", scales[:, 2])]
    columns += [
        ("rot_0", quats[:, 0]),
        ("rot_1", quats[:, 1]),
        ("rot_2", quats[:, 2]),
        ("rot_3", quats[:, 3]),
    ]

    _write_ply_vertex(path, columns)


# --------------------------------------------------------------------------------------
# gsplat bridge
# --------------------------------------------------------------------------------------
def from_gsplat(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    log_scales: bool = False,
) -> Gaussians:
    """Wrap gsplat-style rasteriser tensors as a :class:`Gaussians` (no copy of valid inputs).

    This is the entry point for "I already have my splat as gsplat tensors." It performs only
    light normalisation (shape/contiguity), leaving values untouched.

    Args:
        means: ``(N, 3)`` centres.
        quats: ``(N, 4)`` rotations, ``wxyz`` (gsplat's convention).
        scales: ``(N, 3)``. Linear by default; pass ``log_scales=True`` if these are log-scales.
        opacities: ``(N,)`` or ``(N, 1)``. Whatever activation state your gsplat call expects,
            splatreg treats this as opaque and passes it back unchanged in :func:`to_gsplat`.
        colors: optional ``(N, 3)`` RGB or ``(N, K, 3)`` SH coefficients.
        log_scales: set True if ``scales`` are already log-transformed.

    Returns:
        Gaussians.
    """
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"means must be (N, 3); got {tuple(means.shape)}.")
    if quats.shape[-1] != 4:
        raise ValueError(f"quats must be (N, 4) wxyz; got {tuple(quats.shape)}.")
    if scales.shape[-1] != 3:
        raise ValueError(f"scales must be (N, 3); got {tuple(scales.shape)}.")
    return Gaussians(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        log_scales=log_scales,
    )


def to_gsplat(g: Gaussians) -> dict:
    """Unpack a :class:`Gaussians` into the keyword bundle gsplat's rasteriser consumes.

    Scales are returned **linear** (``exp`` applied if the ``Gaussians`` holds log-scales), since
    ``gsplat.rasterization`` expects linear scales. ``opacities`` and ``colors`` are passed
    through unchanged. The result is intended for ``gsplat.rasterization(..., **to_gsplat(g))``::

        from gsplat import rasterization
        out = rasterization(viewmats=..., Ks=..., width=W, height=H, **to_gsplat(g))

    Args:
        g: The splat to unpack.

    Returns:
        dict: ``{"means", "quats", "scales", "opacities", "colors"}`` (``colors`` omitted if
        ``g.colors is None``).
    """
    scales = g.scales
    if g.log_scales:
        scales = torch.exp(scales)
    out = {
        "means": g.means,
        "quats": g.quats,
        "scales": scales,
        "opacities": g.opacities,
    }
    if g.colors is not None:
        out["colors"] = g.colors
    return out
