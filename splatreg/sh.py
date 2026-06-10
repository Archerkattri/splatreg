"""Real spherical-harmonic rotation (Wigner-D) for 3DGS colour coefficients.

When a rotation ``R`` is baked into a splat (:func:`splatreg.api.apply_transform`, ``merge``,
the ``align`` CLI), the geometry is easy — means moved, quats composed — but the **colour** is a
function on the view-direction sphere: ``c(d) = Σ_k f_k Y_k(d)`` with the real-SH basis ``Y_k``.
Rotating the splat means the new coefficients ``f'`` must satisfy::

    Σ_k f'_k Y_k(d)  =  Σ_k f_k Y_k(R⁻¹ d)      for every direction d,

i.e. ``f' = D(R) f`` with ``D(R)`` the block-diagonal **Wigner-D matrix in the real basis**
(one ``(2l+1)×(2l+1)`` block per degree ``l``; the DC block is the scalar 1, which is why
DC-only splats never noticed). This module builds ``D(R)`` for any degree via the
Ivanic–Ruedenberg recurrence:

    J. Ivanic, K. Ruedenberg, *Rotation Matrices for Real Spherical Harmonics. Direct
    Determination by Recursion*, J. Phys. Chem. A 100 (1996) 6342, **with the corrections of
    the 1998 erratum** (J. Phys. Chem. A 102 (1998) 9099) applied to the ``u/v/w`` coefficient
    table and the ``V_{m,n}`` functions.

Basis convention
----------------
The recurrence is formulated in the "plain" real-SH basis (no Condon–Shortley phase), whose
``l = 1`` band is ordered ``(y, z, x)`` — there the Wigner block is simply the permuted rotation
matrix ``M¹_{ij} = R_{q(i) q(j)}`` with ``q = (y, z, x)``. The 3DGS / gsplat / plenoxels basis
(the one ``f_dc/f_rest`` PLY coefficients live in) carries an extra sign ``(-1)^{|m|}`` per
coefficient relative to that plain basis (e.g. its ``l = 1`` band is ``(-y, +z, -x)``-
proportional). A diagonal ±1 basis change conjugates the block::

    D_l^{3DGS}[m, n] = (-1)^{|m| + |n|} · M_l[m, n]

:func:`sh_rotation_matrix` returns ``D`` **in the 3DGS convention**, ready to multiply the
``(N, K, 3)`` coefficient tensor splatreg's :class:`~splatreg.core.types.Gaussians` carries
(coefficient-major, channel-last — the layout :mod:`splatreg.io` round-trips with standard PLYs).

Everything is built in float64 for accuracy and cast to the caller's dtype at the end; the
matrices are tiny (K ≤ 16 for degree-3 splats), so cost is negligible next to the transform bake.
"""

from __future__ import annotations

import math

import torch

__all__ = ["sh_rotation_matrix", "rotate_sh"]


def _wigner_band_1(R: list) -> list:
    """The l=1 Wigner block in the plain real-SH ``(y, z, x)`` ordering: a permuted rotation.

    ``R`` is the 3×3 rotation as nested Python floats (row-major, xyz indexing). Index the
    returned 3×3 as ``[m+1][n+1]`` for ``m, n ∈ {-1, 0, 1}``.
    """
    q = (1, 2, 0)  # SH index -1, 0, +1  ->  rotation row/col y, z, x
    return [[R[q[i]][q[j]] for j in range(3)] for i in range(3)]


def _wigner_band_next(band1: list, prev: list, l: int) -> list:
    """Build the degree-``l`` block from the degree-``l-1`` block (Ivanic–Ruedenberg recurrence).

    ``band1`` is the l=1 block (``[m+1][n+1]``), ``prev`` the (2l-1)×(2l-1) degree-``l-1`` block
    (``[m+l-1][n+l-1]``). Returns the (2l+1)×(2l+1) degree-``l`` block (``[m+l][n+l]``). All the
    ``u/v/w`` coefficients and the ``U/V/W`` functions follow the 1996 paper as corrected by the
    1998 erratum (the widely-used corrected tables).
    """

    def P(i: int, a: int, b: int) -> float:
        # R^1_{i,·} contracted with the degree-(l-1) block; the |b| = l columns recurse through
        # the edge columns of `prev`.
        if b == l:
            return band1[i + 1][2] * prev[a + l - 1][2 * l - 2] - band1[i + 1][0] * prev[a + l - 1][0]
        if b == -l:
            return band1[i + 1][2] * prev[a + l - 1][0] + band1[i + 1][0] * prev[a + l - 1][2 * l - 2]
        return band1[i + 1][1] * prev[a + l - 1][b + l - 1]

    def U(m: int, n: int) -> float:
        return P(0, m, n)

    def V(m: int, n: int) -> float:
        if m == 0:
            return P(1, 1, n) + P(-1, -1, n)
        if m > 0:
            if m == 1:  # erratum: the sqrt(1+δ_{m1}) factor lands on the P(1, 0, n) term alone
                return math.sqrt(2.0) * P(1, 0, n)
            return P(1, m - 1, n) - P(-1, -m + 1, n)
        if m == -1:
            return math.sqrt(2.0) * P(-1, 0, n)
        return P(1, m + 1, n) + P(-1, -m - 1, n)

    def W(m: int, n: int) -> float:
        # w_{m,n} is zero for m = 0, so W is never evaluated there.
        if m > 0:
            return P(1, m + 1, n) + P(-1, -m - 1, n)
        return P(1, m - 1, n) - P(-1, -m + 1, n)

    size = 2 * l + 1
    out = [[0.0] * size for _ in range(size)]
    for m in range(-l, l + 1):
        for n in range(-l, l + 1):
            d = 1.0 if m == 0 else 0.0
            denom = float((2 * l) * (2 * l - 1)) if abs(n) == l else float((l + n) * (l - n))
            u = math.sqrt((l + m) * (l - m) / denom)
            v = 0.5 * math.sqrt((1.0 + d) * (l + abs(m) - 1) * (l + abs(m)) / denom) * (1.0 - 2.0 * d)
            w = -0.5 * math.sqrt((l - abs(m) - 1) * (l - abs(m)) / denom) * (1.0 - d)
            val = 0.0
            if u != 0.0:
                val += u * U(m, n)
            if v != 0.0:
                val += v * V(m, n)
            if w != 0.0:
                val += w * W(m, n)
            out[m + l][n + l] = val
    return out


def sh_rotation_matrix(R: torch.Tensor, n_coeffs: int) -> torch.Tensor:
    """Block-diagonal real-SH Wigner-D matrix for rotation ``R``, in the 3DGS basis convention.

    Args:
        R: ``(3, 3)`` pure rotation (orthonormal; de-scale a similarity block first).
        n_coeffs: total SH coefficient count ``K`` including DC — must be a perfect square
            ``(degree+1)²`` (1, 4, 9, 16, ... — the only counts standard 3DGS PLYs produce).

    Returns:
        ``(K, K)`` matrix ``D`` (device/dtype of ``R``) such that ``f' = D @ f`` are the
        coefficients of the rotated colour function: ``Σ f'_k Y_k(d) = Σ f_k Y_k(R⁻¹ d)``
        in the 3DGS/gsplat real-SH basis. The DC block is 1 (rotation-invariant).

    Raises:
        ValueError: if ``R`` is not 3×3 or ``n_coeffs`` is not a perfect square ≥ 1.
    """
    if R.shape != (3, 3):
        raise ValueError(f"sh_rotation_matrix needs a (3, 3) rotation, got {tuple(R.shape)}.")
    if n_coeffs < 1 or int(round(math.isqrt(n_coeffs))) ** 2 != n_coeffs:
        raise ValueError(
            f"n_coeffs must be a perfect square (1, 4, 9, 16, ...) — got {n_coeffs}. "
            "Standard 3DGS SH stacks are complete through their max degree."
        )
    deg = math.isqrt(n_coeffs) - 1

    device, dtype = R.device, R.dtype
    Rf = [[float(R[i, j]) for j in range(3)] for i in range(3)]  # float64 Python scalars

    D = torch.zeros(n_coeffs, n_coeffs, dtype=torch.float64)
    D[0, 0] = 1.0  # l = 0: DC is rotation-invariant
    if deg >= 1:
        band = _wigner_band_1(Rf)
        prev = band
        for l in range(1, deg + 1):
            if l >= 2:
                prev = _wigner_band_next(band, prev, l)
            base = l * l
            for m in range(-l, l + 1):
                sm = -1.0 if (abs(m) % 2) else 1.0
                for n in range(-l, l + 1):
                    sn = -1.0 if (abs(n) % 2) else 1.0
                    # (-1)^{|m|+|n|} conjugation: plain real basis -> 3DGS basis (see module doc).
                    D[base + m + l, base + n + l] = sm * sn * prev[m + l][n + l]
    return D.to(device=device, dtype=dtype)


def rotate_sh(colors: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate an ``(N, K, 3)`` SH coefficient tensor by ``R`` (3DGS convention; see module doc).

    ``colors`` is coefficient-major / channel-last — exactly what :class:`Gaussians` carries and
    :mod:`splatreg.io` round-trips. The DC row passes through unchanged; every higher-degree band
    is multiplied by its Wigner-D block, so the view-dependent lobes turn WITH the splat.

    Args:
        colors: ``(N, K, 3)`` SH coefficients (``K`` a perfect square).
        R: ``(3, 3)`` pure rotation.

    Returns:
        ``(N, K, 3)`` rotated coefficients (new tensor; input untouched).
    """
    if colors.dim() != 3 or colors.shape[-1] != 3:
        raise ValueError(f"rotate_sh expects (N, K, 3) SH coefficients, got {tuple(colors.shape)}.")
    K = int(colors.shape[1])
    if K == 1:
        return colors.clone()  # DC-only: rotation-invariant
    D = sh_rotation_matrix(R.to(dtype=colors.dtype), K).to(device=colors.device)
    return torch.einsum("kj,njc->nkc", D, colors)
