"""Real-SH Wigner-D rotation (`splatreg.sh`) — math + bake-in wiring tests.

The math is verified WITHOUT any renderer, against an INDEPENDENT real-SH basis evaluator
hand-coded here from the 3DGS/plenoxels polynomial table (the basis ``f_dc``/``f_rest``
coefficients live in). The defining property: for the rotated coefficients ``f' = D(R) f``,

    Σ f'_k Y_k(d)  ==  Σ f_k Y_k(R⁻¹ d)        for every direction d,

i.e. the view-dependent colour function turns WITH the splat. Closed-form degree-1, identity,
orthogonality, and composition ``D(R1 R2) = D(R1) D(R2)`` are checked separately, then the
wiring into ``apply_transform`` / ``merge`` / PLY round-trip.
"""

from __future__ import annotations

import math

import pytest
import torch

from splatreg.api import _apply_transform_to_gaussians, apply_transform
from splatreg.core.types import Gaussians
from splatreg.io import load_ply, save_ply
from splatreg.sh import rotate_sh, sh_rotation_matrix

# ---------------------------------------------------------------------------- helpers

# 3DGS / gsplat / plenoxels real-SH constants (INRIA reference implementation).
_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def sh_basis_deg3(d: torch.Tensor) -> torch.Tensor:
    """Independent 3DGS real-SH basis evaluator: unit dirs ``(M, 3)`` -> ``(M, 16)``.

    Hand-coded from the published plenoxels/INRIA polynomial table — NOT from splatreg.sh —
    so the property test below is a genuine cross-check, not the library against itself.
    """
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    return torch.stack(
        [
            torch.full_like(x, _C0),
            -_C1 * y,
            _C1 * z,
            -_C1 * x,
            _C2[0] * xy,
            _C2[1] * yz,
            _C2[2] * (2 * zz - xx - yy),
            _C2[3] * xz,
            _C2[4] * (xx - yy),
            _C3[0] * y * (3 * xx - yy),
            _C3[1] * xy * z,
            _C3[2] * y * (4 * zz - xx - yy),
            _C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
            _C3[4] * x * (4 * zz - xx - yy),
            _C3[5] * z * (xx - yy),
            _C3[6] * x * (xx - 3 * yy),
        ],
        dim=1,
    )


def random_rotation(gen: torch.Generator) -> torch.Tensor:
    """Uniform-ish random rotation via QR of a Gaussian matrix (det fixed to +1)."""
    A = torch.randn(3, 3, generator=gen, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def random_dirs(n: int, gen: torch.Generator) -> torch.Tensor:
    d = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    return d / d.norm(dim=1, keepdim=True)


def make_sh_splat(n: int, K: int, seed: int) -> Gaussians:
    g = torch.Generator().manual_seed(seed)
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=1)
    return Gaussians(
        means=torch.randn(n, 3, generator=g) * 0.1,
        quats=quats,
        scales=torch.randn(n, 3, generator=g) * 0.1 - 3.0,
        opacities=torch.rand(n, generator=g),
        colors=torch.randn(n, K, 3, generator=g) * 0.3,
        log_scales=True,
    )


# ---------------------------------------------------------------------------- (a) defining property


def test_rotated_coeffs_evaluate_as_inverse_rotated_direction():
    """f' = D(R) f  must satisfy  Σ f'_k Y_k(d) == Σ f_k Y_k(R⁻¹ d)  (deg 3, tol 1e-5)."""
    gen = torch.Generator().manual_seed(7)
    for _ in range(10):
        R = random_rotation(gen)
        f = torch.randn(16, generator=gen, dtype=torch.float64)
        D = sh_rotation_matrix(R, 16)
        f_rot = D @ f
        d = random_dirs(64, gen)
        lhs = sh_basis_deg3(d) @ f_rot
        # row-vector form of R⁻¹ d (= Rᵀ d) is d @ R
        rhs = sh_basis_deg3(d @ R) @ f
        assert float((lhs - rhs).abs().max()) < 1e-5


def test_property_holds_per_degree_block():
    """Same property at K = 4 (deg 1) and K = 9 (deg 2) — partial stacks rotate correctly too."""
    gen = torch.Generator().manual_seed(11)
    for K in (4, 9):
        R = random_rotation(gen)
        f16 = torch.zeros(16, dtype=torch.float64)
        f16[:K] = torch.randn(K, generator=gen, dtype=torch.float64)
        D = sh_rotation_matrix(R, K)
        f_rot = torch.zeros(16, dtype=torch.float64)
        f_rot[:K] = D @ f16[:K]
        d = random_dirs(64, gen)
        lhs = sh_basis_deg3(d) @ f_rot
        rhs = sh_basis_deg3(d @ R) @ f16
        assert float((lhs - rhs).abs().max()) < 1e-5


# ---------------------------------------------------------------------------- (b) degree-1 closed form


def test_degree1_block_is_signed_permutation_of_R():
    """The l=1 block has the known closed form: signed (y,z,x) permutation of R's entries.

    In the 3DGS basis (l=1 functions ∝ (-y, +z, -x)) the block is S P(R) S with
    P(R)[i][j] = R[q(i)][q(j)], q = (y,z,x) and S = diag(-1, +1, -1).
    """
    gen = torch.Generator().manual_seed(3)
    for _ in range(5):
        R = random_rotation(gen)
        D = sh_rotation_matrix(R, 4)
        q = (1, 2, 0)
        S = (-1.0, 1.0, -1.0)
        expected = torch.tensor(
            [[S[i] * S[j] * float(R[q[i], q[j]]) for j in range(3)] for i in range(3)],
            dtype=torch.float64,
        )
        assert torch.allclose(D[1:4, 1:4], expected, atol=1e-12)
        # DC row/col: invariant and uncoupled
        assert float(D[0, 0]) == pytest.approx(1.0)
        assert float(D[0, 1:].abs().max()) == 0.0 and float(D[1:, 0].abs().max()) == 0.0


# ---------------------------------------------------------------------------- (c) identity


def test_identity_rotation_gives_identity_matrix():
    D = sh_rotation_matrix(torch.eye(3, dtype=torch.float64), 16)
    assert torch.allclose(D, torch.eye(16, dtype=torch.float64), atol=1e-12)


# ---------------------------------------------------------------------------- (d) composition + orthogonality


def test_composition_and_orthogonality():
    gen = torch.Generator().manual_seed(13)
    for _ in range(5):
        R1, R2 = random_rotation(gen), random_rotation(gen)
        D1 = sh_rotation_matrix(R1, 16)
        D2 = sh_rotation_matrix(R2, 16)
        D12 = sh_rotation_matrix(R1 @ R2, 16)
        assert float((D12 - D1 @ D2).abs().max()) < 1e-10  # D is a representation
        eye = torch.eye(16, dtype=torch.float64)
        assert float((D1 @ D1.T - eye).abs().max()) < 1e-10  # real-basis D is orthogonal


# ---------------------------------------------------------------------------- validation


def test_invalid_inputs_raise():
    R = torch.eye(3)
    with pytest.raises(ValueError, match="perfect square"):
        sh_rotation_matrix(R, 7)
    with pytest.raises(ValueError, match="rotation"):
        sh_rotation_matrix(torch.eye(4), 4)
    with pytest.raises(ValueError, match="N, K, 3"):
        rotate_sh(torch.zeros(5, 3), R)


def test_rotate_sh_dc_only_passthrough():
    colors = torch.randn(8, 1, 3)
    gen = torch.Generator().manual_seed(1)
    out = rotate_sh(colors, random_rotation(gen).to(torch.float32))
    assert torch.equal(out, colors) and out is not colors


# ---------------------------------------------------------------------------- bake-in wiring


def _se3(R: torch.Tensor, t) -> torch.Tensor:
    T = torch.eye(4)
    T[:3, :3] = R.to(torch.float32)
    T[:3, 3] = torch.as_tensor(t)
    return T


def test_apply_transform_rotates_f_rest_dc_invariant():
    gen = torch.Generator().manual_seed(5)
    R = random_rotation(gen)
    g = make_sh_splat(32, 16, seed=0)
    out = apply_transform(g, _se3(R, [0.1, -0.2, 0.3]))
    # DC band: untouched (rotation-invariant)
    assert torch.allclose(out.colors[:, 0, :], g.colors[:, 0, :], atol=1e-6)
    # Higher bands: exactly the Wigner-D rotation (and visibly different from a ride-along)
    expected = rotate_sh(g.colors, R.to(torch.float32))
    assert torch.allclose(out.colors, expected, atol=1e-6)
    assert float((out.colors[:, 1:, :] - g.colors[:, 1:, :]).abs().max()) > 0.01


def test_apply_transform_translation_only_leaves_sh_unchanged():
    g = make_sh_splat(16, 9, seed=2)
    out = apply_transform(g, _se3(torch.eye(3), [0.5, 0.0, -0.1]))
    assert torch.allclose(out.colors, g.colors, atol=1e-6)


def test_apply_transform_rgb_colors_unchanged():
    g = make_sh_splat(16, 16, seed=3)
    rgb = Gaussians(
        means=g.means, quats=g.quats, scales=g.scales, opacities=g.opacities,
        colors=torch.rand(16, 3), log_scales=True,
    )
    gen = torch.Generator().manual_seed(9)
    out = apply_transform(rgb, _se3(random_rotation(gen), [0.0, 0.0, 0.0]))
    assert torch.allclose(out.colors, rgb.colors, atol=1e-7)


def test_apply_transform_sim3_uses_descaled_rotation():
    """Under Sim(3) the colour rotation must come from the PURE rotation (scale ∉ colour)."""
    gen = torch.Generator().manual_seed(21)
    R = random_rotation(gen)
    s = 1.4
    T = torch.eye(4)
    T[:3, :3] = (s * R).to(torch.float32)
    g = make_sh_splat(16, 16, seed=4)
    out = _apply_transform_to_gaussians(g, T, scale=s)
    expected = rotate_sh(g.colors, R.to(torch.float32))
    assert torch.allclose(out.colors, expected, atol=1e-5)


def test_apply_transform_view_function_property_end_to_end():
    """End-to-end: the baked splat's per-anchor colour FUNCTION equals the original at R⁻¹d."""
    gen = torch.Generator().manual_seed(17)
    R = random_rotation(gen)
    g = make_sh_splat(8, 16, seed=6)
    out = apply_transform(g, _se3(R, [0.0, 0.1, 0.0]))
    d = random_dirs(32, gen).to(torch.float32)
    B_d = sh_basis_deg3(d.to(torch.float64)).to(torch.float32)  # (M, 16)
    B_rinv = sh_basis_deg3((d.to(torch.float64) @ R)).to(torch.float32)
    for c in range(3):
        lhs = out.colors[:, :, c] @ B_d.T  # (N, M) rotated splat evaluated at d
        rhs = g.colors[:, :, c] @ B_rinv.T  # original evaluated at R⁻¹ d
        assert float((lhs - rhs).abs().max()) < 1e-5


def test_rotated_sh_roundtrips_through_ply(tmp_path):
    """apply_transform -> save_ply -> load_ply preserves the rotated f_rest exactly."""
    gen = torch.Generator().manual_seed(23)
    g = make_sh_splat(24, 16, seed=8)
    out = apply_transform(g, _se3(random_rotation(gen), [0.05, 0.0, 0.0]))
    p = tmp_path / "rot.ply"
    save_ply(out, p)
    re = load_ply(p)
    assert re.colors.shape == out.colors.shape == (24, 16, 3)
    assert torch.allclose(re.colors, out.colors, atol=1e-6)
