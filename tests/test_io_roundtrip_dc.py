"""DC-only PLY round-trip: load->save->load must preserve colour values
(previously double-encoded: load returned raw SH-DC in the RGB slot)."""
import torch

from splatreg.io import load_ply, save_ply, rgb_to_sh_dc
from splatreg.core.types import Gaussians


def _mk(n=64):
    g = torch.Generator().manual_seed(0)
    return Gaussians(
        means=torch.randn(n, 3, generator=g),
        quats=torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=1),
        scales=torch.randn(n, 3, generator=g) * 0.1 - 3.0,
        opacities=torch.rand(n, generator=g),
        colors=torch.rand(n, 3, generator=g),  # RGB
        log_scales=True,
    )


def test_dc_only_roundtrip_preserves_rgb(tmp_path):
    g0 = _mk()
    p1, p2 = tmp_path / "a.ply", tmp_path / "b.ply"
    save_ply(g0, p1)
    g1 = load_ply(p1)
    assert g1.colors.dim() == 2, "DC-only load must return RGB (N,3)"
    assert torch.allclose(g1.colors, g0.colors, atol=1e-5), "first round-trip drifted"
    save_ply(g1, p2)
    g2 = load_ply(p2)
    # the old bug double-encoded here: rgb_to_sh_dc applied to already-DC values
    assert torch.allclose(g2.colors, g0.colors, atol=1e-5), "second round-trip drifted (double-encode)"


def test_full_sh_roundtrip_still_bitexact(tmp_path):
    g0 = _mk()
    K = 16
    sh = torch.zeros(g0.means.shape[0], K, 3)
    sh[:, 0, :] = rgb_to_sh_dc(g0.colors)
    sh[:, 1:, :] = torch.randn(g0.means.shape[0], K - 1, 3) * 0.01
    g0 = Gaussians(means=g0.means, quats=g0.quats, scales=g0.scales,
                   opacities=g0.opacities, colors=sh, log_scales=True)
    p = tmp_path / "sh.ply"
    save_ply(g0, p)
    g1 = load_ply(p)
    assert g1.colors.shape == sh.shape
    assert torch.allclose(g1.colors.float(), sh, atol=1e-6)
