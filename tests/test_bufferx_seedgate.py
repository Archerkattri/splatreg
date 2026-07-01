"""BUFFER-X zero-shot seed (``init="bufferx"``) + Decision-PCR-style seed gate — CPU unit tests.

Neither feature needs a GPU on the tested path:

* BUFFER-X (ICCV 2025, MIT-SPARK/BUFFER-X) is an OPTIONAL, lazily-loaded backend whose CUDA
  extensions + pretrained weights are absent on a CPU box, so ``init="bufferx"`` here exercises the
  GRACEFUL FALLBACK to the classical ``"robust"`` seed (the same contract ``init="learned"`` has
  when GeoTransformer is absent).  What is asserted: the mode registers, the paths resolver reports
  the backend absent, the aligner falls back cleanly and still recovers a synthetic known transform,
  and the API surface / init dispatch accept ``"bufferx"``.

* The seed gate (``_seed_gate_score``, a training-free stand-in for Decision PCR's confidence head,
  arXiv 2507.14965) is tested directly on synthetic known-transform pairs: it scores a CORRECT seed
  high (never rejected) and a planted DECOY seed low (rejected), and the ``register(init="learned",
  seed_gate=True)`` plumbing runs end-to-end.
"""

from __future__ import annotations

import math

import pytest
import torch

from splatreg import register
from splatreg.core.types import Gaussians
from splatreg.align_features import (
    _bufferx_paths,
    _bufferx_seed,
    _seed_gate_score,
    bufferx_feature_align,
)

DT = torch.float32


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_T(deg: float, axis, t) -> torch.Tensor:
    ax = torch.tensor(axis, dtype=torch.float64)
    ax = ax / ax.norm()
    a = math.radians(deg)
    K = torch.tensor([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], dtype=torch.float64)
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = torch.eye(3, dtype=torch.float64) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)
    T[:3, 3] = torch.tensor(t, dtype=torch.float64)
    return T


_T_TRUE = _make_T(40.0, [0.3, 1.0, 0.2], [0.1, -0.05, 0.2])


def _rot_err_deg(Ta: torch.Tensor, Tb: torch.Tensor) -> float:
    Ra, Rb = Ta[:3, :3].double(), Tb[:3, :3].double()
    Ra = Ra / Ra.det().abs().clamp_min(1e-12) ** (1.0 / 3.0)
    Rb = Rb / Rb.det().abs().clamp_min(1e-12) ** (1.0 / 3.0)
    c = float(((Ra @ Rb.T).trace() - 1.0) * 0.5)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _splat_from(means: torch.Tensor) -> Gaussians:
    n = means.shape[0]
    q = torch.zeros(n, 4, dtype=DT)
    q[:, 0] = 1.0
    return Gaussians(
        means=means.to(DT),
        quats=q,
        scales=torch.full((n, 3), 0.005, dtype=DT),
        opacities=torch.ones(n, dtype=DT),
    )


def _known_pair(seed: int = 7):
    """A full-overlap (target, source) splat pair related by ``_T_TRUE`` (source→target)."""
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(400, 3, generator=g, dtype=torch.float64) * 0.3
    means[:, 2] *= 0.5
    target = _splat_from(means.to(DT))
    src_means = ((means - _T_TRUE[:3, 3]) @ _T_TRUE[:3, :3]).to(DT)  # T_true maps source→target
    source = _splat_from(src_means)
    return target, source


# ---------------------------------------------------------------------------
# Task 1 — BUFFER-X init mode (fallback path on CPU)
# ---------------------------------------------------------------------------


def test_bufferx_paths_absent_on_cpu_box():
    """No built CUDA-ext + downloaded weights → the resolver reports the backend absent (None)."""
    assert _bufferx_paths() is None


def test_bufferx_seed_returns_none_when_backend_absent():
    """The lazy seed returns None (never raises) when BUFFER-X is unavailable → caller falls back."""
    tgt, src = _known_pair()
    out = _bufferx_seed(src.means.to(torch.float32), tgt.means.to(torch.float32), torch.device("cpu"))
    assert out is None


def test_bufferx_align_falls_back_cleanly_and_recovers():
    """bufferx_feature_align falls back to the classical robust seed and still recovers the pose."""
    tgt, src = _known_pair()
    T, info = bufferx_feature_align(tgt, src, transform="se3")
    # API surface: fallback flagged honestly.
    assert info["used_bufferx"] is False
    assert info["seed"] == "robust-fallback"
    # The robust fallback still recovers the 40 deg synthetic offset.
    assert _rot_err_deg(T, _T_TRUE) < 3.0


def test_register_init_bufferx_end_to_end():
    """register(init='bufferx') runs end-to-end and recovers a synthetic known transform."""
    tgt, src = _known_pair()
    result = register(tgt, src, init="bufferx", transform="se3")
    assert _rot_err_deg(result.T, _T_TRUE) < 3.0
    assert (result.T[:3, 3] - _T_TRUE[:3, 3].to(DT)).norm() < 0.03


def test_bufferx_invalid_transform_raises():
    tgt, src = _known_pair()
    with pytest.raises(ValueError):
        bufferx_feature_align(tgt, src, transform="affine")


def test_unknown_init_still_rejected():
    """The init dispatch grew a mode but still rejects garbage strings with the updated message."""
    tgt, src = _known_pair()
    with pytest.raises(ValueError, match="bufferx"):
        register(tgt, src, init="not-a-mode")


# ---------------------------------------------------------------------------
# Task 2 — Decision-PCR-style seed gate
# ---------------------------------------------------------------------------


def test_seed_gate_accepts_a_good_seed():
    """The CORRECT seed scores high (high inlier ratio + SC² consistency) → never rejected."""
    tgt, src = _known_pair()
    gs = _seed_gate_score(
        src.means.to(torch.float32), tgt.means.to(torch.float32), _T_TRUE, inlier_tol=0.03
    )
    assert gs["n_corr"] >= 3
    assert gs["inlier_ratio"] > 0.8
    assert gs["confidence"] > 0.10  # comfortably above the default gate threshold


def test_seed_gate_rejects_a_decoy_seed():
    """A planted DECOY seed (a wrong rotation) scatters the source off the target → rejected."""
    tgt, src = _known_pair()
    decoy = _make_T(150.0, [0.1, 0.2, 1.0], [0.5, 0.5, -0.5])  # wrong pose
    gs = _seed_gate_score(
        src.means.to(torch.float32), tgt.means.to(torch.float32), decoy, inlier_tol=0.03
    )
    assert gs["inlier_ratio"] < 0.2
    assert gs["confidence"] < 0.10  # below the default gate threshold → would be reseeded


def test_seed_gate_good_beats_decoy():
    """Separation is decisive: the good seed's confidence is far above the decoy's."""
    tgt, src = _known_pair()
    sf, tf = src.means.to(torch.float32), tgt.means.to(torch.float32)
    good = _seed_gate_score(sf, tf, _T_TRUE, inlier_tol=0.03)["confidence"]
    decoy = _seed_gate_score(
        sf, tf, _make_T(150.0, [0.1, 0.2, 1.0], [0.5, 0.5, -0.5]), inlier_tol=0.03
    )["confidence"]
    assert good > decoy + 0.05


def test_register_learned_seed_gate_plumbing_runs():
    """register(init='learned', seed_gate=True) runs end-to-end (GeoTransformer absent → robust
    fallback) and still recovers the synthetic transform; the flag is accepted, no crash."""
    tgt, src = _known_pair()
    result = register(tgt, src, init="learned", seed_gate=True, transform="se3")
    assert _rot_err_deg(result.T, _T_TRUE) < 3.0
