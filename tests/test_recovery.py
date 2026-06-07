#!/usr/bin/env python
"""Synthetic Sim(3)/SE(3) RECOVERY tests — the load-bearing accuracy claim, in pytest.

This promotes the out-of-band ``examples/validate_recovery.py`` harness into the unit suite so
splatreg's central correctness claim — *take a realistic object splat, apply a KNOWN Sim(3)/SE(3),
recover it with* ``register`` *to within tolerance* — is regression-gated on every push/PR.

It is the same protocol as the example (build A -> apply M_gt -> recover with ``register`` ->
score rotation / translation / scale error), trimmed to run fast on a CPU GitHub runner: a small
anchor count, a few iterations, and a handful of representative grid cells (small/large rotation,
up/down scale, SE(3) and Sim(3)). Each cell asserts the recovered pose is within tolerance.

Two P5 regression gates ride alongside the per-cell checks:

* ``test_register_is_deterministic`` — ``register`` called twice on identical input must return
  the *exact* same transform (``max|dT| < 1e-9``; it is observed to be 0.0). Guards the no-hidden-
  RNG / fixed-seed-sweep contract the global init relies on.
* ``test_no_worst_case_blowup`` — across all recovery cells, no rotation error may exceed ``3x``
  the per-cell success threshold. A single catastrophic cell (basin escape) would trip this even
  if the median stayed healthy.

Everything runs on CPU (``device`` fixture defaults to CPU; GPU is opt-in via
``SPLATREG_TEST_DEVICE=cuda``) and is deterministic per cell.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

from splatreg import register

# The recovery geometry + SO(3) metrics live next to the example harness (examples/_example_utils.py),
# not in the package, so the example and this test share one definition of "a realistic object splat".
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXAMPLES_DIR = os.path.join(_REPO_ROOT, "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from _example_utils import (  # noqa: E402  (path injected above)
    axis_angle_R,
    make_object_splat,
    rot_angle_deg,
    sim3_matrix,
)

# ---------------------------------------------------------------- fast CPU-CI sizing
# Smaller than the example (N=1400, iters=60, quality="full"): N=300 + 30 iters + quality="low"
# recovers every cell to << tolerance in a few seconds each on CPU (verified), keeping the whole
# file well under a couple of minutes on a GitHub runner.
N_POINTS = 300
MAX_ITERS = 30
QUALITY = "low"
DTYPE = torch.float32

ROT_AXIS = [0.3, 0.9, 0.25]
TRANS = [0.03, -0.02, 0.025]  # object units (~30 mm), applied after s*R

# Success gates match the example: <2 deg rotation, <2% scale (the soft-SDF zero level-set and the
# RMS-approximate coarse scale put the achievable floor at basin, not machine, precision).
SUCC_ROT_DEG = 2.0
SUCC_SCALE_PCT = 2.0

# Representative grid cells: (transform, rot_deg, scale). Small + large rotation, scale up/down,
# both rigid (se3) and similarity (sim3). Kept small so CI stays fast while still spanning regimes.
RECOVERY_CELLS = [
    ("se3", 30.0, 1.0),
    ("se3", 90.0, 1.0),
    ("sim3", 5.0, 1.0),
    ("sim3", 30.0, 1.3),
    ("sim3", 90.0, 0.8),
]


def _recover(transform: str, rot_deg: float, s_gt: float, device: str) -> dict:
    """Build A, apply the known M_gt to get B, recover with default-residual ``register``, score."""
    A = make_object_splat(N_POINTS, seed=0, device=device, dtype=DTYPE)
    R_gt = axis_angle_R(ROT_AXIS, rot_deg, device=device, dtype=DTYPE)
    t_gt = torch.tensor(TRANS, device=device, dtype=DTYPE)
    M_gt = sim3_matrix(s_gt, R_gt, t_gt)
    B = make_object_splat.apply_to(A, M_gt)

    res = register(B, A, init="global", transform=transform, max_iters=MAX_ITERS, quality=QUALITY)

    s_est = res.scale
    R_est = res.T[:3, :3] / s_est
    rot_err = rot_angle_deg(R_est, R_gt)
    trans_err_mm = 1000.0 * float((res.T[:3, 3] - M_gt[:3, 3]).norm())
    scale_err_pct = 100.0 * abs(s_est - s_gt) / s_gt
    return {
        "transform": transform,
        "rot_deg": rot_deg,
        "scale_gt": s_gt,
        "rot_err": rot_err,
        "trans_err_mm": trans_err_mm,
        "scale_err_pct": scale_err_pct,
        "scale_est": s_est,
    }


@pytest.mark.parametrize(
    "transform,rot_deg,s_gt",
    RECOVERY_CELLS,
    ids=[f"{t}-rot{int(r)}-s{s}" for (t, r, s) in RECOVERY_CELLS],
)
def test_recovery_cell_within_tolerance(transform, rot_deg, s_gt, device):
    """A known Sim(3)/SE(3) applied to a realistic splat is recovered to within tolerance."""
    m = _recover(transform, rot_deg, s_gt, device)
    assert m["rot_err"] < SUCC_ROT_DEG, (
        f"rotation error {m['rot_err']:.4f} deg >= {SUCC_ROT_DEG} for {m['transform']} "
        f"rot={rot_deg} scale={s_gt}"
    )
    if transform == "sim3":
        assert (
            m["scale_err_pct"] < SUCC_SCALE_PCT
        ), f"scale error {m['scale_err_pct']:.4f}% >= {SUCC_SCALE_PCT} for rot={rot_deg} scale={s_gt}"
    else:
        # SE(3): scale DoF is fixed at 1, so the recovered scale must stay (near-)unit.
        assert abs(m["scale_est"] - 1.0) < 1e-6, f"se3 recovered non-unit scale {m['scale_est']}"
    # Translation is in object units (~30 mm); a recovered pose this good keeps it sub-mm-ish.
    assert m["trans_err_mm"] < 5.0, f"translation error {m['trans_err_mm']:.3f} mm too large"


def test_register_is_deterministic(device):
    """P5 gate: ``register`` twice on identical input -> bit-identical transform (max|dT| < 1e-9).

    The global init is a *fixed* super-Fibonacci sweep with no hidden RNG, so repeated calls must
    agree exactly. Observed max|dT| is 0.0; the 1e-9 bound leaves headroom for any benign reduction
    re-ordering without admitting a real nondeterminism regression.
    """
    A = make_object_splat(N_POINTS, seed=0, device=device, dtype=DTYPE)
    R_gt = axis_angle_R(ROT_AXIS, 30.0, device=device, dtype=DTYPE)
    t_gt = torch.tensor(TRANS, device=device, dtype=DTYPE)
    M_gt = sim3_matrix(1.3, R_gt, t_gt)
    B = make_object_splat.apply_to(A, M_gt)

    r1 = register(B, A, init="global", transform="sim3", max_iters=MAX_ITERS, quality=QUALITY)
    r2 = register(B, A, init="global", transform="sim3", max_iters=MAX_ITERS, quality=QUALITY)

    max_dT = float((r1.T - r2.T).abs().max())
    assert max_dT < 1e-9, f"register is non-deterministic: max|dT| = {max_dT}"


def test_no_worst_case_blowup(device):
    """P5 gate: across all recovery cells, no rotation error exceeds 3x the success threshold.

    A healthy median can hide a single basin-escape cell; this catches that catastrophic tail
    directly (worst rot_err must stay < ``3 * SUCC_ROT_DEG``).
    """
    worst_limit = 3.0 * SUCC_ROT_DEG
    worst_rot = 0.0
    worst_cell = None
    for transform, rot_deg, s_gt in RECOVERY_CELLS:
        m = _recover(transform, rot_deg, s_gt, device)
        if m["rot_err"] > worst_rot:
            worst_rot, worst_cell = m["rot_err"], (transform, rot_deg, s_gt)
    assert worst_rot < worst_limit, (
        f"worst-case rotation error {worst_rot:.4f} deg >= {worst_limit} (3x threshold) "
        f"at cell {worst_cell}"
    )
