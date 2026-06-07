#!/usr/bin/env python
"""6-DoF object-pose mode (v0.2) — recovery + ADD/ADD-S metric tests.

Mirrors the synthetic-recovery protocol (build a known model splat, apply a KNOWN SE(3)/Sim(3) to
get an observation, recover the pose with :func:`splatreg.estimate_object_pose`, score it) but
through the *object-pose* API and the standard FoundationPose/YCB metrics (ADD, ADD-S, AUC). Runs on
CPU so it rides the normal CI suite.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

from splatreg import (
    ObjectPoseEstimator,
    add_auc,
    add_metric,
    adds_metric,
    estimate_object_pose,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXAMPLES_DIR = os.path.join(_REPO_ROOT, "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from _example_utils import (  # noqa: E402
    axis_angle_R,
    make_object_splat,
    rot_angle_deg,
    sim3_matrix,
)

N_POINTS = 350
MAX_ITERS = 30
QUALITY = "low"
DTYPE = torch.float32

ROT_AXIS = [0.3, 0.9, 0.25]
TRANS = [0.05, -0.03, 0.04]

# (transform, rot_deg, scale)
CELLS = [
    ("se3", 30.0, 1.0),
    ("se3", 90.0, 1.0),
    ("sim3", 40.0, 1.2),
]


def _make_pair(transform, rot_deg, s_gt, device):
    model = make_object_splat(N_POINTS, seed=0, device=device, dtype=DTYPE)
    R_gt = axis_angle_R(ROT_AXIS, rot_deg, device=device, dtype=DTYPE)
    t_gt = torch.tensor(TRANS, device=device, dtype=DTYPE)
    M_gt = sim3_matrix(s_gt, R_gt, t_gt)
    obs = make_object_splat.apply_to(model, M_gt)
    return model, obs, M_gt, R_gt


@pytest.mark.parametrize(
    "transform,rot_deg,s_gt", CELLS, ids=[f"{t}-rot{int(r)}-s{s}" for (t, r, s) in CELLS]
)
def test_object_pose_recovery(transform, rot_deg, s_gt, device):
    """A known object pose is recovered to within tolerance via estimate_object_pose."""
    model, obs, M_gt, R_gt = _make_pair(transform, rot_deg, s_gt, device)
    op = estimate_object_pose(
        model, obs, init="global", transform=transform, max_iters=MAX_ITERS, quality=QUALITY
    )
    R_est = op.T_SO[:3, :3] / op.scale
    rot_err = rot_angle_deg(R_est, R_gt)
    assert rot_err < 2.0, f"object-pose rot err {rot_err:.3f} deg too large ({transform} {rot_deg})"
    # ADD should be sub-mm on this clean recovery (model extent ~0.28 m).
    add = add_metric(model, op.T_SO, M_gt)
    adds = adds_metric(model, op.T_SO, M_gt)
    assert add < 2e-3, f"ADD {add*1000:.3f} mm too large"
    # ADD-S (closest-point) is <= ADD up to float32 cdist-vs-norm rounding at this sub-0.1mm scale.
    assert adds <= add + 1e-4, "ADD-S must not meaningfully exceed ADD (closest-point match)"
    if transform == "sim3":
        assert abs(op.scale - s_gt) / s_gt < 0.02, f"scale err {op.scale} vs {s_gt}"
    else:
        assert abs(op.scale - 1.0) < 1e-6


def test_add_auc_monotone_and_bounds():
    """add_auc is in [0,1], 1.0 for all-zero error, and decreases as errors grow."""
    assert add_auc([0.0, 0.0, 0.0]) == pytest.approx(1.0, abs=1e-6)
    assert add_auc([]) == 0.0
    good = add_auc([0.001, 0.002, 0.001])
    bad = add_auc([0.05, 0.08, 0.09])
    assert 0.0 <= bad < good <= 1.0
    # An error exactly at the 0.1 m threshold contributes ~0 AUC.
    assert add_auc([0.1, 0.1]) < 0.05


def test_adds_handles_symmetry(device):
    """ADD-S of a 180° flip of a symmetric (sphere) model is ~0, while ADD is large."""
    # A featureless sphere: rotating it maps it onto itself, so ADD-S ~ 0 but ADD is large.
    import math

    n = 600
    g = torch.Generator(device="cpu").manual_seed(0)
    u = torch.rand(n, generator=g)
    v = torch.rand(n, generator=g)
    phi = 2 * math.pi * u
    costh = 2 * v - 1
    sinth = torch.sqrt((1 - costh**2).clamp_min(0))
    pts = torch.stack([sinth * torch.cos(phi), sinth * torch.sin(phi), costh], dim=1) * 0.1
    pts = pts.to(device)
    from splatreg.core.types import Gaussians

    sphere = Gaussians(
        means=pts,
        quats=torch.tensor([1.0, 0, 0, 0], device=device).repeat(n, 1),
        scales=torch.full((n, 3), 0.01, device=device),
        opacities=torch.ones(n, device=device),
    )
    T_gt = torch.eye(4, device=device)
    R_flip = axis_angle_R([0, 1, 0], 180.0, device=device)
    T_pred = torch.eye(4, device=device)
    T_pred[:3, :3] = R_flip
    add = add_metric(sphere, T_pred, T_gt)
    adds = adds_metric(sphere, T_pred, T_gt)
    assert adds < add, "ADD-S should be much smaller than ADD for a symmetric flip"
    assert adds < 0.02, f"ADD-S of a sphere flip should be ~0, got {adds}"


def test_object_pose_estimator_warmstart(device):
    """ObjectPoseEstimator tracks a small inter-frame motion via the warm-start path."""
    model = make_object_splat(N_POINTS, seed=0, device=device, dtype=DTYPE)
    est = ObjectPoseEstimator(model, transform="se3", init="global", quality=QUALITY)

    # Frame 1: a 25-deg rotated observation (cold init).
    R1 = axis_angle_R(ROT_AXIS, 25.0, device=device, dtype=DTYPE)
    t1 = torch.tensor(TRANS, device=device, dtype=DTYPE)
    M1 = sim3_matrix(1.0, R1, t1)
    op1 = est.estimate(make_object_splat.apply_to(model, M1))
    assert rot_angle_deg(op1.T_SO[:3, :3], R1) < 2.0

    # Frame 2: a few degrees further — warm-started track should stay locked.
    R2 = axis_angle_R(ROT_AXIS, 28.0, device=device, dtype=DTYPE)
    M2 = sim3_matrix(1.0, R2, t1)
    op2 = est.estimate(make_object_splat.apply_to(model, M2))
    assert op2.info.get("mode") == "track", "second frame should use the warm-start track path"
    assert rot_angle_deg(op2.T_SO[:3, :3], R2) < 3.0
