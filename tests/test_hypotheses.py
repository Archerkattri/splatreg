"""CPU tests for ambiguity-aware held-out fusion decisions."""

import math

import pytest
import torch

from splatreg.hypotheses import (
    calibrate_p95_threshold,
    cluster_hypotheses,
    decide_fusion,
    score_hypotheses,
)


def _T(theta=0.0, tx=0.0):
    c, s = math.cos(theta), math.sin(theta)
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[0, 3] = tx
    return T


def test_cluster_keeps_distinct_pose_modes_separate():
    clusters = cluster_hypotheses([_T(), _T(theta=math.pi), _T(tx=0.001)])
    assert len(clusters) == 2
    assert set(clusters[0].members) in ({0, 2}, {1})


def test_heldout_geometry_selects_good_pose_and_rejects_bad_pose():
    source = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.2]], dtype=torch.float64)
    target = source + torch.tensor([0.25, -0.1, 0.05])
    good = _T(tx=0.25)
    good[1, 3] = -0.1
    good[2, 3] = 0.05
    bad = _T(tx=-0.3)
    scores = score_hypotheses(source, target, [good, bad], inlier_tol=0.02)
    assert scores[0].heldout_p95 < scores[1].heldout_p95
    decision = decide_fusion(source, target, [good, bad], max_heldout_p95=0.03)
    assert decision.fuse and decision.selected == 0


def test_symmetric_alternatives_abstain_instead_of_fusing():
    source = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float64)
    decision = decide_fusion(
        source, source, [_T(), _T(theta=math.pi)], max_heldout_p95=0.03,
        cluster_kwargs={"rotation_tol_deg": 5.0, "translation_tol": 0.01},
    )
    assert not decision.fuse
    assert decision.ambiguity
    assert decision.reason == "multiple_pose_clusters_fit_heldout_evidence"


def test_calibration_threshold_is_finite_sample_and_validates_inputs():
    assert calibrate_p95_threshold([0.1, 0.2, 0.3], alpha=0.1) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        calibrate_p95_threshold([])
    with pytest.raises(ValueError):
        calibrate_p95_threshold([0.1], alpha=1.0)

