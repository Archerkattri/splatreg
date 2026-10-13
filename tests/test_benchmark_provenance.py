"""Input identity and stage-control checks without real-scene claim inflation."""

import numpy as np
import pytest
import torch

from benchmarks import threedmatch_bench as bench


@pytest.mark.parametrize(
    "arguments",
    [
        ["--matched-controls", "--init", "fast"],
        ["--matched-controls", "--no-basin-sweep"],
        ["--no-basin-sweep", "--init", "robust"],
    ],
)
def test_stage_controls_reject_mismatched_initialization(arguments):
    with pytest.raises(SystemExit):
        bench.main(arguments)


def test_point_digest_tracks_actual_float32_geometry():
    points = np.arange(18).reshape(6, 3)
    assert bench.point_digest(points) == bench.point_digest(points.astype(np.float32))
    assert bench.point_digest(points) != bench.point_digest(points[::-1])
    assert bench.point_digest(points) != bench.point_digest(points + 1)


def test_matched_stage_controls_share_the_robust_seed_and_budget(monkeypatch):
    calls = []
    seed = torch.eye(4, dtype=torch.float64)
    refined = seed.clone()
    refined[0, 3] = 1.
    monkeypatch.setattr(bench, "_cloud_voxel", lambda points: .05)

    def coarse(source, target, voxel, *, rng_seed):
        calls.append((voxel, rng_seed, source.dtype, target.dtype))
        return seed, 4

    def local(source, target, initial, voxel, *, iters):
        assert initial is seed
        assert voxel == .05
        calls.append(iters)
        return refined

    monkeypatch.setattr(bench, "_open3d_fpfh_ransac_seed", coarse)
    monkeypatch.setattr(bench, "_open3d_icp_refine", local)
    coarse_pose, refined_pose, voxel = bench.matched_open3d_controls(
        np.zeros((6, 3)), np.ones((6, 3))
    )
    assert calls == [(.05, 42, torch.float32, torch.float32), 30]
    np.testing.assert_array_equal(coarse_pose, seed.numpy())
    np.testing.assert_array_equal(refined_pose, refined.numpy())
    assert voxel == .05
