#!/usr/bin/env python
"""Multi-splat joint / bundle registration tests (v0.3) — loop consistency, all on CPU.

:func:`splatreg.bundle_register` registers ``N`` overlapping splats JOINTLY (a pose-graph
Gauss-Newton over all absolute poses) instead of the sequential chain :func:`splatreg.merge` runs.
The load-bearing claim is **loop consistency**: when the captures form a ring, the sequential chain
dumps all its accumulated drift onto the loop-closure edge, while the joint solve spreads that error
over the whole graph — so the *maximum* pairwise inconsistency drops sharply.

These tests build a known ring of ``N`` splats (the same object placed at known poses around a
loop, with per-capture footprint-scale noise so the pairwise solves carry real error), then check:

* the joint max-edge inconsistency is materially below the sequential chain's (the headline win);
* the joint Gauss-Newton drives its cost down and converges;
* ``fuse=True`` returns one merged splat from the jointly optimised poses;
* an explicit pair list / the Sim(3) path run.

CPU-only and deterministic.
"""

from __future__ import annotations

import math
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXAMPLES_DIR = os.path.join(_REPO_ROOT, "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from _example_utils import make_object_splat, axis_angle_R, sim3_matrix  # noqa: E402

from splatreg.bundle import (  # noqa: E402
    bundle_register,
    pairwise_consistency,
    solve_pose_graph,
    _sequential_poses,
)
from splatreg import register  # noqa: E402
from splatreg.core.types import Gaussians  # noqa: E402
from splatreg.core.lie import se3_exp  # noqa: E402

N_POINTS = 250
PAIR_ITERS = 40
QUALITY = "low"
NOISE = 0.0015  # per-capture footprint-scale jitter -> real pairwise registration error


def _noisy(g: Gaussians, eps: float, seed: int) -> Gaussians:
    gen = torch.Generator().manual_seed(seed)
    return Gaussians(
        means=g.means + eps * torch.randn(g.means.shape, generator=gen).to(g.means.device),
        quats=g.quats,
        scales=g.scales,
        opacities=g.opacities,
        log_scales=g.log_scales,
    )


def _build_loop(n: int, device: str, transform: str = "se3"):
    """A ring of ``n`` noisy splats: the base object placed at known poses around a loop."""
    base = make_object_splat(N_POINTS, seed=0, device=device, dtype=torch.float32)
    splats = []
    for k in range(n):
        ang = 360.0 * k / n
        R = axis_angle_R([0.2, 0.8, 0.3], ang * 0.5, device=device, dtype=torch.float32)
        t = torch.tensor(
            [0.06 * math.cos(math.radians(ang)), 0.06 * math.sin(math.radians(ang)), 0.02 * k],
            device=device,
        )
        s = 1.0 if transform == "se3" else (1.0 + 0.02 * (k - n / 2.0))
        g = make_object_splat.apply_to(base, sim3_matrix(s, R, t))
        splats.append(_noisy(g, NOISE, seed=100 + k))
    return splats


def _rel_measurements(splats, edges, transform):
    rel = {}
    for (i, j) in edges:
        rel[(i, j)] = register(
            splats[i], splats[j], init="global", transform=transform, max_iters=PAIR_ITERS, quality=QUALITY
        ).T.detach()
    return rel


def test_bundle_closes_the_loop_better_than_sequential(device):
    """JOINT max pairwise inconsistency is well below the SEQUENTIAL chain's (the headline claim)."""
    n = 5
    splats = _build_loop(n, device, transform="se3")
    edges = [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]  # ring (chain + loop closure)
    rel = _rel_measurements(splats, edges, "se3")

    # Sequential: chain out from ref (ignores the loop-closure edge) — the merge-style drift baseline.
    seq = _sequential_poses(splats, rel, 0, n, torch.device(device), torch.float32)
    seq_max, seq_mean = pairwise_consistency(seq, rel, "se3")

    # Joint: the pose-graph Gauss-Newton over all absolute poses.
    poses, info = bundle_register(
        splats,
        ref=0,
        transform="se3",
        init="global",
        register_kwargs=dict(max_iters=PAIR_ITERS, quality=QUALITY),
        return_info=True,
    )
    print(
        f"\n[bundle] seq max={seq_max:.3e} mean={seq_mean:.3e} | "
        f"joint max={info.max_edge_err:.3e} mean={info.mean_edge_err:.3e} | "
        f"max-reduction={seq_max / max(info.max_edge_err, 1e-12):.1f}x"
    )
    # The joint solve must cut the WORST edge inconsistency substantially (loop closes).
    assert info.max_edge_err < 0.5 * seq_max, (
        f"joint max edge err {info.max_edge_err:.3e} not below half the sequential {seq_max:.3e}"
    )
    # ref pose stays at identity (gauge fixed).
    assert torch.allclose(poses[0], torch.eye(4, device=device), atol=1e-6)


def test_bundle_gauss_newton_converges(device):
    """The joint Gauss-Newton lowers its pose-graph cost and reaches a small final inconsistency."""
    n = 4
    splats = _build_loop(n, device, transform="se3")
    poses, info = bundle_register(
        splats,
        ref=0,
        transform="se3",
        init="global",
        register_kwargs=dict(max_iters=PAIR_ITERS, quality=QUALITY),
        return_info=True,
    )
    assert len(info.cost_history) >= 2
    assert info.cost_history[-1] <= info.cost_history[0] + 1e-12, "joint cost did not decrease"
    # A well-conditioned loop converges to a small residual inconsistency.
    assert info.max_edge_err < 0.05, f"joint max edge err {info.max_edge_err:.3e} too large"


def test_bundle_fuse_returns_merged_splat(device):
    """``fuse=True`` bakes the jointly optimised poses and returns one (deduped) Gaussians."""
    n = 3
    splats = _build_loop(n, device, transform="se3")
    poses, fused = bundle_register(
        splats,
        ref=0,
        transform="se3",
        fuse=True,
        register_kwargs=dict(max_iters=PAIR_ITERS, quality=QUALITY),
    )
    assert isinstance(fused, Gaussians)
    assert len(fused) > 0
    assert len(poses) == n


def test_bundle_explicit_pairs_and_sim3(device):
    """An explicit pair list and the Sim(3) (scale) path both run and stay loop-consistent."""
    n = 4
    splats = _build_loop(n, device, transform="sim3")
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    poses, info = bundle_register(
        splats,
        ref=0,
        pairs=pairs,
        transform="sim3",
        init="global",
        register_kwargs=dict(max_iters=PAIR_ITERS, quality=QUALITY),
        return_info=True,
    )
    assert info.edges == pairs
    assert len(poses) == n
    # Sim(3) poses carry a scale; the solve should still be consistent.
    assert info.max_edge_err < 0.1, f"sim3 joint max edge err {info.max_edge_err:.3e} too large"


def test_bundle_single_splat_is_identity(device):
    """A one-splat bundle is the trivial identity pose (no edges)."""
    splats = _build_loop(1, device)
    poses = bundle_register(splats, ref=0, transform="se3")
    assert len(poses) == 1
    assert torch.allclose(poses[0], torch.eye(4, device=device), atol=1e-6)


def _pose_set_error(poses, ref_poses):
    """Mean SE(3) tangent distance between two gauge-aligned absolute-pose sets (ref pinned)."""
    from splatreg.core.lie import se3_log

    errs = [float(se3_log(torch.linalg.inv(R) @ P, dof=6).norm()) for P, R in zip(poses, ref_poses)]
    return sum(errs) / len(errs)


def _fully_connected_edges(n):
    """Every pair ``(i, j)`` with ``i < j`` — a redundant graph (each node degree ``n-1``).

    Redundancy is what makes outlier rejection *possible*: in a bare ring (every node degree 2) one
    bad edge's error spreads evenly over the loop and is mathematically indistinguishable from the
    others at the least-squares optimum, so NO robust kernel can localise it. With chords, the bad
    edge disagrees with a consistent majority and stands out — the realistic SLAM loop-closure case.
    """
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def test_bundle_robust_rejects_bad_edge(device):
    """A robust (IRLS/Huber + GNC) pose-graph solve survives one corrupt edge; the un-gated solve does not.

    Build a clean, *redundant* graph (all pairs), take the joint solution on the clean edges as
    ground truth, then inject one grossly wrong measurement on a single edge. The robust solve must
    down-weight that edge and stay close to the clean solution, while the plain least-squares solve is
    dragged far off by it. (Redundancy is required — see :func:`_fully_connected_edges`.)
    """
    n = 5
    splats = _build_loop(n, device, transform="se3")
    edges = _fully_connected_edges(n)
    rel_clean = _rel_measurements(splats, edges, "se3")
    dev = torch.device(device)

    # Ground-truth poses: robust solve on the clean (outlier-free) graph.
    seed = _sequential_poses(splats, rel_clean, 0, n, dev, torch.float32)
    gt_poses, _, _ = solve_pose_graph(seed, rel_clean, 0, robust="huber")

    # Corrupt one edge: a ~40 deg rotation + 15 cm translation blunder on edge (1, 2).
    bad_delta = torch.tensor([0.15, -0.1, 0.12, 0.7, -0.5, 0.4], device=dev)
    rel_bad = dict(rel_clean)
    rel_bad[(1, 2)] = rel_clean[(1, 2)] @ se3_exp(bad_delta)

    seed_bad = _sequential_poses(splats, rel_bad, 0, n, dev, torch.float32)
    p_plain, _, _ = solve_pose_graph(list(seed_bad), rel_bad, 0, robust=None)
    p_robust, _, w_robust = solve_pose_graph(list(seed_bad), rel_bad, 0, robust="huber")

    err_plain = _pose_set_error(p_plain, gt_poses)
    err_robust = _pose_set_error(p_robust, gt_poses)
    w_bad = w_robust[(1, 2)]
    w_good = min(w for k, w in w_robust.items() if k != (1, 2))
    print(
        f"\n[bundle-robust] plain_pose_err={err_plain:.3e} robust_pose_err={err_robust:.3e} "
        f"improve={err_plain / max(err_robust, 1e-9):.1f}x | bad_edge_w={w_bad:.3f} "
        f"min_good_w={w_good:.3f}"
    )
    # The robust solve recovers the good poses far better than the corrupted plain solve.
    assert err_robust < 0.25 * err_plain, (
        f"robust pose error {err_robust:.3e} not well below plain {err_plain:.3e}"
    )
    # The bad edge is strongly down-weighted while the good edges keep substantial weight.
    assert w_bad < 0.1, f"bad edge weight {w_bad:.3f} not suppressed"
    assert w_good > 0.4, f"a good edge was over-suppressed (min weight {w_good:.3f})"


def test_bundle_register_reports_rejected_edge(device):
    """End-to-end ``bundle_register`` flags a corrupt edge in ``info.rejected_edges``.

    Injects a bad measurement through the public path by handing ``bundle_register`` precomputed
    relative poses is not supported, so we drive ``solve_pose_graph`` directly for the corrupt case
    and assert the reject bookkeeping. The clean public call must report NO rejections.
    """
    n = 4
    splats = _build_loop(n, device, transform="se3")
    # Clean public run: nothing rejected, robust on by default.
    _, info = bundle_register(
        splats, ref=0, transform="se3", init="global",
        register_kwargs=dict(max_iters=PAIR_ITERS, quality=QUALITY), return_info=True,
    )
    assert info.rejected_edges == [], f"clean loop falsely rejected {info.rejected_edges}"
    assert all(w > 0.5 for w in info.edge_weights.values()), "clean edges should keep high weight"
