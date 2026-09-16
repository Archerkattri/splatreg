from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks import threedmatch_bench as bench


def test_official_gt_log_maps_second_fragment_into_first(tmp_path: Path, monkeypatch) -> None:
    """The official header is target/source, not source/target."""
    for index in (0, 1):
        (tmp_path / f"cloud_bin_{index}.ply").touch()

    transform = np.eye(4)
    transform[0, 3] = 1.0
    rows = ["0 1 2", *(" ".join(map(str, row)) for row in transform)]
    log_path = tmp_path / "gt.log"
    log_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    source = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    target = source + np.array([1.0, 0.0, 0.0])

    def fake_read_points(path: str) -> np.ndarray:
        return target if Path(path).stem == "cloud_bin_0" else source

    monkeypatch.setattr(bench, "read_points", fake_read_points)
    pairs = bench.gather_pairs(
        str(tmp_path),
        voxel=0.01,
        overlap_thresh=0.05,
        min_overlap=0.99,
        n_pairs=1,
        seed=0,
        gt_log=str(log_path),
    )

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["source"] == 1
    assert pair["target"] == 0
    np.testing.assert_allclose(pair["T_gt"], transform)
    np.testing.assert_allclose(
        pair["src"] @ transform[:3, :3].T + transform[:3, 3],
        pair["tgt"],
    )
    assert pair["overlap"] == 1.0


def test_wilson_interval_contains_observed_recall() -> None:
    low, high = bench.wilson_interval(4, 10)
    assert 0.0 < low < 0.4 < high < 1.0
    assert bench.wilson_interval(0, 0) == (0.0, 1.0)
