#!/usr/bin/env python
"""End-to-end tests for the ``splatreg`` command-line interface (:mod:`splatreg.cli`).

The CLI is a thin wrapper over :func:`splatreg.api.register` / :func:`splatreg.api.merge` /
:mod:`splatreg.io`, so these tests focus on the wrapper's own contract:

* ``align`` round-trips PLY -> register -> apply -> PLY and the output actually lands on the
  target (Chamfer drop, checked against the known synthetic ground truth);
* ``merge`` fuses two offset copies into one splat with the overlap deduped (count between
  ``max`` and ``sum`` of the inputs);
* ``info`` reports count / SH layout;
* the error paths (missing file, single-input merge, bad quality) exit non-zero with a
  helpful message instead of a traceback;
* ``--version`` and ``--help`` work and the console entry point resolves.

CPU-only, deterministic, small clouds + ``--quality low`` so the whole file stays fast.
``main(argv)`` is invoked in-process (no subprocess) so failures give real tracebacks and
coverage sees the code.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXAMPLES_DIR = os.path.join(_REPO_ROOT, "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from _example_utils import make_object_splat, axis_angle_R, chamfer_mm, sim3_matrix  # noqa: E402

from splatreg.cli import build_parser, main  # noqa: E402
from splatreg.io import load_ply, save_ply  # noqa: E402

N_POINTS = 400


# --------------------------------------------------------------------------------------
# Fixtures: a synthetic target/source PLY pair with a KNOWN SE(3) offset
# --------------------------------------------------------------------------------------
@pytest.fixture
def ply_pair(tmp_path):
    """Write ``(target.ply, source.ply)`` where source = known SE(3) of target."""
    A = make_object_splat(N_POINTS, seed=0)
    R = axis_angle_R([0.2, 1.0, 0.1], 12.0)
    M = sim3_matrix(1.0, R, torch.tensor([0.05, -0.03, 0.02]))
    B = make_object_splat.apply_to(A, M)
    target, source = tmp_path / "target.ply", tmp_path / "source.ply"
    save_ply(A, target)
    save_ply(B, source)
    return str(target), str(source)


# --------------------------------------------------------------------------------------
# align
# --------------------------------------------------------------------------------------
def test_align_end_to_end(ply_pair, tmp_path, capsys):
    target, source = ply_pair
    out = str(tmp_path / "aligned.ply")
    rc = main(["align", target, source, "-o", out, "--device", "cpu", "--quality", "low"])
    assert rc == 0
    assert os.path.exists(out)

    t, s, a = load_ply(target), load_ply(source), load_ply(out)
    assert len(a) == len(s)
    before = chamfer_mm(s.means, t.means)
    after = chamfer_mm(a.means, t.means)
    assert after < before * 0.2, f"align did not land: {before:.2f} -> {after:.2f} mm"

    printed = capsys.readouterr().out
    assert "T (4x4" in printed
    assert "scale" in printed
    assert "rmse" in printed


def test_align_sim3_and_max_iters(ply_pair, tmp_path):
    """The sim3 / --max-iters path runs and writes a same-size splat."""
    target, source = ply_pair
    out = str(tmp_path / "aligned_sim3.ply")
    rc = main(
        [
            "align",
            target,
            source,
            "-o",
            out,
            "--device",
            "cpu",
            "--quality",
            "low",
            "--transform",
            "sim3",
            "--max-iters",
            "10",
        ]
    )
    assert rc == 0
    assert len(load_ply(out)) == N_POINTS


# --------------------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------------------
def test_merge_end_to_end(ply_pair, tmp_path, capsys):
    target, source = ply_pair
    out = str(tmp_path / "fused.ply")
    # init=fast (not the slow blind sweep) keeps this CPU test quick; the offset is small.
    rc = main(
        [
            "merge",
            target,
            source,
            "-o",
            out,
            "--device",
            "cpu",
            "--quality",
            "low",
            "--init",
            "fast",
        ]
    )
    assert rc == 0
    fused = load_ply(out)
    # Two registered copies of the same object: dedupe must collapse the overlap —
    # strictly fewer than the naive cat, at least as many as one input.
    assert N_POINTS <= len(fused) < 2 * N_POINTS
    assert "fused" in capsys.readouterr().out


def test_merge_no_dedupe_is_plain_cat_count(ply_pair, tmp_path):
    target, source = ply_pair
    out = str(tmp_path / "cat.ply")
    rc = main(
        [
            "merge",
            target,
            source,
            "-o",
            out,
            "--no-dedupe",
            "--device",
            "cpu",
            "--quality",
            "low",
            "--init",
            "fast",
        ]
    )
    assert rc == 0
    assert len(load_ply(out)) == 2 * N_POINTS


# --------------------------------------------------------------------------------------
# info
# --------------------------------------------------------------------------------------
def test_info(ply_pair, capsys):
    target, _ = ply_pair
    rc = main(["info", target])
    assert rc == 0
    printed = capsys.readouterr().out
    assert f"gaussians : {N_POINTS}" in printed
    assert "SH degree 0" in printed  # the synthetic splat is DC-only
    assert "bounds" in printed


# --------------------------------------------------------------------------------------
# error paths + plumbing
# --------------------------------------------------------------------------------------
def test_align_missing_file_exits_nonzero(tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["align", str(tmp_path / "missing.ply"), str(tmp_path / "also.ply"), "-o", "x.ply"])
    assert e.value.code != 0
    assert "not found" in str(e.value.code)


def test_merge_single_input_exits_nonzero(ply_pair):
    target, _ = ply_pair
    with pytest.raises(SystemExit) as e:
        main(["merge", target, "-o", "x.ply"])
    assert "at least two" in str(e.value.code)


def test_bad_quality_rejected(ply_pair):
    target, source = ply_pair
    with pytest.raises(SystemExit):
        main(["align", target, source, "-o", "x.ply", "--quality", "ultra"])
    with pytest.raises(SystemExit):
        main(["align", target, source, "-o", "x.ply", "--quality", "1.5"])


def test_version_and_help(capsys):
    import splatreg

    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert splatreg.__version__ in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for sub in ("align", "merge", "info"):
        assert sub in out
    assert build_parser().prog == "splatreg"


def test_console_script_registered():
    """`[project.scripts] splatreg` resolves to splatreg.cli:main."""
    from importlib.metadata import entry_points

    eps = entry_points()
    scripts = (
        eps.select(group="console_scripts", name="splatreg")
        if hasattr(eps, "select")
        else [e for e in eps.get("console_scripts", []) if e.name == "splatreg"]
    )
    matches = list(scripts)
    assert matches, "console script 'splatreg' not registered (pip install -e . to refresh)"
    assert matches[0].value == "splatreg.cli:main"
