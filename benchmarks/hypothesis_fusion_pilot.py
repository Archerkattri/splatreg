#!/usr/bin/env python3
"""Deterministic CPU pilot for ambiguity-aware splat fusion evidence.

The pilot is deliberately small: it validates that a correct asymmetric pose
passes held-out geometry, a translated decoy fails, and a symmetric pair
abstains.  It is not a 3DMatch/ScanNet leaderboard run and carries no learned
or clinical claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from splatreg.hypotheses import decide_fusion


def _T(theta=0.0, t=(0.0, 0.0, 0.0)):
    c, s = math.cos(theta), math.sin(theta)
    out = torch.eye(4, dtype=torch.float64)
    out[:3, :3] = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out[:3, 3] = torch.tensor(t, dtype=torch.float64)
    return out


def _digest() -> str:
    h = hashlib.sha256()
    for path in sorted((ROOT / "splatreg").glob("*.py")):
        h.update(path.relative_to(ROOT).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def run() -> dict:
    source = torch.tensor(
        [[x, y, 0.1 * (x * x + y)] for x, y in (
            (-1.0, -0.6), (-0.5, -0.2), (0.0, 0.1), (0.6, -0.4),
            (1.0, 0.7), (-0.7, 0.9), (0.3, 0.8), (0.9, -0.8),
        )], dtype=torch.float64
    )
    truth = _T(theta=0.35, t=(0.2, -0.1, 0.04))
    target = source @ truth[:3, :3].T + truth[:3, 3]
    good = truth
    decoy = _T(theta=-0.35, t=(-0.15, 0.1, 0.0))
    accepted = decide_fusion(source, target, [good, decoy], max_heldout_p95=0.03)

    symmetric = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=torch.float64,
    )
    abstained = decide_fusion(
        symmetric, symmetric, [_T(), _T(theta=math.pi)], max_heldout_p95=0.03,
        cluster_kwargs={"rotation_tol_deg": 5.0, "translation_tol": 0.01},
    )
    return {
        "protocol": "synthetic_asymmetric_and_symmetric_pose_alternatives",
        "real_dataset": False,
        "source_digest": _digest(),
        "asymmetric": accepted.as_dict(),
        "symmetric": abstained.as_dict(),
    }


def main() -> int:
    result = run()
    out = ROOT / "benchmarks" / "hypothesis_fusion_pilot.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "source_digest": result["source_digest"],
        "asymmetric": {
            "fuse": result["asymmetric"]["fuse"],
            "selected": result["asymmetric"]["selected"],
            "reason": result["asymmetric"]["reason"],
        },
        "symmetric": {
            "fuse": result["symmetric"]["fuse"],
            "reason": result["symmetric"]["reason"],
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
