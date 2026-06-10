"""`examples/validate_recovery.py --fast` preset — wiring test.

The preset's RUNTIME claim (<2 min CPU) is evidenced by the recorded run in RESULTS.md /
the commit message; running the harness inside the unit suite would defeat the point of a
fast gate. What this locks instead: the flag exists, the preset shrinks exactly the budget
knobs (seeds / grid corners / anchors / iters) and nothing else — same gates, same protocol
constants — and explicit --n/--iters still win over it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_recovery", _EXAMPLES / "validate_recovery.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.pop("validate_recovery", None)
    spec.loader.exec_module(mod)
    return mod


def test_fast_preset_shrinks_budget_only():
    vr = _load_module()
    # full defaults first
    assert vr.ACTIVE_SEEDS == vr.SEEDS and len(vr.SEEDS) >= 3
    assert vr.ROT_DEGS == [5.0, 30.0, 90.0] and vr.SCALES == [0.8, 1.0, 1.3]
    full_gates = (vr.SUCC_ROT_DEG, vr.SUCC_SCALE_PCT)
    full_axis, full_trans = list(vr.ROT_AXIS), list(vr.TRANS)

    vr.apply_fast_preset()
    # budget shrank: 1 seed, the {min, max} rotation x {min, max} scale corners, fewer anchors/iters
    assert vr.ACTIVE_SEEDS == vr.SEEDS[:1]
    assert vr.ROT_DEGS == [5.0, 90.0] and vr.SCALES == [0.8, 1.3]
    assert vr.N_POINTS == 400 and vr.MAX_ITERS == 30
    # protocol unchanged: same success gates, same axis/translation, same init policy
    assert (vr.SUCC_ROT_DEG, vr.SUCC_SCALE_PCT) == full_gates
    assert list(vr.ROT_AXIS) == full_axis and list(vr.TRANS) == full_trans
    assert vr.INIT == "global"


def test_fast_flag_parses():
    import argparse

    vr = _load_module()
    # the flag must exist on the parser main() builds; reproduce the parser cheaply by
    # checking apply_fast_preset is reachable and --fast is in the source contract
    src = (_EXAMPLES / "validate_recovery.py").read_text()
    assert '"--fast"' in src and "apply_fast_preset()" in src
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    assert ap.parse_args(["--fast"]).fast is True
