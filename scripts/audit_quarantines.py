"""Audit tests/QUARANTINES.md references against the test tree.

QUARANTINES.md inventories every skip/skipif/xfail marker in tests/ with
``[`path`:line]`` references. Those line numbers rot silently when quarantined
tests move; this script re-resolves every reference and checks that each landed
line is actually a marker (``pytest.mark.skip/skipif/xfail/slow``,
``pytest.importorskip``, ``pytestmark``, ``pytest.param``). Whole-file rows
listed without line numbers are checked by grepping for their module-level
marker instead.

Run:  uv run python scripts/audit_quarantines.py
Exit: 0 with an OK summary when every reference lands on a marker line;
      1 listing the BAD references otherwise.

When a quarantined test moves (edits above it, file renames), update its
QUARANTINES.md line reference in the same change and re-run this script.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1] / "tests"

MARKER = re.compile(
    r"pytest\.mark\.(skip|skipif|xfail|slow)|pytest\.importorskip|pytestmark|pytest\.param"
)

# Line-referenced rows look like [`path`:N,M,...](path) — capture path + line list.
REF = re.compile(r"\[`([^`]+?)`:([0-9][0-9,]*)\]")

# Whole-file rows carry no line numbers in the doc; pin their module-level markers here
# (mirrors the no-line rows in QUARANTINES.md — keep the two in sync).
MODULE_LEVEL_MARKERS = [
    ("acceptance/model_bridge/test_bridge_multigpu.py", "pytestmark = MULTIGPU_MARKS"),
    ("acceptance/model_bridge/test_bridge_multigpu_device_map.py", "pytestmark = MULTIGPU_MARKS"),
    ("mps/test_mps_basic.py", "pytestmark = pytest.mark.skipif("),
    ("mps/test_mps_ssm_eager_scan.py", "pytestmark = pytest.mark.skipif("),
    ("unit/model_bridge/test_vllm_driver.py", 'pytest.importorskip("vllm")'),
    ("unit/test_lit.py", "skipif(not LIT_AVAILABLE"),
    ("acceptance/model_bridge/test_vllm_multigpu.py", 'pytest.importorskip("vllm")'),
    ("acceptance/model_bridge/test_vllm_multigpu_pp.py", 'pytest.importorskip("vllm")'),
]


def main() -> int:
    doc = (TESTS / "QUARANTINES.md").read_text()
    ok: list[str] = []
    bad: list[str] = []

    for match in REF.finditer(doc):
        rel, lines = match.group(1), match.group(2)
        path = TESTS / rel
        if not path.exists():
            bad.append(f"MISSING FILE: {rel}")
            continue
        content = path.read_text().splitlines()
        for n in (int(x) for x in lines.split(",")):
            if n > len(content):
                bad.append(f"{rel}:{n} -> beyond EOF ({len(content)} lines)")
                continue
            text = content[n - 1].strip()
            if MARKER.search(text):
                ok.append(f"{rel}:{n} -> {text[:90]}")
            else:
                bad.append(f"{rel}:{n} -> NOT A MARKER LINE: {text[:90]}")

    for rel, needle in MODULE_LEVEL_MARKERS:
        path = TESTS / rel
        if not path.exists():
            bad.append(f"MISSING FILE: {rel}")
            continue
        count = path.read_text().count(needle)
        if count:
            ok.append(f"{rel} :: {needle} (x{count})")
        else:
            bad.append(f"{rel} :: MODULE-LEVEL MARKER NOT FOUND: {needle}")

    print(f"OK: {len(ok)}  BAD: {len(bad)}\n")
    for line in ok:
        print("OK ", line)
    for line in bad:
        print("BAD", line)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
