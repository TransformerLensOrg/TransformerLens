#!/usr/bin/env python3
"""Format .adapter-workspace/timeline.jsonl entries for human-readable display.

Reads JSONL entries from stdin and writes one formatted line per entry to
stdout. Malformed lines are annotated rather than dropped.

Typical usage:
    tail -1 timeline.jsonl | format-timeline.py          # last event (status)
    tail -f  timeline.jsonl | format-timeline.py          # live stream (logs)

Each entry is expected to look like the structure written by
agents/hooks/timeline-capture.sh:
    {"ts": "...", "event": "...", "tool": "...", "tool_input": {...}, ...}

Kept as a standalone script (rather than embedded inside bash -c) so the
Python/bash boundary is debuggable and unit-testable.
"""
from __future__ import annotations

import json
import sys
from typing import Any


def format_entry(entry: dict[str, Any]) -> str:
    """Return a one-line human-readable summary of a single timeline entry."""
    ts = entry.get("ts", "?")
    # Extract HH:MM:SS from ISO 8601 timestamp; fall back to whatever we have.
    ts_short = ts[11:19] if isinstance(ts, str) and len(ts) >= 19 else str(ts)

    event = entry.get("event", "?") or "?"
    tool = entry.get("tool") or ""
    tool_input = entry.get("tool_input") or {}

    # Prefer the agent-authored description when present; it's already
    # human-readable ("Run mypy", "Read file X", etc.).
    description = None
    if isinstance(tool_input, dict):
        description = tool_input.get("description")

    if description:
        label = str(description)
    elif tool:
        if tool == "Bash" and isinstance(tool_input, dict):
            cmd = (tool_input.get("command") or "").strip().split("\n")[0]
            label = f"{tool}: {cmd[:80]}"
        elif isinstance(tool_input, dict) and tool_input.get("file_path"):
            label = f"{tool}: {tool_input['file_path']}"
        else:
            label = tool
    else:
        label = event  # Fall back to the raw event name (SessionStart, etc.)

    if len(label) > 100:
        label = label[:97] + "..."

    return f"{ts_short}  {event:14}  {label}"


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            print(format_entry(json.loads(line)), flush=True)
        except Exception:
            print(f"(malformed) {line}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
