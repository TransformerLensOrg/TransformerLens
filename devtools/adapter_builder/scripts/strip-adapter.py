#!/usr/bin/env python3
"""Strip an existing adapter from a checkout for golden-master rebuild tests.

Removes everything the adapter builder would have to recreate for an
architecture — the adapter module, its registrations, its unit tests, and
its model-registry entries — so the build system can be pointed at the
architecture as if it were unsupported, and the result diffed against the
original (`git diff <base-branch> -- <paths>`).

Run from inside the checkout (worktree) to strip. Leaves shared
infrastructure alone (HT-side files, model_type maps, scan configs,
comments) — those are legitimately part of the environment a fresh
adapter author would see.

Usage:
    python3 strip-adapter.py --module neox --adapter-class NeoxArchitectureAdapter \\
        --arch-class GPTNeoXForCausalLM --arch-class NeoXForCausalLM

Example (round 2):
    python3 strip-adapter.py --module codegen --adapter-class CodeGenArchitectureAdapter \\
        --arch-class CodeGenForCausalLM
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    )
    return Path(out.stdout.strip())


def remove_lines(path: Path, predicate, label: str) -> int:
    lines = path.read_text().splitlines(keepends=True)
    kept = [ln for ln in lines if not predicate(ln)]
    removed = len(lines) - len(kept)
    if removed:
        path.write_text("".join(kept))
    print(f"  {path.name}: removed {removed} line(s) ({label})")
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, help="adapter module name, e.g. neox")
    parser.add_argument(
        "--adapter-class", required=True, help="adapter class name, e.g. NeoxArchitectureAdapter"
    )
    parser.add_argument(
        "--arch-class",
        action="append",
        required=True,
        help="HF architecture class(es) registered to this adapter (repeatable)",
    )
    args = parser.parse_args()

    root = repo_root()
    arch_dir = root / "transformer_lens" / "model_bridge" / "supported_architectures"
    factory = root / "transformer_lens" / "factories" / "architecture_adapter_factory.py"
    data_dir = root / "transformer_lens" / "tools" / "model_registry" / "data"

    print(f"Stripping adapter '{args.module}' ({args.adapter_class}) from {root}")

    # 1. Adapter module
    adapter_file = arch_dir / f"{args.module}.py"
    if adapter_file.exists():
        adapter_file.unlink()
        print(f"  deleted {adapter_file.relative_to(root)}")
    else:
        print(f"  WARNING: {adapter_file.relative_to(root)} not found")

    # 2. Package exports
    remove_lines(
        arch_dir / "__init__.py",
        lambda ln: f".{args.module} import" in ln or f'"{args.adapter_class}"' in ln,
        "import + __all__",
    )

    # 3. Factory registration (import line + one dict entry per arch class)
    remove_lines(
        factory,
        lambda ln: args.adapter_class in ln,
        "import + SUPPORTED_ARCHITECTURES entries",
    )

    # 4. Unit tests
    test_dir = root / "tests" / "unit" / "model_bridge" / "supported_architectures"
    for test_file in sorted(test_dir.glob(f"test_{args.module}*.py")):
        test_file.unlink()
        print(f"  deleted {test_file.relative_to(root)}")

    # 5. Registry: supported_models.json (entries + top-level counters)
    sm_path = data_dir / "supported_models.json"
    sm = json.loads(sm_path.read_text())
    before = len(sm["models"])
    removed_models = [m for m in sm["models"] if m["architecture_id"] in args.arch_class]
    sm["models"] = [m for m in sm["models"] if m["architecture_id"] not in args.arch_class]
    removed_archs = {m["architecture_id"] for m in removed_models}
    sm["total_models"] = sm.get("total_models", before) - len(removed_models)
    sm["total_verified"] = sm.get("total_verified", 0) - sum(
        1 for m in removed_models if m["status"] == 1
    )
    sm["total_architectures"] = sm.get("total_architectures", 0) - len(removed_archs)
    sm_path.write_text(json.dumps(sm, indent=2) + "\n")
    print(f"  supported_models.json: removed {len(removed_models)} model entries")

    # 6. Registry: verification_history.json
    vh_path = data_dir / "verification_history.json"
    vh = json.loads(vh_path.read_text())
    n_rec = len(vh["records"])
    vh["records"] = [r for r in vh["records"] if r.get("architecture_id") not in args.arch_class]
    vh_path.write_text(json.dumps(vh, indent=2) + "\n")
    print(f"  verification_history.json: removed {n_rec - len(vh['records'])} records")

    # 7. Sanity: the adapter class must be gone from the package
    leftovers = subprocess.run(
        ["grep", "-rl", args.adapter_class, str(root / "transformer_lens")],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if leftovers:
        print(f"FAIL: '{args.adapter_class}' still referenced in:\n{leftovers}")
        return 1
    print(f"OK: no remaining references to {args.adapter_class} in the package")
    return 0


if __name__ == "__main__":
    sys.exit(main())
