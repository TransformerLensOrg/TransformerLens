"""Guard: no module's doctests may construct HookedTransformer.

HookedTransformer was removed in 4.0. No surviving module's docstring examples
may build it — doing so would fail the docstring test tier — so this guard
fails the instant such an example is (re)introduced. With the class gone there
are no exempt files left; every ``.py`` under the package is checked.
"""

import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[2] / "transformer_lens"

# The Hooked* implementation files are deleted, so nothing is exempt anymore.
EXEMPT: set[str] = set()
EXEMPT_DIRS: set[str] = set()

# A doctest example line (>>> or ... continuation) that references the class at
# all — construction, import, or isinstance: any of them turns the docstring
# tier red once the class is gone. The lookahead excludes surviving classes
# whose names extend it (HookedTransformerConfig, ...KeyValueCache).
_CONSTRUCTION = re.compile(r"^\s*(?:>>>|\.\.\.).*HookedTransformer(?![A-Za-z_])")


def _doctest_sources(path: Path):
    """Yield (lineno, line) for every doctest example line in the file.

    Grep-level by design: example lines are the only place `>>> ` appears, so a
    line scan finds them without importing the module or parsing docstrings
    (DocTestParser rejects pseudo-examples embedded in comments).
    """
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith(">>>") or stripped.startswith("..."):
            yield lineno, line


def test_no_surviving_doctest_constructs_hooked_transformer():
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT)
        if relative.name in EXEMPT and len(relative.parts) == 1:
            continue
        if relative.parts[0] in EXEMPT_DIRS:
            continue
        for lineno, source in _doctest_sources(path):
            if _CONSTRUCTION.search(source):
                offenders.append(f"{relative}:{lineno}: {source.strip()}")

    assert not offenders, (
        "Doctest examples in surviving modules reference HookedTransformer; "
        "port them to TransformerBridge before the v4 removal:\n  " + "\n  ".join(offenders)
    )
