# UV Migration Summary

This document summarizes the migration from Poetry to UV for the TransformerLens project.

## Changes Made

### 1. pyproject.toml Updates

#### Updated `requires-python`
- **Before**: `>=3.8,<4.0`
- **After**: `>=3.10,<4.0`
- **Reason**: Dev dependencies (pytest-xdist) require Python >= 3.9, and Poetry config already required 3.10+

#### Simplified Python Version-Specific Dependencies
Removed Python 3.8 and 3.9 specific dependencies since the minimum is now 3.10:
- Simplified `numpy` version specifications
- Simplified `torch` version specification (now just `>=2.6`)
- Simplified `transformers` version specification (now just `>=4.51`)

#### Added `[dependency-groups]` Section
Converted Poetry groups to UV's dependency-groups format:
```toml
[dependency-groups]
    dev = [
        "black>=23.3.0",
        "circuitsvis>=1.38.1",
        "isort==5.8.0",
        "jupyter>=1.0.0",
        "mypy>=1.10.0",
        "nbval>=0.10.0",
        "plotly>=5.12.0",
        "pycln>=2.1.3",
        "pytest>=7.2.0",
        "pytest-cov>=4.0.0",
        "pytest-doctestplus>=1.0.0",
        "pytest-xdist>=3.8.0",
    ]
    jupyter = [
        "ipywidgets>=8.1.1",
        "jupyterlab>=3.5.0",
    ]
    docs = [
        "muutils>=0.6.13",
    ]
```

#### Updated `[tool.uv]` Section
```toml
[tool.uv]
    default-groups=["dev", "jupyter", "docs"]
```

### 2. GitHub Actions Workflow Updates

Updated [.github/workflows/checks.yml](.github/workflows/checks.yml) to use UV instead of Poetry for the following jobs:

#### format-check Job (Lines 105-122)
- **Before**: Used `snok/install-poetry@v1` and `poetry install --with dev`
- **After**: Uses `astral-sh/setup-uv@v6` with `uv sync`

#### type-check Job (Lines 124-141)
- **Before**: Used Poetry for dependency installation
- **After**: Uses UV with `uv run mypy .`

#### docstring-test Job (Lines 143-160)
- **Before**: Used Poetry for dependency installation
- **After**: Uses UV with `uv sync`

**Note**: The following jobs were already using UV:
- compatibility-checks (lines 46-103)
- coverage-test (lines 162-215)
- notebook-checks (lines 217-263)
- build-docs (lines 266-310)
- release-python in release.yml (lines 38-67)

### 3. Makefile
Already configured correctly - uses `uv run` for all commands.

### 4. Local Environment Setup

Installed UV locally:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Successfully synced dependencies:
```bash
export PATH="$HOME/.local/bin:$PATH"
uv sync
```

## Benefits of UV

1. **Faster dependency resolution**: UV resolves 200+ packages in ~56ms vs Poetry's minutes
2. **Faster installation**: Installed 178 packages in 1.14s
3. **Better caching**: UV has built-in caching that's more efficient
4. **Modern tooling**: UV is actively developed and follows modern Python packaging standards
5. **CI/CD optimization**: Significantly faster CI runs due to faster dependency resolution

## Breaking Changes

- **Minimum Python version**: Now requires Python >= 3.10 (was >= 3.8)
  - This aligns with the actual dev dependency requirements
  - The Poetry config already required 3.10+ anyway

## Testing

- ✅ Local import test passed: `import transformer_lens` works
- ✅ Test collection works: 556 unit tests collected successfully
- ✅ UV build works: `uv build` command functions correctly
- ✅ All GitHub Actions workflows updated and ready to use UV

## Migration Checklist

- [x] Update pyproject.toml with dependency-groups
- [x] Update requires-python to >=3.10
- [x] Simplify version-specific dependencies
- [x] Update format-check job in GitHub Actions
- [x] Update type-check job in GitHub Actions
- [x] Update docstring-test job in GitHub Actions
- [x] Add docs dependency group with muutils
- [x] Install UV locally
- [x] Test UV sync
- [x] Test UV run
- [x] Verify import works

## Next Steps

1. **Optional cleanup**: Remove Poetry-specific sections from pyproject.toml if no longer needed
2. **Update documentation**: Update any developer documentation that mentions Poetry
3. **Team communication**: Inform team members about the switch to UV

## Commands Reference

### Common UV Commands

```bash
# Sync dependencies (like poetry install)
uv sync

# Run a command in the virtual environment
uv run pytest

# Run a Python script
uv run python script.py

# Build the package
uv build

# Add a dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name
```

### Makefile Commands (unchanged)
```bash
make dep              # Install dependencies
make test             # Run all tests
make unit-test        # Run unit tests
make integration-test # Run integration tests
make format           # Format code
make check-format     # Check code format
```

## Notes

- The uv.lock file is already present and up to date
- All Makefile commands already use `uv run` so no changes needed there
- The Poetry sections in pyproject.toml are still present for backward compatibility but are no longer used
