# Plan: Split Full Code Coverage Test into parallel shards

## Context

The `Full Code Coverage Test` job in [.github/workflows/checks.yml](.github/workflows/checks.yml) runs the full test suite (`tests/integration + tests/benchmarks + tests/unit + tests/acceptance`) with `--cov` instrumentation and produces the `test-coverage` artifact that the `docs-build` job depends on.

Recent measurement (run 25564944516): **39.2 minutes**. This is the critical-path job for the workflow — every other job finishes in ≤18 min, so coverage gates the whole run. A 39-minute feedback loop is too long for active development.

The fix is to matrix-shard the tests, write per-shard `.coverage.<id>` data files, and merge in a final job via `coverage combine`. Coverage.py supports this natively, including with `--cov-branch`.

Target after this plan: **~20 minutes wall-clock**, no loss of coverage data.

## Files to modify

- `.github/workflows/checks.yml` — replace the single `coverage-test` job with a 3-shard matrix + a `coverage-combine` job. Update `needs:` on `docs-build`.

That's it. No source or `makefile` changes needed; pytest invocations live in the workflow YAML.

## Implementation

### Step 1 — Replace `coverage-test` with a sharded matrix

Three shards, sized roughly evenly based on observed runtime:

| Shard id | Path | Estimated time | xdist? |
|---|---|---|---|
| `unit-acceptance` | `tests/unit tests/acceptance` | ~20 min | No (model loads will OOM) |
| `integration` | `tests/integration` | ~13 min | No (same reason) |
| `benchmarks` | `tests/benchmarks` | ~3 min | `-n 2` if helpful |

Each matrix entry runs the same setup as the current job (uv sync, HF model cache restore, HF auth) and then:

```yaml
- name: Run shard
  run: |
    uv run pytest \
      --cov=transformer_lens \
      --cov-branch \
      --cov-report= \
      ${{ matrix.shard.path }}
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
    COVERAGE_FILE: .coverage.${{ matrix.shard.id }}

- name: Upload partial coverage data
  uses: actions/upload-artifact@v4
  with:
    name: coverage-data-${{ matrix.shard.id }}
    path: .coverage.${{ matrix.shard.id }}
    include-hidden-files: true   # required: .coverage.* is dotfile
    retention-days: 1
```

Key details:
- `COVERAGE_FILE=.coverage.<id>` per shard so each writes to a distinct file
- `--cov-report=` (empty) suppresses per-shard HTML; only data file is needed
- `include-hidden-files: true` is mandatory on `upload-artifact@v4+` for dotfiles
- Keep `timeout-minutes: 30` (down from 60) since longest shard is ~20 min

### Step 2 — Add `coverage-combine` job

```yaml
coverage-combine:
  name: Combine coverage and build report
  runs-on: ubuntu-latest
  needs: coverage-test
  timeout-minutes: 5
  steps:
    - uses: actions/checkout@v4
    - name: Install uv
      uses: astral-sh/setup-uv@v7
      with:
        python-version: "3.12"
        activate-environment: true
        enable-cache: true
    - name: Install dependencies
      run: |
        uv lock --check
        uv sync
    - name: Download all partial coverage artifacts
      uses: actions/download-artifact@v4
      with:
        pattern: coverage-data-*
        merge-multiple: true
    - name: Combine + build report
      run: |
        uv run coverage combine
        uv run coverage html -d htmlcov
        uv run coverage report --skip-empty
    - name: Upload Coverage Report Artifact
      uses: actions/upload-artifact@v4
      with:
        name: test-coverage   # preserve name; docs-build expects this
        path: htmlcov
```

### Step 3 — Update `docs-build` dependency

In the existing `docs-build` job (currently `needs: coverage-test`):

```yaml
docs-build:
  needs: coverage-combine   # was: coverage-test
```

The downstream `download-artifact` call (line 392 of checks.yml) is unchanged — it still pulls the `test-coverage` artifact, just from `coverage-combine` instead of `coverage-test`.

### Step 4 (optional, only if needed) — Sentinel job for branch protection

If "Full Code Coverage Test" is currently configured as a required status check on `main` / `dev*`, the matrix split will rename it to multiple entries (e.g. `Full Code Coverage Test (unit-acceptance)`). Branch protection rules will need to either:

- (a) Be updated to require all three matrix entries individually, or
- (b) Require a new sentinel job:

```yaml
coverage-required:
  name: Coverage (required)
  runs-on: ubuntu-latest
  needs: [coverage-test, coverage-combine]
  if: always()
  steps:
    - run: |
        # Fail if any upstream failed
        if [[ "${{ needs.coverage-test.result }}" != "success" ]]; then exit 1; fi
        if [[ "${{ needs.coverage-combine.result }}" != "success" ]]; then exit 1; fi
        echo "All coverage shards passed and were combined."
```

Then point branch protection at `Coverage (required)` instead.

## Verification

Run on a test branch and check:

1. **Wall-clock**: longest shard should be ~20 min (unit-acceptance). Combine job adds ~2 min. Total ≈ 22 min vs 39 min baseline.
2. **Coverage parity**: compare `coverage report` output line-by-line against the pre-split run. Branch coverage numbers should match exactly — `coverage combine` with `--cov-branch` data is well-tested.
3. **Artifact contract**: `docs-build` still finds and consumes `test-coverage`. No change visible to downstream consumers.
4. **Empty shard handling**: if any shard has zero tests collected (shouldn't happen here), `coverage combine` ignores empty data files cleanly.

## Risks and mitigations

- **HF model cache cost**: each shard restores from the same cache key. After first warm-up, all shards hit the cache; first cold run pays the download once per shard. Acceptable cost.
- **Coverage merging edge cases**: branch coverage data can drift between shards if a single line is exercised in two shards via different code paths. `coverage combine` handles this — verified in the tool's own test suite.
- **pytest-xdist on shards that load models**: not applied. If we want xdist on the unit shard later, gate it on whether that shard's tests use big models.
- **Test interaction across shards**: tests in `tests/integration` and `tests/unit` don't share state across processes (no shared fixtures across directories). Sharding is safe.

## Follow-up work (out of scope for this plan)

- **Split integration with `pytest-split`**: if 20 min is still too slow, split `tests/integration` into 2 balanced shards using `pytest-split`'s `--splits N --group i`. Drops wall-clock further to ~12-15 min.
- **Split unit too**: `tests/unit` has 115 files; 2-way split via `pytest-split` would balance well. Combined with integration split, wall-clock ~10-12 min.
- **Compat checks parallelization**: after coverage drops, "Compatibility Checks" at ~17 min becomes the new critical path. Separate question.
- **Drop redundancy between coverage and Python 3.12 compat**: currently both run unit+acceptance on 3.12. Could be dedup'd, but complicates the matrix; not urgent.
