# Cohere processed parity fix and test evidence

Tested October 14, 2026 (America/Guatemala).
Branch: `fix/cohere-processed-parity`.
Base: `5eed2712f9217565acfcf6ee2ca8d0e1d8dd48ab` on `feature/cohere-adapter`.

## Problem and changes

The original passing logits comparison never processed weights. The added tests
exercise normalization folding with non-identity norm weights, independent HF
reference and bridge copies, four input cases, and float32 computation. Raw
logits are compared without centering; log-probabilities are compared with the
default centering options. Both enforce absolute error below 1e-4.

These tests exposed two defects: the loader dropped the checkpoint's logit_scale,
and processing folded an output scale into weights while native forward applied
its scale again. The loader now propagates logit_scale. After weight installation,
an adapter finalization callback disables Cohere's redundant native multiplier.
Other adapters inherit a no-op callback. A guard prevents another output-scale
fold after finalization. The native checkpoint scale in this run was 0.125; the
old adapter incorrectly fell back to 0.0625.

## Recorded evidence

- `baseline-tests.txt`: new forward tests against unchanged base production code;
  all six fail, with maximum errors approximately 0.00366 to 0.00389.
- `fixed-tests.txt`: complete Cohere integration and unit tests plus adapter-base
  unit tests on the fix. See the final pytest summary for counts.
- `epsilon-mutation.txt`: both targeted normalization tests fail after epsilon is
  deliberately moved outside the square root. Maximum error is about 1.94.
- `environment.txt`: runtime versions, device, and cached checkpoint revision.
- `requirements-tested.txt`: installed distributions for reproducing this run.

The deliberate failures are the expected result, not unresolved suite failures.
The epsilon mutation exists only in its Python process; it does not edit source.
The new normalization test passes without that mutation. This is an analogous
normalization fault, not the original June Gemma-3 reproducer.

## Reproduce the passing suite

Use a Python 3.12 virtual environment, install the recorded distributions (the
CPU PyTorch wheel may require the PyTorch CPU index), and install this checkout
with `pip install --no-deps -e .`. From the repository root:

```bash
python -m pytest -o addopts='' \
  tests/integration/model_bridge/test_cohere_adapter.py \
  tests/unit/model_bridge/supported_architectures/test_cohere_adapter.py \
  tests/unit/model_bridge/test_architecture_adapter.py -q --tb=short
```

The test needs `trl-internal-testing/tiny-CohereForCausalLM` cached or reachable.
Recorded runs used `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_PROGRESS_BARS=1` with
checkpoint revision `90bc56bd12c642f6bf5e710ebd7c626dd7c0201c` already cached.
The tests now pin that revision for config, tokenizer, and weights together.
`addopts` is cleared to avoid requiring unrelated notebook/doctest plugins.

## Reproduce the negative controls

Run the in-memory epsilon mutation from the fixed checkout:

```bash
python review_evidence/cohere/check_epsilon_mutation.py
```

Expected: exactly four assertion failures, process exit code 1, and no source
changes. The script guards its replacement target and stops if that target changes.

To check the baseline, create a separate worktree at the base commit, copy only
`tests/integration/model_bridge/test_cohere_adapter.py` from the fixed checkout
into the same path there, and run this command from that baseline worktree using
the same virtual environment:

```bash
python -m pytest -o addopts='' \
  tests/integration/model_bridge/test_cohere_adapter.py \
  -k folded_forward -q --tb=short --show-capture=no
```

The original captured baseline used the six-case test at commit 263462c;
use that version to reproduce its six failures and exit code 1. Do not restore the old production code
into the fixed checkout. The captured baseline traceback paths identify the
separate baseline worktree.

## Scope and release limits

The follow-up run installs runtime dependencies exported from uv.lock, including
torch 2.7.1 and transformers 5.0.0, with pytest 8.4.1. See
`requirements-locked-tested.txt`, `lock-check.txt`, and `locked-tests.txt`.
The focused run passed 132 tests with one unrelated skip; shared-component tests
passed 133 tests. The intentional epsilon mutation failed all four targeted cases.
The original `fixed-tests.txt` records the earlier, different environment.

Coverage now includes 64-token sequences and deterministic synthetic QK-normalized
models, as well as the pinned tiny checkpoint. These checks exposed incorrect
QK tensor ordering in the shared attention bridge. Cohere normalization now runs
before transposing heads and sequence; the processed attention path also uses
Cohere's adjacent-pair rotary implementation. Other architectures retain their
existing path. `shared-components-tests.txt` records additional component tests.
`locked-epsilon-mutation.txt` records the four expected negative-control failures.

The finalization callback is invoked after weight distribution, is a no-op for
other adapters, and has a Cohere unit test for the scale reset and repeat-fold
guard. It does not guarantee transaction rollback if weight distribution fails.

The existing CI did not run this integration test. The new
`.github/workflows/cohere-parity.yml` uses frozen dependencies and uploads output.
It has not run remotely: upstream is read-only and no writable fork was found.

The one skipped adapter test explicitly targets a SoLU MLP norm absent from its
Gemma3 fixture; it is unrelated to Cohere parity and remains skipped.

These are CPU checks, not release certification. GPU, reduced precision, the
actual release checkpoint, and the full repository suite remain unvalidated.
No GPU is available in this environment. The June triage note
`triage_1121_gemma3_logits.md` reports about 1.3e-4 error on a longer prompt and
an epsilon-placement fix, but supplies no executable reproducer or exact prompt.
The analogous mutation above is not a replay of that incident.
Rerun on the actual release configuration before merging. The email's PR number
did not match the live PR; this change uses the branch confirmed by the user.
