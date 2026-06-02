# GitHub Copilot Instructions

**Read [AGENTS.md](../AGENTS.md) at the repo root for the full set of project conventions, quickstart commands, repo layout, and hard rules.** This file mirrors the top-of-mind essentials so Copilot Chat surfaces them inline.

## Top rules to remember

1. **Use `uv`, not `pip` or `poetry`.** Install with `uv sync`. Run commands with `uv run …` or via `make` targets.
2. **Two parallel systems**: `HookedTransformer` (legacy, in maintenance) and `TransformerBridge` (v3, default for new work). If you change behavior on HookedTransformer that has an equivalent in TransformerBridge, mirror the change to TransformerBridge in the same PR. New TransformerBridge features do not need to be mirrored in HookedTransformer. The HT model registry [transformer_lens/supported_models.py](../transformer_lens/supported_models.py) is **HT-only** — Bridge-only models go in the Bridge registry under [transformer_lens/tools/model_registry/](../transformer_lens/tools/model_registry/).
3. **Base PRs against `dev`**, not `main`. PRs targeting `main` are done by Maintainers only – Request permission before basing off `main`.
4. **No pre-commit hook is installed.** Run `make format` and `uv run mypy .` manually before pushing — CI will fail otherwise.
5. **Source `.env` before any HF-Hub command** (`set -a; source .env; set +a`). `HF_TOKEN` is required for gated models (Llama, Mistral, Gemma, gated Qwen variants).

## Common commands

```bash
uv sync                  # install
make unit-test           # fast tests
make format              # pycln + isort + black
uv run mypy .            # type check
uv run docs-hot-reload   # live docs preview
```

## Hard "do not"s

- Don't dismiss failing tests as "pre-existing" — investigate every failure.
- Don't add `# type: ignore`. Prefer `isinstance` / `cast`.
- Don't run model verification or benchmarks in parallel (device OOM).
- Don't claim drift is "fp noise" without empirical evidence.
- Don't add skips / `xfail` to make failing CI green.
- Don't reference plan-file details in source comments or docstrings.

See [AGENTS.md](../AGENTS.md) for the full set with rationale.
