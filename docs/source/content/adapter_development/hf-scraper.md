# HuggingFace Model Scraper

The HuggingFace model scraper (`transformer_lens.tools.model_registry.hf_scraper`) discovers HF models, extracts each model's architecture from its config, and writes two registry files: `supported_models.json` (models whose architecture has a registered TransformerLens adapter) and `architecture_gaps.json` (unsupported architectures, scored by relevancy). Most adapter contributors will only need to run it after merging a new architecture adapter to populate the registry with that architecture's models.

## When to run

- **After merging a new adapter** — to populate `supported_models.json` with HF models of that architecture so `verify_models` has things to verify.
- **Periodically (maintainers)** — to refresh the full registry as the HF Hub evolves and the relevancy ranking in `architecture_gaps.json` shifts.
- **When investigating an unsupported architecture** — to see how many models exist for it and what the canonical examples are before deciding to write an adapter.

## Setup

Sphinx, `verify_models`, and the scraper all need an HF token to read gated models. Source from the repo's `.env` (see [feedback_hf_token_env.md](../../../../../.claude/projects/-Users-jlarson-Documents-PROJECTS-TransformerLens/memory/feedback_hf_token_env.md) for the project convention):

```bash
set -a; source .env; set +a
```

Then run from the repo root with `uv`:

```bash
uv run python -m transformer_lens.tools.model_registry.hf_scraper [flags]
```

## Common invocations

### Targeted scrape — a single architecture

The most common workflow for adapter contributors. After registering a new adapter (see the four registration sites in [contributing.md](../contributing.md)), populate the registry with models of just that architecture:

```bash
uv run python -m transformer_lens.tools.model_registry.hf_scraper \
    --architecture LlamaForCausalLM --full-scan
```

**How it queries the Hub.** When `--architecture` names an architecture with registered canonical orgs (in `CANONICAL_AUTHORS_BY_ARCH`), the scraper **skips the global `--task` pagination entirely** and runs only the per-author canonical sweep. That's the only HF-side narrowing available — the Hub API doesn't expose `config.architectures[0]` as a filter, but it does expose `author`, and the canonical-orgs map is exact. For `--architecture LlamaForCausalLM` this means ~4 paginated `list_models(author=...)` calls instead of iterating every text-generation model on the Hub.

If the architecture is **not** in `CANONICAL_AUTHORS_BY_ARCH` (e.g., a new arch where you haven't yet populated the canonical-orgs entry), the scraper falls back to the global scan with a client-side filter — slower, but still complete. You'll see a log line at startup indicating which path was taken.

Existing entries in `supported_models.json` are preserved — the scraper appends, it does not overwrite. So a targeted scrape lands new entries for the requested architecture alongside everything that was already there.

The architecture string must match `config.architectures[0]` from the HF model's config exactly — i.e., the same string the adapter is keyed under in `SUPPORTED_ARCHITECTURES`.

**Trade-off vs. a full scan.** The canonical-sweep-only path is fast and exhaustive within the canonical orgs, but it misses **community fine-tunes from non-canonical orgs** (e.g., a random user fine-tuning Llama-2-7b for a specific task). For populating the registry after adding a new adapter, that's usually fine — foundation-org checkpoints are what `verify_models` should run against first. If you want fine-tune coverage too, follow up with a periodic full scan (no `--architecture` flag).

### Full scan — all architectures

The comprehensive refresh. Iterates every model on the Hub matching `--task`, extracts each architecture, and updates both the supported and gaps reports. Saves checkpoints periodically; safe to interrupt and resume.

```bash
uv run python -m transformer_lens.tools.model_registry.hf_scraper --full-scan
```

### Quick scan — top-N by downloads

Smoke test the scraper or refresh just the most-popular models:

```bash
uv run python -m transformer_lens.tools.model_registry.hf_scraper --limit 10000
```

### Isolated/exploratory scrape

Write to a scratch directory to avoid touching the committed registry:

```bash
uv run python -m transformer_lens.tools.model_registry.hf_scraper \
    --architecture <ArchClass> --full-scan -o ./tmp/scrape/
```

## Output

Two JSON files are written to `--output` (default: `transformer_lens/tools/model_registry/data/`):

- **`supported_models.json`** — one entry per model whose `config.architectures[0]` matches a key in `SUPPORTED_ARCHITECTURES`. Each entry has `architecture_id`, `model_id`, `status`, per-phase verification scores, and metadata. Schema: `SupportedModelsReport` in `schemas.py`.
- **`architecture_gaps.json`** — one entry per *un*supported architecture, with model count, total downloads, smallest known parameter count, sample model IDs, and a computed `relevancy_score`. Sorted by relevancy descending. Schema: `ArchitectureGapsReport` in `schemas.py`.

A `verification_history.json` placeholder is also written if it doesn't already exist; `verify_models` is what actually populates it.

## Flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--architecture ARCH` | none | Only include models whose `config.architectures[0]` matches this exact class name. Applies to main scan and canonical sweep. |
| `--full-scan` | off | Scan every model matching `--task`. Hours-long; checkpoints periodically. |
| `--limit N` | 10000 | Cap scan at N models (ignored with `--full-scan`). |
| `--task TASK` | `text-generation` | HF tag to filter by. Use `text2text-generation` for seq2seq architectures (T5, mT5). |
| `--output DIR` | `transformer_lens/tools/model_registry/data/` | Where to write JSON files. |
| `--min-downloads N` | 500 | Skip models below this download threshold. Canonical-org models bypass this via the sweep. |
| `--checkpoint-interval N` | 5000 | Save a checkpoint every N scanned models. |
| `--no-canonical-sweep` | off | Skip the post-scan sweep that admits canonical-org models below the download threshold. |

## Resumption

The scraper saves checkpoints periodically and on Ctrl-C / network error / HTTP 429. Re-running with the same arguments resumes from the last checkpoint. Checkpoints live at `<output>/scrape_checkpoint.json` and are deleted on successful completion.

If you see a 429 mid-run, the scraper waits and retries automatically (up to 10 attempts, exponentially backed off). No action needed.

## Workflow: adding a new adapter

1. Implement the adapter — see [adapter-creation-guide.md](adapter-creation-guide.md).
2. Register it in the four places listed in [contributing.md](../contributing.md): `supported_architectures/__init__.py`, `factories/architecture_adapter_factory.py`, `tools/model_registry/__init__.py` (both `HF_SUPPORTED_ARCHITECTURES` and `CANONICAL_AUTHORS_BY_ARCH`), and `tools/model_registry/generate_report.py`.
3. Run the registry-sync test to confirm the four sites agree:

   ```bash
   uv run pytest tests/unit/tools/test_model_registry.py -k TestRegistrySyncedWithFactory
   ```

4. Run a targeted scrape to populate the registry with HF models of the new architecture:

   ```bash
   set -a; source .env; set +a
   uv run python -m transformer_lens.tools.model_registry.hf_scraper \
       --architecture <YourArchClass> --full-scan
   ```

5. Verify the discovered models with `verify_models`, smallest-first:

   ```bash
   uv run python -m transformer_lens.tools.model_registry.verify_models --model <hf_repo>
   ```

6. Commit the updated `supported_models.json` (and any `verification_history.json` changes from step 5).

## Caveats

- **Rate limiting.** HF Hub allows ~1000 requests per 5 minutes. The scraper uses `list_models(expand=['config', 'safetensors'])` to fetch inline metadata, so it spends ~200 paginated calls on a full ~200K-model scan — well under the limit. The retry/backoff is there for transient blips, not as a workaround.
- **Quantized variants are filtered.** AWQ, GPTQ, GGUF, bnb, FP8 checkpoints are dropped at the `is_quantized_model` check. TransformerLens requires full-precision weights.
- **`--task` matches HF tags, not `pipeline_tag`.** Encoder-decoder models tagged `text2text-generation` are discoverable under the default `--task text-generation` only via tag overlap. For seq2seq architectures (T5, mT5), pass `--task text2text-generation` explicitly to be safe.
- **The architecture filter is exact-match against `config.architectures[0]`.** It does not accept aliases or partial matches. A targeted scrape for `LlamaForCausalLM` will not surface `LlamaModel` checkpoints (which have a different primary architecture string), nor variants like `Llama4ForCausalLM` if they appear in the future.
- **Existing registry data is preserved, not filtered.** A targeted scrape adds new entries matching the filter; it does not remove unrelated entries from `supported_models.json`. To inspect just the targeted architecture's results in isolation, write to a scratch directory with `-o`.
