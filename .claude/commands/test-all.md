---
description: Run the full test suite (unit + integration + acceptance + benchmark + docstring + notebook). Slow.
---

Run every test tier in TransformerLens via the top-level `make test` target:

```
make test
```

This is slow — it runs unit, integration, acceptance, benchmark, docstring, and notebook tests sequentially. It hits HuggingFace Hub and loads multiple models. Before running, confirm:

1. `.env` is sourced so `HF_TOKEN` is set (`set -a; source .env; set +a`).
2. No other heavy GPU/MPS jobs are running on this machine — model verification cannot run concurrently (see [AGENTS.md §10](../../AGENTS.md#10-hard-rules)).

Report the actual command output. Investigate any failures rather than dismissing them.
