# Artifact Templates

All structured output files produced during an adapter build. Both the
Programmer and Reviewer reference these templates when writing artifacts to
`.adapter-workspace/`. Copy the relevant template, fill in the placeholders,
and write to the specified path.

---

## Architecture Brief

**Path:** `.adapter-workspace/adapter-brief.md`
**Written by:** Programmer (Step 1a)
**Reviewed by:** Reviewer (Brief Review Mode)

```markdown
# Architecture Brief: <Architecture Name>

## Source Files
- modeling: `models/<arch>/modeling_<arch>.py`
- config:   `models/<arch>/configuration_<arch>.py`

## Module Hierarchy (from __init__ methods)
- Embedding: `model.embed_tokens` (line N)
- Blocks:    `model.layers` (line N)
  - Attention: `self_attn` → q_proj, k_proj, v_proj, o_proj
  - Norm 1:    `input_layernorm`
  - Norm 2:    `post_attention_layernorm`
  - MLP:       `mlp` → gate_proj, up_proj, down_proj
- Final norm: `model.norm` (line N)
- LM head:    `lm_head` (line N)

## Config Fields
hidden_size, num_attention_heads, num_key_value_heads, intermediate_size,
num_hidden_layers, vocab_size, max_position_embeddings, rms_norm_eps, ...

## Architectural Properties
- Normalization: <RMSNorm/LayerNorm>, eps attr: <name> (line N)
- Position embeddings: <rotary/learned/ALiBi>, module: <path>
- Attention: <MHA/GQA/MQA>, n_kv_heads=<N>
- MLP: <gated/standard>, activation: <silu/gelu/...>
- Biases: <which projections have/lack bias>

## Forward Pass Flow
<order of operations in model, attention, MLP forward methods>

## Reference Adapter
- Closest match: <adapter file>
- Key differences from this architecture: <list>

## Representative Models (≤7B only)
| Model | Params | Est. Memory | Notes |
|-------|--------|-------------|-------|
| ...   | ...    | ...         | ...   |
```

---

## Implementation Plan

**Path:** `.adapter-workspace/adapter-plan.md`
**Written by:** Programmer (Step 1b)
**Reviewed by:** Reviewer (Plan Review Mode)

```markdown
# Adapter Plan: <Architecture Name>

Brief: .adapter-workspace/adapter-brief.md

## Phase A: <title>
<details — reference brief for module paths and config fields>

## Phase B: <title>
<details>

## Phase C: <title> (if needed)
<details>

## Phase D: Registration
<details>

## Verification Strategy
- Models: <from brief, all ≤7B>
- Expected phase scores: <targets>
```

Phase merges are fine: name them `Phase A+B: ...` and use that label
everywhere (progress file, phase report filename, signals).

---

## Phase Report

**Path:** `.adapter-workspace/phase-reports/phase-<X>-report.md`
**Written by:** Programmer (Step 2, after each phase)

`<X>` is the plan's phase label: `phase-A-report.md` for Phase A,
`phase-A+B-report.md` for a merged Phase A+B. One report per plan phase,
regardless of merge — never split a merged phase into separate report files.

```markdown
# Phase <letter>: <title>

## Changes made
- <file>: <what changed and why>

## Tests run
- <what was tested and the outcome>

## Verification against HF source
- <module paths, config fields, weight shapes confirmed>
```

---

## Verification Results

**Path:** `.adapter-workspace/verification-results.md`
**Written by:** Programmer (Step 3.4, after all models pass)

```markdown
# Verification Results: <Architecture>

## Models verified
| Model | Params | Memory (GB) | P1 | P2 | P3 | P4 | Status |
|-------|--------|-------------|----|----|----|----|--------|
| ...   | ...    | ...         | ...| ...| ...| ...| ...    |

## Summary
- Total: N models tested
- Passed: N/N
- All phase scores meet thresholds
- mypy: clean
- format: clean
```

---

## Verification Failure Analysis

**Path:** `.adapter-workspace/verification-failure.md`
**Written by:** Programmer (Step 3, when models fail)

```markdown
# Verification Failure Analysis: <Architecture>

## Failed models
| Model | Phase | Score | Threshold | Failed Tests |
|-------|-------|-------|-----------|-------------|
| ...   | ...   | ...   | ...       | ...         |

## Root cause analysis
- <what went wrong and why>

## Planned fix approach
- <what needs to change>
```

---

## Review Files

**Path:** `.adapter-workspace/reviews/<type>-review-<N>.md`
**Written by:** Reviewer (all review modes)

File naming:
- Brief reviews: `brief-review-<N>.md`
- Plan reviews:  `plan-review-<N>.md`
- Phase reviews: `phase-<letter>-review-<N>.md`

`<N>` is the round number — check existing files with
`ls .adapter-workspace/reviews/` before writing.

### Requesting changes

```markdown
# <Brief|Plan|Phase A|...> Review — Round <N>

## Overall assessment
<1-3 sentence summary>

## Issues

### CRITICAL (blocks approval — crashes or wrong results)
1. **[File: path, ~line N]** Description.
   - Verified against: <what source you checked>
   - Why it matters: ...
   - Suggestion: ...

### MEDIUM (works but adds risk or maintenance cost)
...

### LOW (documentation, style, optional improvements)
...

## What was done well
- ...
```

### Approving a phase

```markdown
# Phase <letter> Review — Round <N> — APPROVED

## Review summary
<what was reviewed>

## Verification performed
- Phase 1: <factual verification against HF source>
- Phase 2: <numerical/semantic checks>
- Phase 3: <design assessment>
- Phase 4: <completeness check>
- Phase 5: <plan-to-code match>
```

---

## Completion Report (Final Review)

**Path:** `.adapter-workspace/completion-report.md`
**Written by:** Reviewer (final review after verification passes)

```markdown
# Adapter Completion Report: <Architecture Name>

### Architecture Overview
- Architecture class: <HF class name>
- Transformers source: <modeling file path>
- Pattern: <Llama-like/GPT2-like/...>
- Key features: <normalization, attention, MLP, etc.>

### Adapter Implementation
- Adapter file: <path to new adapter .py>
- Closest reference used: <existing adapter it was based on>
- Unique challenges: <what was non-trivial>
- Solutions: <how they were resolved>

### New Bridge Components (if any)
- <name>: <what it does, why it was needed, file path>
- Tests: <test file path, what is covered>

### New Utilities (if any)
- <name>: <what it does, file path, tests>

### Files Changed
- <complete list of files created or modified>

### Verification Results
| Model | Params | Memory (GB) | P1 | P2 | P3 | P4 | Status |
|-------|--------|-------------|----|----|----|----|--------|
| ...   | ...    | ...         | ...| ...| ...| ...| ...    |

- Total models tested: <N>
- Pass rate: <N/N>
- All phase score thresholds met: yes/no

### Review Summary
- Planning iterations: <count>
- Implementation phases: <count>
- Review iterations per phase: <summary>
- Total issues found and resolved: <count by severity>

### Notes for Future Work (optional)
- ...
```
