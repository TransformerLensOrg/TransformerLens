# Natural language autoencoder inference

## Availability and validation boundary

Natural language autoencoder (NLA) execution is currently private experimental
infrastructure. There is no supported public `NaturalLanguageAutoencoder` export,
and the bundled support registry contains no verified pair. Calling its private
`from_pretrained` resolver fails for unregistered names rather than silently
loading an unverified artifact. Private interfaces may change.

Local synthetic checkpoints exercise the inference mechanics. This does not
establish numerical compatibility with the released Qwen checkpoints. The slow
released-artifact checks must pass before any pair is advertised as supported.
The implementation currently uses raw fp32, eager-attention, single-device
execution; this is an execution restriction, not a universal precision/device
compatibility claim.

## What the round trip does

An Activation Verbalizer (AV) translates one residual-stream activation into
explanation text. An Activation Reconstructor (AR) translates that text back into
a vector using a truncated backbone and its separate trained linear value head.
TransformerBridge hooks provide local execution without an inference server.
Training, explanation-edit steering, corpus execution, and dataset-level FVE are
not part of this inference interface.

The candidate Qwen artifacts expect the raw output of zero-based block 20:
`blocks.20.hook_out`, corresponding to HF `hidden_states[21]`. Their hidden size
is 3584. This block is artifact-fixed, not a freely interchangeable layer.
Compatibility weight processing changes the expected residual basis and is
rejected. Merely matching architecture and dimensions does not establish that
an activation belongs to the training distribution.

The AV sidecar records its mandatory injection L2 norm (150 for the inspected
Qwen candidate). The vector replaces one reserved token's post-embedding output
in the canonical chat template. The tokenizer encoding, placeholder count, and
neighbor IDs are checked rather than guessed. Cached execution injects on
prefill only; uncached execution reapplies at the original position on each
prefix recomputation. One-token prefill is valid.

AR takes a raw prompt, not a chat template. For the inspected Qwen candidate it
is `Summary of the following text: <text>{explanation}</text> <summary>`.
Special-token and suffix conventions must match the sidecar. AR reads the raw
final-token output of block 20 before final normalization, then applies the
released bias-free linear head. The private loader validates the omitted final
norm/LM-head weights and runs only the backbone under Bridge hooks. The omitted
LM head is never used to reconstruct a vector.

## Explanation and scoring contracts

An explanation result preserves raw generated continuation text and continuation
token IDs separately from canonical inner explanation content. Exactly one
complete `<explanation>...</explanation>` region is required. Missing, multiple,
nested, malformed, incomplete, and empty regions yield `text=None` with an
explicit parse status; raw text is not substituted as a success.

Reconstruction accepts nonempty untagged explanation content. Explicit or malformed
explanation wrappers are rejected. It preserves authored interior whitespace and
does not claim semantic detection of arbitrary untagged full responses. Do not
pass raw failed generation to AR automatically.

Scoring normalizes both finite nonzero vectors independently and reports cosine
and normalized per-coordinate MSE:

```text
normalized_mse = 2 * (1 - cosine)
```

Equivalently, normalize each vector to L2 norm `sqrt(d)` before taking the
per-coordinate squared-error mean. There is no additional dimension multiplier.
CPU float64 accumulation with scale-safe norms handles finite extreme inputs;
only roundoff-sized cosine overshoot is clamped. Unsupported AR normalization
modes, including null/raw-MSE metadata, are rejected.

These scores are directional diagnostics, not raw vector MSE, dataset FVE, or
proof of human-readable faithfulness. A low reconstruction error can coexist
with misleading text. Do not infer causal validity from fluent explanations.

## Experimental execution examples

The following examples are for contributors with an explicitly constructed,
immutable `NLAPairConfig` whose artifacts they are validating. They deliberately
use the private constructor rather than claiming a working public registry name.
Replace `pair` with that explicit configuration; do not copy test fixture identities
into a real analysis or interpret these examples as verified model support.

### Convenience workflow

This retains the caller-owned target while AV and AR execute and may require
multiple models to fit in memory:

```python
from transformer_lens.tools.analysis._nla import NaturalLanguageAutoencoder

with NaturalLanguageAutoencoder(pair, "experimental-pair", device="cpu") as nla:
    target = nla.load_target()
    try:
        activation = nla.extract(target, prompt="The auditor asked about reasoning.")
        explanation = nla.verbalize(activation, max_new_tokens=512)
        if explanation.text is None:
            raise ValueError(f"Explanation parse failed: {explanation.parse_status}")
        reconstruction = nla.reconstruct(explanation.text)
        score = nla.score(activation, reconstruction)
        print(explanation.text, score.cosine, score.normalized_mse)
    finally:
        del target
```

### Sequential low-memory workflow

Extraction and reconstruction return detached owned vectors on CPU by default.
Release the target before loading AV, and unload AV before AR:

```python
from transformer_lens.tools.analysis._nla import NaturalLanguageAutoencoder

with NaturalLanguageAutoencoder(pair, "experimental-pair", device="cpu") as nla:
    target = nla.load_target()
    try:
        activation = nla.extract(
            target, prompt="The auditor asked about reasoning.", output_device="cpu"
        )
    finally:
        del target
    try:
        explanation = nla.verbalize(activation)
    finally:
        nla.unload("av")
    if explanation.text is None:
        raise ValueError(f"Explanation parse failed: {explanation.parse_status}")
    try:
        reconstruction = nla.reconstruct(explanation.text)
    finally:
        nla.unload("ar")
    score = nla.score(activation, reconstruction.vector)
```

The context owns AV/AR only, not returned targets or caller-held tensors. Other
references can keep a target alive after `del target`. Unload releases owned
references, not all allocator-reserved memory or externally held allocations.
An unloaded role can reload lazily; `close()` permanently closes the instance.
Concurrent operations, including unload during inference, are rejected.

## Inputs, provenance, and bounded generation

- One unpadded string or explicit integer token sequence is supported. Negative
  positions resolve against the original prompt. Oversized inputs raise rather
  than silently truncate. Target strings are not automatically chat-templated.
- `max_new_tokens=512`, temperature 1, and KV caching are the finite AV defaults.
  Greedy decoding uses `do_sample=False`; sampling filters require sampling.
  The prompt plus generation budget must fit the context even if EOS could stop
  earlier. The default length is a convenience policy, not a paper replication.
- `source_verified` means recorded pinned-loading provenance, not continuously
  verified live weights. In-place weight edits can invalidate artifact assumptions.
  Unknown target origins fail by default; explicit `allow_unverified_target=True`
  warns and marks results unverified without bypassing known incompatibilities.
- Bare vectors have unknown source provenance. Structured extraction results record
  identity, layer, position, basis, and tokenization policy without retaining models.
- Cache/offline options reuse HF snapshots. Metadata/tokenizers are checked before
  full AV/AR weight loading. Remote-code execution is disabled. Import itself does
  not initiate NLA downloads.

## Resource requirements and verification

The workflow involves three separate 7B-class checkpoints. A pinned-artifact file
inventory measured 38.5 GiB of missing safetensor files for target, AV, and AR on
an empty-weight cache, excluding tokenizer assets and temporary/runtime overhead.
The slow checker adds a 2 GiB free-space reserve. Sequential execution reduces
peak residency, not retained checkpoint storage. Released-checkpoint peak RAM/VRAM
and timing have not been measured; no accelerator fit claim is made.

Run validation serially on a machine with sufficient storage and memory. Source
`.env` before Hub access. Set `HF_HOME` to a larger cache volume before starting
Python if needed:

```bash
set -a; source .env; set +a
uv run pytest tests/integration/test_nla_oracle_parity.py -m slow -s
```

These tests compare AV prefill logits and AR vectors to pinned direct-HF reference
paths and exercise a complete sequential round trip. They do not execute downloaded
reference Python code. Exact fp32 equality is the initial strict criterion; any
nonzero drift requires investigation and measured tolerance justification.
Current standard test targets deselect `slow`, so their passing does not imply
released-artifact parity. Normal-CI synthetic/cached-model tests remain necessary
for maintaining hook execution and lifecycle behavior.

## Package-resource check

The registry must be present in a built wheel even while it is empty. A local
build/check uses no model downloads:

```bash
uv build
uv run pytest tests/integration/test_nla_packaging.py
```

The packaging test builds a wheel in a temporary directory, reads its resources,
and imports its artifact helper from an isolated extracted-wheel package with
network calls forbidden. This verifies resource delivery, not model compatibility.

## References

- [Natural Language Autoencoders paper](https://transformer-circuits.pub/2026/nla/)
- [Authors' reference repository](https://github.com/kitft/natural_language_autoencoders)

The test oracle records its pinned reference revision and artifact identities.
The integration consumes trained artifacts and does not import the authors'
training or serving stack.
