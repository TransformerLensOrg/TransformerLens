# Natural language autoencoder inference

## Availability and validation boundary

`NaturalLanguageAutoencoder` is available from `transformer_lens.tools.analysis`.
The supported registry name is `qwen2.5-7b-instruct-L20`, with immutable target,
AV, and AR revisions. The released pair passed direct-HF AV-logit and AR-vector
checks on CPU in fp32 with eager attention, with zero maximum and mean absolute
error for two inputs per path. The sequential released-model round trip also
completed. These finite test inputs do not guarantee explanation quality.

Only raw fp32/eager single-device execution is implemented. CPU is the validated
configuration; CUDA/MPS, quantization, arbitrary dispatch, and other released
pairs are not advertised as verified. Local synthetic tests complement, but do
not replace, slow released-artifact checks.

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
prefix recomputation. One-token prefill is valid. On ordinary fp32 inputs,
injection normalization matches the released client's fp32 norm and division
order. A scale-safe fallback handles norm underflow/overflow outside that input
domain; this fallback is not claimed as reference-client equivalence.

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

## Execution examples

The registry pins the target and both role checkpoints. Metadata and tokenizer
contracts are checked before full role-model loading. Use the context manager to
release owned AV/AR resources, including on exceptions.

### Convenience workflow

This retains the caller-owned target while AV and AR execute and may require
multiple models to fit in memory:

```python
from transformer_lens.tools.analysis import NaturalLanguageAutoencoder

with NaturalLanguageAutoencoder.from_pretrained(
    "qwen2.5-7b-instruct-L20", device="cpu"
) as nla:
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
from transformer_lens.tools.analysis import NaturalLanguageAutoencoder

with NaturalLanguageAutoencoder.from_pretrained(
    "qwen2.5-7b-instruct-L20", device="cpu"
) as nla:
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
peak residency, not retained checkpoint storage. A cached serial CPU/fp32 run of
all three released tests, including public `from_pretrained()` and the round trip,
completed in 281.10 seconds (284.33 seconds wall time including process overhead).
The child process maximum resident set size reported by macOS `time -l` was
36,417,159,168 bytes, approximately 33.92 GiB, on a machine with 48 GiB RAM.
This measures the complete test process, not a per-role minimum or accelerator
VRAM requirement. Downloads are excluded from this cached-run timing; no CUDA/MPS
fit or timing claim is made.

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

The packaging test builds a wheel with a fresh uv cache, allowing build dependency
resolution, and reads its resources. A separate isolated subprocess imports the
actual extracted-wheel package and analysis initializers with network calls
forbidden. Build dependencies may require network access; the package import
check does not. These checks verify package/resource delivery, not model
compatibility.

## References

- [Natural Language Autoencoders paper](https://transformer-circuits.pub/2026/nla/)
- [Authors' reference repository](https://github.com/kitft/natural_language_autoencoders)

The test oracle records its pinned reference revision and artifact identities.
The integration consumes trained artifacts and does not import the authors'
training or serving stack.
