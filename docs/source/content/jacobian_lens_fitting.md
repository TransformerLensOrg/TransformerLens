# Fitting a Jacobian lens

A [Jacobian lens](https://github.com/TransformerLensOrg/TransformerLens/blob/dev/demos/Jacobian_Lens_Demo.ipynb)
maps each intermediate residual stream to the final residual stream. Use a published
artifact when one exists:

```python
from transformer_lens.tools.analysis import JacobianLens

lens = JacobianLens.from_pretrained("gemma-2-2b")
```

Fit a new lens when your model is not in the artifact registry or when you need a
different prompt distribution. Fitting is deterministic for a fixed model, prompt
manifest, and set of estimator options.

## Requirements

`JacobianLens.fit` requires a freshly booted, causal decoder-only
`TransformerBridge` with raw Hugging Face weights:

```python
import torch

from transformer_lens import TransformerBridge

model = TransformerBridge.boot_transformers(
    "openai-community/gpt2",
    device="cuda",
    dtype=torch.float32,
    revision="YOUR_MODEL_COMMIT_SHA",
)
```

Do not enable compatibility mode or process the model weights. Pin a model revision
for a reproducible fit. The resolved revision, model name, dtype, TransformerLens
version, corpus identifier, and estimator options are recorded in the artifact.

Fitting one prompt requires one forward pass and
`ceil(d_model / dim_batch)` backward passes. Increasing `dim_batch` is faster but
replicates the prompt that many times in memory. Start with `dim_batch=8` and lower
it after an out-of-memory error. Float32 gives the highest-fidelity estimator;
bfloat16 and float16 use less memory but emit a reduced-precision warning.

## Prepare a prompt manifest

Store the exact fitting texts in a versioned JSON Lines file, one object per line:

```json
{"text": "The first sufficiently long document..."}
{"text": "The second sufficiently long document..."}
```

Keep the order fixed and give the corpus a stable identifier containing the dataset
and preprocessing revision, for example
`wikitext-103-raw-v1@2.0.0:train:min-600-chars`. The `corpus` argument is provenance;
it does not load or transform the texts.

Around 100 prompts of 128 tokens gives a usable first fit, while published lenses
use up to 1,000 prompts. Very short prompts are skipped because the estimator
excludes the first few positions and the final position. Review warnings after each
run and keep the same manifest for every shard.

## Fit one shard

Save this as `fit_jacobian_lens_shard.py`:

```python
import argparse
import hashlib
import json
from pathlib import Path

import torch

from transformer_lens import TransformerBridge
from transformer_lens.tools.analysis import JacobianLens


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--revision", required=True)
parser.add_argument("--prompts", type=Path, required=True)
parser.add_argument("--corpus", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--shard-index", type=int, required=True)
parser.add_argument("--num-shards", type=int, required=True)
parser.add_argument("--device", default="cuda")
parser.add_argument("--dim-batch", type=int, default=8)
parser.add_argument("--max-seq-len", type=int, default=128)
args = parser.parse_args()

if args.num_shards < 1:
    parser.error("--num-shards must be at least 1")
if not 0 <= args.shard_index < args.num_shards:
    parser.error("--shard-index must be in [0, num-shards)")

manifest_bytes = args.prompts.read_bytes()
records = [
    json.loads(line)
    for line in manifest_bytes.decode("utf-8").splitlines()
    if line.strip()
]
prompts = [record["text"] for record in records]
shard_prompts = prompts[args.shard_index :: args.num_shards]
if not shard_prompts:
    parser.error("this shard has no prompts")

model = TransformerBridge.boot_transformers(
    args.model,
    revision=args.revision,
    device=args.device,
    dtype=torch.float32,
)
lens = JacobianLens.fit(
    model,
    shard_prompts,
    corpus=args.corpus,
    dim_batch=args.dim_batch,
    max_seq_len=args.max_seq_len,
    metadata={
        "prompt_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "num_shards": args.num_shards,
    },
)
args.output.parent.mkdir(parents=True, exist_ok=True)
lens.save(str(args.output))
```

Every worker must use the same model revision, manifest, corpus identifier,
`num_shards`, TransformerLens version, dtype, and estimator options. Only
`shard-index`, device, and output path should differ:

```bash
uv run python fit_jacobian_lens_shard.py \
  --model openai-community/gpt2 \
  --revision YOUR_MODEL_COMMIT_SHA \
  --prompts prompts.jsonl \
  --corpus wikitext-103-raw-v1@2.0.0:train:min-600-chars \
  --num-shards 8 \
  --shard-index 0 \
  --output fits/shard-00.pt
```

Run indices `0` through `7` on separate processes or machines. Do not run multiple
workers on one GPU unless their combined model and fitting batches fit in memory.
Round-robin slicing means the union of all shards contains every manifest row
exactly once.

Do not add shard-specific values such as the shard index or hostname to `metadata`.
`JacobianLens.merge` rejects mismatched provenance so it cannot silently combine
different fits. Put worker-specific information in logs or output filenames instead.

## Merge the shards

Copy all shard artifacts to one machine, then merge them:

```python
from pathlib import Path

from transformer_lens.tools.analysis import JacobianLens

paths = sorted(Path("fits").glob("shard-*.pt"))
if len(paths) != 8:
    raise RuntimeError(f"expected 8 shards, found {len(paths)}")

shards = [JacobianLens.load(str(path)) for path in paths]
lens = JacobianLens.merge(shards)
lens.save("gpt2_jacobian_lens.pt")

print(f"merged {len(paths)} shards and {lens.n_prompts} accepted prompts")
```

The merge is an exact prompt-count-weighted average of the shard matrices. It
requires matching source layers, model width, and provenance apart from
`n_prompts`. A mismatch is normally evidence that workers used different inputs,
versions, or fitting options; fix and rerun the inconsistent shard rather than
editing its metadata.

## Validate and publish

Load the saved artifact, validate it against a fresh bridge, and run a readout before
publishing:

```python
import torch

from transformer_lens import TransformerBridge
from transformer_lens.tools.analysis import JacobianLens

model = TransformerBridge.boot_transformers(
    "openai-community/gpt2",
    device="cuda",
    dtype=torch.float32,
    revision="YOUR_MODEL_COMMIT_SHA",
)
lens = JacobianLens.load("gpt2_jacobian_lens.pt").validate_model(model)

assert lens.n_prompts > 0
assert all(torch.isfinite(matrix).all() for matrix in lens.jacobians.values())
readout = lens.readout(model, "The capital of France is", top_k=10)
print(readout)
```

Record the manifest hash, resolved model revision, fitting command, shard count, and
accepted prompt count with the artifact. Upload the `.pt` file to a Hugging Face
model repository:

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo("your-org/your-jacobian-lenses", repo_type="model", exist_ok=True)
api.upload_file(
    repo_id="your-org/your-jacobian-lenses",
    repo_type="model",
    path_or_fileobj="gpt2_jacobian_lens.pt",
    path_in_repo="gpt2_jacobian_lens.pt",
)
```

Consumers can then load and validate it directly:

```python
lens = JacobianLens.from_pretrained(
    "your-org/your-jacobian-lenses",
    filename="gpt2_jacobian_lens.pt",
    model=model,
)
```

To propose a short-name entry in TransformerLens, open a pull request that adds the
published file to `transformer_lens/tools/analysis/jacobian_lens_registry.json` and
include the fitting provenance and validation results.

## Importing an existing lens

`JacobianLens.load()` accepts two file schemas: the standard artifact format (four
official keys — `J`, `n_prompts`, `source_layers`, `d_model`) written by
`JacobianLens.save()` and compatible with the Anthropic reference package, and the
fit-checkpoint format described below.

### Fit checkpoint format

A fit checkpoint holds the running Jacobian sums from an interrupted or staged fit,
before the final per-prompt average is applied. `JacobianLens.load()` detects a
checkpoint by the presence of a `jacobian_sum` key and reconstructs the per-layer
means automatically:

```python
lens = JacobianLens.load("path/to/checkpoint.pt")
```

The expected top-level keys are:

| Key | Type | Required | Description |
|---|---|---|---|
| `jacobian_sum` | `dict[int, Tensor[d, d]]` | yes | Running sum of per-prompt Jacobians (float32) |
| `n_prompts` | `int` | yes | Number of prompts accumulated into the sums (must be > 0) |
| `d_model` | `int` | yes | Residual stream width |
| `source_layers` | `list[int]` | no | Documented layer indices (informational only) |
| `metadata` | `dict[str, ...]` | no | Nested provenance (scalars, lists, string-keyed dicts) |
| `model_name` | `str` | no | Flat provenance shortcut (harvested into `metadata`) |
| `model_revision` | `str` | no | Flat provenance shortcut (harvested into `metadata`) |
| `corpus` | `str` | no | Flat provenance shortcut (harvested into `metadata`) |

`load()` always sets `converted_from: "jacobian_lens_checkpoint"` in the resulting
lens's metadata. This sentinel prevents `merge()` from silently combining a
converted checkpoint with a natively TL-fitted lens, since the two provenance
dictionaries will differ. To merge shards that were all loaded from checkpoints, they
must otherwise share identical provenance (same `model_name`, `corpus`, etc.):

```python
shards = [JacobianLens.load(str(p)) for p in checkpoint_paths]
# All shards get converted_from="jacobian_lens_checkpoint";
# merge() requires identical provenance apart from n_prompts.
merged = JacobianLens.merge(shards)
merged.save("merged_lens.pt")
```

### Metadata handling

Three things are guaranteed when loading a checkpoint:

**Fit-reserved keys are stripped.** Keys written by `JacobianLens.fit()` to record
TransformerLens internals (`transformer_lens_fit`, `transformer_lens_version`,
`model_system`, `hook_convention`, etc.) are not carried over, since they describe
the *native* fitting pipeline and would be wrong for an imported lens.

**Tensor-valued fields are dropped.** `JacobianLens.save()` uses
`weights_only=True` for reload safety, which restricts metadata to plain Python
scalars, lists, and string-keyed dicts. Tensor-valued fields from the source
checkpoint would silently corrupt a future load, so they are dropped and their names
are recorded in a `dropped_fields` list in the lens's metadata:

```python
lens = JacobianLens.load("checkpoint_with_tensors.pt")
if "dropped_fields" in lens.metadata:
    print("fields not carried over:", lens.metadata["dropped_fields"])
```

**Dtype is not preserved.** `JacobianLens.__init__` casts all matrices to float32
and `save()` defaults to float16, so the source checkpoint's storage dtype cannot
be preserved end-to-end. Value precision is preserved within those two conversions.

### Note on tuned-lens

Tuned-lens checkpoints are not currently supported. Tuned-lens translators are
affine (weight + bias), but the Jacobian artifact format has no bias slot to
receive the translation component; importing would therefore silently drop the
bias and produce wrong-but-plausible readouts. Layer indexing also differs
(input-to-layer-ℓ vs. output-of-block-ℓ), which would yield misaligned
readouts if not corrected. These issues are deferred; support can be added in a
follow-up once a lossless mapping is established.
