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

## Sparse decomposition (J-space coordinates)

A fitted lens also decomposes an activation into the concepts it is *disposed to say*.
`JacobianLens.decompose` writes an activation `x` at layer ℓ as a sparse **nonnegative**
combination of J-lens vectors `v_t = J_ℓ^T W_U[:, t]` (one direction per vocabulary token),
selected greedily. `k` is an **upper bound**, not a target: selection stops early once no
unselected vector is materially positively correlated with the residual (under nonnegativity a
negatively-correlated vector cannot reduce it), so fewer than `k` vectors may be selected and
fewer still may be numerically active.

```python
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis import JacobianLens

model = TransformerBridge.boot_transformers("gpt2", device="cpu")
lens = JacobianLens.from_pretrained("gpt2-small", model=model)

# decompose the activation at a prompt position ...
result = lens.decompose(model, "The Eiffel Tower is in the city of", layer=6, position=-1, k=8)
# ... or a raw [d_model] activation you already have (leave position=None):
#   result = lens.decompose(model, activation, layer=6, k=8)

tokens = [model.to_string(int(t)) for t in result.support]  # the (up to k) *active* J-lens vectors
coordinates = result.coordinates                            # their nonnegative coefficients
```

The result exposes **two supports**, because the paper uses two inconsistent operationalizations
(a main-text sparse nonnegative reconstruction and an appendix projection onto a selected span):

- `support` -- the numerically **active** vectors: the selected vectors whose contribution
  `coordinates[i] * ||v_t||` is a materially nonzero fraction of `||x||`. `coordinates` is aligned
  with `support`, and `reconstruction = sum(coordinates * v_t)` over `support`.
- `selected_support` -- **every** greedily selected vector, including any whose coordinate the
  nonnegativity constraint drove to zero. It defines the span for `j_space_component`. Hence
  `len(support) <= len(selected_support) <= k`.

So two vector outputs also need not coincide:

- `reconstruction` -- the nonnegative combination over the active `support`.
- `j_space_component` (the *J-space component*) -- the orthogonal projection of the activation onto
  the span of `selected_support` -- with `non_j_space_component = x - j_space_component`, the
  residual the interventions leave unchanged.

For the default exact NNLS re-solve the `reconstruction` equals the projection onto the *active*
support (KKT stationarity), so it differs from `j_space_component` exactly when a selected vector
has a zero coordinate (the projection then uses a strictly larger span).

```
x  in R^d_model
   |-- decompose(x, layer, k)
        |-- support / coordinates  a_t >= 0      (active vectors; reconstruction = sum a_t v_t)
        |-- selected_support       S             (all selected vectors; defines the span below)
        |-- j_space_component      Pi_S x        (orthogonal projection onto span of selected v_t)
        \-- non_j_space_component  x - Pi_S x     (orthogonal to the selected vectors)
```

Two algorithms are available via `algorithm=`. The default,
`"nonnegative_orthogonal_matching_pursuit"`, solves a nonnegative least-squares (NNLS) problem
over the selected atoms in float64 after each step. It checks the result against the KKT
conditions and raises `RuntimeError` if the check fails. `"gradient_pursuit"` skips that solve
and uses the directional update from Blumensath & Davies (2008), matching the update used in
the paper; its projected step is accepted only when it does not increase the residual. The two
algorithms share the same greedy selection *rule* but, because their coefficient residuals
differ, may select different vectors at later steps and so return a different `support` and
`reconstruction`.

### Interpreting the numbers honestly

The quantitative findings below are from Gurnee et al. (2026) and were measured on **closed
Anthropic models** (Sonnet / Haiku / Opus); on open-weight models the *shape* may hold but the
exact values will not necessarily transfer.

- The decomposition is **not** a top-k logit-lens readout: because the J-lens vectors are
  overcomplete and non-orthogonal, it gives "a different (and typically less redundant) set of
  active concepts than simply taking the top-k by inner product."
- The J-space is a **small fraction** of the activation: the paper's span projection (the
  `selected_support` operationalization here) "never [exceeds] more than 10%" of total activation
  variance, and for concept vectors carries "a median of only 6-7% ... the remaining ~93% lying
  outside the J-space." Those figures are the paper's own measurements on its models; do not read
  them off this implementation's `j_space_component` without matching the operationalization.
- `k` defaults to 25 because the paper "typically choose[s] it to be no more than 25, which we
  empirically observed to be the number of J-lens vectors that are meaningfully active at a given
  time." Here `k` is an **upper bound**: `support` returns *at most* `k` active vectors (often
  fewer), never `k` padded with zero-coefficient slots.

The full-vocabulary dictionary is cached on the model's device and is vocabulary-sized
(gigabytes for large models); release it with `lens.clear_device_cache()`.
