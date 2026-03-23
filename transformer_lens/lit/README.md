# TransformerLens LIT Integration

This module provides integration between [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and Google's [Learning Interpretability Tool (LIT)](https://pair-code.github.io/lit/).

## Features

- **Interactive Model Exploration**: Visualize attention patterns, embeddings, and predictions
- **Token Salience**: Gradient-based importance scores for input tokens
- **Embedding Projector**: UMAP/t-SNE visualization of token and sequence embeddings
- **Attention Visualization**: Multi-head attention patterns across all layers
- **Top-K Predictions**: Token predictions with probabilities at each position
- **Built-in Datasets**: IOI, Induction, and custom dataset support

## Installation

Install TransformerLens with LIT support:

```bash
pip install transformer-lens[lit]
```

Or install LIT separately:

```bash
pip install transformer-lens lit-nlp
```

## Quick Start

### In a Jupyter/Colab Notebook

```python
from transformer_lens import HookedTransformer
from transformer_lens.lit import (
    HookedTransformerLIT,
    HookedTransformerLITConfig,
    SimpleTextDataset,
    LITWidget,
)

# Load model
model = HookedTransformer.from_pretrained("gpt2-small")

# Create LIT wrapper
config = HookedTransformerLITConfig(
    max_seq_length=256,
    compute_gradients=True,
    output_attention=True,
)
lit_model = HookedTransformerLIT(model, config=config)

# Create dataset
dataset = SimpleTextDataset.from_strings([
    "The capital of France is Paris.",
    "The quick brown fox jumps over the lazy dog.",
])

# Launch widget
widget = LITWidget(
    models={"gpt2": lit_model},
    datasets={"examples": dataset},
)
widget.render()
```

### As a Standalone Server

```python
from transformer_lens import HookedTransformer
from transformer_lens.lit import (
    HookedTransformerLIT,
    SimpleTextDataset,
    serve,
)

model = HookedTransformer.from_pretrained("gpt2-small")
lit_model = HookedTransformerLIT(model)

serve(
    models={"gpt2": lit_model},
    datasets={"examples": SimpleTextDataset.from_strings(["Hello world!"])},
    port=5432,
)
# Navigate to http://localhost:5432
```

## Configuration

### HookedTransformerLITConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_seq_length` | int | 512 | Maximum sequence length |
| `batch_size` | int | 8 | Batch size for inference |
| `top_k` | int | 10 | Number of top predictions per position |
| `compute_gradients` | bool | True | Enable gradient-based salience |
| `output_attention` | bool | True | Output attention patterns |
| `output_embeddings` | bool | True | Output embeddings |
| `output_all_layers` | bool | False | Output all layer embeddings |
| `prepend_bos` | bool | True | Prepend BOS token |
| `device` | str | None | Device (auto-detected if None) |

## Datasets

### SimpleTextDataset

Basic dataset for text inputs:

```python
dataset = SimpleTextDataset.from_strings([
    "Text 1",
    "Text 2",
])
```

### PromptCompletionDataset

For prompt-completion pairs:

```python
dataset = PromptCompletionDataset.from_pairs([
    ("The capital of France is", " Paris"),
    ("2 + 2 =", " 4"),
])
```

### IOIDataset

Indirect Object Identification task:

```python
dataset = IOIDataset.generate(n_examples=100, seed=42)
# Generates: "When Mary and John went to the store, John gave a book to"
# Answer: "Mary"
```

### InductionDataset

For analyzing induction heads:

```python
dataset = InductionDataset.generate_simple(n_examples=50)
# Generates patterns like: "A B C D A B" -> expects "C"
```

## Output Fields

The wrapper produces these outputs for LIT:

| Field | Type | Description |
|-------|------|-------------|
| `tokens` | Tokens | Tokenized input |
| `probabilities` | MulticlassPreds | Token probabilities |
| `top_k_tokens` | TokenTopKPreds | Top-K predictions per position |
| `cls_embedding` | Embeddings | First token embedding |
| `mean_embedding` | Embeddings | Mean-pooled embedding |
| `input_embeddings` | TokenEmbeddings | Per-token embeddings [seq, emb_dim] |
| `layer_N/attention` | AttentionHeads | Attention per layer |
| `layer_N/embeddings` | TokenEmbeddings | Embeddings per layer |
| `grad_l2` | TokenGradients | Per-token gradients [seq, emb_dim] |
| `grad_dot_input` | TokenGradients | Per-token gradients [seq, emb_dim] |

Note: LIT internally computes L2 norms and dot products from the gradient arrays.

## LIT Features Supported

- **Attention Visualization**: See which tokens attend to which
- **Embedding Projector**: UMAP/t-SNE of embeddings
- **Token Salience**: Gradient-based importance
- **Prediction Analysis**: Top-K token predictions
- **Data Table**: Browse and filter examples
- **Counterfactual Generation**: Test modified inputs

## Examples

### Visualizing Attention Patterns

```python
import matplotlib.pyplot as plt

# Get outputs
outputs = list(lit_model.predict([{"text": "Hello world!"}]))[0]

# Plot attention for layer 5, head 0
attn = outputs["layer_5/attention"]  # [heads, q, k]
plt.imshow(attn[0])
plt.colorbar()
plt.show()
```

### Computing Token Salience

LIT's built-in salience modules (GradientNorm, GradientDotInput) compute 
token importance from the raw gradient arrays automatically.

```python
import numpy as np

outputs = list(lit_model.predict([{"text": "The answer is 42"}]))[0]

# Raw gradients [seq_len, emb_dim]
gradients = outputs["grad_l2"]

# Compute L2 norm manually (LIT does this internally)
salience = np.linalg.norm(gradients, axis=1)
for token, score in zip(outputs["tokens"], salience):
    print(f"{token}: {score:.4f}")
```

## API Reference

### Classes

- `HookedTransformerLIT`: Main LIT model wrapper
- `HookedTransformerLITBatched`: Batched inference wrapper
- `HookedTransformerLITConfig`: Configuration dataclass
- `SimpleTextDataset`: Basic text dataset
- `PromptCompletionDataset`: Prompt-completion pairs
- `IOIDataset`: IOI benchmark dataset
- `InductionDataset`: Induction head dataset
- `LITWidget`: Jupyter/Colab widget

### Functions

- `serve(models, datasets, ...)`: Start LIT server
- `check_lit_installed()`: Check if LIT is available
- `wrap_for_lit(examples)`: Wrap examples for LIT

## Troubleshooting

### LIT not found

```
ImportError: lit-nlp is not installed
```

Install LIT: `pip install lit-nlp`

### CUDA out of memory

Reduce batch size or use a smaller model:

```python
config = HookedTransformerLITConfig(batch_size=1)
```

### Widget not rendering

Make sure you're in a Jupyter/Colab environment with JavaScript enabled.

## Contributing

See the main TransformerLens [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## References

- [LIT Paper](https://arxiv.org/abs/2008.05122): Tenney et al., "The Language Interpretability Tool" (EMNLP 2020)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- [LIT Documentation](https://pair-code.github.io/lit/)
- [IOI Paper](https://arxiv.org/abs/2211.00593): Wang et al., "Interpretability in the Wild" (2022)
