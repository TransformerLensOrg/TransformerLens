from transformer_lens import HookedTransformer
import torch
from typing import cast

print("Loading model...")
model = HookedTransformer.from_pretrained("gpt2-small")

print("Running model...")
logits, activations = model.run_with_cache("Hello World")

# runtime type + repr
print("type(logits) =", type(logits))
print("repr(logits)[:200] =", repr(logits)[:200])

# safe checks and printing shape
if isinstance(logits, torch.Tensor):
    print("logits.shape (runtime):", logits.shape)
else:
    # cast for type-checkers (see next section)
    logits = cast(torch.Tensor, logits)
    print("After cast, logits.shape:", getattr(logits, "shape", None))
