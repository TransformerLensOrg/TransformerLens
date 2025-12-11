from transformer_lens import HookedTransformer
import torch

model = HookedTransformer.from_pretrained("gpt2-small")
logits, cache = model.run_with_cache("Hello world")

print("type(logits) =", type(logits))
if isinstance(logits, torch.Tensor):
    print("logits:", logits.shape)
else:
    print("logits is not a tensor!")

for name, value in cache.items():
    if isinstance(value, torch.Tensor):
        print(f"{name:40s} -> {tuple(value.shape)}")
    else:
        print(f"{name:40s} -> NON-TENSOR type: {type(value)}")

print(logits.shape)  # type: ignore[attr-defined]
