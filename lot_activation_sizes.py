from transformer_lens import HookedTransformer
import torch
import matplotlib.pyplot as plt
import numpy as np

model = HookedTransformer.from_pretrained("gpt2-small")
_, cache = model.run_with_cache("Hello world")

names = []
dim0 = []
dim1 = []
dim2 = []

for name, value in cache.items():
    if isinstance(value, torch.Tensor):
        shp = tuple(value.shape)
        names.append(name)
        # record up to three dims, use 1 if missing to avoid plotting gaps
        dim0.append(shp[0] if len(shp) > 0 else 1)
        dim1.append(shp[1] if len(shp) > 1 else 1)
        dim2.append(shp[2] if len(shp) > 2 else 1)
    else:
        # non-tensor entries get 0 dims
        names.append(name)
        dim0.append(0)
        dim1.append(0)
        dim2.append(0)

# Keep order stable; you may want to trim long lists for readability
MAX = 60
indices = list(range(min(len(names), MAX)))

x = np.arange(len(indices))

plt.figure(figsize=(12, 6))
plt.plot(x, [dim0[i] for i in indices], marker='o', label='dim0 (batch)')
plt.plot(x, [dim1[i] for i in indices], marker='o', label='dim1 (seq)')
plt.plot(x, [dim2[i] for i in indices], marker='o', label='dim2 (channels/heads/...)')

plt.xticks(x, [names[i] for i in indices], rotation=90, fontsize=8)
plt.xlabel("Activation name (truncated to first {})".format(len(indices)))
plt.ylabel("Dimension size")
plt.title("Activation tensor dimensions (first {})".format(len(indices)))
plt.legend()
plt.tight_layout()
plt.show()
