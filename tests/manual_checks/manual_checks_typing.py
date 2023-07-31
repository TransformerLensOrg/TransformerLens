# Adapted from [HookedTransformer_Demo.ipynb]. Useful for testing that all the typing mechanisms work
# out.

# %%

import torch as t
from jaxtyping import Float

from transformer_lens import HookedTransformer, utils

DEVICE = utils.get_device()
MODEL = "gpt2"

# %%
model = HookedTransformer.from_pretrained(MODEL)
model.to(DEVICE)

# %%

prompt = "Hello World!"
tokens = model.to_tokens(prompt, prepend_bos=False)
logits_tokens = model(tokens)
logits_text: Float[t.Tensor, "1 n_tokens d_vocab"] = model(prompt, prepend_bos=False)

# %%

logits_text.shape
# %%
