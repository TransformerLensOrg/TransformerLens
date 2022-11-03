# Adapted from [EasyTransformer_Demo.ipynb]. Useful for testing that all the typing mechanisms work
# out.

# %%

import torch as t
from easy_transformer import EasyTransformer, EasyTransformerConfig
from torchtyping import TensorType as TT, patch_typeguard

patch_typeguard()

DEVICE = "cuda" if t.cuda.is_available() else "cpu"
MODEL = "gpt2"

# %%
model = EasyTransformer.from_pretrained(MODEL)
model.to(DEVICE)

# %%

prompt = "Hello World!"
tokens = model.to_tokens(prompt, prepend_bos=False)
logits_tokens = model(tokens)
logits_text: TT[1, "n_tokens", "d_vocab"] = model(prompt, prepend_bos=False)

# %%

logits_text.shape
# %%
