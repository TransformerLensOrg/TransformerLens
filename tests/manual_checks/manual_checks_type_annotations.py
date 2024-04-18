import torch
from jaxtyping import Float

from transformer_lens import HookedTransformer

MODEL = "gpt2"
model = HookedTransformer.from_pretrained(MODEL)

prompt = "Hello World!"
tokens = model.to_tokens(prompt, prepend_bos=False)
logits_tokens = model(tokens)
logits_text: Float[torch.Tensor, "1 n_tokens d_vocab"] = model(prompt, prepend_bos=False)

# n.b. that i used this file to see if my type annotations were working- they were! i occasionally
# changed one of the sizes and saw that the type checker caught it.
