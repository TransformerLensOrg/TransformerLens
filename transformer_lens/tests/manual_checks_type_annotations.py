from typeguard.importhook import install_import_hook

install_import_hook("transformer_lens")

from transformer_lens import HookedTransformer
from torchtyping import TensorType as TT, patch_typeguard

patch_typeguard()

MODEL = "gpt2"
model = HookedTransformer.from_pretrained(MODEL)

prompt = "Hello World!"
tokens = model.to_tokens(prompt, prepend_bos=False)
logits_tokens = model(tokens)
logits_text: TT[1, "n_tokens", "d_vocab"] = model(prompt, prepend_bos=False)

# n.b. that i used this file to see if my type annotations were working- they were! i occasionally
# changed one of the sizes and saw that the type checker caught it.
