from typeguard.importhook import install_import_hook

install_import_hook("easy_transformer")

from easy_transformer import EasyTransformer
from torchtyping import TensorType as TT, patch_typeguard
import torch

patch_typeguard()

MODEL = "gpt2"
model = EasyTransformer.from_pretrained(MODEL)

prompt = "Hello World!"
tokens = model.to_tokens(prompt, prepend_bos=False)
logits_tokens = model(tokens)
logits_text: TT[1, "n_tokens", "d_vocab"] = model(prompt, prepend_bos=False)

embed_or_first_layer = lambda name: (name[:6] != "blocks" or name[:8] == "blocks.0")

class Counter:
    def __init__(self):
        self.count = 0

    def inc(self, *args):
        self.count += 1

c = Counter()

random_tokens = torch.randint(1000, 10000, (4, 50))

logits = model.run_with_hooks(
    random_tokens, fwd_hooks=[(embed_or_first_layer, c.inc)]
)

assert c.count == 1

model.add_perma_hook(random_tokens, fwd_hooks=[(embed_or_first_layer, c.inc)])

model.run_with_hooks(
    prompt,
    fwd_hooks=[],
)

assert c.count == 2


