from typeguard.importhook import install_import_hook

install_import_hook("easy_transformer")

from easy_transformer import EasyTransformer
from torchtyping import TensorType as TT, patch_typeguard
import torch

patch_typeguard()

MODEL = "gpt2"

prompt = "Hello World!"
model = EasyTransformer.from_pretrained(MODEL)
embed = lambda name: name == "hook_embed"

class Counter:
    def __init__(self):
        self.count = 0

    def inc(self, *args, **kwargs):
        self.count += 1

def hook_attaches_normally_test():
    c = Counter()
    _ = model.run_with_hooks(prompt, fwd_hooks=[(embed, c.inc)])
    assert all([len(hp.fwd_hooks) == 0 for _, hp in model.hook_dict.items()])
    assert c.count == 1
    model.remove_all_hook_fns(including_permanent=True)

def perma_hook_attaches_normally_test():
    c = Counter()
    model.add_perma_hook(embed, c.inc)
    assert len(model.hook_dict['hook_embed'].fwd_hooks) == 1
    model.run_with_hooks(prompt, fwd_hooks=[])
    assert len(model.hook_dict['hook_embed'].fwd_hooks) == 1
    assert c.count == 1
    model.remove_all_hook_fns(including_permanent=True)

def remove_hook_test():
    c = Counter()
    model.add_perma_hook(embed, c.inc)
    assert len(model.hook_dict['hook_embed'].fwd_hooks) == 1 # 1 after adding
    model.remove_all_hook_fns()
    assert len(model.hook_dict['hook_embed'].fwd_hooks) == 1 # permanent not removed without flag
    model.remove_all_hook_fns(including_permanent=True)
    assert len(model.hook_dict['hook_embed'].fwd_hooks) == 0 # removed now
    model.run_with_hooks(prompt, fwd_hooks=[])
    assert c.count == 0
    model.remove_all_hook_fns(including_permanent=True)

hook_attaches_normally_test()
perma_hook_attaches_normally_test()
remove_hook_test()


