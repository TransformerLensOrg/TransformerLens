from typeguard.importhook import install_import_hook

install_import_hook("transformer_lens")

from transformer_lens import HookedTransformer
from torchtyping import TensorType as TT, patch_typeguard

patch_typeguard()

MODEL = "solu-1l"

prompt = "Hello World!"
model = HookedTransformer.from_pretrained(MODEL)
embed = lambda name: name == "hook_embed"

class Counter:
    def __init__(self):
        self.count = 0

    def inc(self, *args, **kwargs):
        self.count += 1

def test_hook_attaches_normally():
    c = Counter()
    _ = model.run_with_hooks(prompt, fwd_hooks=[(embed, c.inc)])
    assert all([len(hp.fwd_hooks) == 0 for _, hp in model.hook_dict.items()])
    assert c.count == 1
    model.remove_all_hook_fns(including_permanent=True)

def test_perma_hook_attaches_normally():
    c = Counter()
    model.add_perma_hook(embed, c.inc)
    assert len(model.hook_dict['hook_embed'].fwd_hooks) == 1
    model.run_with_hooks(prompt, fwd_hooks=[])
    assert len(model.hook_dict['hook_embed'].fwd_hooks) == 1
    assert c.count == 1
    model.remove_all_hook_fns(including_permanent=True)

def test_remove_hook():
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

