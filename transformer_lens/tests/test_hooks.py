import pytest
from typeguard.importhook import install_import_hook

install_import_hook("transformer_lens")

from transformer_lens import HookedTransformer
from torchtyping import TensorType as TT, patch_typeguard

patch_typeguard()

MODEL = "gpt2"

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

def test_conditional_hooks():
    """Test that it's only possible to add certain hooks when certain conditions are met"""

    def identity_hook(z, hook):
        return z

    model.reset_hooks()
    model.set_use_attn_result(False)
    with pytest.raises(AssertionError):
        model.add_hook("blocks.0.attn.hook_result", identity_hook)


    model.reset_hooks()
    model.set_use_split_qkv_input(False)
    with pytest.raises(AssertionError):
        model.add_hook("blocks.0.attn.hook_q_input", identity_hook)

    # now when we set these conditions to true, should be no errors!

    model.reset_hooks()
    model.set_use_attn_result(True)
    model.add_hook("blocks.0.attn.hook_result", identity_hook)

    model.reset_hooks()
    model.set_use_split_qkv_input(True)
    model.add_hook("blocks.0.attn.hook_q_input", identity_hook)

def test_strided_hooks():
    """Test that certain hooks that expect strided tensors do have strided tensors"""

    model.reset_hooks()
    model.set_use_split_qkv_input(False)
    def no_split_strided_hook(z, hook):
        print(z.shape, z.stride())
        assert list(z.shape) == [1, 4, 768, 3], list(z.shape) # [batch, pos, d_model, qkv]
        assert list(z.stride())[-1] == 0, z.stride()[-1] # for GPT-2 small (which doesn't use shortformer style attention) Q, K and V inputs are the same
        assert 0 not in z.stride()[:-1], z.stride()[:-1] # no other dimensions should be strided
        raise Exception("Checked not strided hook")

    with pytest.raises(Exception) as exc_info:   
        model.run_with_hooks(prompt, fwd_hooks=[("blocks.0.attn.hook_attn_input", no_split_strided_hook)])
    assert str(exc_info.value) == "Checked not strided hook"

    model.reset_hooks()
    model.set_use_split_qkv_input(True)
    def split_strided_hook(z, hook):
        print(z.shape, z.stride())
        assert list(z.shape) == [1, 4, 12, 768], list(z.shape) # [batch, pos, head_index, d_model]
        assert 0 not in z.stride(), z.stride() # no dimensions should be strided
        raise Exception("Checked strided hook")

    with pytest.raises(Exception) as exc_info:
        model.run_with_hooks(prompt, fwd_hooks=[("blocks.0.attn.hook_q_input", split_strided_hook)])
    assert str(exc_info.value) == "Checked strided hook"