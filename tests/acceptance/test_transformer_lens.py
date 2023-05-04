from jaxtyping import install_import_hook
hook = install_import_hook("transformer_lens", ("typeguard", "typechecked"))

import pytest
import torch

from transformer_lens import HookedTransformer

model_names = [
    "attn-only-demo",
    "gpt2-small",
    "opt-125m",
    "gpt-neo-125M",
    "stanford-gpt2-small-a",
    "solu-4l-old",
    "solu-6l",
    "attn-only-3l",
    "pythia",
    "gelu-2l",
]
text = "Hello world!"
""" 
# Code to regenerate loss store
store = {}
for name in model_names:
    model = HookedTransformer.from_pretrained(name, device='cuda')
    loss = model(text,return_type="loss")
    store[name] = loss.item()
print(store)
"""
loss_store = {
    "attn-only-demo": 5.701841354370117,
    "gpt2-small": 5.331855773925781,
    "opt-125m": 6.159054279327393,
    "gpt-neo-125M": 4.900552272796631,
    "stanford-gpt2-small-a": 5.652035713195801,
    "solu-4l-old": 5.6021833419799805,
    "solu-6l": 5.7042999267578125,
    "attn-only-3l": 5.747507095336914,
    "pythia": 4.659344673156738,
    "gelu-2l": 6.501802444458008,
    "redwood_attn_2l": 10.530948638916016,
    "solu-1l": 5.256411552429199,
}
no_processing = {
    "solu-1l": loss_store["solu-1l"],
    "redwood_attn_2l": loss_store["redwood_attn_2l"],
}


@pytest.mark.parametrize("name,expected_loss", list(loss_store.items()))
def test_model(name, expected_loss):
    # Runs the model on short text and checks if the loss is as expected
    model = HookedTransformer.from_pretrained(name)
    loss = model(text, return_type="loss")
    assert (loss.item() - expected_loss) < 4e-5


@pytest.mark.parametrize("name,expected_loss", list(no_processing.items()))
def test_from_pretrained_no_processing(name, expected_loss):
    # Checks if manually overriding the boolean flags in from_pretrained
    # is equivalent to using from_pretrained_no_processing

    model_ref = HookedTransformer.from_pretrained_no_processing(name)
    model_override = HookedTransformer.from_pretrained(name, fold_ln=False, center_writing_weights=False, center_unembed=False, refactor_factored_attn_matrices=False)
    assert model_ref.cfg == model_override.cfg

    # also check losses
    loss = model_ref(text, return_type="loss")
    print(loss.item())
    assert (loss.item() - expected_loss) < 4e-5

@torch.no_grad()
def test_pos_embed_hook():
    """
    Checks that pos embed hooks:
    - do not permanently change the pos embed
    - can be used to alter the pos embed for a specific batch element
    """
    model = HookedTransformer.from_pretrained("gpt2-small")
    initial_W_pos = model.W_pos.detach().clone()

    def remove_pos_embed(z, hook):
        z[:] = 0.0
        return z
    
    _ = model.run_with_hooks(
        "Hello, world",
        fwd_hooks=[("hook_pos_embed", remove_pos_embed)])

    # Check that pos embed has not been permanently changed
    assert (model.W_pos == initial_W_pos).all()
    
    def edit_pos_embed(z, hook):
        sequence_length = z.shape[1]
        z[1, :] = 0.0
        # Check that the second batch element is zeroed
        assert (z[1, :] == 0.0).all()
        # Check that the first batch element is unchanged
        assert (z[0, :] == initial_W_pos[:sequence_length]).all()
        return z

    _ = model.run_with_hooks(
        ["Hello, world", "Goodbye, world"],
        fwd_hooks=[("hook_pos_embed", edit_pos_embed)])

hook.uninstall()