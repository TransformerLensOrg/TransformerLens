import gc
import os

import pytest
import torch
from transformers import AutoConfig

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES
from transformer_lens.utils import clear_huggingface_cache

TINY_STORIES_MODEL_NAMES = [
    name for name in OFFICIAL_MODEL_NAMES if name.startswith("roneneldan/TinyStories")
]

text = "Hello world!"
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
    "tiny-stories-33M": 12.203617095947266,
}
""" 
# Code to regenerate loss store
new_loss_store = {}
for name in loss_store.keys():
    model = HookedTransformer.from_pretrained(name, device='cuda')
    loss = model(text, return_type="loss")
    new_loss_store[name] = loss.item()
print(new_store)
"""


@pytest.mark.parametrize("name,expected_loss", list(loss_store.items()))
def test_model(name, expected_loss):
    # Runs the model on short text and checks if the loss is as expected
    model = HookedTransformer.from_pretrained(name)
    loss = model(text, return_type="loss")
    assert (loss.item() - expected_loss) < 4e-5
    del model
    gc.collect()

    if "GITHUB_ACTIONS" in os.environ:
        clear_huggingface_cache()


def test_othello_gpt():
    # like test model but Othello GPT has a weird input format
    # so we need to test it separately

    model = HookedTransformer.from_pretrained("othello-gpt")
    sample_input = torch.tensor(
        [
            [
                # fmt: off
                20, 19, 18, 10, 2, 1, 27, 3, 41, 42, 34, 12, 4, 40, 11, 29, 43, 13, 48, 56, 33,
                39, 22, 44, 24, 5, 46, 6, 32, 36, 51, 58, 52, 60, 21, 53, 26, 31, 37, 9, 25, 38,
                23, 50, 45, 17, 47, 28, 35, 30, 54, 16, 59, 49, 57, 14, 15, 55, 7,
                # fmt: on
            ]
        ]
    )
    loss = model(sample_input, return_type="loss")
    expected_loss = 1.9079375267028809
    assert (loss.item() - expected_loss) < 4e-5


@pytest.mark.parametrize(
    "name,expected_loss",
    [(k, v) for k, v in loss_store.items() if k in ["solu-1l", "redwood_attn_2l"]],
)
def test_from_pretrained_no_processing(name, expected_loss):
    # Checks if manually overriding the boolean flags in from_pretrained
    # is equivalent to using from_pretrained_no_processing

    model_ref = HookedTransformer.from_pretrained_no_processing(name)
    model_ref_config = model_ref.cfg
    reff_loss = model_ref(text, return_type="loss")
    del model_ref
    model_override = HookedTransformer.from_pretrained(
        name,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        refactor_factored_attn_matrices=False,
    )
    assert model_ref_config == model_override.cfg

    # Do the converse check, i.e. check that overriding boolean flags in
    # from_pretrained_no_processing is equivalent to using from_pretrained
    model_ref = HookedTransformer.from_pretrained(name)
    model_ref_config = model_ref.cfg
    reff_loss = model_ref(text, return_type="loss")
    del model_ref
    model_override = HookedTransformer.from_pretrained_no_processing(
        name,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        refactor_factored_attn_matrices=False,
    )
    assert model_ref_config == model_override.cfg

    # also check losses
    print(reff_loss.item())
    assert (reff_loss.item() - expected_loss) < 4e-5


def test_from_pretrained_dtype():
    """Check that the parameter `torch_dtype` works"""
    model = HookedTransformer.from_pretrained("solu-1l", torch_dtype=torch.bfloat16)
    assert model.W_K.dtype == torch.bfloat16


def test_future_attention_scores_neg_inf():
    model = HookedTransformer.from_pretrained("pythia")
    hook_name = str(get_act_name("attn_scores", 0))
    # fmt: off
    sample_input = torch.tensor([[1073, 1089, 1098, 1084, 1097, 1098, 1098, 1098, 1098, 1098]])
    # fmt: on
    _, cache = model.run_with_cache(
        sample_input,
        names_filter=lambda name: name == hook_name,
    )
    assert (cache[hook_name] == (-torch.inf)).int().sum().item() == cache[
        hook_name
    ].shape[0] * cache[hook_name].shape[1] * (
        cache[hook_name].shape[2] * (cache[hook_name].shape[2] - 1)
    ) // 2, (
        "The attention paid to future tokens should be -inf in all cases and attention paid to current tokens should not be -inf",
        (cache[hook_name] == (-torch.inf)).int().sum().item(),
        cache[hook_name][0, -1],
    )


def test_from_pretrained_revision():
    """
    Check that the from_pretrained parameter `revision` (= git version) works
    """

    model_name = "roneneldan/TinyStories-1M"  # this model has no revisions
    _ = HookedTransformer.from_pretrained(model_name, revision="main")

    try:
        _ = HookedTransformer.from_pretrained(
            model_name, revision="inexistent_branch_name"
        )
    except:
        pass
    else:
        raise AssertionError("Should have raised an error")


@torch.no_grad()
def test_pos_embed_hook():
    """
    Checks that pos embed hooks:
    - do not permanently change the pos embed
    - can be used to alter the pos embed for a specific batch element
    """
    model = HookedTransformer.from_pretrained("solu-1l")
    initial_W_pos = model.W_pos.detach().clone()

    def remove_pos_embed(z, hook):
        z[:] = 0.0
        return z

    _ = model.run_with_hooks(
        "Hello, world", fwd_hooks=[("hook_pos_embed", remove_pos_embed)]
    )

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
        fwd_hooks=[("hook_pos_embed", edit_pos_embed)],
    )


def test_all_tinystories_models_exist():
    for model in TINY_STORIES_MODEL_NAMES:
        try:
            AutoConfig.from_pretrained(model)
        except OSError:
            pytest.fail(
                f"Could not download model '{model}' from Huggingface."
                " Maybe the name was changed or the model has been removed."
            )
