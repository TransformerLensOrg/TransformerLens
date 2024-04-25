import einops
import pytest
import torch

from transformer_lens import HookedSAE, HookedSAEConfig, HookedSAETransformer

MODEL = "solu-1l"
prompt = "Hello World!"


class Counter:
    def __init__(self):
        self.count = 0

    def inc(self, *args, **kwargs):
        self.count += 1


@pytest.fixture(scope="module")
def model():
    model = HookedSAETransformer.from_pretrained(MODEL)
    yield model
    model.reset_saes()


def get_sae_config(model, act_name):
    site_to_size = {
        "hook_z": model.cfg.d_head * model.cfg.n_heads,
        "hook_mlp_out": model.cfg.d_model,
        "hook_resid_pre": model.cfg.d_model,
        "hook_post": model.cfg.d_mlp,
    }
    site = act_name.split(".")[-1]
    d_in = site_to_size[site]
    return HookedSAEConfig(d_in=d_in, d_sae=d_in * 2, hook_name=act_name)


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_forward_reconstructs_input(model, act_name):
    """Verfiy that the HookedSAE returns an output with the same shape as the input activations."""
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)

    _, cache = model.run_with_cache(prompt, names_filter=act_name)
    x = cache[act_name]

    sae_output = hooked_sae(x)
    assert sae_output.shape == x.shape


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_run_with_cache(model, act_name):
    """Verifies that run_with_cache caches SAE activations"""
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)

    _, cache = model.run_with_cache(prompt, names_filter=act_name)
    x = cache[act_name]

    sae_output, cache = hooked_sae.run_with_cache(x)
    assert sae_output.shape == x.shape

    assert "hook_sae_input" in cache
    assert "hook_sae_acts_pre" in cache
    assert "hook_sae_acts_post" in cache
    assert "hook_sae_recons" in cache
    assert "hook_sae_output" in cache


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_run_with_hooks(model, act_name):
    """Verifies that run_with_hooks works with SAE activations"""
    c = Counter()
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)

    _, cache = model.run_with_cache(prompt, names_filter=act_name)
    x = cache[act_name]

    sae_hooks = [
        "hook_sae_input",
        "hook_sae_acts_pre",
        "hook_sae_acts_post",
        "hook_sae_recons",
        "hook_sae_output",
    ]

    sae_output = hooked_sae.run_with_hooks(
        x, fwd_hooks=[(sae_hook_name, c.inc) for sae_hook_name in sae_hooks]
    )
    assert sae_output.shape == x.shape

    assert c.count == len(sae_hooks)


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_error_term(model, act_name):
    """Verifies that that if we use error_terms, HookedSAE returns an output that is equal to the input activations."""
    sae_cfg = get_sae_config(model, act_name)
    sae_cfg.use_error_term = True
    hooked_sae = HookedSAE(sae_cfg)

    _, cache = model.run_with_cache(prompt, names_filter=act_name)
    x = cache[act_name]

    sae_output = hooked_sae(x)
    assert sae_output.shape == x.shape
    assert torch.allclose(sae_output, x, atol=1e-6)


# %%
@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_feature_grads_with_error_term(model, act_name):
    """Verifies that pytorch backward computes the correct feature gradients when using error_terms. Motivated by the need to compute feature gradients for attribution patching."""

    # Load SAE
    sae_cfg = get_sae_config(model, act_name)
    sae_cfg.use_error_term = True
    hooked_sae = HookedSAE(sae_cfg)

    # Get input activations
    _, cache = model.run_with_cache(prompt, names_filter=act_name)
    x = cache[act_name]

    # Cache gradients with respect to feature acts
    hooked_sae.reset_hooks()
    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    hooked_sae.add_hook("hook_sae_acts_post", backward_cache_hook, "bwd")
    hooked_sae.add_hook("hook_sae_output", backward_cache_hook, "bwd")

    sae_output = hooked_sae(x)
    assert torch.allclose(sae_output, x, atol=1e-6)
    value = sae_output.sum()
    value.backward()
    hooked_sae.reset_hooks()

    # Compute gradient analytically
    if act_name.endswith("hook_z"):
        reshaped_output_grad = einops.rearrange(
            grad_cache["hook_sae_output"], "... n_heads d_head -> ... (n_heads d_head)"
        )
        analytic_grad = reshaped_output_grad @ hooked_sae.W_dec.T
    else:
        analytic_grad = grad_cache["hook_sae_output"] @ hooked_sae.W_dec.T

    # Compare analytic gradient with pytorch computed gradient
    assert torch.allclose(grad_cache["hook_sae_acts_post"], analytic_grad, atol=1e-6)
