import pytest
import torch

from transformer_lens import (
    HookedSAE,
    HookedSAEConfig,
    HookedSAETransformer,
    HookedTransformer,
)
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint  # Hooking utilities
from transformer_lens.HookedSAETransformer import get_deep_attr

MODEL = "solu-1l"
prompt = "Hello World!"


class Counter:
    def __init__(self):
        self.count = 0

    def inc(self, *args, **kwargs):
        self.count += 1


@pytest.fixture(scope="module")
def original_logits():
    original_model = HookedTransformer.from_pretrained(MODEL)
    return original_model(prompt)


@pytest.fixture(scope="module")
def model():
    model = HookedSAETransformer.from_pretrained(MODEL)
    yield model
    model.remove_all_saes()


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


def test_model_with_no_saes_matches_original_model(model, original_logits):
    """Verifies that HookedSAETransformer behaves like a normal HookedTransformer model when no SAEs are attached."""
    assert len(model.acts_to_saes) == 0
    logits = model(prompt)
    assert torch.allclose(original_logits, logits)


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_model_with_saes_does_not_match_original_model(model, act_name, original_logits):
    """Verifies that the attached (and turned on) SAEs actually affect the models output logits"""
    assert len(model.acts_to_saes) == 0
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)
    model.attach_sae(hooked_sae)
    assert len(model.acts_to_saes) == 1
    logits_with_saes = model(prompt)
    assert not torch.allclose(original_logits, logits_with_saes)
    model.remove_all_saes()


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_attach_sae(model, act_name):
    """Verifies that attach_sae correctly updates the model's acts_to_saes dictionary."""
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)
    model.attach_sae(hooked_sae)
    assert len(model.acts_to_saes) == 1
    assert model.acts_to_saes[act_name] == hooked_sae
    model.remove_all_saes()


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_attach_sae_turns_on_sae_by_default(model, act_name):
    """Verifies that model.attach_sae turns on that SAE by default"""
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)
    assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.attach_sae(hooked_sae)
    assert isinstance(get_deep_attr(model, act_name), HookedSAE)
    assert len(model.acts_to_saes) == 1
    assert model.acts_to_saes[act_name] == hooked_sae
    model.remove_all_saes()


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_turn_saes_on(model, act_name):
    """Verifies that model.turn_saes_on(act_names) successfully turns on the SAEs for the specified activation name"""
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)
    model.attach_sae(hooked_sae, turn_on=False)
    assert len(model.acts_to_saes) == 1
    assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.turn_saes_on(act_name)
    assert isinstance(get_deep_attr(model, act_name), HookedSAE)
    model.remove_all_saes()


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_turn_saes_off(model, act_name):
    """Verifies that model.turn_saes_off(act_names) successfully turns off the SAEs for the specified activation name"""
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)
    model.attach_sae(hooked_sae)
    assert len(model.acts_to_saes) == 1
    assert isinstance(get_deep_attr(model, act_name), HookedSAE)
    model.turn_saes_off(act_name)
    assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.remove_all_saes()


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_saes_context_manager(model, act_name):
    """Verifies that the model.saes context manager successfully turns on the SAEs for the specified activation name in the context manager and turns them off after the context manager exits."""
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)
    model.attach_sae(hooked_sae, turn_on=False)
    assert len(model.acts_to_saes) == 1
    assert isinstance(get_deep_attr(model, act_name), HookPoint)
    with model.saes(act_names=act_name):
        assert isinstance(get_deep_attr(model, act_name), HookedSAE)
        model.forward(prompt)
    assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.remove_all_saes()


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_turn_saes_off_in_context_manager(model, act_name):
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)
    model.attach_sae(hooked_sae, turn_on=False)
    assert len(model.acts_to_saes) == 1
    assert isinstance(get_deep_attr(model, act_name), HookPoint)
    with model.saes(act_names=act_name):
        assert isinstance(get_deep_attr(model, act_name), HookedSAE)
        model.turn_saes_off(act_name)
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.remove_all_saes()


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_saes_context_manager_run_with_cache(model, act_name):
    """Verifies that the model.run_with_cache method works correctly in the context manager."""
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)
    model.attach_sae(hooked_sae, turn_on=False)
    assert len(model.acts_to_saes) == 1
    assert isinstance(get_deep_attr(model, act_name), HookPoint)
    with model.saes(act_names=act_name):
        assert isinstance(get_deep_attr(model, act_name), HookedSAE)
        model.run_with_cache(prompt)
    assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.remove_all_saes()


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_run_with_saes(model, act_name, original_logits):
    """Verifies that the model.run_with_saes method works correctly. The logits with SAEs should be different from the original logits, but the SAE should be removed immediately after the forward pass."""
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)
    model.attach_sae(hooked_sae, turn_on=False)
    assert len(model.acts_to_saes) == 1
    assert isinstance(get_deep_attr(model, act_name), HookPoint)
    logits_with_saes = model.run_with_saes(prompt, act_names=act_name)
    assert not torch.allclose(logits_with_saes, original_logits)
    assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.remove_all_saes()


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_run_with_cache(model, act_name, original_logits):
    """Verifies that the model.run_with_cache method works correctly. The logits with SAEs should be different from the original logits and the cache should contain SAE activations for the attached SAE."""
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)
    model.attach_sae(hooked_sae)
    assert len(model.acts_to_saes) == 1
    assert isinstance(get_deep_attr(model, act_name), HookedSAE)
    logits_with_saes, cache = model.run_with_cache(prompt)
    assert not torch.allclose(logits_with_saes, original_logits)
    assert isinstance(cache, ActivationCache)
    assert act_name + ".hook_sae_acts_post" in cache
    assert isinstance(get_deep_attr(model, act_name), HookedSAE)
    model.remove_all_saes()


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_run_with_cache_with_saes(model, act_name, original_logits):
    """Verifies that the model.run_with_cache_with_saes method works correctly. The logits with SAEs should be different from the original logits and the cache should contain SAE activations for the attached SAE."""
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)
    model.attach_sae(hooked_sae)
    assert len(model.acts_to_saes) == 1
    assert isinstance(get_deep_attr(model, act_name), HookedSAE)
    logits_with_saes, cache = model.run_with_cache_with_saes(prompt, act_names=act_name)
    assert not torch.allclose(logits_with_saes, original_logits)
    assert isinstance(cache, ActivationCache)
    assert act_name + ".hook_sae_acts_post" in cache
    assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.remove_all_saes()


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_run_with_hooks(model, act_name, original_logits):
    """Verifies that the model.run_with_hooks method works correctly when SAEs are attached. The count should be incremented by 1 when the hooked SAE is called, and the SAE should stay attached after the forward pass"""
    c = Counter()
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)
    model.attach_sae(hooked_sae)
    assert len(model.acts_to_saes) == 1
    assert isinstance(get_deep_attr(model, act_name), HookedSAE)
    logits_with_saes = model.run_with_hooks(
        prompt, fwd_hooks=[(act_name + ".hook_sae_acts_post", c.inc)]
    )
    assert isinstance(get_deep_attr(model, act_name), HookedSAE)
    assert not torch.allclose(logits_with_saes, original_logits)
    assert isinstance(get_deep_attr(model, act_name), HookedSAE)
    assert c.count == 1
    model.remove_all_saes()
    model.remove_all_hook_fns(including_permanent=True)


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_run_with_hooks_with_saes(model, act_name, original_logits):
    """Verifies that the model.run_with_hooks_with_saes method works correctly when SAEs are attached. The count should be incremented by 1 when the hooked SAE is called, but the SAE should be removed immediately after the forward pass."""
    c = Counter()
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)
    model.attach_sae(hooked_sae, turn_on=False)
    assert len(model.acts_to_saes) == 1
    assert isinstance(get_deep_attr(model, act_name), HookPoint)
    logits_with_saes = model.run_with_hooks_with_saes(
        prompt,
        act_names=act_name,
        fwd_hooks=[(act_name + ".hook_sae_acts_post", c.inc)],
    )
    assert not torch.allclose(logits_with_saes, original_logits)
    assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.remove_all_saes()
    model.remove_all_hook_fns(including_permanent=True)
