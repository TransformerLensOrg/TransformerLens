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
    model.add_sae(hooked_sae)
    assert len(model.acts_to_saes) == 1
    logits_with_saes = model(prompt)
    assert not torch.allclose(original_logits, logits_with_saes)
    model.reset_saes()


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_add_sae(model, act_name):
    """Verifies that add_sae correctly updates the model's acts_to_saes dictionary and replaces the HookPoint."""
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)
    model.add_sae(hooked_sae)
    assert len(model.acts_to_saes) == 1
    assert model.acts_to_saes[act_name] == hooked_sae
    assert get_deep_attr(model, act_name) == hooked_sae
    model.reset_saes()


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_add_sae_overwrites_prev_sae(model, act_name):
    """Verifies that add_sae correctly updates the model's acts_to_saes dictionary and replaces the HookPoint."""
    prev_sae_cfg = get_sae_config(model, act_name)
    prev_hooked_sae = HookedSAE(prev_sae_cfg)
    model.add_sae(prev_hooked_sae)
    assert len(model.acts_to_saes) == 1
    assert model.acts_to_saes[act_name] == prev_hooked_sae
    assert get_deep_attr(model, act_name) == prev_hooked_sae

    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)
    model.add_sae(hooked_sae)
    assert len(model.acts_to_saes) == 1
    assert model.acts_to_saes[act_name] == hooked_sae
    assert get_deep_attr(model, act_name) == hooked_sae
    model.reset_saes()


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_reset_sae_removes_sae_by_default(model, act_name):
    """Verifies that reset_sae correctly removes the SAE from the model's acts_to_saes dictionary and replaces the HookedSAE with a HookPoint."""
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)
    model.add_sae(hooked_sae)
    assert len(model.acts_to_saes) == 1
    assert model.acts_to_saes[act_name] == hooked_sae
    assert get_deep_attr(model, act_name) == hooked_sae
    model._reset_sae(act_name)
    assert len(model.acts_to_saes) == 0
    assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.reset_saes()


@pytest.mark.parametrize(
    "act_name",
    [
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_resid_pre",
    ],
)
def test_reset_sae_replaces_sae(model, act_name):
    """Verifies that reset_sae correctly removes the SAE from the model's acts_to_saes dictionary and replaces the HookedSAE with a HookPoint."""
    sae_cfg = get_sae_config(model, act_name)
    hooked_sae = HookedSAE(sae_cfg)

    prev_sae_cfg = get_sae_config(model, act_name)
    prev_sae = HookedSAE(prev_sae_cfg)

    model.add_sae(hooked_sae)
    assert len(model.acts_to_saes) == 1
    assert model.acts_to_saes[act_name] == hooked_sae
    assert get_deep_attr(model, act_name) == hooked_sae
    model._reset_sae(act_name, prev_sae)
    assert len(model.acts_to_saes) == 1
    assert get_deep_attr(model, act_name) == prev_sae
    model.reset_saes()


@pytest.mark.parametrize(
    "act_names",
    [
        ["blocks.0.attn.hook_z"],
        ["blocks.0.hook_mlp_out"],
        ["blocks.0.mlp.hook_post"],
        ["blocks.0.hook_resid_pre"],
        [
            "blocks.0.attn.hook_z",
            "blocks.0.hook_mlp_out",
            "blocks.0.mlp.hook_post",
            "blocks.0.hook_resid_pre",
        ],
    ],
)
def test_reset_saes_removes_all_saes_by_default(model, act_names):
    """Verifies that reset_saes correctly removes all SAEs from the model's acts_to_saes dictionary and replaces the HookedSAEs with HookPoints."""
    sae_cfgs = [get_sae_config(model, act_name) for act_name in act_names]
    hooked_saes = [HookedSAE(sae_cfg) for sae_cfg in sae_cfgs]
    for hooked_sae in hooked_saes:
        model.add_sae(hooked_sae)
    assert len(model.acts_to_saes) == len(act_names)
    for act_name, hooked_sae in zip(act_names, hooked_saes):
        assert model.acts_to_saes[act_name] == hooked_sae
        assert get_deep_attr(model, act_name) == hooked_sae
    model.reset_saes()
    assert len(model.acts_to_saes) == 0
    for act_name in act_names:
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.reset_saes()


@pytest.mark.parametrize(
    "act_names",
    [
        ["blocks.0.attn.hook_z"],
        ["blocks.0.hook_mlp_out"],
        ["blocks.0.mlp.hook_post"],
        ["blocks.0.hook_resid_pre"],
        [
            "blocks.0.attn.hook_z",
            "blocks.0.hook_mlp_out",
            "blocks.0.mlp.hook_post",
            "blocks.0.hook_resid_pre",
        ],
    ],
)
def test_reset_saes_replaces_saes(model, act_names):
    """Verifies that reset_saes correctly removes all SAEs from the model's acts_to_saes dictionary and replaces the HookedSAEs with HookPoints."""
    sae_cfgs = [get_sae_config(model, act_name) for act_name in act_names]
    hooked_saes = [HookedSAE(sae_cfg) for sae_cfg in sae_cfgs]
    for hooked_sae in hooked_saes:
        model.add_sae(hooked_sae)

    prev_sae_cfgs = [get_sae_config(model, act_name) for act_name in act_names]
    prev_hooked_saes = [HookedSAE(prev_sae_cfg) for prev_sae_cfg in prev_sae_cfgs]

    assert len(model.acts_to_saes) == len(act_names)
    for act_name, hooked_sae in zip(act_names, hooked_saes):
        assert model.acts_to_saes[act_name] == hooked_sae
        assert get_deep_attr(model, act_name) == hooked_sae
    model.reset_saes(act_names, prev_hooked_saes)
    assert len(model.acts_to_saes) == len(prev_hooked_saes)
    for act_name, prev_hooked_sae in zip(act_names, prev_hooked_saes):
        assert get_deep_attr(model, act_name) == prev_hooked_sae
    model.reset_saes()


@pytest.mark.parametrize(
    "act_names",
    [
        ["blocks.0.attn.hook_z"],
        ["blocks.0.hook_mlp_out"],
        ["blocks.0.mlp.hook_post"],
        ["blocks.0.hook_resid_pre"],
        [
            "blocks.0.attn.hook_z",
            "blocks.0.hook_mlp_out",
            "blocks.0.mlp.hook_post",
            "blocks.0.hook_resid_pre",
        ],
    ],
)
def test_saes_context_manager_removes_saes_after(model, act_names):
    """Verifies that the model.saes context manager successfully adds the SAEs for the specified activation name in the context manager and resets off after the context manager exits."""
    sae_cfgs = [get_sae_config(model, act_name) for act_name in act_names]
    hooked_saes = [HookedSAE(sae_cfg) for sae_cfg in sae_cfgs]
    assert len(model.acts_to_saes) == 0
    for act_name in act_names:
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    with model.saes(saes=hooked_saes):
        for act_name, hooked_sae in zip(act_names, hooked_saes):
            assert model.acts_to_saes[act_name] == hooked_sae
            assert isinstance(get_deep_attr(model, act_name), HookedSAE)
            assert get_deep_attr(model, act_name) == hooked_sae
        model.forward(prompt)
    assert len(model.acts_to_saes) == 0
    for act_name in act_names:
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.reset_saes()


@pytest.mark.parametrize(
    "act_names",
    [
        ["blocks.0.attn.hook_z"],
        ["blocks.0.hook_mlp_out"],
        ["blocks.0.mlp.hook_post"],
        ["blocks.0.hook_resid_pre"],
        [
            "blocks.0.attn.hook_z",
            "blocks.0.hook_mlp_out",
            "blocks.0.mlp.hook_post",
            "blocks.0.hook_resid_pre",
        ],
    ],
)
def test_saes_context_manager_restores_previous_sae_state(model, act_names):
    """Verifies that the model.saes context manager successfully adds the SAEs for the specified activation name in the context manager and resets off after the context manager exits."""
    # First add SAEs statefully
    prev_sae_cfgs = [get_sae_config(model, act_name) for act_name in act_names]
    prev_hooked_saes = [HookedSAE(sae_cfg) for sae_cfg in prev_sae_cfgs]
    for act_name, prev_hooked_sae in zip(act_names, prev_hooked_saes):
        model.add_sae(prev_hooked_sae)
        assert get_deep_attr(model, act_name) == prev_hooked_sae
    assert len(model.acts_to_saes) == len(prev_hooked_saes)

    # Now temporarily run with new SAEs
    sae_cfgs = [get_sae_config(model, act_name) for act_name in act_names]
    hooked_saes = [HookedSAE(sae_cfg) for sae_cfg in sae_cfgs]
    with model.saes(saes=hooked_saes):
        for act_name, hooked_sae in zip(act_names, hooked_saes):
            assert model.acts_to_saes[act_name] == hooked_sae
            assert isinstance(get_deep_attr(model, act_name), HookedSAE)
            assert get_deep_attr(model, act_name) == hooked_sae
        model.forward(prompt)

    # Check that the previously attached SAEs have been restored
    assert len(model.acts_to_saes) == len(prev_hooked_saes)
    for act_name, prev_hooked_sae in zip(act_names, prev_hooked_saes):
        assert isinstance(get_deep_attr(model, act_name), HookedSAE)
        assert get_deep_attr(model, act_name) == prev_hooked_sae
    model.reset_saes()


@pytest.mark.parametrize(
    "act_names",
    [
        ["blocks.0.attn.hook_z"],
        ["blocks.0.hook_mlp_out"],
        ["blocks.0.mlp.hook_post"],
        ["blocks.0.hook_resid_pre"],
        [
            "blocks.0.attn.hook_z",
            "blocks.0.hook_mlp_out",
            "blocks.0.mlp.hook_post",
            "blocks.0.hook_resid_pre",
        ],
    ],
)
def test_saes_context_manager_run_with_cache(model, act_names):
    """Verifies that the model.run_with_cache method works correctly in the context manager."""
    sae_cfgs = [get_sae_config(model, act_name) for act_name in act_names]
    hooked_saes = [HookedSAE(sae_cfg) for sae_cfg in sae_cfgs]
    assert len(model.acts_to_saes) == 0
    for act_name in act_names:
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    with model.saes(saes=hooked_saes):
        for act_name, hooked_sae in zip(act_names, hooked_saes):
            assert model.acts_to_saes[act_name] == hooked_sae
            assert isinstance(get_deep_attr(model, act_name), HookedSAE)
            assert get_deep_attr(model, act_name) == hooked_sae
        model.run_with_cache(prompt)
    assert len(model.acts_to_saes) == 0
    for act_name in act_names:
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.reset_saes()


@pytest.mark.parametrize(
    "act_names",
    [
        ["blocks.0.attn.hook_z"],
        ["blocks.0.hook_mlp_out"],
        ["blocks.0.mlp.hook_post"],
        ["blocks.0.hook_resid_pre"],
        [
            "blocks.0.attn.hook_z",
            "blocks.0.hook_mlp_out",
            "blocks.0.mlp.hook_post",
            "blocks.0.hook_resid_pre",
        ],
    ],
)
def test_run_with_saes(model, act_names, original_logits):
    """Verifies that the model.run_with_saes method works correctly. The logits with SAEs should be different from the original logits, but the SAE should be removed immediately after the forward pass."""
    sae_cfgs = [get_sae_config(model, act_name) for act_name in act_names]
    hooked_saes = [HookedSAE(sae_cfg) for sae_cfg in sae_cfgs]
    assert len(model.acts_to_saes) == 0
    logits_with_saes = model.run_with_saes(prompt, saes=hooked_saes)
    assert not torch.allclose(logits_with_saes, original_logits)
    assert len(model.acts_to_saes) == 0
    for act_name in act_names:
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.reset_saes()


@pytest.mark.parametrize(
    "act_names",
    [
        ["blocks.0.attn.hook_z"],
        ["blocks.0.hook_mlp_out"],
        ["blocks.0.mlp.hook_post"],
        ["blocks.0.hook_resid_pre"],
        [
            "blocks.0.attn.hook_z",
            "blocks.0.hook_mlp_out",
            "blocks.0.mlp.hook_post",
            "blocks.0.hook_resid_pre",
        ],
    ],
)
def test_run_with_cache(model, act_names, original_logits):
    """Verifies that the model.run_with_cache method works correctly. The logits with SAEs should be different from the original logits and the cache should contain SAE activations for the attached SAE."""
    sae_cfgs = [get_sae_config(model, act_name) for act_name in act_names]
    hooked_saes = [HookedSAE(sae_cfg) for sae_cfg in sae_cfgs]
    for hooked_sae in hooked_saes:
        model.add_sae(hooked_sae)
    assert len(model.acts_to_saes) == len(hooked_saes)
    logits_with_saes, cache = model.run_with_cache(prompt)
    assert not torch.allclose(logits_with_saes, original_logits)
    assert isinstance(cache, ActivationCache)
    for act_name, hooked_sae in zip(act_names, hooked_saes):
        assert act_name + ".hook_sae_acts_post" in cache
        assert isinstance(get_deep_attr(model, act_name), HookedSAE)
        assert get_deep_attr(model, act_name) == hooked_sae
    model.reset_saes()


@pytest.mark.parametrize(
    "act_names",
    [
        ["blocks.0.attn.hook_z"],
        ["blocks.0.hook_mlp_out"],
        ["blocks.0.mlp.hook_post"],
        ["blocks.0.hook_resid_pre"],
        [
            "blocks.0.attn.hook_z",
            "blocks.0.hook_mlp_out",
            "blocks.0.mlp.hook_post",
            "blocks.0.hook_resid_pre",
        ],
    ],
)
def test_run_with_cache_with_saes(model, act_names, original_logits):
    """Verifies that the model.run_with_cache_with_saes method works correctly. The logits with SAEs should be different from the original logits and the cache should contain SAE activations for the attached SAE."""
    sae_cfgs = [get_sae_config(model, act_name) for act_name in act_names]
    hooked_saes = [HookedSAE(sae_cfg) for sae_cfg in sae_cfgs]
    logits_with_saes, cache = model.run_with_cache_with_saes(prompt, saes=hooked_saes)
    assert not torch.allclose(logits_with_saes, original_logits)
    assert isinstance(cache, ActivationCache)

    assert len(model.acts_to_saes) == 0
    for act_name, hooked_sae in zip(act_names, hooked_saes):
        assert act_name + ".hook_sae_acts_post" in cache
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.reset_saes()


@pytest.mark.parametrize(
    "act_names",
    [
        ["blocks.0.attn.hook_z"],
        ["blocks.0.hook_mlp_out"],
        ["blocks.0.mlp.hook_post"],
        ["blocks.0.hook_resid_pre"],
        [
            "blocks.0.attn.hook_z",
            "blocks.0.hook_mlp_out",
            "blocks.0.mlp.hook_post",
            "blocks.0.hook_resid_pre",
        ],
    ],
)
def test_run_with_hooks(model, act_names, original_logits):
    """Verifies that the model.run_with_hooks method works correctly when SAEs are attached. The count should be incremented by 1 when the hooked SAE is called, and the SAE should stay attached after the forward pass"""
    c = Counter()
    sae_cfgs = [get_sae_config(model, act_name) for act_name in act_names]
    hooked_saes = [HookedSAE(sae_cfg) for sae_cfg in sae_cfgs]

    for hooked_sae in hooked_saes:
        model.add_sae(hooked_sae)

    logits_with_saes = model.run_with_hooks(
        prompt, fwd_hooks=[(act_name + ".hook_sae_acts_post", c.inc) for act_name in act_names]
    )
    assert not torch.allclose(logits_with_saes, original_logits)

    for act_name, hooked_sae in zip(act_names, hooked_saes):
        assert isinstance(get_deep_attr(model, act_name), HookedSAE)
        assert get_deep_attr(model, act_name) == hooked_sae
    assert c.count == len(act_names)
    model.reset_saes()
    model.remove_all_hook_fns(including_permanent=True)


@pytest.mark.parametrize(
    "act_names",
    [
        ["blocks.0.attn.hook_z"],
        ["blocks.0.hook_mlp_out"],
        ["blocks.0.mlp.hook_post"],
        ["blocks.0.hook_resid_pre"],
        [
            "blocks.0.attn.hook_z",
            "blocks.0.hook_mlp_out",
            "blocks.0.mlp.hook_post",
            "blocks.0.hook_resid_pre",
        ],
    ],
)
def test_run_with_hooks_with_saes(model, act_names, original_logits):
    """Verifies that the model.run_with_hooks_with_saes method works correctly when SAEs are attached. The count should be incremented by 1 when the hooked SAE is called, but the SAE should be removed immediately after the forward pass."""
    c = Counter()
    sae_cfgs = [get_sae_config(model, act_name) for act_name in act_names]
    hooked_saes = [HookedSAE(sae_cfg) for sae_cfg in sae_cfgs]

    logits_with_saes = model.run_with_hooks_with_saes(
        prompt,
        saes=hooked_saes,
        fwd_hooks=[(act_name + ".hook_sae_acts_post", c.inc) for act_name in act_names],
    )
    assert not torch.allclose(logits_with_saes, original_logits)
    assert c.count == len(act_names)

    assert len(model.acts_to_saes) == 0
    for act_name in act_names:
        assert isinstance(get_deep_attr(model, act_name), HookPoint)
    model.reset_saes()
    model.remove_all_hook_fns(including_permanent=True)
