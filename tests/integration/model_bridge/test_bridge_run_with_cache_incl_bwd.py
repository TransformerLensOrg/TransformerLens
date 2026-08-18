"""Regression coverage for `TransformerBridge.run_with_cache(incl_bwd=True)`.

The bridge reimplements caching instead of inheriting `HookedRootModule`, so
`incl_bwd` used to fall through `**kwargs` into the HuggingFace forward, which
discards unknown kwargs — the caller asked for gradients and silently got none
(issue #1680). Gradients are cached under a `_grad` suffix, matching
`HookedRootModule.run_with_cache`.
"""

import pytest
import torch

NAMES = ["blocks.0.attn.hook_pattern", "blocks.1.attn.hook_z", "blocks.2.hook_resid_post"]
PROMPT = "The capital of France is Paris."


@pytest.fixture()
def bridge(distilgpt2_bridge):
    """Alias the session fixture for concise test signatures."""
    return distilgpt2_bridge


# Lives here, not conftest: single consumer, and a second resident distilgpt2
# is exactly what the golden fixtures exist to avoid.
@pytest.fixture(scope="module")
def distilgpt2_hooked_processed():
    """HookedTransformer distilgpt2 with default weight processing."""
    import gc

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained("distilgpt2", device="cpu")
    yield model
    del model
    gc.collect()


def test_incl_bwd_caches_gradients(bridge):
    """Every cached forward activation gets a matching `_grad` entry."""
    tokens = bridge.to_tokens(PROMPT)

    loss, cache = bridge.run_with_cache(
        tokens, return_type="loss", names_filter=NAMES, incl_bwd=True
    )

    assert loss.ndim == 0
    for name in NAMES:
        grad_name = f"{name}_grad"
        assert grad_name in cache, f"missing gradient for {name}; cached: {sorted(cache.keys())}"
        assert cache[grad_name].shape == cache[name].shape
        assert torch.isfinite(cache[grad_name]).all()
        assert cache[grad_name].abs().max() > 0, f"{grad_name} is all zeros"


def test_incl_bwd_default_caches_no_gradients(bridge):
    """The default run is unchanged — forward activations only."""
    _, cache = bridge.run_with_cache(
        bridge.to_tokens(PROMPT), return_type="loss", names_filter=NAMES
    )

    assert [key for key in cache.keys() if key.endswith("_grad")] == []


def test_incl_bwd_removes_its_backward_hooks(bridge):
    """Backward hooks are torn down like the forward ones, so the call leaks nothing.

    Counts the delta rather than absolute emptiness: the bridge fixture is session-scoped, so
    hooks other tests left behind are not this call's business.
    """

    def hook_counts():
        return {
            name: (len(hp.fwd_hooks), len(hp.bwd_hooks)) for name, hp in bridge.hook_dict.items()
        }

    before = hook_counts()
    bridge.run_with_cache(bridge.to_tokens(PROMPT), return_type="loss", incl_bwd=True)

    assert hook_counts() == before


def test_incl_bwd_with_dict_cache_and_remove_batch_dim(bridge):
    """Gradients follow the same dict/`remove_batch_dim` handling as activations."""
    tokens = bridge.to_tokens(PROMPT)
    kwargs = dict(return_type="loss", names_filter=NAMES, incl_bwd=True, return_cache_object=False)
    _, batched = bridge.run_with_cache(tokens, **kwargs)
    _, squeezed = bridge.run_with_cache(tokens, remove_batch_dim=True, **kwargs)

    for name in NAMES:
        for key in (name, f"{name}_grad"):
            assert squeezed[key].shape == batched[key].shape[1:], key
            torch.testing.assert_close(squeezed[key], batched[key][0])


def test_incl_bwd_requires_scalar_output(bridge):
    """Backward on logits is a shape error deep in autograd; catch it up front."""
    with pytest.raises(ValueError, match="scalar output"):
        bridge.run_with_cache(
            bridge.to_tokens(PROMPT), return_type="logits", names_filter=NAMES, incl_bwd=True
        )


def test_incl_bwd_rejects_disabled_grad_mode(bridge):
    """Notebooks often sit inside set_grad_enabled(False); say so instead of failing in autograd."""
    with torch.set_grad_enabled(False):
        with pytest.raises(ValueError, match="gradient tracking is off"):
            bridge.run_with_cache(
                bridge.to_tokens(PROMPT), return_type="loss", names_filter=NAMES, incl_bwd=True
            )


def test_incl_bwd_rejects_frozen_parameters(bridge):
    """A model with every parameter frozen has no grad_fn to differentiate."""
    frozen = [p for p in bridge.original_model.parameters() if p.requires_grad]
    for param in frozen:
        param.requires_grad_(False)
    try:
        with pytest.raises(ValueError, match="no grad_fn"):
            bridge.run_with_cache(
                bridge.to_tokens(PROMPT), return_type="loss", names_filter=NAMES, incl_bwd=True
            )
    finally:
        for param in frozen:
            param.requires_grad_(True)


def test_incl_bwd_rejects_stop_at_layer(bridge):
    """`stop_at_layer` returns an intermediate activation, not something to backward."""
    with pytest.raises(ValueError, match="stop_at_layer"):
        bridge.run_with_cache(
            bridge.to_tokens(PROMPT),
            return_type="loss",
            names_filter=NAMES,
            incl_bwd=True,
            stop_at_layer=2,
        )


def test_incl_bwd_gradients_reach_compatibility_aliases(distilgpt2_bridge_compat):
    """Compat mode aliases `_grad` entries too, not just forward activations."""
    _, cache = distilgpt2_bridge_compat.run_with_cache(
        distilgpt2_bridge_compat.to_tokens(PROMPT),
        return_type="loss",
        names_filter=NAMES,
        incl_bwd=True,
    )

    for name in NAMES:
        assert f"{name}_grad" in cache, f"missing {name}_grad; cached: {sorted(cache.keys())}"


def test_incl_bwd_gradients_match_hooked_transformer(
    distilgpt2_bridge_compat, distilgpt2_hooked_processed
):
    """Cached gradients agree with HookedTransformer's, not just in shape."""
    prompt_tokens = distilgpt2_hooked_processed.to_tokens(PROMPT)

    _, bridge_cache = distilgpt2_bridge_compat.run_with_cache(
        prompt_tokens, return_type="loss", names_filter=NAMES, incl_bwd=True
    )
    _, hooked_cache = distilgpt2_hooked_processed.run_with_cache(
        prompt_tokens, return_type="loss", names_filter=NAMES, incl_bwd=True
    )

    for name in NAMES:
        grad_name = f"{name}_grad"
        bridge_grad = bridge_cache[grad_name]
        hooked_grad = hooked_cache[grad_name]
        assert bridge_grad.shape == hooked_grad.shape
        scale = hooked_grad.abs().max()
        max_diff = (bridge_grad - hooked_grad).abs().max()
        assert max_diff < 1e-3 * max(scale, 1.0), (
            f"{grad_name} diverges from HookedTransformer: max diff {max_diff:.3e} "
            f"against gradient scale {scale:.3e}"
        )
