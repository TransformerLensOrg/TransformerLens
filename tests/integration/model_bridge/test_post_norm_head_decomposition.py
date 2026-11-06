"""Head- and neuron-level residual decompositions on post-sublayer-norm architectures.

Gemma 2/3 (sandwich norms) and OLMo 2/3 (post-norms only) normalize each sublayer
output before adding it to the residual stream, so a head or neuron contributes
``ln_post(x_i)`` rather than ``x_i``. With the cached scale the norm is linear,
which makes the per-component contributions exact:

    sum(get_full_resid_decomposition(layer)) == resid_pre[layer]

``test_residual_decomposition_identities.py`` covers the block-level aliases on
these architectures (#1648); this file covers the per-head and per-neuron stacks
built on top of them.
"""

import copy
from typing import Any

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    Gemma2Config,
    Gemma3TextConfig,
    LlamaConfig,
    Olmo2Config,
)

from transformer_lens.model_bridge.sources import build_bridge_from_module

_TINY = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=128,
    vocab_size=128,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)

CASES = [
    pytest.param("Gemma2ForCausalLM", Gemma2Config, _TINY, id="gemma2"),
    pytest.param("Gemma3ForCausalLM", Gemma3TextConfig, _TINY, id="gemma3"),
    pytest.param("Olmo2ForCausalLM", Olmo2Config, _TINY, id="olmo2"),
    # Pre-norm control: no post-sublayer norm, so the stacks are unchanged.
    pytest.param("LlamaForCausalLM", LlamaConfig, _TINY, id="llama"),
]

_CACHE: dict[tuple[str, bool, torch.dtype, bool], tuple[Any, Any]] = {}


def _run(
    hf_architecture: str,
    config_cls: Any,
    config_kwargs: dict[str, Any],
    attn_result: bool = False,
    dtype: torch.dtype = torch.float32,
    o_bias: bool = False,
):
    """Boot a seeded tiny bridge and cache one forward."""
    key = (hf_architecture, attn_result, dtype, o_bias)
    if key not in _CACHE:
        config = config_cls(**config_kwargs, **({"attention_bias": True} if o_bias else {}))
        config._attn_implementation = "eager"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            hf = AutoModelForCausalLM.from_config(config).eval()
            if o_bias:
                # HF zero-inits biases, which would leave the bias correction unexercised.
                with torch.no_grad():
                    for layer in hf.model.layers:
                        layer.self_attn.o_proj.bias.normal_(std=0.5)
            hf = hf.to(dtype)
            bridge = build_bridge_from_module(
                hf,
                hf_architecture,
                hf_config=copy.deepcopy(config),
                tokenizer=None,
                device="cpu",
                dtype=dtype,
            ).eval()
            if attn_result:
                bridge.set_use_attn_result(True)
            torch.manual_seed(0)
            tokens = torch.randint(3, config_kwargs["vocab_size"], (2, 10))
            with torch.no_grad():
                _, cache = bridge.run_with_cache(tokens)
        _CACHE[key] = (bridge, cache)
    return _CACHE[key]


def _target(cache, n_layers: int, layer: int, mlp_input: bool) -> torch.Tensor:
    if layer == n_layers:
        return cache[f"blocks.{n_layers - 1}.hook_resid_post"]
    return cache[f"blocks.{layer}.hook_resid_{'mid' if mlp_input else 'pre'}"]


@pytest.mark.parametrize("hf_architecture,config_cls,config_kwargs", CASES)
@pytest.mark.parametrize("expand_neurons", [False, True])
def test_full_decomposition_sums_to_residual_stream(
    hf_architecture: str, config_cls: Any, config_kwargs: dict[str, Any], expand_neurons: bool
) -> None:
    bridge, cache = _run(hf_architecture, config_cls, config_kwargs)
    n_layers = bridge.cfg.n_layers
    for layer in range(1, n_layers + 1):
        for mlp_input in (False, True) if layer < n_layers else (False,):
            full = cache.get_full_resid_decomposition(
                layer=layer, mlp_input=mlp_input, expand_neurons=expand_neurons
            )
            torch.testing.assert_close(
                full.sum(0), _target(cache, n_layers, layer, mlp_input), atol=2e-5, rtol=1e-5
            )


@pytest.mark.parametrize("hf_architecture,config_cls,config_kwargs", CASES)
def test_cached_head_results_give_the_same_decomposition(
    hf_architecture: str, config_cls: Any, config_kwargs: dict[str, Any]
) -> None:
    """Head results cached in the forward pass (use_attn_result) are the pre-norm module
    outputs too, so the stacks must map them through the post-norm the same way."""
    _, computed = _run(hf_architecture, config_cls, config_kwargs)
    _, forward = _run(hf_architecture, config_cls, config_kwargs, attn_result=True)
    torch.testing.assert_close(
        forward.stack_head_results(), computed.stack_head_results(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("hf_architecture,config_cls,config_kwargs", CASES)
def test_projected_ln_decomposition_matches_unprojected(
    hf_architecture: str, config_cls: Any, config_kwargs: dict[str, Any]
) -> None:
    """The analytic apply_ln + projection path folds the post-norm into W_out like the dense one."""
    bridge, cache = _run(hf_architecture, config_cls, config_kwargs)
    direction = torch.linspace(-1.0, 1.0, bridge.cfg.d_model)[:, None]
    for layer in range(1, bridge.cfg.n_layers + 1):
        if layer < bridge.cfg.n_layers and bridge.blocks[layer]._modules.get("ln1") is None:
            continue  # post-norm-only blocks have no mid-layer pre-norm to apply
        projected = cache.get_full_resid_decomposition(
            layer=layer, apply_ln=True, project_output_onto=direction
        )
        dense = cache.apply_ln_to_stack(cache.get_full_resid_decomposition(layer=layer), layer)
        torch.testing.assert_close(projected, dense @ direction, atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "hf_architecture,config_cls",
    [
        pytest.param("Gemma2ForCausalLM", Gemma2Config, id="gemma2"),
        pytest.param("Olmo2ForCausalLM", Olmo2Config, id="olmo2"),
    ],
)
def test_output_bias_is_mapped_through_post_norm(hf_architecture: str, config_cls: Any) -> None:
    """A post-norm block adds norm(b_O), not b_O, so the bias row needs the per-position fix."""
    bridge, cache = _run(hf_architecture, config_cls, _TINY, o_bias=True)
    assert bridge.blocks[0].attn.o.bias.abs().max() > 0.1
    n_layers = bridge.cfg.n_layers
    for layer in range(1, n_layers + 1):
        for mlp_input in (False, True) if layer < n_layers else (False,):
            full = cache.get_full_resid_decomposition(layer=layer, mlp_input=mlp_input)
            torch.testing.assert_close(
                full.sum(0), _target(cache, n_layers, layer, mlp_input), atol=2e-5, rtol=1e-5
            )


def test_half_precision_stacks_keep_model_dtype() -> None:
    bridge, cache = _run("Gemma2ForCausalLM", Gemma2Config, _TINY, dtype=torch.bfloat16)
    assert cache.stack_head_results().dtype == torch.bfloat16
    assert cache.get_neuron_results(0).dtype == torch.bfloat16
    n_layers = bridge.cfg.n_layers
    target = _target(cache, n_layers, n_layers, False).float()
    full = cache.get_full_resid_decomposition(layer=n_layers, expand_neurons=True)
    torch.testing.assert_close(
        full.float().sum(0), target, atol=2e-2 * target.abs().max().item(), rtol=0
    )


def test_filtered_cache_names_the_missing_post_norm_scale() -> None:
    bridge, _ = _run("Gemma2ForCausalLM", Gemma2Config, _TINY)
    tokens = torch.randint(
        3, _TINY["vocab_size"], (1, 6), generator=torch.Generator().manual_seed(0)
    )
    with torch.no_grad():
        _, cache = bridge.run_with_cache(
            tokens, names_filter=lambda n: n.endswith(("hook_z", "hook_resid_post", "hook_embed"))
        )
    with pytest.raises(KeyError, match=r"blocks\.0\.ln1_post\.hook_scale.*names_filter"):
        cache.stack_head_results()
