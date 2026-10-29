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

_CACHE: dict[tuple[str, bool], tuple[Any, Any]] = {}


def _run(hf_architecture: str, config_cls: Any, config_kwargs: dict[str, Any], attn_result=False):
    """Boot a seeded tiny bridge and cache one forward."""
    key = (hf_architecture, attn_result)
    if key not in _CACHE:
        config = config_cls(**config_kwargs)
        config._attn_implementation = "eager"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            hf = AutoModelForCausalLM.from_config(config).eval()
            bridge = build_bridge_from_module(
                hf, hf_architecture, hf_config=copy.deepcopy(config), tokenizer=None, device="cpu"
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
