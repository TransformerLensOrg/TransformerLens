"""`get_caching_hooks(remove_batch_dim=True, pos_slice=...)` slices the position axis.

Every case compares against `run_with_cache` with the same arguments, and the
layout-specific cases also against the unsliced cache indexed by hand, so a slice on
a neighbouring axis (d_model, heads, channels) or a skipped slice fails on values.
"""

from __future__ import annotations

import copy

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge

# Token ids [b, p], a resid stream [b, p, d], a head-split tensor [b, p, h, d_head] and
# an attention map [b, h, dest, src].
NAMES = [
    "embed.hook_in",
    "blocks.0.hook_out",
    "blocks.0.attn.o.hook_in",
    "blocks.0.attn.hook_pattern",
]


def _bridge() -> TransformerBridge:
    cfg = TransformerBridgeConfig(
        d_model=32,
        d_head=16,
        n_heads=2,
        n_layers=1,
        n_ctx=8,
        d_vocab=16,
        d_mlp=64,
        act_fn="gelu",
        normalization_type="LN",
        seed=0,
    )
    return TransformerBridge.boot_native(cfg)


def _run_caching_hooks(bridge, tokens, names=NAMES, forward_kwargs=None, **kwargs) -> dict:
    cache, fwd_hooks, _ = bridge.get_caching_hooks(names_filter=names, **kwargs)
    with torch.no_grad(), bridge.hooks(fwd_hooks=fwd_hooks):
        bridge.forward(tokens, **(forward_kwargs or {}))
    return cache


def _assert_matches_run_with_cache(bridge, tokens, names, forward_kwargs=None, **kwargs):
    with torch.no_grad():
        _, ref = bridge.run_with_cache(
            tokens, names_filter=names, **kwargs, **(forward_kwargs or {})
        )
    cache = _run_caching_hooks(bridge, tokens, names, forward_kwargs, **kwargs)
    for name in names:
        assert cache[name].shape == ref[name].shape, f"{name} shape differs from run_with_cache"
        torch.testing.assert_close(cache[name], ref[name])
    return cache


@pytest.mark.parametrize("pos_slice", [-1, 2, (1, 4), [0, 2, 5]])
def test_remove_batch_dim_with_pos_slice_matches_run_with_cache(pos_slice):
    bridge = _bridge()
    torch.manual_seed(0)
    tokens = torch.randint(0, bridge.cfg.d_vocab, (1, 6))
    _assert_matches_run_with_cache(
        bridge, tokens, NAMES, remove_batch_dim=True, pos_slice=pos_slice
    )


def test_remove_batch_dim_with_int_pos_slice_shapes():
    bridge = _bridge()
    tokens = torch.randint(0, bridge.cfg.d_vocab, (1, 6))
    n_heads, d_head, d_model = bridge.cfg.n_heads, bridge.cfg.d_head, bridge.cfg.d_model

    cache = _run_caching_hooks(bridge, tokens, remove_batch_dim=True, pos_slice=-1)

    assert cache["embed.hook_in"].shape == (1,)
    assert cache["blocks.0.hook_out"].shape == (1, d_model)
    assert cache["blocks.0.attn.o.hook_in"].shape == (1, n_heads, d_head)
    assert cache["blocks.0.attn.hook_pattern"].shape == (n_heads, 1, 6)


def test_pos_slice_without_remove_batch_dim_is_unchanged():
    bridge = _bridge()
    tokens = torch.randint(0, bridge.cfg.d_vocab, (1, 6))
    _assert_matches_run_with_cache(bridge, tokens, NAMES, pos_slice=(1, 4))


def test_remove_batch_dim_with_pos_slice_trims_gradients_on_the_same_axis():
    """`incl_bwd` `_grad` entries are sliced like the activations they belong to."""
    bridge = _bridge()
    torch.manual_seed(0)
    tokens = torch.randint(0, bridge.cfg.d_vocab, (1, 6))
    names = ["blocks.0.hook_out", "blocks.0.attn.o.hook_in", "blocks.0.attn.hook_pattern"]
    kwargs = dict(names_filter=names, remove_batch_dim=True, pos_slice=(1, 4))

    _, ref = bridge.run_with_cache(tokens, return_type="loss", incl_bwd=True, **kwargs)
    cache, fwd_hooks, bwd_hooks = bridge.get_caching_hooks(incl_bwd=True, **kwargs)
    with bridge.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
        bridge.forward(tokens, return_type="loss").backward()

    for name in names:
        grad_name = f"{name}_grad"
        assert grad_name in cache, f"missing {grad_name}; cached: {sorted(cache)}"
        assert cache[grad_name].shape == cache[name].shape, grad_name
        assert cache[grad_name].shape == ref[grad_name].shape, f"{grad_name} differs"
        torch.testing.assert_close(cache[grad_name], ref[grad_name], rtol=1e-3, atol=1e-5)


# --- Hooks whose position axis is not dim 1 in the batched layout ---------------------


def _tiny_gemma3_bridge() -> TransformerBridge:
    """Gemma-3 applies q_norm/k_norm after the head split, on [batch, heads, pos, d_head]."""
    from transformers import Gemma3TextConfig
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_from_module,
    )

    cfg = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        sliding_window=8,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    hf = Gemma3ForCausalLM(cfg).eval()
    return build_bridge_from_module(
        hf, "Gemma3ForCausalLM", hf_config=copy.deepcopy(cfg), tokenizer=None, device="cpu"
    ).eval()


def _tiny_mamba_bridge() -> TransformerBridge:
    """Mamba-1's conv1d and eager-scan hooks are channel-first: [batch, channels, pos, ...]."""
    from transformers import AutoModelForCausalLM
    from transformers.models.mamba import MambaConfig

    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_config_from_hf,
    )
    from transformer_lens.model_bridge.supported_architectures.mamba import (
        MambaArchitectureAdapter,
    )

    torch.manual_seed(0)
    cfg = MambaConfig(
        vocab_size=64,
        hidden_size=16,
        state_size=8,
        num_hidden_layers=1,
        expand=2,
        time_step_rank=4,
        conv_kernel=4,
    )
    cfg.architectures = ["MambaForCausalLM"]
    hf = AutoModelForCausalLM.from_config(cfg).eval()
    bridge_cfg = build_bridge_config_from_hf(hf.config, "MambaForCausalLM", "tiny", torch.float32)
    return TransformerBridge(hf, MambaArchitectureAdapter(bridge_cfg), tokenizer=object())


def _assert_slices_position(bridge, tokens, names, pos_dim, forward_kwargs=None):
    """Both APIs slice `pos_dim` of the batched layout and agree with manual indexing."""
    with torch.no_grad():
        _, full = bridge.run_with_cache(tokens, names_filter=names, **(forward_kwargs or {}))
    for pos_slice, index in [(-1, [5]), ((1, 4), [1, 2, 3])]:
        cache = _assert_matches_run_with_cache(
            bridge, tokens, names, forward_kwargs, remove_batch_dim=True, pos_slice=pos_slice
        )
        for name in names:
            expected = full[name].index_select(pos_dim, torch.tensor(index))[0]
            assert cache[name].shape == expected.shape, f"{name} sliced on the wrong axis"
            torch.testing.assert_close(cache[name], expected)


def test_post_reshape_qk_norm_hooks_slice_position():
    bridge = _tiny_gemma3_bridge()
    assert bridge.blocks[0].attn._qk_norm_phase == "post_reshape"
    tokens = torch.randint(3, 64, (1, 6))
    names = [
        "blocks.0.attn.hook_q_normed",
        "blocks.0.attn.hook_k_normed",
        "blocks.0.attn.q_norm.hook_in",
        "blocks.0.attn.k_norm.hook_out",
    ]
    _assert_slices_position(bridge, tokens, names, pos_dim=2)


def test_mamba_channel_first_hooks_slice_position():
    bridge = _tiny_mamba_bridge()
    bridge.blocks[0].mixer.eager_scan = True
    tokens = torch.randint(0, 64, (1, 6))
    names = [
        "blocks.0.mixer.conv1d.hook_in",
        "blocks.0.mixer.hook_ssm_write",
        "blocks.0.mixer.hook_ssm_state",
    ]
    _assert_slices_position(bridge, tokens, names, pos_dim=2, forward_kwargs={"use_cache": False})
