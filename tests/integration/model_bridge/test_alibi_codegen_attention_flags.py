"""Attention-hook flags on the ALiBi (Falcon, MPT) and CodeGen attention bridges.

These bridges override JointQKVAttentionBridge's forward, and the bridge
accepted use_split_qkv_input / use_attn_in / use_attn_result for them because
they inherit its support flags, but their forwards ignored the flags: the fork
hooks and hook_result never fired, so an edit there silently did nothing. Each
flag is checked against the same edit made on the Hugging Face model with plain
PyTorch hooks. Tiny programmatic configs, no weight downloads.
"""

import copy

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from transformer_lens.model_bridge.sources import build_bridge_from_module

_SHARED = dict(vocab_size=128, pad_token_id=0, bos_token_id=1, eos_token_id=2)

# (architecture, model_type, config, layer-0 path, its input norm, its attention, attention output proj)
CASES = {
    "falcon_alibi": (
        "FalconForCausalLM",
        "falcon",
        dict(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            alibi=True,
            new_decoder_architecture=False,
            parallel_attn=False,
            multi_query=False,
            bias=True,
            **_SHARED,
        ),
        ("transformer.h.0", "input_layernorm", "self_attention", "dense"),
    ),
    # One shared K/V head: the K and V forks carry n_key_value_heads copies.
    "falcon_alibi_mqa": (
        "FalconForCausalLM",
        "falcon",
        dict(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            alibi=True,
            new_decoder_architecture=False,
            parallel_attn=True,
            multi_query=True,
            bias=False,
            **_SHARED,
        ),
        ("transformer.h.0", "input_layernorm", "self_attention", "dense"),
    ),
    "mpt": (
        "MptForCausalLM",
        "mpt",
        dict(d_model=64, n_layers=2, n_heads=4, max_seq_len=64, **_SHARED),
        ("transformer.blocks.0", "norm_1", "attn", "out_proj"),
    ),
    "codegen": (
        "CodeGenForCausalLM",
        "codegen",
        dict(n_embd=64, n_layer=2, n_head=4, n_positions=64, rotary_dim=8, **_SHARED),
        ("transformer.h.0", "ln_1", "attn", "out_proj"),
    ),
}

HEAD = 2


@pytest.fixture(scope="module", params=sorted(CASES), ids=sorted(CASES))
def models(request):
    architecture, model_type, kwargs, paths = CASES[request.param]
    hf_config = AutoConfig.for_model(model_type, **kwargs)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(hf_config, attn_implementation="eager").eval()
    with torch.no_grad():
        # Non-trivial norm gains and biases.
        for param in model.parameters():
            if param.ndim == 1:
                param.add_(0.2 * torch.randn_like(param))
    reference = copy.deepcopy(model)
    bridge = build_bridge_from_module(model, architecture, hf_config=hf_config).eval()
    return bridge, reference, paths


@pytest.fixture(scope="module")
def tokens():
    return torch.randint(3, 128, (2, 7), generator=torch.Generator().manual_seed(1))


def _reference_logits(reference, paths, tokens, attn_pre_hook=None, out_proj_pre_hook=None):
    layer_path, norm_name, attn_name, out_proj_name = paths
    layer = reference.get_submodule(layer_path)
    handles = []
    captured = {}

    def capture_layer_input(module, args, kwargs):
        captured["resid"] = args[0] if args else kwargs["hidden_states"]

    handles.append(layer.register_forward_pre_hook(capture_layer_input, with_kwargs=True))
    if attn_pre_hook is not None:
        norm = getattr(layer, norm_name)

        def replace_attention_input(module, args, kwargs):
            new = attn_pre_hook(norm, captured["resid"])
            if args:
                return (new,) + args[1:], kwargs
            return args, dict(kwargs, hidden_states=new)

        attn = getattr(layer, attn_name)
        handles.append(attn.register_forward_pre_hook(replace_attention_input, with_kwargs=True))
    if out_proj_pre_hook is not None:
        out_proj = getattr(getattr(layer, attn_name), out_proj_name)
        handles.append(out_proj.register_forward_pre_hook(out_proj_pre_hook))
    try:
        with torch.no_grad():
            return reference(tokens).logits
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("flag", ["use_split_qkv_input", "use_attn_in"])
def test_attention_input_fork_matches_hf(models, tokens, flag):
    bridge, reference, paths = models
    delta = 0.5 * torch.randn(2, 7, bridge.cfg.d_model, generator=torch.Generator().manual_seed(2))
    names = (
        ["hook_q_input", "hook_k_input", "hook_v_input"]
        if flag == "use_split_qkv_input"
        else ["hook_attn_in"]
    )

    def add_delta(x, hook):
        return x + delta.unsqueeze(2)

    getattr(bridge, f"set_{flag}")(True)
    try:
        with torch.no_grad():
            plain, cache = bridge.run_with_cache(tokens)
            edited = bridge.run_with_hooks(
                tokens, fwd_hooks=[(f"blocks.0.attn.{name}", add_delta) for name in names]
            )
    finally:
        getattr(bridge, f"set_{flag}")(False)
    n_kv_heads = getattr(bridge.cfg, "n_key_value_heads", None) or bridge.cfg.n_heads
    for name in names:
        heads = n_kv_heads if name in ("hook_k_input", "hook_v_input") else bridge.cfg.n_heads
        assert cache[f"blocks.0.attn.{name}"].shape == (2, 7, heads, bridge.cfg.d_model)
    # Only attention reads the shifted residual, through its own norm.
    expected = _reference_logits(
        reference, paths, tokens, attn_pre_hook=lambda norm, resid: norm(resid + delta)
    )
    torch.testing.assert_close(plain, _reference_logits(reference, paths, tokens))
    assert (edited - plain).abs().max() > 1e-3
    torch.testing.assert_close(edited, expected)


def test_attn_result_matches_hf_head_ablation(models, tokens):
    bridge, reference, paths = models
    d_head = bridge.cfg.d_head

    def zero_head(x, hook):
        x = x.clone()
        x[:, :, HEAD] = 0.0
        return x

    bridge.set_use_attn_result(True)
    try:
        with torch.no_grad():
            plain, cache = bridge.run_with_cache(tokens)
            ablated = bridge.run_with_hooks(
                tokens, fwd_hooks=[("blocks.0.attn.hook_result", zero_head)]
            )
    finally:
        bridge.set_use_attn_result(False)
    result = cache["blocks.0.attn.hook_result"]
    assert result.shape == (2, 7, bridge.cfg.n_heads, bridge.cfg.d_model)
    out_proj = getattr(getattr(reference.get_submodule(paths[0]), paths[2]), paths[3])
    bias = out_proj.bias if out_proj.bias is not None else 0.0
    torch.testing.assert_close(result.sum(dim=2) + bias, cache["blocks.0.attn.hook_out"])

    def zero_head_input(module, args):
        z = args[0].clone()
        z[..., HEAD * d_head : (HEAD + 1) * d_head] = 0.0
        return (z,)

    expected = _reference_logits(reference, paths, tokens, out_proj_pre_hook=zero_head_input)
    torch.testing.assert_close(plain, _reference_logits(reference, paths, tokens))
    assert (ablated - plain).abs().max() > 1e-3
    torch.testing.assert_close(ablated, expected)
