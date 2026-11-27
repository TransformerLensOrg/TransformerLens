"""Flag-gated hooks on the ALiBi (Falcon, MPT, BLOOM) and CodeGen bridges.

The ALiBi and CodeGen attention bridges override JointQKVAttentionBridge's
forward, and the bridge accepted use_split_qkv_input / use_attn_in /
use_attn_result for them because they inherit its support flags, but their
forwards ignored the flags: the fork hooks and hook_result never fired, so an
edit there silently did nothing. BloomBlockBridge overrides BlockBridge's
forward without installing the block's capture hooks, so on BLOOM hook_mlp_in
never fired and the attention forks read ln1's output instead of its input.
Each flag is checked against the same edit made on the Hugging Face model with
plain PyTorch hooks. Tiny programmatic configs, no weight downloads.
"""

import copy

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from transformer_lens.model_bridge.sources import build_bridge_from_module

_SHARED = dict(vocab_size=128, pad_token_id=0, bos_token_id=1, eos_token_id=2)
_FALCON_ALIBI = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    alibi=True,
    new_decoder_architecture=False,
    **_SHARED,
)
_FALCON_MODULES = dict(
    layer="transformer.h.0", ln1="input_layernorm", attn="self_attention", out_proj="dense"
)

# (architecture, model_type, config, HF module names in layer 0; ln2 only on sequential blocks)
CASES = {
    "falcon_alibi": (
        "FalconForCausalLM",
        "falcon",
        dict(_FALCON_ALIBI, parallel_attn=False, multi_query=False, bias=True),
        dict(_FALCON_MODULES, ln2="post_attention_layernorm"),
    ),
    # One shared K/V head: the K and V forks carry n_key_value_heads copies.
    "falcon_alibi_mqa": (
        "FalconForCausalLM",
        "falcon",
        dict(_FALCON_ALIBI, parallel_attn=True, multi_query=True, bias=False),
        _FALCON_MODULES,
    ),
    "mpt": (
        "MptForCausalLM",
        "mpt",
        dict(d_model=64, n_layers=2, n_heads=4, max_seq_len=64, **_SHARED),
        dict(
            layer="transformer.blocks.0",
            ln1="norm_1",
            attn="attn",
            out_proj="out_proj",
            ln2="norm_2",
        ),
    ),
    "bloom": (
        "BloomForCausalLM",
        "bloom",
        dict(hidden_size=64, n_layer=2, n_head=4, **_SHARED),
        dict(
            layer="transformer.h.0",
            ln1="input_layernorm",
            attn="self_attention",
            out_proj="dense",
            ln2="post_attention_layernorm",
        ),
    ),
    "codegen": (
        "CodeGenForCausalLM",
        "codegen",
        dict(n_embd=64, n_layer=2, n_head=4, n_positions=64, rotary_dim=8, **_SHARED),
        dict(layer="transformer.h.0", ln1="ln_1", attn="attn", out_proj="out_proj"),
    ),
}
SEQUENTIAL = sorted(name for name, case in CASES.items() if "ln2" in case[3])

HEAD = 2


def _build(name):
    architecture, model_type, kwargs, modules = CASES[name]
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
    return bridge, reference, modules


@pytest.fixture(scope="module", params=sorted(CASES), ids=sorted(CASES))
def models(request):
    return _build(request.param)


@pytest.fixture(scope="module", params=SEQUENTIAL, ids=SEQUENTIAL)
def sequential_models(request):
    return _build(request.param)


@pytest.fixture(scope="module")
def tokens():
    return torch.randint(3, 128, (2, 7), generator=torch.Generator().manual_seed(1))


def _delta(bridge):
    return 0.5 * torch.randn(2, 7, bridge.cfg.d_model, generator=torch.Generator().manual_seed(2))


def _reference_logits(reference, modules, tokens, pre_hooks=()):
    """HF logits with each (submodule of layer 0, edit(layer, x, resid)) replacing that
    submodule's first input; resid is the layer's own input."""
    layer = reference.get_submodule(modules["layer"])
    handles = []
    captured = {}

    def capture_layer_input(module, args, kwargs):
        captured["resid"] = args[0] if args else kwargs["hidden_states"]

    handles.append(layer.register_forward_pre_hook(capture_layer_input, with_kwargs=True))
    for name, edit in pre_hooks:

        def replace_first_input(module, args, kwargs, edit=edit):
            x = args[0] if args else kwargs["hidden_states"]
            new = edit(layer, x, captured["resid"])
            if args:
                return (new,) + args[1:], kwargs
            return args, dict(kwargs, hidden_states=new)

        target = layer.get_submodule(name)
        handles.append(target.register_forward_pre_hook(replace_first_input, with_kwargs=True))
    try:
        with torch.no_grad():
            return reference(tokens).logits
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("flag", ["use_split_qkv_input", "use_attn_in"])
def test_attention_input_fork_matches_hf(models, tokens, flag):
    bridge, reference, modules = models
    delta = _delta(bridge)
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
    # Only attention reads the shifted residual, through its own pre-attention norm.
    shift_attention_input = (
        modules["attn"],
        lambda layer, x, resid: getattr(layer, modules["ln1"])(resid + delta),
    )
    expected = _reference_logits(reference, modules, tokens, [shift_attention_input])
    torch.testing.assert_close(plain, _reference_logits(reference, modules, tokens))
    assert (edited - plain).abs().max() > 1e-3
    torch.testing.assert_close(edited, expected)


def test_attn_result_matches_hf_head_ablation(models, tokens):
    bridge, reference, modules = models
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
    out_proj_name = f"{modules['attn']}.{modules['out_proj']}"
    out_proj = reference.get_submodule(modules["layer"]).get_submodule(out_proj_name)
    bias = out_proj.bias if out_proj.bias is not None else 0.0
    with torch.no_grad():
        projected = out_proj(cache["blocks.0.attn.hook_z"].flatten(-2))
    torch.testing.assert_close(result.sum(dim=2) + bias, projected)

    def zero_head_input(layer, z, resid):
        z = z.clone()
        z[..., HEAD * d_head : (HEAD + 1) * d_head] = 0.0
        return z

    expected = _reference_logits(reference, modules, tokens, [(out_proj_name, zero_head_input)])
    torch.testing.assert_close(plain, _reference_logits(reference, modules, tokens))
    assert (ablated - plain).abs().max() > 1e-3
    torch.testing.assert_close(ablated, expected)


def test_hook_mlp_in_matches_hf(sequential_models, tokens):
    bridge, reference, modules = sequential_models
    delta = _delta(bridge)
    bridge.set_use_hook_mlp_in(True)
    try:
        with torch.no_grad():
            plain, cache = bridge.run_with_cache(tokens)
            edited = bridge.run_with_hooks(
                tokens, fwd_hooks=[("blocks.0.hook_mlp_in", lambda x, hook: x + delta)]
            )
    finally:
        bridge.set_use_hook_mlp_in(False)
    # Pre-norm: the MLP branch entry is ln2's input, the residual after attention.
    torch.testing.assert_close(cache["blocks.0.hook_mlp_in"], cache["blocks.0.hook_resid_mid"])
    # Only the MLP branch reads the shift; the residual stream itself is untouched.
    shift_mlp_input = (modules["ln2"], lambda layer, x, resid: x + delta)
    expected = _reference_logits(reference, modules, tokens, [shift_mlp_input])
    assert (edited - plain).abs().max() > 1e-3
    torch.testing.assert_close(edited, expected)
