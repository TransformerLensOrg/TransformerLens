"""Tests for MuseGlimmerArchitectureAdapter on a tiny random config."""

import copy

import pytest
import torch

pytest.importorskip("transformers.models.muse_glimmer")

from transformers import AutoModelForImageTextToText
from transformers.models.muse_glimmer import MuseGlimmerConfig

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.supported_architectures.muse_glimmer import (
    MuseGlimmerArchitectureAdapter,
)

ARCH = "MuseGlimmerForConditionalGeneration"
N_LAYERS = 4
TOKENS = torch.tensor([[5, 17, 29, 3, 11, 42, 7, 23], [8, 9, 10, 11, 12, 13, 14, 15]])


class _Tok:
    pass


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    cfg = MuseGlimmerConfig(
        text_config=dict(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=N_LAYERS,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            sliding_window=4,
        ),
        vision_config=dict(
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=2,
            pos_emb_height=4,
            pos_emb_width=4,
        ),
        out_hidden_size=128,
        projector_hidden_size=32,
    )
    cfg.architectures = [ARCH]
    hf = AutoModelForImageTextToText.from_config(cfg, attn_implementation="eager")
    hf = hf.to(torch.float32).eval()
    reference = copy.deepcopy(hf)
    bridge_cfg = build_bridge_config_from_hf(hf.config, ARCH, "muse-glimmer-tiny", torch.float32)
    bridge = TransformerBridge(hf, MuseGlimmerArchitectureAdapter(bridge_cfg), tokenizer=_Tok())
    return bridge, reference


def test_forward_matches_hf(models):
    bridge, reference = models
    with torch.no_grad():
        bridge_logits = bridge(TOKENS)
        hf_logits = reference(input_ids=TOKENS).logits
    assert torch.allclose(bridge_logits, hf_logits, atol=1e-4, rtol=0)


def test_run_with_cache_hooks(models):
    bridge, reference = models
    with torch.no_grad():
        _, cache = bridge.run_with_cache(TOKENS)
        hf_attn = reference(input_ids=TOKENS, output_attentions=True).attentions
    batch, seq = TOKENS.shape
    for i in range(N_LAYERS):
        for name in ("hook_resid_pre", "hook_attn_out", "hook_mlp_out", "hook_resid_post"):
            assert cache[f"blocks.{i}.{name}"].shape == (batch, seq, 64)
        assert cache[f"blocks.{i}.attn.hook_z"].shape == (batch, seq, 4, 16)
        torch.testing.assert_close(cache[f"blocks.{i}.attn.hook_pattern"], hf_attn[i])


def test_vision_projector_hook_out_is_the_merged_embedding(models):
    """HF scatters ``perception_emb_norm``'s output into the text embeddings.

    The projector bridge therefore wraps that module rather than the raw
    ``vision_projection`` Linear, so ``vision_projector.hook_out`` is the value
    HF actually merges (matching the other multimodal adapters).
    """
    bridge, _ = models
    projector = bridge.vision_projector
    assert projector.name == "model.perception_emb_norm"
    assert type(projector.original_component).__name__ == "MuseGlimmerRMSNorm"


def _hf_norm_scale(norm: torch.nn.Module, d_model: int) -> torch.Tensor:
    """Effective per-channel scale of an HF RMSNorm, measured from its output."""
    x = torch.randn(8, d_model)
    x_hat = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)
    with torch.no_grad():
        return (norm(x) / x_hat).mean(0)


@pytest.fixture(scope="module")
def perturbed_models():
    """Bridge + HF reference with non-identity norms (HF zero-inits the block norms)."""
    torch.manual_seed(1)
    cfg = MuseGlimmerConfig(
        text_config=dict(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            sliding_window=4,
            post_norm_eps=1e-3,
        ),
        vision_config=dict(
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=2,
            pos_emb_height=4,
            pos_emb_width=4,
        ),
        out_hidden_size=128,
        projector_hidden_size=32,
    )
    cfg.architectures = [ARCH]
    hf = AutoModelForImageTextToText.from_config(cfg, attn_implementation="eager")
    hf = hf.to(torch.float32).eval()
    with torch.no_grad():
        for name, param in hf.named_parameters():
            if "norm" in name and param.ndim == 1:
                param.add_(0.3 * torch.randn_like(param))
    reference = copy.deepcopy(hf)
    bridge_cfg = build_bridge_config_from_hf(hf.config, ARCH, "muse-glimmer-tiny", torch.float32)
    bridge = TransformerBridge(hf, MuseGlimmerArchitectureAdapter(bridge_cfg), tokenizer=_Tok())
    return bridge, reference


def test_svd_w_in_uses_hf_ln2_scale(perturbed_models):
    """Block norms scale by 1 + w while the final norm (and cfg) use raw w."""
    from transformer_lens.SVDInterpreter import SVDInterpreter

    bridge, _ = perturbed_models
    interpreter = SVDInterpreter(bridge)
    for layer in range(bridge.cfg.n_layers):
        hf_ln2 = bridge.blocks[layer].ln2.original_component
        scale = _hf_norm_scale(hf_ln2, bridge.cfg.d_model)
        expected = interpreter.params[f"blocks.{layer}.mlp.W_in"].T * scale
        torch.testing.assert_close(interpreter._get_w_in_matrix(layer), expected)


@pytest.mark.parametrize(
    "hook_name", ["blocks.0.ln2.hook_normalized", "blocks.1.ln1_post.hook_scale"]
)
def test_identity_norm_hook_edit_keeps_hf_logits(perturbed_models, hook_name):
    """A replaced norm intermediate is rebuilt with the module's 1 + w scale and own eps."""
    bridge, reference = perturbed_models
    with torch.no_grad():
        hf_logits = reference(input_ids=TOKENS).logits
        with pytest.warns(UserWarning, match="edited hook_scale/hook_normalized"):
            logits = bridge.run_with_hooks(
                TOKENS, fwd_hooks=[(hook_name, lambda t, hook: t.clone())]
            )
    torch.testing.assert_close(logits, hf_logits, atol=1e-5, rtol=0)


def test_post_norm_head_and_neuron_decompositions_sum_to_sublayer_outputs(perturbed_models):
    """Per-head and per-neuron stacks route through ln*_post and reproduce attn/mlp out."""
    bridge, _ = perturbed_models
    n_heads, d_mlp = bridge.cfg.n_heads, bridge.cfg.d_mlp
    with torch.no_grad():
        _, cache = bridge.run_with_cache(TOKENS)
        for layer in range(bridge.cfg.n_layers):
            heads = cache.stack_head_results(layer=layer + 1)[-n_heads:].sum(0)
            torch.testing.assert_close(
                heads, cache[f"blocks.{layer}.hook_attn_out"], atol=1e-4, rtol=0
            )
            neurons = cache.stack_neuron_results(layer=layer + 1)[-d_mlp:].sum(0)
            torch.testing.assert_close(
                neurons, cache[f"blocks.{layer}.hook_mlp_out"], atol=1e-4, rtol=0
            )
