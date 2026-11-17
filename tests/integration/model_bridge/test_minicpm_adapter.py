"""Integration tests for the MiniCPM architecture adapter (MiniCPMForCausalLM).

Model: openbmb/MiniCPM4-0.5B
  - Remote code (auto_map -> modeling_minicpm), no tiny random checkpoint; the
    real 0.43B weights are ~0.8GB to download and ~2GB in fp32. Gated out of CI;
    run locally with:
        uv run pytest tests/integration/model_bridge/test_minicpm_adapter.py -v -m slow
  - CI coverage of the same code paths lives in test_minicpm_tiny.py.

Only one copy of the model is resident at a time: the HF reference outputs are
captured and the reference freed before the bridge loads. The reference is an
independent load patched by the adapter's own prepare_loading/prepare_model
(never ``bridge.original_model``, whose modules are hook-wrapped), loaded with
eager attention like the bridge's reconstructed attention.
"""

import gc
import math
from typing import Any

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.supported_architectures.minicpm import (
    MiniCPMArchitectureAdapter,
)

MODEL = "openbmb/MiniCPM4-0.5B"
PROMPT = "The capital of France is"
TOL = 1e-4

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def hf_reference() -> dict[str, Any]:
    """HF outputs for PROMPT, captured before (and freed ahead of) the bridge load."""
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    adapter = MiniCPMArchitectureAdapter(
        build_bridge_config_from_hf(config, "MiniCPMForCausalLM", MODEL, torch.float32)
    )
    adapter.prepare_loading(MODEL, {"config": config})
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        config=config,
        dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=True,
    ).eval()
    adapter.prepare_model(hf_model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokens = tokenizer(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        out = hf_model(tokens, output_hidden_states=True, use_cache=False)
    reference = {
        "config": config,
        "tokens": tokens,
        "logits": out.logits,
        "hidden_states": out.hidden_states,
    }
    del hf_model, out
    gc.collect()
    return reference


@pytest.fixture(scope="module")
def bridge(hf_reference: dict[str, Any]) -> TransformerBridge:
    """Depends on hf_reference so the reference is freed before this loads."""
    return TransformerBridge.boot_transformers(
        MODEL, device="cpu", dtype=torch.float32, trust_remote_code=True
    )


@pytest.fixture(scope="module")
def cache(bridge: TransformerBridge, hf_reference: dict[str, Any]) -> Any:
    with torch.no_grad():
        _, activation_cache = bridge.run_with_cache(hf_reference["tokens"])
    return activation_cache


class TestMiniCPMBridgeCreation:
    def test_adapter_selected(self, bridge: TransformerBridge) -> None:
        assert isinstance(bridge.adapter, MiniCPMArchitectureAdapter)

    def test_config_flags(self, bridge: TransformerBridge) -> None:
        assert bridge.cfg.normalization_type == "RMS"
        assert bridge.cfg.positional_embedding_type == "rotary"
        assert bridge.cfg.gated_mlp is True

    def test_shape_matches_hf(self, bridge: TransformerBridge, hf_reference: dict) -> None:
        config = hf_reference["config"]
        assert len(bridge.blocks) == config.num_hidden_layers
        assert bridge.cfg.d_model == config.hidden_size
        assert bridge.cfg.n_heads == config.num_attention_heads
        assert bridge.cfg.n_key_value_heads == config.num_key_value_heads

    def test_mup_attrs_propagated(self, bridge: TransformerBridge, hf_reference: dict) -> None:
        config = hf_reference["config"]
        assert bridge.cfg.scale_emb == config.scale_emb
        assert bridge.cfg.scale_depth == config.scale_depth
        assert bridge.cfg.dim_model_base == config.dim_model_base

    def test_dense_attention_only(self, hf_reference: dict) -> None:
        """Sparse (InfLLM v2) MiniCPM4 checkpoints are out of scope."""
        assert getattr(hf_reference["config"], "sparse_config", None) is None


class TestMiniCPMForwardEquivalence:
    def test_logits_match_hf(self, bridge: TransformerBridge, hf_reference: dict) -> None:
        with torch.no_grad():
            got = bridge(hf_reference["tokens"])
        max_diff = (got - hf_reference["logits"]).abs().max().item()
        assert max_diff < TOL, f"Bridge vs HF max diff = {max_diff}"

    def test_residual_stream_matches_hf(self, cache: Any, hf_reference: dict) -> None:
        for i, want in enumerate(hf_reference["hidden_states"][:-1]):
            got = cache[f"blocks.{i}.hook_resid_pre"]
            assert torch.allclose(got, want, atol=TOL), i

    def test_greedy_completion(self, bridge: TransformerBridge) -> None:
        """Garbage rotary tables still run, but complete this prompt with '19999'."""
        out = bridge.generate(PROMPT, max_new_tokens=3, do_sample=False, verbose=False)
        assert "Paris" in out


class TestMiniCPMHFDelegation:
    def test_bridges_are_installed_in_the_hf_modules(self, bridge: TransformerBridge) -> None:
        """The bridge wrappers replace the HF submodules in place."""
        attn = bridge.blocks[0].attn
        assert attn is bridge.original_model.model.layers[0].self_attn
        hf_attn = attn.original_component
        assert type(hf_attn).__name__ == "MiniCPMAttention"
        for bridge_name, hf_name in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")):
            projection = getattr(attn, bridge_name)
            assert projection is getattr(hf_attn, hf_name)
            assert isinstance(projection.original_component, torch.nn.Linear)
        assert attn.o is hf_attn.o_proj

    def test_rotary_is_the_layers_own(self, bridge: TransformerBridge) -> None:
        for i, layer in enumerate(bridge.original_model.model.layers):
            assert bridge.blocks[i].attn.original_component.rotary_emb is layer.self_attn.rotary_emb

    def test_embeddings_stay_tied(self, bridge: TransformerBridge) -> None:
        hf_model = bridge.original_model
        assert hf_model.lm_head.weight.data_ptr() == hf_model.model.embed_tokens.weight.data_ptr()


class TestMiniCPMHookShapes:
    def test_residual_hooks(self, cache: Any, bridge: TransformerBridge) -> None:
        seq = cache["hook_embed"].shape[1]
        for name in ("hook_resid_pre", "hook_resid_mid", "hook_resid_post"):
            assert cache[f"blocks.0.{name}"].shape == (1, seq, bridge.cfg.d_model)

    def test_attention_hooks(self, cache: Any, bridge: TransformerBridge) -> None:
        seq = cache["hook_embed"].shape[1]
        assert cache["blocks.0.attn.hook_pattern"].shape == (1, bridge.cfg.n_heads, seq, seq)
        assert cache["blocks.0.hook_attn_out"].shape == (1, seq, bridge.cfg.d_model)
        assert cache["blocks.0.hook_mlp_out"].shape == (1, seq, bridge.cfg.d_model)


class TestMiniCPMEmbeddingScale:
    def test_hook_embed_is_hfs_scaled_embedding(self, cache: Any, hf_reference: dict) -> None:
        assert torch.allclose(cache["hook_embed"], hf_reference["hidden_states"][0], atol=TOL)


class TestMiniCPMResidualScale:
    def test_contributions_sum_to_the_residual_stream(
        self, cache: Any, bridge: TransformerBridge
    ) -> None:
        for i in range(bridge.cfg.n_layers):
            resid_mid = cache[f"blocks.{i}.hook_resid_pre"] + cache[f"blocks.{i}.hook_attn_out"]
            assert torch.allclose(resid_mid, cache[f"blocks.{i}.hook_resid_mid"], atol=TOL), i
            resid_post = resid_mid + cache[f"blocks.{i}.hook_mlp_out"]
            assert torch.allclose(resid_post, cache[f"blocks.{i}.hook_resid_post"], atol=TOL), i

    def test_scale_is_scale_depth_over_sqrt_layers(
        self, bridge: TransformerBridge, hf_reference: dict
    ) -> None:
        config = hf_reference["config"]
        expected = config.scale_depth / math.sqrt(config.num_hidden_layers)
        assert bridge.blocks[0].residual_contribution_scale == expected


class TestMiniCPMLogitScale:
    def test_transform_reproduces_the_logits(
        self, bridge: TransformerBridge, cache: Any, hf_reference: dict
    ) -> None:
        with torch.no_grad():
            unembedded = bridge.unembed(cache["ln_final.hook_out"])
        recomputed = bridge.adapter.apply_output_logits_transform(unembedded)
        assert torch.allclose(recomputed, hf_reference["logits"], atol=TOL)


class TestMiniCPMTokenizer:
    def test_single_bos_prepended(self, bridge: TransformerBridge) -> None:
        tokens = bridge.to_tokens(PROMPT)
        hf_tokens = bridge.tokenizer(PROMPT, return_tensors="pt").input_ids
        assert tokens[0, 0].item() == bridge.tokenizer.bos_token_id
        assert tokens[0, 1].item() != bridge.tokenizer.bos_token_id
        assert torch.equal(tokens.cpu(), hf_tokens)
