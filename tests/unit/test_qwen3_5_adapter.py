"""Unit tests for the Qwen3_5 architecture adapter (Phase A+B).

Tests cover:
1. Registration: adapter importable, in SUPPORTED_ARCHITECTURES, in HF_SUPPORTED_ARCHITECTURES
2. Component mapping: correct bridge hierarchy with only universal submodules (no self_attn),
   GatedMLPBridge with gate/in/out LinearBridge submodules
3. Config attributes: all cfg attributes set correctly
4. Weight conversions: preprocess_weights correctly slices q_proj.weight per-head
5. Integration: end-to-end tests with a tiny programmatically-constructed model

Note: Qwen3_5 is supported only via TransformerBridge, not HookedTransformer.
"""

import pytest

from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.tools.model_registry import HF_SUPPORTED_ARCHITECTURES

try:
    from transformers import Qwen3_5ForCausalLM as _Qwen3_5ForCausalLM
    from transformers import Qwen3_5TextConfig

    _QWEN3_5_AVAILABLE = True
except ImportError:
    _QWEN3_5_AVAILABLE = False

# ============================================================================
# Test: Registration
# ============================================================================

@pytest.mark.skipif(
    not _QWEN3_5_AVAILABLE,
    reason="Qwen3_5TextConfig / Qwen3_5ForCausalLM not available in installed transformers",
)
class TestQwen3_5Registration:
    """Verify the adapter is properly registered in all lookup tables."""

    def test_adapter_importable(self):
        """Qwen3_5ArchitectureAdapter must be importable."""
        from transformer_lens.model_bridge.supported_architectures import (
            Qwen3_5ArchitectureAdapter,
        )

        assert Qwen3_5ArchitectureAdapter is not None

    def test_in_supported_architectures(self):
        """Qwen3_5ForCausalLM must be in SUPPORTED_ARCHITECTURES."""
        assert "Qwen3_5ForCausalLM" in SUPPORTED_ARCHITECTURES

    def test_in_hf_supported_architectures(self):
        """Qwen3_5ForCausalLM must be in HF_SUPPORTED_ARCHITECTURES."""
        assert "Qwen3_5ForCausalLM" in HF_SUPPORTED_ARCHITECTURES

    def test_adapter_class_correct(self):
        """The adapter class must be Qwen3_5ArchitectureAdapter."""
        from transformer_lens.model_bridge.supported_architectures import (
            Qwen3_5ArchitectureAdapter,
        )

        assert SUPPORTED_ARCHITECTURES["Qwen3_5ForCausalLM"] is Qwen3_5ArchitectureAdapter


# ============================================================================
# Helpers: TransformerBridgeConfig for adapter instantiation
# ============================================================================


def _make_bridge_cfg(**overrides):
    """Create a minimal TransformerBridgeConfig for Qwen3_5 adapter tests."""
    from transformer_lens.config.TransformerBridgeConfig import TransformerBridgeConfig

    defaults = dict(
        d_model=1024,
        d_head=256,
        n_heads=8,
        n_layers=24,
        n_ctx=2048,
        d_vocab=248320,
        n_key_value_heads=2,
        architecture="Qwen3_5ForCausalLM",
    )
    defaults.update(overrides)
    return TransformerBridgeConfig(**defaults)


# ============================================================================
# Test: Component Mapping (Phase A+B)
# ============================================================================


@pytest.mark.skipif(
    not _QWEN3_5_AVAILABLE,
    reason="Qwen3_5TextConfig / Qwen3_5ForCausalLM not available in installed transformers",
)
class TestQwen3_5ComponentMapping:
    """Verify the component_mapping structure for Qwen3_5.

    The key invariant: self_attn is NOT mapped as a block submodule because
    linear-attention layers lack self_attn. Only universally present submodules
    (norms, dense MLP) are mapped. Unlike Qwen3Next, the MLP is a GatedMLPBridge
    with enumerated gate/in/out LinearBridge submodules.
    """

    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

        cfg = _make_bridge_cfg()
        return Qwen3_5ArchitectureAdapter(cfg)

    # ---- Top-level keys ----

    def test_component_mapping_keys(self, adapter):
        """component_mapping must have exactly the expected top-level keys."""
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    # ---- HF path names ----

    def test_embed_path(self, adapter):
        """embed maps to model.embed_tokens."""
        assert adapter.component_mapping["embed"].name == "model.embed_tokens"

    def test_rotary_emb_path(self, adapter):
        """rotary_emb maps to model.rotary_emb."""
        assert adapter.component_mapping["rotary_emb"].name == "model.rotary_emb"

    def test_blocks_path(self, adapter):
        """blocks maps to model.layers."""
        assert adapter.component_mapping["blocks"].name == "model.layers"

    def test_ln_final_path(self, adapter):
        """ln_final maps to model.norm."""
        assert adapter.component_mapping["ln_final"].name == "model.norm"

    def test_unembed_path(self, adapter):
        """unembed maps to lm_head."""
        assert adapter.component_mapping["unembed"].name == "lm_head"

    # ---- Block submodules ----

    def test_block_submodules_keys(self, adapter):
        """blocks submodules must contain ln1, ln2, mlp, and optional attn + linear_attn."""
        submodules = adapter.component_mapping["blocks"].submodules
        assert set(submodules.keys()) == {"ln1", "ln2", "mlp", "attn", "linear_attn"}

    def test_attn_is_optional(self, adapter):
        """attn must be marked optional (absent on linear-attention layers)."""
        submodules = adapter.component_mapping["blocks"].submodules
        assert submodules["attn"].optional is True

    def test_linear_attn_is_optional(self, adapter):
        """linear_attn must be marked optional (absent on full-attention layers)."""
        submodules = adapter.component_mapping["blocks"].submodules
        assert submodules["linear_attn"].optional is True

    def test_linear_attn_bridge_type(self, adapter):
        """linear_attn must be a GatedDeltaNetBridge."""
        from transformer_lens.model_bridge.generalized_components.gated_delta_net import (
            GatedDeltaNetBridge,
        )

        submodules = adapter.component_mapping["blocks"].submodules
        assert isinstance(submodules["linear_attn"], GatedDeltaNetBridge)

    def test_ln1_path(self, adapter):
        """ln1 maps to input_layernorm."""
        assert adapter.component_mapping["blocks"].submodules["ln1"].name == "input_layernorm"

    def test_ln2_path(self, adapter):
        """ln2 maps to post_attention_layernorm."""
        assert (
            adapter.component_mapping["blocks"].submodules["ln2"].name == "post_attention_layernorm"
        )

    def test_mlp_path(self, adapter):
        """mlp maps to mlp."""
        assert adapter.component_mapping["blocks"].submodules["mlp"].name == "mlp"

    # ---- MLP submodules ----

    def test_mlp_submodule_keys(self, adapter):
        """mlp submodules must be exactly {gate, in, out}."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"gate", "in", "out"}

    def test_mlp_gate_path(self, adapter):
        """mlp.gate maps to gate_proj."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["gate"].name == "gate_proj"

    def test_mlp_in_path(self, adapter):
        """mlp.in maps to up_proj."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "up_proj"

    def test_mlp_out_path(self, adapter):
        """mlp.out maps to down_proj."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["out"].name == "down_proj"

    # ---- Bridge types ----

    def test_blocks_bridge_type(self, adapter):
        """blocks uses BlockBridge."""
        from transformer_lens.model_bridge.generalized_components import BlockBridge

        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_rotary_emb_bridge_type(self, adapter):
        """rotary_emb uses RotaryEmbeddingBridge."""
        from transformer_lens.model_bridge.generalized_components import (
            RotaryEmbeddingBridge,
        )

        assert isinstance(adapter.component_mapping["rotary_emb"], RotaryEmbeddingBridge)

    def test_ln1_bridge_type(self, adapter):
        """ln1 uses RMSNormalizationBridge."""
        from transformer_lens.model_bridge.generalized_components import (
            RMSNormalizationBridge,
        )

        ln1 = adapter.component_mapping["blocks"].submodules["ln1"]
        assert isinstance(ln1, RMSNormalizationBridge)

    def test_ln2_bridge_type(self, adapter):
        """ln2 uses RMSNormalizationBridge."""
        from transformer_lens.model_bridge.generalized_components import (
            RMSNormalizationBridge,
        )

        ln2 = adapter.component_mapping["blocks"].submodules["ln2"]
        assert isinstance(ln2, RMSNormalizationBridge)

    def test_mlp_bridge_type(self, adapter):
        """mlp uses GatedMLPBridge (dense gated MLP, not MoE)."""
        from transformer_lens.model_bridge.generalized_components import GatedMLPBridge

        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)

    def test_mlp_gate_bridge_type(self, adapter):
        """mlp.gate uses LinearBridge."""
        from transformer_lens.model_bridge.generalized_components import LinearBridge

        gate = adapter.component_mapping["blocks"].submodules["mlp"].submodules["gate"]
        assert isinstance(gate, LinearBridge)

    def test_mlp_in_bridge_type(self, adapter):
        """mlp.in uses LinearBridge."""
        from transformer_lens.model_bridge.generalized_components import LinearBridge

        up = adapter.component_mapping["blocks"].submodules["mlp"].submodules["in"]
        assert isinstance(up, LinearBridge)

    def test_mlp_out_bridge_type(self, adapter):
        """mlp.out uses LinearBridge."""
        from transformer_lens.model_bridge.generalized_components import LinearBridge

        down = adapter.component_mapping["blocks"].submodules["mlp"].submodules["out"]
        assert isinstance(down, LinearBridge)

    # ---- weight_processing_conversions ----

    def test_weight_processing_conversions_empty(self, adapter):
        """weight_processing_conversions is empty (no attention submodules mapped)."""
        assert adapter.weight_processing_conversions == {}


# ============================================================================
# Test: Config Attributes (Phase A+B)
# ============================================================================


@pytest.mark.skipif(
    not _QWEN3_5_AVAILABLE,
    reason="Qwen3_5TextConfig / Qwen3_5ForCausalLM not available in installed transformers",
)
class TestQwen3_5ConfigAttributes:
    """Verify all cfg attributes are set correctly by the adapter."""

    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

        cfg = _make_bridge_cfg()
        return Qwen3_5ArchitectureAdapter(cfg)

    def test_normalization_type(self, adapter):
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type(self, adapter):
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms(self, adapter):
        assert adapter.cfg.final_rms is True

    def test_gated_mlp(self, adapter):
        assert adapter.cfg.gated_mlp is True

    def test_attn_only(self, adapter):
        assert adapter.cfg.attn_only is False

    def test_uses_rms_norm(self, adapter):
        assert adapter.cfg.uses_rms_norm is True

    def test_default_prepend_bos(self, adapter):
        assert adapter.cfg.default_prepend_bos is False

    def test_supports_fold_ln_false(self, adapter):
        """supports_fold_ln must be False: hybrid layers break fold_ln."""
        assert adapter.supports_fold_ln is False

    def test_attn_implementation_eager(self, adapter):
        """attn_implementation must be 'eager' for output_attentions support."""
        assert adapter.cfg.attn_implementation == "eager"

    def test_n_key_value_heads_set_when_gqa(self):
        """n_key_value_heads is set on cfg when the input config has it."""
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

        cfg = _make_bridge_cfg(n_key_value_heads=2)
        adapter = Qwen3_5ArchitectureAdapter(cfg)
        assert adapter.cfg.n_key_value_heads == 2

    def test_n_key_value_heads_not_set_when_absent(self):
        """n_key_value_heads is not set when the config doesn't have it."""
        from transformer_lens.config.TransformerBridgeConfig import (
            TransformerBridgeConfig,
        )
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

        # Config without n_key_value_heads
        cfg = TransformerBridgeConfig(
            d_model=1024,
            d_head=256,
            n_heads=8,
            n_layers=24,
            n_ctx=2048,
            d_vocab=248320,
            architecture="Qwen3_5ForCausalLM",
        )
        adapter = Qwen3_5ArchitectureAdapter(cfg)
        # n_key_value_heads should equal n_heads (standard MHA default)
        assert not (
            hasattr(adapter.cfg, "n_key_value_heads")
            and adapter.cfg.n_key_value_heads is not None
            and adapter.cfg.n_key_value_heads != adapter.cfg.n_heads
        )


# ============================================================================
# Test: preprocess_weights (Phase A+B)
# ============================================================================


@pytest.mark.skipif(
    not _QWEN3_5_AVAILABLE,
    reason="Qwen3_5TextConfig / Qwen3_5ForCausalLM not available in installed transformers",
)
class TestQwen3_5PreprocessWeights:
    """Verify preprocess_weights correctly slices q_proj.weight per-head.

    Background: In Qwen3_5, q_proj.weight has shape (n_heads * head_dim * 2, hidden_size)
    where rows are organized as interleaved per-head pairs:
      head_0_query (d_head rows), head_0_gate (d_head rows),
      head_1_query (d_head rows), head_1_gate (d_head rows), ...

    A naive first-half slice would be wrong. The correct approach reshapes by
    head and takes only the first d_head rows per head (the query half).
    """

    N_HEADS = 4
    D_HEAD = 8
    HIDDEN_SIZE = 32

    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

        cfg = _make_bridge_cfg(
            n_heads=self.N_HEADS,
            d_head=self.D_HEAD,
            d_model=self.HIDDEN_SIZE,
            n_key_value_heads=self.N_HEADS,  # MHA for simplicity
        )
        return Qwen3_5ArchitectureAdapter(cfg)

    def _make_q_proj_weight(self):
        """Create a q_proj.weight tensor with distinct per-head-row values."""
        import torch

        total_rows = self.N_HEADS * self.D_HEAD * 2
        w = torch.zeros(total_rows, self.HIDDEN_SIZE)
        for row_idx in range(total_rows):
            w[row_idx] = float(row_idx)
        return w

    def test_q_proj_output_shape(self, adapter):
        """preprocess_weights reduces q_proj rows from n_heads*d_head*2 to n_heads*d_head."""
        import torch

        w = self._make_q_proj_weight()
        state_dict = {"model.layers.3.self_attn.q_proj.weight": w}
        result = adapter.preprocess_weights(state_dict)
        out = result["model.layers.3.self_attn.q_proj.weight"]
        assert out.shape == (self.N_HEADS * self.D_HEAD, self.HIDDEN_SIZE)

    def test_q_proj_selects_query_rows_not_naive_first_half(self, adapter):
        """For each head i, output rows [i*d_head:(i+1)*d_head] == input rows
        [i*d_head*2 : i*d_head*2 + d_head] (per-head interleaved layout)."""
        import torch

        w = self._make_q_proj_weight()
        state_dict = {"model.layers.0.self_attn.q_proj.weight": w}
        result = adapter.preprocess_weights(state_dict)
        out = result["model.layers.0.self_attn.q_proj.weight"]

        for head_idx in range(self.N_HEADS):
            out_rows = out[head_idx * self.D_HEAD : (head_idx + 1) * self.D_HEAD]
            expected_start = head_idx * self.D_HEAD * 2
            expected_rows = w[expected_start : expected_start + self.D_HEAD]
            assert torch.equal(out_rows, expected_rows), (
                f"Head {head_idx}: output rows do not match expected query rows. "
                f"Got row values starting at {out_rows[0, 0].item()}, "
                f"expected starting at {expected_rows[0, 0].item()}"
            )

    def test_naive_slice_would_be_wrong(self, adapter):
        """Naive first-half slice gives different (wrong) results for n_heads > 1."""
        import torch

        w = self._make_q_proj_weight()
        state_dict = {"model.layers.0.self_attn.q_proj.weight": w}
        result = adapter.preprocess_weights(state_dict)
        correct_out = result["model.layers.0.self_attn.q_proj.weight"]
        naive_out = w[: self.N_HEADS * self.D_HEAD]

        if self.N_HEADS > 1:
            assert not torch.equal(correct_out, naive_out), (
                "Naive first-half slice gave the same result as per-head slice — "
                "test setup may be wrong"
            )

    def test_non_q_proj_weights_unchanged(self, adapter):
        """k_proj, v_proj, and down_proj weights are NOT modified by preprocess_weights."""
        import torch

        k_proj = torch.randn(self.N_HEADS * self.D_HEAD, self.HIDDEN_SIZE)
        down_proj = torch.randn(self.HIDDEN_SIZE, self.N_HEADS * self.D_HEAD)
        state_dict = {
            "model.layers.0.self_attn.k_proj.weight": k_proj.clone(),
            "model.layers.0.mlp.down_proj.weight": down_proj.clone(),
        }
        result = adapter.preprocess_weights(state_dict)
        assert torch.equal(result["model.layers.0.self_attn.k_proj.weight"], k_proj)
        assert torch.equal(result["model.layers.0.mlp.down_proj.weight"], down_proj)

    def test_multiple_layers_all_processed(self, adapter):
        """q_proj.weight tensors across multiple layers are all sliced correctly."""
        import torch

        w0 = self._make_q_proj_weight()
        w3 = self._make_q_proj_weight() * 2
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": w0,
            "model.layers.3.self_attn.q_proj.weight": w3,
        }
        result = adapter.preprocess_weights(state_dict)
        expected_shape = (self.N_HEADS * self.D_HEAD, self.HIDDEN_SIZE)
        assert result["model.layers.0.self_attn.q_proj.weight"].shape == expected_shape
        assert result["model.layers.3.self_attn.q_proj.weight"].shape == expected_shape

    def test_empty_state_dict_returns_empty(self, adapter):
        """preprocess_weights with an empty state dict returns an empty dict."""
        assert adapter.preprocess_weights({}) == {}

    def test_state_dict_without_q_proj_unchanged(self, adapter):
        """A state dict with no q_proj keys is returned unmodified."""
        import torch

        state_dict = {"model.embed_tokens.weight": torch.randn(100, self.HIDDEN_SIZE)}
        original_keys = set(state_dict.keys())
        result = adapter.preprocess_weights(state_dict)
        assert set(result.keys()) == original_keys

    def test_weight_processing_conversions_is_empty_dict(self, adapter):
        """weight_processing_conversions is {} — q_proj slicing is in preprocess_weights."""
        assert adapter.weight_processing_conversions == {}


# ============================================================================
# Test: Integration (Phase A+B)
# ============================================================================


def _make_tiny_hf_model():
    """Create a tiny Qwen3_5ForCausalLM for integration testing.

    8 layers: layers 3 and 7 are full-attention (full_attention_interval=4),
    layers 0-2 and 4-6 are linear-attention (GatedDeltaNet).
    Dense gated MLP on all layers.
    """
    cfg = Qwen3_5TextConfig(
        hidden_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256,
        vocab_size=512,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        rope_parameters={
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        },
    )
    model = _Qwen3_5ForCausalLM(cfg)
    model.eval()
    return model


def _make_tiny_bridge():
    """Create a Qwen3_5 bridge from a tiny HF model."""
    from unittest.mock import MagicMock

    from transformer_lens.config.TransformerBridgeConfig import TransformerBridgeConfig
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
        Qwen3_5ArchitectureAdapter,
    )

    hf_model = _make_tiny_hf_model()

    bridge_cfg = TransformerBridgeConfig(
        d_model=128,
        d_head=32,
        n_heads=4,
        n_layers=8,
        n_ctx=2048,
        d_vocab=512,
        n_key_value_heads=2,
        architecture="Qwen3_5ForCausalLM",
    )
    adapter = Qwen3_5ArchitectureAdapter(bridge_cfg)
    return TransformerBridge(hf_model, adapter, tokenizer=MagicMock()), hf_model


@pytest.mark.skipif(
    not _QWEN3_5_AVAILABLE,
    reason="Qwen3_5TextConfig / Qwen3_5ForCausalLM not available in installed transformers",
)
class TestQwen3_5Integration:
    """End-to-end integration tests using a tiny programmatic Qwen3_5 model.

    The linear attention layers run via the torch fallback path when
    flash-linear-attention / causal-conv1d are not installed.
    """

    @pytest.fixture(scope="class")
    def bridge_and_model(self):
        """Create a tiny bridge + HF model pair, shared across the class."""
        return _make_tiny_bridge()

    @pytest.fixture(scope="class")
    def bridge(self, bridge_and_model):
        br, _ = bridge_and_model
        return br

    @pytest.fixture(scope="class")
    def hf_model(self, bridge_and_model):
        _, hf = bridge_and_model
        return hf

    def test_bridge_creation(self, bridge):
        """TransformerBridge construction from a tiny Qwen3_5 model must succeed."""
        from transformer_lens.model_bridge import TransformerBridge

        assert isinstance(bridge, TransformerBridge)

    def test_hook_names_present(self, bridge):
        """Key hook names must be present; blocks.0.attn.* must NOT be present.

        Verified:
        - blocks.0.hook_resid_pre: linear-attention layer (layer 0)
        - blocks.3.hook_resid_pre: first full-attention layer (layer 3)
        - blocks.0.ln1.*: norm present on all layers (universal submodule)
        - blocks.0.mlp.*: MLP present on all layers (universal submodule)
        - blocks.0.attn.*: NOT present (self_attn absent on linear-attn layers)
        """
        hook_keys = set(bridge.hook_dict.keys())

        assert "blocks.0.hook_resid_pre" in hook_keys, "linear-attn layer must have hook_resid_pre"
        assert "blocks.3.hook_resid_pre" in hook_keys, "full-attn layer must have hook_resid_pre"
        assert any(
            "blocks.0.ln1" in k for k in hook_keys
        ), "blocks.0.ln1 submodule hooks must be present"
        assert any(
            "blocks.0.mlp" in k for k in hook_keys
        ), "blocks.0.mlp submodule hooks must be present"
        assert not any(
            "blocks.0.attn" in k for k in hook_keys
        ), "blocks.0.attn hooks must NOT be present (hybrid architecture)"

    def test_forward_pass_consistency(self, bridge, hf_model):
        """Bridge output logits must match HF model output logits within atol=1e-4."""
        import torch

        tokens = torch.randint(0, 512, (1, 4))
        with torch.no_grad():
            hf_logits = hf_model(tokens).logits
            bridge_logits = bridge(tokens)

        assert (
            hf_logits.shape == bridge_logits.shape
        ), f"Shape mismatch: HF={hf_logits.shape}, bridge={bridge_logits.shape}"
        assert torch.allclose(
            hf_logits, bridge_logits, atol=1e-4
        ), f"Logit mismatch: max diff = {(hf_logits - bridge_logits).abs().max().item():.6f}"

    def test_hook_activation_shapes(self, bridge):
        """A hook on blocks.0.mlp.hook_out must capture a (batch, seq, d_model) tensor."""
        import torch

        captured: list[torch.Tensor] = []

        def capture_hook(tensor: torch.Tensor, hook: object) -> torch.Tensor:
            captured.append(tensor.detach().clone())
            return tensor

        tokens = torch.randint(0, 512, (1, 4))
        with torch.no_grad():
            bridge.run_with_hooks(tokens, fwd_hooks=[("blocks.0.mlp.hook_out", capture_hook)])

        assert len(captured) == 1, "Hook must fire exactly once per forward pass"
        output = captured[0]
        batch, seq, d_model = 1, 4, 128
        assert output.shape == (
            batch,
            seq,
            d_model,
        ), f"Expected MLP output shape ({batch}, {seq}, {d_model}), got {output.shape}"
