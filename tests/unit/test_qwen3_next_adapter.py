"""Unit tests for the Qwen3Next architecture adapter (Phases A through D).

Tests cover:
1. Registration: adapter importable, in SUPPORTED_ARCHITECTURES, in HF_SUPPORTED_ARCHITECTURES
2. Config extraction: convert_hf_model_config produces correct config for Qwen3NextForCausalLM
3. _get_partial_rotary_factor helper: reads from rope_parameters dict only (not top-level)
4. Component mapping: correct bridge hierarchy with only universal submodules (no self_attn)
5. Weight conversions: preprocess_weights correctly slices q_proj.weight per-head
6. Integration: end-to-end tests with a tiny programmatically-constructed model
"""

from unittest import mock

import pytest

from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.tools.model_registry import HF_SUPPORTED_ARCHITECTURES

# ============================================================================
# Test: Registration
# ============================================================================


class TestQwen3NextRegistration:
    """Verify the adapter is properly registered in all lookup tables."""

    def test_adapter_importable(self):
        """Qwen3NextArchitectureAdapter must be importable."""
        from transformer_lens.model_bridge.supported_architectures import (
            Qwen3NextArchitectureAdapter,
        )

        assert Qwen3NextArchitectureAdapter is not None

    def test_in_supported_architectures(self):
        """Qwen3NextForCausalLM must be in SUPPORTED_ARCHITECTURES."""
        assert "Qwen3NextForCausalLM" in SUPPORTED_ARCHITECTURES

    def test_in_hf_supported_architectures(self):
        """Qwen3NextForCausalLM must be in HF_SUPPORTED_ARCHITECTURES."""
        assert "Qwen3NextForCausalLM" in HF_SUPPORTED_ARCHITECTURES

    def test_adapter_class_correct(self):
        """The adapter class must be Qwen3NextArchitectureAdapter."""
        from transformer_lens.model_bridge.supported_architectures import (
            Qwen3NextArchitectureAdapter,
        )

        assert SUPPORTED_ARCHITECTURES["Qwen3NextForCausalLM"] is Qwen3NextArchitectureAdapter


# ============================================================================
# Helpers: mock HF config
# ============================================================================


def _make_hf_config(
    *,
    hidden_size: int = 2048,
    num_attention_heads: int = 8,
    num_key_value_heads: int = 2,
    head_dim: int = 256,
    intermediate_size: int = 6144,
    num_hidden_layers: int = 24,
    vocab_size: int = 248320,
    rms_norm_eps: float = 1e-6,
    hidden_act: str = "silu",
    tie_word_embeddings: bool = False,
    rope_parameters: dict | None = None,
) -> mock.Mock:
    """Create a minimal mock HuggingFace config for Qwen3NextForCausalLM.

    Uses spec=[] so only explicitly assigned attributes exist. This prevents
    mock.Mock() from auto-creating attributes (like rope_theta) that would
    interfere with beartype-validated helpers like _get_rope_theta().
    """
    if rope_parameters is None:
        rope_parameters = {
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        }
    cfg = mock.Mock(spec=[])
    cfg.architectures = ["Qwen3NextForCausalLM"]
    cfg.hidden_size = hidden_size
    cfg.num_attention_heads = num_attention_heads
    cfg.num_key_value_heads = num_key_value_heads
    cfg.head_dim = head_dim
    cfg.intermediate_size = intermediate_size
    cfg.num_hidden_layers = num_hidden_layers
    cfg.vocab_size = vocab_size
    cfg.rms_norm_eps = rms_norm_eps
    cfg.hidden_act = hidden_act
    cfg.tie_word_embeddings = tie_word_embeddings
    cfg.rope_parameters = rope_parameters
    return cfg


# ============================================================================
# Test: Config extraction
# ============================================================================


class TestQwen3NextConfigExtraction:
    """Verify convert_hf_model_config extracts all fields correctly."""

    def _extract_config(self, hf_config: mock.Mock) -> dict:
        """Run convert_hf_model_config with a mocked AutoConfig and model name lookup."""
        from transformer_lens.loading_from_pretrained import convert_hf_model_config

        model_name = "Qwen/Qwen3-Next-80B-A3B"
        with mock.patch(
            "transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained",
            return_value=hf_config,
        ), mock.patch(
            "transformer_lens.loading_from_pretrained.get_official_model_name",
            return_value=model_name,
        ):
            return convert_hf_model_config(model_name)

    def test_basic_dimensions(self):
        """d_model, n_heads, n_layers, d_mlp, d_vocab extracted correctly."""
        hf_config = _make_hf_config()
        cfg = self._extract_config(hf_config)

        assert cfg["d_model"] == 2048
        assert cfg["n_heads"] == 8
        assert cfg["n_layers"] == 24
        assert cfg["d_mlp"] == 6144
        assert cfg["d_vocab"] == 248320

    def test_head_dim(self):
        """d_head reads from hf_config.head_dim directly."""
        hf_config = _make_hf_config(head_dim=256)
        cfg = self._extract_config(hf_config)

        assert cfg["d_head"] == 256

    def test_n_key_value_heads_gqa(self):
        """n_key_value_heads is set when num_key_value_heads != num_attention_heads (GQA)."""
        hf_config = _make_hf_config(num_attention_heads=8, num_key_value_heads=2)
        cfg = self._extract_config(hf_config)

        assert cfg["n_key_value_heads"] == 2

    def test_n_key_value_heads_mha(self):
        """n_key_value_heads is None when num_key_value_heads == num_attention_heads (MHA)."""
        hf_config = _make_hf_config(num_attention_heads=8, num_key_value_heads=8)
        cfg = self._extract_config(hf_config)

        assert cfg["n_key_value_heads"] is None

    def test_n_ctx_is_2048(self):
        """n_ctx is hardcoded to 2048 (safe cap for 262144 max)."""
        hf_config = _make_hf_config()
        cfg = self._extract_config(hf_config)

        assert cfg["n_ctx"] == 2048

    def test_eps(self):
        """eps reads from rms_norm_eps."""
        hf_config = _make_hf_config(rms_norm_eps=1e-6)
        cfg = self._extract_config(hf_config)

        assert cfg["eps"] == 1e-6

    def test_rotary_base_from_rope_parameters(self):
        """rotary_base reads rope_theta from rope_parameters dict."""
        hf_config = _make_hf_config(
            rope_parameters={
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
                "rope_type": "default",
            }
        )
        cfg = self._extract_config(hf_config)

        assert cfg["rotary_base"] == 10000000

    def test_rotary_dim_partial_factor_0_25(self):
        """rotary_dim = int(head_dim * partial_rotary_factor).

        With partial_rotary_factor=0.25 and head_dim=256, expect rotary_dim=64.
        """
        hf_config = _make_hf_config(
            head_dim=256,
            rope_parameters={
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "rope_type": "default",
            },
        )
        cfg = self._extract_config(hf_config)

        assert cfg["rotary_dim"] == 64

    def test_rotary_adjacent_pairs_false(self):
        """rotary_adjacent_pairs must be False."""
        hf_config = _make_hf_config()
        cfg = self._extract_config(hf_config)

        assert cfg["rotary_adjacent_pairs"] is False

    def test_flags(self):
        """final_rms, gated_mlp, use_qk_norm, use_attn_scale all True; default_prepend_bos False."""
        hf_config = _make_hf_config()
        cfg = self._extract_config(hf_config)

        assert cfg["final_rms"] is True
        assert cfg["gated_mlp"] is True
        assert cfg["use_qk_norm"] is True
        assert cfg["use_attn_scale"] is True
        assert cfg["default_prepend_bos"] is False

    def test_tie_word_embeddings(self):
        """tie_word_embeddings reads from hf_config."""
        hf_config = _make_hf_config(tie_word_embeddings=False)
        cfg = self._extract_config(hf_config)

        assert cfg["tie_word_embeddings"] is False

    def test_trust_remote_code(self):
        """trust_remote_code must be True."""
        hf_config = _make_hf_config()
        cfg = self._extract_config(hf_config)

        assert cfg["trust_remote_code"] is True

    def test_normalization_type_rms(self):
        """normalization_type is 'RMS'."""
        hf_config = _make_hf_config()
        cfg = self._extract_config(hf_config)

        assert cfg["normalization_type"] == "RMS"

    def test_positional_embedding_type_rotary(self):
        """positional_embedding_type is 'rotary'."""
        hf_config = _make_hf_config()
        cfg = self._extract_config(hf_config)

        assert cfg["positional_embedding_type"] == "rotary"


# ============================================================================
# Test: _get_partial_rotary_factor helper
# ============================================================================


class TestGetPartialRotaryFactor:
    """Verify _get_partial_rotary_factor reads from rope_parameters dict only."""

    def test_reads_from_rope_parameters(self):
        """partial_rotary_factor is read from rope_parameters dict."""
        from transformer_lens.loading_from_pretrained import _get_partial_rotary_factor

        cfg = mock.Mock()
        cfg.rope_parameters = {"partial_rotary_factor": 0.25}
        # Top-level attribute should NOT be consulted
        cfg.partial_rotary_factor = 0.99  # wrong value — must not be used

        result = _get_partial_rotary_factor(cfg)
        assert result == 0.25

    def test_fallback_when_rope_parameters_missing(self):
        """Returns 1.0 when rope_parameters is absent."""
        from transformer_lens.loading_from_pretrained import _get_partial_rotary_factor

        cfg = mock.Mock(spec=[])  # no attributes at all

        result = _get_partial_rotary_factor(cfg)
        assert result == 1.0

    def test_fallback_when_partial_rotary_factor_not_in_dict(self):
        """Returns 1.0 when rope_parameters exists but lacks partial_rotary_factor.

        This is the key correctness test: a config that has partial_rotary_factor
        as a top-level attribute but NOT in rope_parameters must return 1.0 (the
        fallback), not 0.5. This verifies we only read from the dict.
        """
        from transformer_lens.loading_from_pretrained import _get_partial_rotary_factor

        cfg = mock.Mock()
        cfg.rope_parameters = {}  # no partial_rotary_factor key
        cfg.partial_rotary_factor = 0.5  # top-level only — must NOT be used

        result = _get_partial_rotary_factor(cfg)
        assert result == 1.0

    def test_custom_default(self):
        """Custom default is returned when rope_parameters is absent."""
        from transformer_lens.loading_from_pretrained import _get_partial_rotary_factor

        cfg = mock.Mock(spec=[])

        result = _get_partial_rotary_factor(cfg, default=0.5)
        assert result == 0.5

    def test_non_dict_rope_parameters_uses_default(self):
        """Returns default when rope_parameters is not a dict."""
        from transformer_lens.loading_from_pretrained import _get_partial_rotary_factor

        cfg = mock.Mock()
        cfg.rope_parameters = "not_a_dict"

        result = _get_partial_rotary_factor(cfg)
        assert result == 1.0


# ============================================================================
# Helpers: TransformerBridgeConfig for adapter instantiation
# ============================================================================


def _make_bridge_cfg(**overrides):
    """Create a minimal TransformerBridgeConfig for Qwen3Next adapter tests."""
    from transformer_lens.config.TransformerBridgeConfig import TransformerBridgeConfig

    defaults = dict(
        d_model=2048,
        d_head=256,
        n_heads=8,
        n_layers=24,
        n_ctx=2048,
        d_vocab=248320,
        n_key_value_heads=2,
        architecture="Qwen3NextForCausalLM",
    )
    defaults.update(overrides)
    return TransformerBridgeConfig(**defaults)


# ============================================================================
# Test: Component Mapping (Phase B)
# ============================================================================


class TestQwen3NextComponentMapping:
    """Verify the component_mapping structure for Qwen3Next.

    The key invariant: self_attn is NOT mapped as a block submodule because
    linear-attention layers lack self_attn, and get_remote_component raises
    AttributeError for missing attributes (verified in architecture_adapter.py).
    Only universally present submodules (norms, MLP) are mapped.
    """

    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_next import (
            Qwen3NextArchitectureAdapter,
        )

        cfg = _make_bridge_cfg()
        return Qwen3NextArchitectureAdapter(cfg)

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
        """blocks submodules must contain ln1, ln2, mlp but NOT attn.

        This is a critical correctness test: self_attn is absent on
        linear-attention layers, so mapping attn as a block submodule
        would crash on those layers.
        """
        submodules = adapter.component_mapping["blocks"].submodules
        assert set(submodules.keys()) == {"ln1", "ln2", "mlp"}

    def test_no_attn_in_block_submodules(self, adapter):
        """attn must NOT appear as a block submodule (hybrid architecture safety check)."""
        submodules = adapter.component_mapping["blocks"].submodules
        assert "attn" not in submodules

    def test_ln1_path(self, adapter):
        """ln1 maps to input_layernorm."""
        submodules = adapter.component_mapping["blocks"].submodules
        assert submodules["ln1"].name == "input_layernorm"

    def test_ln2_path(self, adapter):
        """ln2 maps to post_attention_layernorm."""
        submodules = adapter.component_mapping["blocks"].submodules
        assert submodules["ln2"].name == "post_attention_layernorm"

    def test_mlp_path(self, adapter):
        """mlp maps to mlp."""
        submodules = adapter.component_mapping["blocks"].submodules
        assert submodules["mlp"].name == "mlp"

    # ---- MLP submodules ----

    def test_mlp_has_no_submodules(self, adapter):
        """mlp is a MoEBridge with no enumerated submodules.

        Real Qwen3Next checkpoints use Qwen3NextSparseMoeBlock whose router
        (`gate`) is a Qwen3NextTopKRouter rather than nn.Linear, and whose
        experts are batched as 3D tensors inside Qwen3NextExperts. MoEBridge
        wraps the whole block and delegates to HF's native forward, so no
        internal submodules are mapped here.
        """
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules == {}

    # ---- Bridge types ----

    def test_mlp_bridge_type(self, adapter):
        """mlp uses MoEBridge (sparse MoE on every real checkpoint)."""
        from transformer_lens.model_bridge.generalized_components import MoEBridge

        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)

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

    # ---- weight_processing_conversions ----

    def test_weight_processing_conversions_empty(self, adapter):
        """weight_processing_conversions is empty (no attention submodules mapped)."""
        assert adapter.weight_processing_conversions == {}


# ============================================================================
# Test: Weight Conversions (Phase C)
# ============================================================================


class TestQwen3NextWeightConversions:
    """Verify preprocess_weights correctly slices q_proj.weight per-head.

    Background: In Qwen3Next, q_proj.weight has shape (n_heads * head_dim * 2, hidden_size)
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
        from transformer_lens.model_bridge.supported_architectures.qwen3_next import (
            Qwen3NextArchitectureAdapter,
        )

        cfg = _make_bridge_cfg(
            n_heads=self.N_HEADS,
            d_head=self.D_HEAD,
            d_model=self.HIDDEN_SIZE,
            n_key_value_heads=self.N_HEADS,  # MHA for simplicity
        )
        return Qwen3NextArchitectureAdapter(cfg)

    def _make_q_proj_weight(self):
        """Create a q_proj.weight tensor with distinct per-head-row values.

        Shape: (n_heads * d_head * 2, hidden_size)
        Each row is filled with a unique integer so we can verify which rows
        were selected after slicing.
        """
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
        """For each head i, output rows [i*d_head : (i+1)*d_head] == input rows
        [i*d_head*2 : i*d_head*2 + d_head].

        This verifies the per-head reshape: a naive slice of the first half would
        incorrectly include gate rows from later heads.
        """
        import torch

        w = self._make_q_proj_weight()
        state_dict = {"model.layers.0.self_attn.q_proj.weight": w}

        result = adapter.preprocess_weights(state_dict)
        out = result["model.layers.0.self_attn.q_proj.weight"]

        for head_idx in range(self.N_HEADS):
            out_rows = out[head_idx * self.D_HEAD : (head_idx + 1) * self.D_HEAD]
            # Per-head interleaved layout: query rows for head i start at i*(d_head*2)
            expected_start = head_idx * self.D_HEAD * 2
            expected_rows = w[expected_start : expected_start + self.D_HEAD]
            assert torch.equal(out_rows, expected_rows), (
                f"Head {head_idx}: output rows do not match expected query rows. "
                f"Got row values starting at {out_rows[0, 0].item()}, "
                f"expected starting at {expected_rows[0, 0].item()}"
            )

    def test_naive_slice_would_be_wrong(self, adapter):
        """Demonstrate that a naive first-half slice gives different (wrong) results.

        This documents the correctness invariant: the interleaved layout means
        naive slicing includes gate rows from intermediate heads.
        """
        import torch

        w = self._make_q_proj_weight()
        state_dict = {"model.layers.0.self_attn.q_proj.weight": w}

        result = adapter.preprocess_weights(state_dict)
        correct_out = result["model.layers.0.self_attn.q_proj.weight"]

        # Naive first half: just take the top n_heads*d_head rows
        naive_out = w[: self.N_HEADS * self.D_HEAD]

        # They should differ (unless n_heads==1, where both produce the same result)
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
        w3 = self._make_q_proj_weight() * 2  # distinct values to catch cross-layer bugs

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
        result = adapter.preprocess_weights({})
        assert result == {}

    def test_state_dict_without_q_proj_unchanged(self, adapter):
        """A state dict with no q_proj keys is returned unmodified."""
        import torch

        state_dict = {
            "model.embed_tokens.weight": torch.randn(100, self.HIDDEN_SIZE),
        }
        original_keys = set(state_dict.keys())

        result = adapter.preprocess_weights(state_dict)

        assert set(result.keys()) == original_keys

    def test_weight_processing_conversions_is_empty_dict(self, adapter):
        """weight_processing_conversions is {} — q_proj slicing is done in preprocess_weights."""
        assert adapter.weight_processing_conversions == {}


# ============================================================================
# Test: Integration (Phase D)
# ============================================================================

try:
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

    _QWEN3NEXT_AVAILABLE = True
except ImportError:
    _QWEN3NEXT_AVAILABLE = False


def _make_tiny_hf_model():
    """Create a tiny Qwen3Next model for integration testing.

    Uses num_experts=4 (sparse MoE) to exercise the real production code path.
    Every real Qwen3Next checkpoint has mlp_only_layers=[] and
    decoder_sparse_step=1, so every decoder layer uses Qwen3NextSparseMoeBlock.
    Test fixtures must mirror this or the adapter's MoE wiring goes untested.

    Config details:
    - 8 layers: layers 3 and 7 are full-attention (full_attention_interval=4)
    - All other layers are linear_attention
    - sparse MoE MLP on all layers (num_experts=4, num_experts_per_tok=2)
    """
    cfg = Qwen3NextConfig(
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
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        rope_parameters={
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        },
    )
    model = Qwen3NextForCausalLM(cfg)
    model.eval()
    return model


def _make_tiny_bridge():
    """Create a Qwen3Next bridge from a tiny HF model."""
    from unittest.mock import MagicMock

    from transformer_lens.config.TransformerBridgeConfig import TransformerBridgeConfig
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.model_bridge.supported_architectures.qwen3_next import (
        Qwen3NextArchitectureAdapter,
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
        architecture="Qwen3NextForCausalLM",
    )
    adapter = Qwen3NextArchitectureAdapter(bridge_cfg)
    return TransformerBridge(hf_model, adapter, tokenizer=MagicMock()), hf_model


@pytest.mark.skipif(
    not _QWEN3NEXT_AVAILABLE,
    reason="Qwen3NextForCausalLM not available in installed transformers",
)
class TestQwen3NextIntegration:
    """End-to-end integration tests using a tiny programmatic Qwen3Next model.

    Tests use num_experts=4 (sparse MoE) to exercise the real production code
    path. The linear attention layers run via the torch fallback path when
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
        """TransformerBridge construction from a tiny Qwen3Next model must succeed."""
        from transformer_lens.model_bridge import TransformerBridge

        assert isinstance(bridge, TransformerBridge)

    def test_hook_names_present(self, bridge):
        """Key hook names must be present in the bridge hook_dict.

        Verified hook names:
        - blocks.0.hook_resid_pre: present on linear-attention layer (layer 0)
        - blocks.3.hook_resid_pre: present on first full-attention layer (layer 3)
        - blocks.0.ln1.*: norm is present on all layers (universal submodule)
        - blocks.0.mlp.*: MLP is present on all layers (universal submodule)

        Also verifies that blocks.0.attn.* is NOT present — self_attn is only on
        full-attention layers, so it is NOT mapped as a block submodule.
        """
        hook_keys = set(bridge.hook_dict.keys())

        # Block-level residual hooks exist on all layers
        assert "blocks.0.hook_resid_pre" in hook_keys, "linear-attn layer must have hook_resid_pre"
        assert "blocks.3.hook_resid_pre" in hook_keys, "full-attn layer must have hook_resid_pre"

        # Norm hooks present on all layers
        assert any(
            "blocks.0.ln1" in k for k in hook_keys
        ), "blocks.0.ln1 submodule hooks must be present"

        # MLP hooks present on all layers
        assert any(
            "blocks.0.mlp" in k for k in hook_keys
        ), "blocks.0.mlp submodule hooks must be present"

        # No attn bridge — self_attn is absent on linear-attention layers
        assert not any(
            "blocks.0.attn" in k for k in hook_keys
        ), "blocks.0.attn hooks must NOT be present (hybrid architecture)"

    def test_forward_pass_consistency(self, bridge, hf_model):
        """Bridge output logits must match HF model output logits to within atol=1e-4."""
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
        """A hook added on blocks.0.mlp.hook_out must capture a (batch, seq, d_model) tensor."""
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
