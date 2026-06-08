"""Unit tests for the Qwen3_5 architecture adapter.

Qwen3_5 is supported only via TransformerBridge, not HookedTransformer.
"""

from types import SimpleNamespace

import pytest
import torch.nn as nn

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


@pytest.fixture
def qwen3_5_dependency_available(monkeypatch):
    """Make adapter-only tests independent of the installed Transformers build."""
    import transformers

    monkeypatch.setattr(transformers, "__version__", "5.10.0")
    monkeypatch.setattr(transformers, "Qwen3_5ForCausalLM", object(), raising=False)


class TestQwen3_5ArchitectureDetection:
    """Tests that do not require a Transformers build with Qwen3.5 classes."""

    def test_model_type_qwen3_5_routes_to_text_only_architecture(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="qwen3_5", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Qwen3_5ForCausalLM"

    def test_model_type_qwen3_5_text_routes_to_text_only_architecture(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="qwen3_5_text", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Qwen3_5ForCausalLM"

    def test_full_conditional_generation_routes_to_multimodal_not_text_only(self):
        # Qwen3.5 multimodal is now the default route for ForConditionalGeneration
        # checkpoints; the text-only adapter is deliberately not selected for them.
        from transformer_lens.model_bridge.supported_architectures import (
            Qwen3_5ArchitectureAdapter,
        )
        from transformer_lens.model_bridge.supported_architectures.qwen3_5_multimodal import (
            Qwen3_5MultimodalArchitectureAdapter,
        )

        assert "Qwen3_5ForConditionalGeneration" in SUPPORTED_ARCHITECTURES
        assert "Qwen3_5ForConditionalGeneration" in HF_SUPPORTED_ARCHITECTURES
        assert (
            SUPPORTED_ARCHITECTURES["Qwen3_5ForConditionalGeneration"]
            is Qwen3_5MultimodalArchitectureAdapter
        )
        assert (
            SUPPORTED_ARCHITECTURES["Qwen3_5ForConditionalGeneration"]
            is not Qwen3_5ArchitectureAdapter
        )


class TestQwen3_5LoadingGuards:
    """Text-only routing and preloaded-model guards."""

    def test_prepare_loading_swaps_top_level_config_for_text_config(
        self, qwen3_5_dependency_available
    ):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

        adapter = Qwen3_5ArchitectureAdapter(_make_bridge_cfg())
        text_config = SimpleNamespace(model_type="qwen3_5_text")
        full_config = SimpleNamespace(model_type="qwen3_5", text_config=text_config)
        model_kwargs = {"config": full_config}

        adapter.prepare_loading("Qwen/Qwen3.5-0.8B", model_kwargs)

        assert model_kwargs["config"] is text_config

    def test_prepare_model_accepts_text_only_model(self, qwen3_5_dependency_available):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

        adapter = Qwen3_5ArchitectureAdapter(_make_bridge_cfg())
        hf_model = SimpleNamespace(config=SimpleNamespace(architectures=["Qwen3_5ForCausalLM"]))

        adapter.prepare_model(hf_model)

    def test_prepare_model_rejects_conditional_generation_model(self, qwen3_5_dependency_available):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

        adapter = Qwen3_5ArchitectureAdapter(_make_bridge_cfg())
        hf_model = SimpleNamespace(
            config=SimpleNamespace(architectures=["Qwen3_5ForConditionalGeneration"])
        )

        with pytest.raises(ValueError, match="text-only"):
            adapter.prepare_model(hf_model)

    def test_prepare_model_rejects_unswapped_top_level_config(self, qwen3_5_dependency_available):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

        adapter = Qwen3_5ArchitectureAdapter(_make_bridge_cfg())
        hf_model = SimpleNamespace(
            config=SimpleNamespace(
                architectures=["Qwen3_5ForCausalLM"],
                text_config=SimpleNamespace(model_type="qwen3_5_text"),
            )
        )

        with pytest.raises(ValueError, match="text-only"):
            adapter.prepare_model(hf_model)

    def test_load_weights_false_uses_prepared_text_config(
        self, monkeypatch, qwen3_5_dependency_available
    ):
        from transformer_lens.model_bridge.bridge import TransformerBridge
        from transformer_lens.model_bridge.sources import transformers as source

        # boot() lives in the submodule after the package split; module-level
        # name lookups for TransformerBridge / setup_tokenizer happen there, not
        # in the package __init__.
        from transformer_lens.model_bridge.sources.transformers import (
            source as boot_module,
        )

        text_config = SimpleNamespace(
            model_type="qwen3_5_text",
            architectures=["Qwen3_5ForCausalLM"],
            hidden_size=128,
            head_dim=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            max_position_embeddings=64,
            intermediate_size=256,
            vocab_size=512,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            pad_token_id=0,
            eos_token_id=1,
        )
        # A bare qwen3_5 config routes text-only; an explicit ForConditionalGeneration arch
        # now goes to the multimodal adapter, so this exercises the text-only text_config swap.
        full_config = SimpleNamespace(
            model_type="qwen3_5",
            architectures=[],
            text_config=text_config,
            pad_token_id=0,
            eos_token_id=1,
        )

        class DummyModel(nn.Module):
            # nn.Module subclass: TransformersDriver's beartype-validated init
            # requires the underlying model to be an nn.Module after the type
            # split. Plain objects no longer suffice.
            def __init__(self, config):
                super().__init__()
                self.config = config

        class DummyModelClass:
            seen_config = None

            @classmethod
            def from_config(cls, config, **kwargs):
                cls.seen_config = config
                return DummyModel(config)

        class DummyBridge(TransformerBridge):
            # **kwargs absorbs the ``driver=`` kwarg boot() now passes after the
            # type-split refactor (TransformersDriver gets constructed in boot()
            # and threaded through to the bridge). nn.Module.__init__() must run
            # before any nn.Module-valued attribute assignment (e.g. hf_model).
            def __init__(self, hf_model, adapter, tokenizer, **kwargs):
                nn.Module.__init__(self)
                self.hf_model = hf_model
                self.adapter = adapter
                self.tokenizer = tokenizer

        class DummyTokenizer:
            bos_token_id = 2
            eos_token_id = 1

            def encode(self, text):
                return [10]

        monkeypatch.setattr(
            source.AutoConfig,
            "from_pretrained",
            staticmethod(lambda *args, **kwargs: full_config),
        )
        monkeypatch.setattr(
            source.AutoTokenizer,
            "from_pretrained",
            staticmethod(lambda *args, **kwargs: DummyTokenizer()),
        )
        monkeypatch.setattr(boot_module, "TransformerBridge", DummyBridge)
        monkeypatch.setattr(boot_module, "setup_tokenizer", lambda tokenizer, **kwargs: tokenizer)

        bridge = source.boot(
            "Qwen/Qwen3.5-0.8B",
            device="cpu",
            load_weights=False,
            model_class=DummyModelClass,
        )

        assert DummyModelClass.seen_config is text_config
        assert bridge.hf_model.config is text_config


def _make_bridge_cfg(**overrides):
    """Minimal TransformerBridgeConfig for Qwen3_5 adapter tests."""
    from transformer_lens.config.transformer_bridge_config import (
        TransformerBridgeConfig,
    )

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


class TestQwen3_5ComponentMapping:
    """self_attn is not a block submodule (absent on linear-attn layers); dense GatedMLP only."""

    @pytest.fixture
    def adapter(self, qwen3_5_dependency_available):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

        cfg = _make_bridge_cfg()
        return Qwen3_5ArchitectureAdapter(cfg)

    def test_component_mapping_keys(self, adapter):
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_embed_path(self, adapter):
        assert adapter.component_mapping["embed"].name == "model.embed_tokens"

    def test_rotary_emb_path(self, adapter):
        assert adapter.component_mapping["rotary_emb"].name == "model.rotary_emb"

    def test_blocks_path(self, adapter):
        assert adapter.component_mapping["blocks"].name == "model.layers"

    def test_ln_final_path(self, adapter):
        assert adapter.component_mapping["ln_final"].name == "model.norm"

    def test_unembed_path(self, adapter):
        assert adapter.component_mapping["unembed"].name == "lm_head"

    def test_block_submodules_keys(self, adapter):
        submodules = adapter.component_mapping["blocks"].submodules
        assert set(submodules.keys()) == {"ln1", "ln2", "mlp", "attn", "linear_attn"}

    def test_attn_is_optional(self, adapter):
        """attn is absent on linear-attention layers."""
        submodules = adapter.component_mapping["blocks"].submodules
        assert submodules["attn"].optional is True

    def test_linear_attn_is_optional(self, adapter):
        """linear_attn is absent on full-attention layers."""
        submodules = adapter.component_mapping["blocks"].submodules
        assert submodules["linear_attn"].optional is True

    def test_linear_attn_bridge_type(self, adapter):
        from transformer_lens.model_bridge.generalized_components.gated_delta_net import (
            GatedDeltaNetBridge,
        )

        submodules = adapter.component_mapping["blocks"].submodules
        assert isinstance(submodules["linear_attn"], GatedDeltaNetBridge)

    def test_ln1_path(self, adapter):
        assert adapter.component_mapping["blocks"].submodules["ln1"].name == "input_layernorm"

    def test_ln2_path(self, adapter):
        assert (
            adapter.component_mapping["blocks"].submodules["ln2"].name == "post_attention_layernorm"
        )

    def test_mlp_path(self, adapter):
        assert adapter.component_mapping["blocks"].submodules["mlp"].name == "mlp"

    def test_mlp_submodule_keys(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"gate", "in", "out"}

    def test_mlp_gate_path(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["gate"].name == "gate_proj"

    def test_mlp_in_path(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "up_proj"

    def test_mlp_out_path(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["out"].name == "down_proj"

    def test_blocks_bridge_type(self, adapter):
        from transformer_lens.model_bridge.generalized_components import BlockBridge

        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_rotary_emb_bridge_type(self, adapter):
        from transformer_lens.model_bridge.generalized_components import (
            RotaryEmbeddingBridge,
        )

        assert isinstance(adapter.component_mapping["rotary_emb"], RotaryEmbeddingBridge)

    def test_ln1_bridge_type(self, adapter):
        from transformer_lens.model_bridge.generalized_components import (
            RMSNormalizationBridge,
        )

        ln1 = adapter.component_mapping["blocks"].submodules["ln1"]
        assert isinstance(ln1, RMSNormalizationBridge)

    def test_ln2_bridge_type(self, adapter):
        from transformer_lens.model_bridge.generalized_components import (
            RMSNormalizationBridge,
        )

        ln2 = adapter.component_mapping["blocks"].submodules["ln2"]
        assert isinstance(ln2, RMSNormalizationBridge)

    def test_mlp_bridge_type(self, adapter):
        from transformer_lens.model_bridge.generalized_components import GatedMLPBridge

        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)

    def test_mlp_gate_bridge_type(self, adapter):
        from transformer_lens.model_bridge.generalized_components import LinearBridge

        gate = adapter.component_mapping["blocks"].submodules["mlp"].submodules["gate"]
        assert isinstance(gate, LinearBridge)

    def test_mlp_in_bridge_type(self, adapter):
        from transformer_lens.model_bridge.generalized_components import LinearBridge

        up = adapter.component_mapping["blocks"].submodules["mlp"].submodules["in"]
        assert isinstance(up, LinearBridge)

    def test_mlp_out_bridge_type(self, adapter):
        from transformer_lens.model_bridge.generalized_components import LinearBridge

        down = adapter.component_mapping["blocks"].submodules["mlp"].submodules["out"]
        assert isinstance(down, LinearBridge)

    def test_weight_processing_conversions_empty(self, adapter):
        """No attention submodules mapped, so no conversions."""
        assert adapter.weight_processing_conversions == {}


class TestQwen3_5ConfigAttributes:
    """cfg attributes set by the adapter."""

    @pytest.fixture
    def adapter(self, qwen3_5_dependency_available):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

        cfg = _make_bridge_cfg()
        return Qwen3_5ArchitectureAdapter(cfg)

    def test_supports_fold_ln_false(self, adapter):
        """Hybrid layers break fold_ln."""
        assert adapter.supports_fold_ln is False

    def test_n_key_value_heads_not_set_when_absent(self, qwen3_5_dependency_available):
        from transformer_lens.config.transformer_bridge_config import (
            TransformerBridgeConfig,
        )
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

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
        # When unset, n_key_value_heads must default to n_heads (standard MHA).
        assert not (
            hasattr(adapter.cfg, "n_key_value_heads")
            and adapter.cfg.n_key_value_heads is not None
            and adapter.cfg.n_key_value_heads != adapter.cfg.n_heads
        )


class TestQwen3_5PreprocessWeights:
    """q_proj rows are interleaved per-head (query, gate, query, gate, ...) — naive first-half slice is wrong."""

    N_HEADS = 4
    D_HEAD = 8
    HIDDEN_SIZE = 32

    @pytest.fixture
    def adapter(self, qwen3_5_dependency_available):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

        cfg = _make_bridge_cfg(
            n_heads=self.N_HEADS,
            d_head=self.D_HEAD,
            d_model=self.HIDDEN_SIZE,
            n_key_value_heads=self.N_HEADS,
        )
        return Qwen3_5ArchitectureAdapter(cfg)

    def _make_q_proj_weight(self):
        import torch

        total_rows = self.N_HEADS * self.D_HEAD * 2
        w = torch.zeros(total_rows, self.HIDDEN_SIZE)
        for row_idx in range(total_rows):
            w[row_idx] = float(row_idx)
        return w

    def test_q_proj_output_shape(self, adapter):
        import torch

        w = self._make_q_proj_weight()
        state_dict = {"model.layers.3.self_attn.q_proj.weight": w}
        result = adapter.preprocess_weights(state_dict)
        out = result["model.layers.3.self_attn.q_proj.weight"]
        assert out.shape == (self.N_HEADS * self.D_HEAD, self.HIDDEN_SIZE)

    def test_q_proj_selects_query_rows_not_naive_first_half(self, adapter):
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
        assert adapter.preprocess_weights({}) == {}

    def test_state_dict_without_q_proj_unchanged(self, adapter):
        import torch

        state_dict = {"model.embed_tokens.weight": torch.randn(100, self.HIDDEN_SIZE)}
        original_keys = set(state_dict.keys())
        result = adapter.preprocess_weights(state_dict)
        assert set(result.keys()) == original_keys


@pytest.mark.skipif(
    not _QWEN3_5_AVAILABLE,
    reason="Qwen3_5TextConfig / Qwen3_5ForCausalLM not available in installed transformers",
)
class TestQwen3_5ComponentTypes:
    """Top-level bridge classes — guards against silent type substitution."""

    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

        return Qwen3_5ArchitectureAdapter(_make_bridge_cfg())

    def test_embed_is_embedding_bridge(self, adapter):
        from transformer_lens.model_bridge.generalized_components import EmbeddingBridge

        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_ln_final_is_rms_norm_bridge(self, adapter):
        from transformer_lens.model_bridge.generalized_components import (
            RMSNormalizationBridge,
        )

        assert isinstance(adapter.component_mapping["ln_final"], RMSNormalizationBridge)

    def test_unembed_is_unembedding_bridge(self, adapter):
        from transformer_lens.model_bridge.generalized_components import (
            UnembeddingBridge,
        )

        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)


@pytest.mark.skipif(
    not _QWEN3_5_AVAILABLE,
    reason="Qwen3_5TextConfig / Qwen3_5ForCausalLM not available in installed transformers",
)
class TestQwen3_5AttnSubmodules:
    """Full-attention layers wire Qwen3-pattern submodules; gated q_proj half is pre-sliced."""

    @pytest.fixture
    def attn(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

        adapter = Qwen3_5ArchitectureAdapter(_make_bridge_cfg())
        return adapter.component_mapping["blocks"].submodules["attn"]

    def test_attn_is_position_embeddings_attention(self, attn):
        from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
            PositionEmbeddingsAttentionBridge,
        )

        assert isinstance(attn, PositionEmbeddingsAttentionBridge)

    def test_attn_path(self, attn):
        assert attn.name == "self_attn"

    def test_attn_qkvo_submodule_paths(self, attn):
        from transformer_lens.model_bridge.generalized_components import LinearBridge

        for sub_name, expected_path in (
            ("q", "q_proj"),
            ("k", "k_proj"),
            ("v", "v_proj"),
            ("o", "o_proj"),
        ):
            sub = attn.submodules[sub_name]
            assert isinstance(sub, LinearBridge)
            assert sub.name == expected_path

    def test_attn_q_norm_k_norm_present(self, attn):
        """Qwen3 family uses per-head Q/K RMSNorm."""
        from transformer_lens.model_bridge.generalized_components import (
            RMSNormalizationBridge,
        )

        assert isinstance(attn.submodules["q_norm"], RMSNormalizationBridge)
        assert isinstance(attn.submodules["k_norm"], RMSNormalizationBridge)
        assert attn.submodules["q_norm"].name == "q_norm"
        assert attn.submodules["k_norm"].name == "k_norm"


@pytest.mark.skipif(
    not _QWEN3_5_AVAILABLE,
    reason="Qwen3_5TextConfig / Qwen3_5ForCausalLM not available in installed transformers",
)
class TestQwen3_5HybridSpecifics:
    """Qwen3.5-specific config invariants."""

    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

        return Qwen3_5ArchitectureAdapter(_make_bridge_cfg())

    def test_gated_q_proj_flag_set(self, adapter):
        """Flag drives preprocess_weights to slice the gated half of q_proj."""
        assert getattr(adapter.cfg, "gated_q_proj", False) is True


@pytest.mark.skipif(
    not _QWEN3_5_AVAILABLE,
    reason="Qwen3_5TextConfig / Qwen3_5ForCausalLM not available in installed transformers",
)
class TestQwen3_5ArchitectureGuards:
    """Guards against drift from Qwen3 conventions."""

    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

        return Qwen3_5ArchitectureAdapter(_make_bridge_cfg())

    def test_no_norm_offset_conversions(self, adapter):
        """LLaMA-style RMSNorm — no +1 offset like Gemma."""
        for key in adapter.weight_processing_conversions:
            assert "ln1" not in key
            assert "ln2" not in key
            assert "ln_final" not in key

    def test_mlp_is_gated_not_moe(self, adapter):
        """Dense GatedMLP, not MoE (Qwen3Next has MoE)."""
        from transformer_lens.model_bridge.generalized_components import (
            GatedMLPBridge,
            MoEBridge,
        )

        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert not isinstance(mlp, MoEBridge)


def _make_tiny_hf_model():
    """Tiny Qwen3_5ForCausalLM: 8 layers, full-attn at 3 and 7 (interval=4), GatedDeltaNet elsewhere."""
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
    """Build a Qwen3_5 bridge from a tiny HF model."""
    from unittest.mock import MagicMock

    from transformer_lens.config.transformer_bridge_config import (
        TransformerBridgeConfig,
    )
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
    """End-to-end tests; linear-attn falls back to torch when flash-linear-attention is absent."""

    @pytest.fixture(scope="class")
    def bridge_and_model(self):
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
        from transformer_lens.model_bridge import TransformerBridge

        assert isinstance(bridge, TransformerBridge)

    def test_hook_names_present(self, bridge):
        """blocks.0.attn.* must NOT appear — self_attn is absent on linear-attn layers."""
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
        """MLP, full-attention, and linear-attention hooks must all fire."""
        import torch

        hook_names = [
            "blocks.0.mlp.hook_out",
            "blocks.3.attn.hook_out",
            "blocks.0.linear_attn.hook_out",
        ]
        captured: dict[str, list[torch.Tensor]] = {name: [] for name in hook_names}

        def capture_hook(name: str):
            def _capture(tensor: torch.Tensor, hook: object) -> torch.Tensor:
                captured[name].append(tensor.detach().clone())
                return tensor

            return _capture

        tokens = torch.randint(0, 512, (1, 4))
        with torch.no_grad():
            bridge.run_with_hooks(
                tokens,
                fwd_hooks=[(name, capture_hook(name)) for name in hook_names],
            )

        batch, seq, d_model = 1, 4, 128
        for hook_name, activations in captured.items():
            assert len(activations) == 1, f"{hook_name} must fire exactly once"
            assert activations[0].shape == (
                batch,
                seq,
                d_model,
            ), (
                f"Expected {hook_name} shape ({batch}, {seq}, {d_model}), "
                f"got {activations[0].shape}"
            )
