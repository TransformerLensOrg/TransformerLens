"""Unit tests for the Qwen3_5 architecture adapter.

Qwen3_5 is supported only via TransformerBridge, not HookedTransformer.
"""

from types import SimpleNamespace

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

        class DummyModel:
            def __init__(self, config):
                self.config = config

            def parameters(self):
                return iter(())

        class DummyModelClass:
            seen_config = None

            @classmethod
            def from_config(cls, config, **kwargs):
                cls.seen_config = config
                return DummyModel(config)

        class DummyBridge(TransformerBridge):
            def __init__(self, hf_model, adapter, tokenizer):
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
        monkeypatch.setattr(source, "TransformerBridge", DummyBridge)
        monkeypatch.setattr(source, "setup_tokenizer", lambda tokenizer, **kwargs: tokenizer)

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

    def test_block_submodules_keys(self, adapter):
        submodules = adapter.component_mapping["blocks"].submodules
        assert set(submodules.keys()) == {"ln1", "ln2", "mlp", "attn", "linear_attn"}

    def test_mlp_submodule_keys(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"gate", "in", "out"}

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


class TestQwen3_5ConfigAttributes:
    """cfg attributes set by the adapter."""

    @pytest.fixture
    def adapter(self, qwen3_5_dependency_available):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
            Qwen3_5ArchitectureAdapter,
        )

        cfg = _make_bridge_cfg()
        return Qwen3_5ArchitectureAdapter(cfg)

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


@pytest.mark.skipif(
    not _QWEN3_5_AVAILABLE,
    reason="Qwen3_5TextConfig / Qwen3_5ForCausalLM not available in installed transformers",
)
class TestQwen3_5GatedDeltaNetEffectiveAttention:
    """Faithfulness gate for GatedDeltaNetBridge.compute_effective_attention.

    The reconstruction is a documented heuristic: (1) interior hooks fire only on
    the hooked prefill path (use_cache=False); (2) the hooked Q/K are pre-norm
    while the kernel L2-normalizes them; (3) the gated-linear-attention form omits
    the delta rule's key-removal term. These tests pin the measured divergences.
    """

    LINEAR_LAYER = 0  # full-attn lands at 3 and 7 (interval=4); 0 is GatedDeltaNet
    SEQ_LEN = 12

    @pytest.fixture(scope="class")
    def bridge(self):
        br, _ = _make_tiny_bridge()
        return br

    @pytest.fixture(scope="class")
    def cache(self, bridge):
        import torch

        torch.manual_seed(0)
        tokens = torch.randint(0, 512, (1, self.SEQ_LEN))
        with torch.no_grad():
            # use_cache=False forces the hooked prefill path so interior hooks fire.
            _, cache = bridge.run_with_cache(tokens, use_cache=False)
        return cache

    @staticmethod
    def _build_M(cache, layer, *, l2norm, dtype):
        """Recompute M from cached hooks, independent of the bridge impl."""
        import torch
        import torch.nn.functional as F

        prefix = f"blocks.{layer}.linear_attn"
        q = cache[f"{prefix}.hook_q"].to(dtype)
        k = cache[f"{prefix}.hook_k"].to(dtype)
        beta = cache[f"{prefix}.hook_beta"].to(dtype)
        g = cache[f"{prefix}.hook_log_decay"].to(dtype)
        if l2norm:
            q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
        if q.shape[2] < beta.shape[-1]:
            rep = beta.shape[-1] // q.shape[2]
            q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        qk = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3).transpose(-2, -1))
        cum = torch.cumsum(g.permute(0, 2, 1), dim=-1)
        L_log = cum[:, :, :, None] - cum[:, :, None, :]
        seq = q.shape[1]
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        L = torch.where(mask[None, None], torch.exp(L_log), torch.zeros_like(L_log))
        return qk * beta.permute(0, 2, 1)[:, :, None, :] * L

    def test_requires_use_cache_false(self, bridge):
        """Default cache path omits interior hooks -> compute_effective_attention raises."""
        import torch

        tokens = torch.randint(0, 512, (1, self.SEQ_LEN))
        with torch.no_grad():
            _, cache = bridge.run_with_cache(tokens)  # default: cache allocated
        mixer = bridge.blocks[self.LINEAR_LAYER].linear_attn
        with pytest.raises(RuntimeError, match="in cache"):
            mixer.compute_effective_attention(cache, layer_idx=self.LINEAR_LAYER)

    def test_shape_causal_finite(self, bridge, cache):
        import torch

        mixer = bridge.blocks[self.LINEAR_LAYER].linear_attn
        M = mixer.compute_effective_attention(cache, layer_idx=self.LINEAR_LAYER)
        assert M.ndim == 4 and M.shape[-1] == M.shape[-2] == self.SEQ_LEN
        assert torch.isfinite(M).all()
        upper = torch.triu(torch.ones(self.SEQ_LEN, self.SEQ_LEN, dtype=torch.bool), diagonal=1)
        assert torch.all(M[..., upper] == 0), "effective attention must be causal"

    def test_impl_matches_fp64_reference(self, bridge, cache):
        """The impl must match its own fp64 recomputation (no numerical/algorithm bug)."""
        import torch

        mixer = bridge.blocks[self.LINEAR_LAYER].linear_attn
        M = mixer.compute_effective_attention(cache, layer_idx=self.LINEAR_LAYER)
        M_ref = self._build_M(cache, self.LINEAR_LAYER, l2norm=False, dtype=torch.float64)
        rel = (M.double() - M_ref).abs().max().item() / max(M_ref.abs().max().item(), 1e-12)
        assert rel < 1e-5, f"impl vs fp64 pre-norm reference rel diff {rel:.2e}"

    def test_l2norm_approximation_divergence_measured(self, bridge, cache):
        """Pin the L2-norm approximation divergence (documented in the docstring).

        The hooked Q/K are pre-norm; the kernel L2-normalizes them. On the random-
        init fixture (small, non-uniform Q/K norms) the relative divergence is ~1.0.
        """
        import torch

        M_prenorm = self._build_M(cache, self.LINEAR_LAYER, l2norm=False, dtype=torch.float64)
        M_l2 = self._build_M(cache, self.LINEAR_LAYER, l2norm=True, dtype=torch.float64)
        rel = (M_prenorm - M_l2).abs().max().item() / max(M_l2.abs().max().item(), 1e-12)
        assert torch.isfinite(torch.tensor(rel))
        assert 0.5 < rel < 1.5, (
            f"L2-norm approximation divergence {rel:.3f} outside the documented ~1.0 "
            "band for the random-init fixture; update the docstring if the model changed."
        )


@pytest.mark.skipif(
    not _QWEN3_5_AVAILABLE,
    reason="Qwen3_5TextConfig / Qwen3_5ForCausalLM not available in installed transformers",
)
class TestQwen3_5GatedDeltaNetState:
    """Faithful recurrent-state reconstruction + eager-scan interventions for GDN.

    Unlike compute_effective_attention (a gated-linear-attention heuristic that drops
    key removal), compute_ssm_state replays the *full* delta rule, so ``S_t^T q_t``
    must reconstruct the fused kernel's hook_recurrence_out to fp tolerance.
    """

    LINEAR_LAYER = 0  # full-attn lands at 3 and 7 (interval=4); 0 is GatedDeltaNet
    SEQ_LEN = 10

    @pytest.fixture(scope="class")
    def bridge(self):
        br, _ = _make_tiny_bridge()
        return br

    @pytest.fixture(scope="class")
    def cache(self, bridge):
        import torch

        torch.manual_seed(0)
        tokens = torch.randint(0, 512, (2, self.SEQ_LEN))
        with torch.no_grad():
            _, cache = bridge.run_with_cache(tokens, use_cache=False)
        return cache

    @staticmethod
    def _ref_scan(cache, layer, dtype):
        """Independent gated-delta-rule scan from cached hooks (not the impl).

        Uses the kernel's L2-norm convention (eps=1e-6) so the only differences vs
        the impl are reduction order (fp32) and precision (fp64).
        """
        import torch

        def l2norm(x):
            return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)

        prefix = f"blocks.{layer}.linear_attn"
        q = cache[f"{prefix}.hook_q"].to(dtype)
        k = cache[f"{prefix}.hook_k"].to(dtype)
        v = cache[f"{prefix}.hook_v"].to(dtype)
        beta = cache[f"{prefix}.hook_beta"].to(dtype)
        g = cache[f"{prefix}.hook_log_decay"].to(dtype)
        n_v = v.shape[2]
        if q.shape[2] < n_v:
            rep = n_v // q.shape[2]
            q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        q = l2norm(q) * (q.shape[-1] ** -0.5)
        k = l2norm(k)
        b, seq, _, kd = q.shape
        vd = v.shape[-1]
        state = torch.zeros(b, n_v, kd, vd, dtype=dtype)
        states, outs = [], []
        for t in range(seq):
            state = state * g[:, t].exp()[:, :, None, None]
            kv = (state * k[:, t][:, :, :, None]).sum(-2)
            delta = (v[:, t] - kv) * beta[:, t][:, :, None]
            state = state + k[:, t][:, :, :, None] * delta[:, :, None, :]
            states.append(state)
            outs.append((state * q[:, t][:, :, :, None]).sum(-2))
        return torch.stack(states, 1), torch.stack(outs, 1)

    def test_requires_use_cache_false(self, bridge):
        """Default cache path omits interior hooks -> compute_ssm_state raises."""
        import torch

        tokens = torch.randint(0, 512, (1, self.SEQ_LEN))
        with torch.no_grad():
            _, cache = bridge.run_with_cache(tokens)  # default: cache allocated
        mixer = bridge.blocks[self.LINEAR_LAYER].linear_attn
        with pytest.raises(RuntimeError, match="in cache"):
            mixer.compute_ssm_state(cache, layer_idx=self.LINEAR_LAYER)

    def test_state_shape(self, bridge, cache):
        mixer = bridge.blocks[self.LINEAR_LAYER].linear_attn
        S = mixer.compute_ssm_state(cache, layer_idx=self.LINEAR_LAYER)
        # [batch, seq, n_v_heads, head_k_dim, head_v_dim]
        assert S.ndim == 5
        assert S.shape[1] == self.SEQ_LEN
        assert S.shape[2] == cache[f"blocks.{self.LINEAR_LAYER}.linear_attn.hook_v"].shape[2]

    def test_reconstructs_recurrence_out(self, bridge, cache):
        """o_t = S_t^T q_t must match the fused kernel's hook_recurrence_out (faithful)."""
        import torch

        mixer = bridge.blocks[self.LINEAR_LAYER].linear_attn
        S = mixer.compute_ssm_state(cache, layer_idx=self.LINEAR_LAYER)
        # o_t from the impl's own state, using the impl's normalized/scaled q.
        prefix = f"blocks.{self.LINEAR_LAYER}.linear_attn"
        q = cache[f"{prefix}.hook_q"].float()
        n_v = cache[f"{prefix}.hook_v"].shape[2]
        if q.shape[2] < n_v:
            q = q.repeat_interleave(n_v // q.shape[2], dim=2)
        q = mixer._l2norm(q) * (q.shape[-1] ** -0.5)
        o = torch.einsum("bshkv,bshk->bshv", S, q)
        rec = cache[f"{prefix}.hook_recurrence_out"].float()
        rel = (o - rec).abs().max().item() / max(rec.abs().max().item(), 1e-8)
        assert rel < 1e-4, f"delta-rule state reconstruction rel diff {rel:.2e}"

    def test_impl_matches_independent_reference(self, bridge, cache):
        """The impl state must match an independent reimplementation of the scan.

        The impl works in fp32 by design, so it is compared to a fp32 reference at
        tight tolerance (independent-algorithm check, only reduction-order noise).
        The residual against a fp64 reference is then shown to be no larger than
        fp32's own rounding gap — i.e. the impl adds no error beyond its working
        precision (an actual algorithm bug would be O(1), not fp32-scale).
        """
        import torch

        mixer = bridge.blocks[self.LINEAR_LAYER].linear_attn
        S = mixer.compute_ssm_state(cache, layer_idx=self.LINEAR_LAYER).float()
        S_ref32, _ = self._ref_scan(cache, self.LINEAR_LAYER, torch.float32)
        S_ref64, _ = self._ref_scan(cache, self.LINEAR_LAYER, torch.float64)

        denom = max(S_ref64.abs().max().item(), 1e-12)
        rel32 = (S - S_ref32).abs().max().item() / denom
        assert rel32 < 1e-5, f"impl vs independent fp32 reference rel diff {rel32:.2e}"

        # fp64 gap of the impl vs fp64 gap of the fp32 reference: comparable ⇒ fp32 noise.
        impl_vs_fp64 = (S.double() - S_ref64).abs().max().item() / denom
        ref_fp32_vs_fp64 = (S_ref32.double() - S_ref64).abs().max().item() / denom
        assert impl_vs_fp64 <= 4 * ref_fp32_vs_fp64 + 1e-6, (
            f"impl fp64 gap {impl_vs_fp64:.2e} exceeds fp32 rounding gap "
            f"{ref_fp32_vs_fp64:.2e} — not attributable to fp32 accumulation"
        )

    def test_time_step_matches_full(self, bridge, cache):
        import torch

        mixer = bridge.blocks[self.LINEAR_LAYER].linear_attn
        S = mixer.compute_ssm_state(cache, layer_idx=self.LINEAR_LAYER)
        for t in (0, self.SEQ_LEN // 2, self.SEQ_LEN - 1):
            St = mixer.compute_ssm_state(cache, layer_idx=self.LINEAR_LAYER, time_step=t)
            assert torch.equal(St, S[:, t])


@pytest.mark.skipif(
    not _QWEN3_5_AVAILABLE,
    reason="Qwen3_5TextConfig / Qwen3_5ForCausalLM not available in installed transformers",
)
class TestQwen3_5GatedDeltaNetEagerScan:
    """Opt-in eager_scan path: fused-kernel parity, hook_ssm_state interventions."""

    LINEAR_LAYER = 0
    SEQ_LEN = 10

    @pytest.fixture(scope="class")
    def bridge(self):
        br, _ = _make_tiny_bridge()
        return br

    @pytest.fixture(scope="class")
    def tokens(self):
        import torch

        torch.manual_seed(0)
        return torch.randint(0, 512, (2, self.SEQ_LEN))

    @staticmethod
    def _eager(bridge):
        import contextlib

        from transformer_lens.model_bridge.generalized_components import (
            GatedDeltaNetBridge,
        )

        @contextlib.contextmanager
        def _ctx():
            mixers = [
                b.linear_attn
                for b in bridge.blocks
                if isinstance(getattr(b, "linear_attn", None), GatedDeltaNetBridge)
            ]
            for m in mixers:
                m.eager_scan = True
            try:
                yield
            finally:
                for m in mixers:
                    m.eager_scan = False

        return _ctx()

    def test_default_path_has_no_state_hook(self, bridge, tokens):
        import torch

        with torch.no_grad():
            _, cache = bridge.run_with_cache(tokens, use_cache=False)
        assert not any("hook_ssm_state" in k for k in cache)

    def test_eager_scan_matches_fused(self, bridge, tokens):
        """Eager delta-rule scan reproduces the fused-kernel logits to fp tolerance."""
        import torch

        with torch.no_grad():
            base = bridge(tokens, use_cache=False)
            with self._eager(bridge):
                eager = bridge(tokens, use_cache=False)
        rel = (eager.float() - base.float()).abs().max().item() / max(
            base.float().abs().max().item(), 1e-8
        )
        assert rel < 1e-3, f"eager vs fused logit parity rel diff {rel:.2e}"

    def test_eager_scan_fires_state_hook(self, bridge, tokens):
        import torch

        captured = {}

        def _grab(t, hook):
            captured["shape"] = tuple(t.shape)
            return t

        with self._eager(bridge):
            with torch.no_grad():
                bridge.run_with_hooks(
                    tokens,
                    use_cache=False,
                    fwd_hooks=[(f"blocks.{self.LINEAR_LAYER}.linear_attn.hook_ssm_state", _grab)],
                )
        assert captured.get("shape") is not None
        assert captured["shape"][1] == self.SEQ_LEN and len(captured["shape"]) == 5

    def test_state_patch_changes_logits(self, bridge, tokens):
        """Zeroing the state trajectory (hook_ssm_state) must change the output."""
        import torch

        def _zero(t, hook):
            return torch.zeros_like(t)

        with self._eager(bridge):
            with torch.no_grad():
                base = bridge(tokens, use_cache=False)
                patched = bridge.run_with_hooks(
                    tokens,
                    use_cache=False,
                    fwd_hooks=[(f"blocks.{self.LINEAR_LAYER}.linear_attn.hook_ssm_state", _zero)],
                )
        assert (patched.float() - base.float()).abs().max().item() > 1e-4

    def test_write_knockout_via_beta_alias(self, bridge, tokens):
        """hook_ssm_write (alias -> hook_beta) knockout propagates through the scan."""
        import torch

        def _zero(t, hook):
            return torch.zeros_like(t)

        with self._eager(bridge):
            with torch.no_grad():
                base = bridge(tokens, use_cache=False)
                ko = bridge.run_with_hooks(
                    tokens,
                    use_cache=False,
                    fwd_hooks=[(f"blocks.{self.LINEAR_LAYER}.linear_attn.hook_ssm_write", _zero)],
                )
        assert (ko.float() - base.float()).abs().max().item() > 1e-4

    def test_disabling_restores_default(self, bridge, tokens):
        """Toggling eager_scan off must return bit-identical fused-path logits."""
        import torch

        with torch.no_grad():
            first = bridge(tokens, use_cache=False)
            with self._eager(bridge):
                _ = bridge(tokens, use_cache=False)
            after = bridge(tokens, use_cache=False)
        assert torch.equal(first, after)


def _make_tiny_gqa_bridge():
    """Qwen3_5 bridge with n_v_heads (4) > n_k_heads (2) to exercise the GDN GQA branch.

    The default _make_tiny_bridge uses n_k==n_v==4, so the repeat_interleave GQA
    expansion in GatedDeltaNetBridge._hooked_forward never runs. Here Q/K carry 2
    heads and V carries 4, so the recurrence path must expand Q/K 2x.
    """
    from unittest.mock import MagicMock

    from transformer_lens.config.transformer_bridge_config import (
        TransformerBridgeConfig,
    )
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
        Qwen3_5ArchitectureAdapter,
    )

    cfg = Qwen3_5TextConfig(
        hidden_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256,
        vocab_size=512,
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,  # < value heads -> GQA expansion required
        linear_num_value_heads=4,
        rope_parameters={
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        },
    )
    hf_model = _Qwen3_5ForCausalLM(cfg).eval()
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
    return (
        TransformerBridge(hf_model, Qwen3_5ArchitectureAdapter(bridge_cfg), tokenizer=MagicMock()),
        hf_model,
    )


@pytest.mark.skipif(
    not _QWEN3_5_AVAILABLE,
    reason="Qwen3_5TextConfig / Qwen3_5ForCausalLM not available in installed transformers",
)
class TestQwen3_5GatedDeltaNetGQA:
    """Cover the GDN GQA expansion (n_v_heads > n_k_heads) in CI, not just env-gated
    real-weight tests. Forward parity requires the Q/K repeat_interleave to be correct."""

    LINEAR_LAYER = 0

    def test_gqa_branch_engaged_and_parity(self):
        import torch

        bridge, hf = _make_tiny_gqa_bridge()
        assert hf.config.linear_num_value_heads > hf.config.linear_num_key_heads  # GQA active
        torch.manual_seed(0)
        tokens = torch.randint(0, 512, (1, 10))
        with torch.no_grad():
            bridge_logits = bridge(tokens, use_cache=False)
            hf_logits = hf(tokens).logits
        rel = (bridge_logits.float() - hf_logits.float()).abs().max().item() / max(
            hf_logits.float().abs().max().item(), 1e-8
        )
        assert rel < 1e-4, f"GQA GDN forward parity rel diff {rel:.2e}"

    def test_gqa_hook_head_counts(self):
        """hook_q is pre-GQA (n_k_heads); hook_recurrence_out is post-expansion (n_v_heads)."""
        import torch

        bridge, hf = _make_tiny_gqa_bridge()
        tokens = torch.randint(0, 512, (1, 8))
        with torch.no_grad():
            _, cache = bridge.run_with_cache(tokens, use_cache=False)
        p = f"blocks.{self.LINEAR_LAYER}.linear_attn"
        assert cache[f"{p}.hook_q"].shape[2] == hf.config.linear_num_key_heads
        assert cache[f"{p}.hook_recurrence_out"].shape[2] == hf.config.linear_num_value_heads
