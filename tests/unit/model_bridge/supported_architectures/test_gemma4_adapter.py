"""Unit tests for the Gemma 4 architecture adapter."""

from types import SimpleNamespace

import pytest
import torch

from tests.unit.model_bridge.supported_architectures.helpers import (
    wire_attention_bridge,
)
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    DelegatedAttentionBlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.gemma4 import (
    Gemma4ArchitectureAdapter,
)

ARCH = "Gemma4ForConditionalGeneration"
ARCH_UNIFIED = "Gemma4UnifiedForConditionalGeneration"


def _cfg(arch: str = ARCH, **kwargs) -> TransformerBridgeConfig:
    cfg = TransformerBridgeConfig(
        d_model=1536,
        d_head=256,
        n_heads=8,
        n_layers=35,
        n_ctx=131072,
        d_vocab=262144,
        n_key_value_heads=1,
        architecture=arch,
        **kwargs,
    )
    # Both variants are multimodal (have vision_config + embed_vision).
    cfg.vision_config = SimpleNamespace(
        hidden_size=2048,
        num_hidden_layers=27,
        num_attention_heads=16,
    )
    cfg.vision_soft_tokens_per_image = 256
    return cfg


def _adapter(arch: str = ARCH) -> Gemma4ArchitectureAdapter:
    return Gemma4ArchitectureAdapter(_cfg(arch))


def test_config_flags():
    a = _adapter()
    # Multimodal (Gemma4ForConditionalGeneration has vision tower + projector).
    assert a.cfg.is_multimodal is True
    # PLE / layer_scalar / MoE residual topology is not fold-safe.
    assert a.supports_fold_ln is False
    assert a.weight_processing_conversions == {}
    assert a.cfg.normalization_type == "RMS"
    # Gemma4RMSNorm scales by weight directly — no (1 + weight) offset, unlike Gemma 1-3.
    assert a.cfg.rmsnorm_uses_offset is False
    assert a.cfg.positional_embedding_type == "rotary"
    assert a.applicable_phases == [1, 2, 4]


def test_config_flags_unified():
    """Gemma4UnifiedForConditionalGeneration (12B) is encoder-free but still multimodal:
    has model.embed_vision (raw-patch projector) but no model.vision_tower."""
    a = _adapter(ARCH_UNIFIED)
    assert a.cfg.is_multimodal is True
    assert "vision_encoder" not in a.component_mapping
    assert "vision_projector" in a.component_mapping
    assert a.component_mapping["vision_projector"].name == "model.embed_vision"


def test_text_path_nested_under_language_model():
    m = _adapter().component_mapping
    assert m["embed"].name == "model.language_model.embed_tokens"
    assert m["rotary_emb"].name == "model.language_model.rotary_emb"
    assert m["blocks"].name == "model.language_model.layers"
    assert m["ln_final"].name == "model.language_model.norm"
    assert m["unembed"].name == "lm_head"
    assert isinstance(m["embed"], EmbeddingBridge)
    assert isinstance(m["rotary_emb"], RotaryEmbeddingBridge)
    assert isinstance(m["blocks"], DelegatedAttentionBlockBridge)
    assert isinstance(m["unembed"], UnembeddingBridge)


def test_vision_components_present_for_multimodal():
    """Gemma4ForConditionalGeneration has vision_tower + embed_vision."""
    m = _adapter().component_mapping
    assert "vision_encoder" in m
    assert "vision_projector" in m
    assert m["vision_encoder"].name == "model.vision_tower"
    assert m["vision_projector"].name == "model.embed_vision"
    assert isinstance(m["vision_projector"], GeneralizedComponent)
    # Vision config fields extracted from vision_config.
    a = _adapter()
    assert a.cfg.vision_hidden_size == 2048
    assert a.cfg.vision_num_layers == 27
    assert a.cfg.vision_num_heads == 16
    assert a.cfg.mm_tokens_per_image == 256


def test_block_decomposition():
    blocks = _adapter().component_mapping["blocks"]
    for name in ("attn", "mlp"):
        assert name in blocks.submodules
    # Sandwich norms (same shape as Gemma 2/3) under canonical keys.
    for norm in ("ln1", "ln1_post", "ln2", "ln2_post"):
        assert norm in blocks.submodules
        assert blocks.submodules[norm].optional is False


def test_split_qkv_fork_aliases_absent():
    """Attention is delegated wholesale to HF; per-layer structure is heterogeneous
    (KV-shared layers have no k/v projections), so the split-qkv fork aliases
    do not apply."""
    blocks = _adapter().component_mapping["blocks"]
    for alias in ("hook_q_input", "hook_k_input", "hook_v_input", "hook_attn_in"):
        assert alias not in blocks.hook_aliases
    # The single-stream residual aliases remain, redirected through the sandwich norms.
    assert blocks.hook_aliases["hook_resid_mid"] == "ln2.hook_in"
    assert blocks.hook_aliases["hook_attn_out"] == "ln1_post.hook_out"
    assert blocks.hook_aliases["hook_mlp_out"] == "ln2_post.hook_out"


def test_kv_shared_and_k_eq_v_submodules_are_optional():
    """KV-shared layers (E2B/E4B) drop k/v proj + norms; K==V global-attention
    layers (31B / 26B-A4B) drop v_proj."""
    attn = _adapter().component_mapping["blocks"].submodules["attn"]
    assert attn.submodules["q"].optional is False
    assert attn.submodules["o"].optional is False
    assert attn.submodules["q_norm"].optional is False
    for shared in ("k", "v", "k_norm", "v_norm"):
        assert attn.submodules[shared].optional is True
    assert isinstance(attn.submodules["q"], LinearBridge)


def test_per_layer_embedding_submodules_are_optional():
    """PLE modules exist only when hidden_size_per_layer_input > 0 (E2B/E4B)."""
    blocks = _adapter().component_mapping["blocks"]
    for name in (
        "per_layer_input_gate",
        "per_layer_projection",
        "post_per_layer_input_norm",
    ):
        assert blocks.submodules[name].optional is True


def test_moe_submodules_are_optional():
    """MoE branch exists only when enable_moe_block (26B-A4B)."""
    blocks = _adapter().component_mapping["blocks"]
    for name in (
        "router",
        "experts",
        "pre_feedforward_layernorm_2",
        "post_feedforward_layernorm_1",
        "post_feedforward_layernorm_2",
    ):
        assert blocks.submodules[name].optional is True


def test_gated_mlp_decomposition():
    mlp = _adapter().component_mapping["blocks"].submodules["mlp"]
    assert isinstance(mlp, GatedMLPBridge)
    assert mlp.submodules["gate"].name == "gate_proj"
    assert mlp.submodules["in"].name == "up_proj"
    assert mlp.submodules["out"].name == "down_proj"
    assert mlp.hook_aliases == {
        "hook_pre": "gate.hook_out",
        "hook_pre_linear": "in.hook_out",
        "hook_post": "out.hook_in",
    }


class TestGemma4AttentionHookSurface:
    """Attention must expose the gemma1/2/3 hook surface.

    It was a bare `GeneralizedComponent` (hook_in/hook_out only), so
    hook_pattern, hook_attn_scores and the hook_q/k/v/z aliases were all
    absent and q/k/v fired flat rather than per-head. HF still does the
    attention math; this is observability only. Gemma4's MLP got the
    equivalent fix in #1650.
    """

    D_MODEL, N_HEADS, N_KV_HEADS, D_HEAD, BATCH, SEQ = 64, 8, 1, 8, 2, 4

    def _small_adapter(self):
        # Built from _cfg (which supplies vision_config) then shrunk: the
        # head-split conversion is a no-op unless the flat width equals
        # n_heads * d_head, so the fake's dims and the cfg must agree.
        cfg = _cfg()
        cfg.d_model, cfg.d_head = self.D_MODEL, self.D_HEAD
        cfg.n_heads, cfg.n_key_value_heads = self.N_HEADS, self.N_KV_HEADS
        cfg.n_layers, cfg.n_ctx, cfg.d_vocab = 2, 64, 128
        return Gemma4ArchitectureAdapter(cfg)

    def _wire(self):
        from tests.unit.model_bridge.supported_architectures.helpers import (
            FakeDelegatedAttention,
        )

        hf_attn = FakeDelegatedAttention(self.D_MODEL, self.N_HEADS, self.N_KV_HEADS, self.D_HEAD)
        return wire_attention_bridge(
            self._small_adapter(),
            hf_attn,
            component="blocks.0.attn",
            expected_type=AttentionBridge,
        )

    def test_q_hook_is_head_split(self):
        bridge = self._wire()
        seen: dict = {}
        bridge.q.hook_out.add_hook(lambda t, hook: seen.__setitem__("q", tuple(t.shape)))
        with torch.no_grad():
            bridge(hidden_states=torch.randn(self.BATCH, self.SEQ, self.D_MODEL))
        assert seen["q"] == (self.BATCH, self.SEQ, self.N_HEADS, self.D_HEAD), seen

    def test_pattern_scores_and_z_hooks_fire(self):
        bridge = self._wire()
        fired: dict = {}
        for name in ("hook_pattern", "hook_attn_scores"):
            assert hasattr(bridge, name), f"{name} missing"
            getattr(bridge, name).add_hook(
                lambda t, hook, n=name: fired.__setitem__(n, tuple(t.shape))
            )
        bridge.o.hook_in.add_hook(lambda t, hook: fired.__setitem__("z", tuple(t.shape)))
        with torch.no_grad():
            bridge(hidden_states=torch.randn(self.BATCH, self.SEQ, self.D_MODEL))
        expected = (self.BATCH, self.N_HEADS, self.SEQ, self.SEQ)
        assert fired.get("hook_pattern") == expected, fired
        assert fired.get("hook_attn_scores") == expected, fired
        assert fired.get("z") == (self.BATCH, self.SEQ, self.N_HEADS, self.D_HEAD), fired


class TestGemma4HeterogeneousKVGeometry:
    """Weight accessors must not silently mis-factorize minority-geometry layers.

    Gemma4's config is genuinely per-layer (num_key_value_heads=4 with
    head_dim=256 on sliding layers vs global_* variants on full-attention
    layers); TL stores the majority scalar. On a minority layer the K width
    still DIVIDES evenly by the majority head count, so einops would return a
    wrong-shaped factorization with no error.
    """

    def _bridge_with_kv_width(self, kv_out_features):
        from tests.unit.model_bridge.supported_architectures.helpers import (
            wire_attention_bridge,
        )

        cfg = _cfg()
        cfg.d_model, cfg.d_head = 64, 8
        cfg.n_heads, cfg.n_key_value_heads = 8, 4
        cfg.n_layers, cfg.n_ctx, cfg.d_vocab = 2, 64, 128
        adapter = Gemma4ArchitectureAdapter(cfg)

        class _Attn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(64, 64, bias=False)
                self.k_proj = torch.nn.Linear(64, kv_out_features, bias=False)
                self.v_proj = torch.nn.Linear(64, kv_out_features, bias=False)
                self.o_proj = torch.nn.Linear(64, 64, bias=False)
                for name in ("q_norm", "k_norm", "v_norm"):
                    setattr(self, name, torch.nn.Identity())

        return wire_attention_bridge(
            adapter, _Attn(), component="blocks.0.attn", expected_type=AttentionBridge
        )

    def test_majority_geometry_factorizes(self) -> None:
        bridge = self._bridge_with_kv_width(4 * 8)  # n_kv * d_head
        assert bridge.W_K.shape == (4, 64, 8)

    def test_minority_geometry_raises_not_misfactorizes(self) -> None:
        """64 divides evenly by n_kv=4 (d_head 16 != 8), so einops would have
        happily produced a wrong [4, 64, 16] with no error."""
        bridge = self._bridge_with_kv_width(64)
        with pytest.raises(ValueError, match="geometry differs"):
            _ = bridge.W_K
