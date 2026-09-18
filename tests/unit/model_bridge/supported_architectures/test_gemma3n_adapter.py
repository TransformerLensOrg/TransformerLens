"""Unit tests for the Gemma 3n text-only architecture adapter."""

import pytest
import torch

from tests.unit.model_bridge.supported_architectures.helpers import (
    FakeDelegatedAttention,
    wire_attention_bridge,
)
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    AltUpBlockBridge,
    AttentionBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.gemma3n import (
    Gemma3nArchitectureAdapter,
)

ARCH = "Gemma3nForConditionalGeneration"


def _adapter():
    cfg = TransformerBridgeConfig(
        d_model=2048,
        d_head=256,
        n_heads=8,
        n_layers=30,
        n_ctx=4096,
        d_vocab=262400,
        n_key_value_heads=2,
        architecture=ARCH,
    )
    return ArchitectureAdapterFactory.select_architecture_adapter(cfg)


def test_missing_required_library_raises_actionable_error():
    """A missing multimodal dep surfaces a clear error, not a deep HF import failure."""

    class _FakeMissing(Gemma3nArchitectureAdapter):
        required_libraries = ["definitely_not_installed_xyz"]
        required_libraries_group = "custom-group"

    cfg = TransformerBridgeConfig(
        d_model=2048,
        d_head=256,
        n_heads=8,
        n_layers=30,
        n_ctx=4096,
        d_vocab=262400,
        n_key_value_heads=2,
        architecture=ARCH,
    )
    # The error names the missing lib and the adapter's declared dependency group (not a
    # hardcoded one).
    with pytest.raises(ImportError, match=r"definitely_not_installed_xyz.*custom-group"):
        _FakeMissing(cfg)


def test_config_flags():
    a = _adapter()
    # Text-only for now; AltUp/PLE topology is not fold-safe.
    assert a.cfg.is_multimodal is False
    assert a.supports_fold_ln is False
    assert a.weight_processing_conversions == {}
    assert a.cfg.normalization_type == "RMS"
    assert a.cfg.rmsnorm_uses_offset is True


def test_text_path_nested_under_language_model():
    m = _adapter().component_mapping
    assert m["embed"].name == "model.language_model.embed_tokens"
    assert m["blocks"].name == "model.language_model.layers"
    assert m["ln_final"].name == "model.language_model.norm"
    assert m["unembed"].name == "lm_head"
    assert isinstance(m["embed"], EmbeddingBridge)
    assert isinstance(m["blocks"], AltUpBlockBridge)
    assert isinstance(m["unembed"], UnembeddingBridge)
    # Vision/audio are referenced but not bridged (text-only adapter for now).
    assert "vision_encoder" not in m and "audio_encoder" not in m


def test_altup_block_decomposition():
    blocks = _adapter().component_mapping["blocks"]
    assert blocks.altup_active_idx == 0
    # AltUp/LAuReL/PLE submodules present alongside attn + mlp + the five norms.
    for name in (
        "altup",
        "laurel",
        "per_layer_input_gate",
        "per_layer_projection",
        "self_attn",
        "mlp",
    ):
        assert name in blocks.submodules
    for norm in (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "post_per_layer_input_norm",
    ):
        assert norm in blocks.submodules


def test_kv_shared_submodules_are_optional():
    """The last num_kv_shared_layers layers drop their own k/v proj + norms."""
    attn = _adapter().component_mapping["blocks"].submodules["self_attn"]
    assert attn.submodules["q"].optional is False
    assert attn.submodules["o"].optional is False
    for shared in ("k", "v", "k_norm", "v_norm"):
        assert attn.submodules[shared].optional is True
    assert isinstance(attn.submodules["q"], LinearBridge)


class TestGemma3nAttentionAndMLPHookSurface:
    """Attention and the gated MLP must expose the gemma1/2/3 hook surface; as
    bare GeneralizedComponents everything beyond hook_in/hook_out was absent and
    q/k/v fired flat. Observability only — HF keeps the math.
    """

    # Must match the adapter cfg below: the head-split conversion only applies
    # when the flat width equals n_heads * d_head, so mismatched dims would
    # silently leave the tensors flat and the assertions would test nothing.
    D_MODEL, N_HEADS, N_KV_HEADS, D_HEAD, BATCH, SEQ = 64, 8, 2, 8, 2, 5

    @staticmethod
    def _small_adapter():
        cfg = TransformerBridgeConfig(
            d_model=64,
            d_head=8,
            n_heads=8,
            n_layers=2,
            n_ctx=64,
            d_vocab=128,
            n_key_value_heads=2,
            architecture=ARCH,
        )
        return ArchitectureAdapterFactory.select_architecture_adapter(cfg)

    def _wire(self):
        hf_attn = FakeDelegatedAttention(self.D_MODEL, self.N_HEADS, self.N_KV_HEADS, self.D_HEAD)
        return (
            wire_attention_bridge(
                self._small_adapter(),
                hf_attn,
                component="blocks.0.self_attn",
                expected_type=AttentionBridge,
            ),
            hf_attn,
        )

    def test_per_head_hooks_fire_with_head_split_shapes(self):
        """hook_q/hook_k/hook_v/hook_z must resolve AND carry
        [batch, seq, n_heads, d_head] — the flat d_model shape is what the bare
        component gave, and it is unusable for head-indexed analysis."""
        bridge, _ = self._wire()
        seen: dict = {}
        for alias, target in (("hook_q", bridge.q), ("hook_v", bridge.v)):
            target.hook_out.add_hook(lambda t, hook, a=alias: seen.__setitem__(a, tuple(t.shape)))
        with torch.no_grad():
            bridge(hidden_states=torch.randn(self.BATCH, self.SEQ, self.D_MODEL))

        assert seen["hook_q"] == (self.BATCH, self.SEQ, self.N_HEADS, self.D_HEAD), seen
        # GQA: v is n_kv_heads wide, so this also pins that the K/V reshape uses
        # n_key_value_heads rather than n_heads.
        assert seen["hook_v"] == (self.BATCH, self.SEQ, self.N_KV_HEADS, self.D_HEAD), seen

    def test_pattern_scores_and_z_hooks_fire(self):
        """The hooks a bare component cannot provide. Under delegation both
        pattern and scores fire the post-softmax weights HF returns (there is
        no pre-softmax tensor to expose) — asserted for BOTH, since asserting
        one leaves the other's firing unverified."""
        bridge, _ = self._wire()
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
        # hook_z must be per-head; a fake with a narrower o_proj makes the
        # reshape silently no-op and hides a flat z.
        assert fired.get("z") == (self.BATCH, self.SEQ, self.N_HEADS, self.D_HEAD), fired

    def test_gated_mlp_exposes_neuron_basis_hooks(self):
        """hook_pre must be the gate projection's output in the neuron basis —
        the same fix gemma4 took in #1650."""
        mlp = _adapter().component_mapping["blocks"].submodules["mlp"]
        # The class carries the alias constants; restating them here would be
        # a declaration test. Resolution is covered for every adapter by
        # test_hook_alias_resolution.py; behavior by the sparsity tests below.
        assert isinstance(mlp, GatedMLPBridge)


class TestGemma3nProcessedWeightsPreserveSparsity:
    """Compat mode must not bypass Gemma3n's `_gaussian_topk` activation sparsity:
    the functional processed branch computes plain act(gate)*up (~10x inflation
    on 0.95-sparsity layers), so this bridge keeps delegating.
    """

    class _SparseMLP(torch.nn.Module):
        """Gemma3nTextMLP-shaped, with the gaussian-topk sparsity inline."""

        def __init__(self, sparsity: float = 0.95) -> None:
            super().__init__()
            self.gate_proj = torch.nn.Linear(8, 16, bias=False)
            self.up_proj = torch.nn.Linear(8, 16, bias=False)
            self.down_proj = torch.nn.Linear(16, 8, bias=False)
            self.act_fn = torch.nn.functional.gelu
            self.activation_sparsity = sparsity

        def _gaussian_topk(self, x):
            import math

            std_multiplier = math.sqrt(2.0) * torch.erfinv(
                torch.tensor(2 * self.activation_sparsity - 1, dtype=x.dtype)
            )
            cutoff = x.mean(dim=-1, keepdim=True) + std_multiplier * x.std(dim=-1, keepdim=True)
            return torch.nn.functional.relu(x - cutoff)

        def forward(self, x):
            gate = self.gate_proj(x)
            if self.activation_sparsity > 0.0:
                gate = self._gaussian_topk(gate)
            return self.down_proj(self.act_fn(gate) * self.up_proj(x))

    def _wired_bridge(self, module):
        import copy

        from transformer_lens.model_bridge.component_setup import setup_submodules

        adapter = TestGemma3nAttentionAndMLPHookSurface._small_adapter()
        bridge = copy.deepcopy(adapter.component_mapping["blocks"].submodules["mlp"])
        bridge.set_original_component(module)
        setup_submodules(bridge, adapter, module)
        return bridge

    def test_processed_mode_still_applies_sparsity(self) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            module = self._SparseMLP()
            x = torch.randn(2, 5, 8)
        bridge = self._wired_bridge(module)

        # Enter processed mode the way process_weights does.
        bridge.set_processed_weights(
            {
                "gate.weight": module.gate_proj.weight.detach().clone(),
                "in.weight": module.up_proj.weight.detach().clone(),
                "out.weight": module.down_proj.weight.detach().clone(),
            }
        )
        with torch.no_grad():
            torch.testing.assert_close(bridge(x), module(x))

    def test_sparsity_is_observable_in_this_fixture(self) -> None:
        """Negative control: with sparsity bypassed the output must differ, or
        the parity assertion above cannot catch the functional branch."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            module = self._SparseMLP()
            x = torch.randn(2, 5, 8)
        dense = self._SparseMLP(sparsity=0.0)
        dense.load_state_dict(module.state_dict())
        with torch.no_grad():
            assert not torch.allclose(module(x), dense(x), atol=1e-4)

    def test_hooks_still_fire_in_delegated_processed_mode(self) -> None:
        """The whole point of the bridge swap: hook_pre must keep firing."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            module = self._SparseMLP()
            x = torch.randn(2, 5, 8)
        bridge = self._wired_bridge(module)
        bridge.set_processed_weights({"gate.weight": module.gate_proj.weight.detach().clone()})

        seen: list = []
        bridge.submodules["gate"].hook_out.add_hook(lambda t, hook: seen.append(t.shape))
        with torch.no_grad():
            bridge(x)
        assert seen and seen[0] == (2, 5, 16)
