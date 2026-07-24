"""Family-agnostic SSM effective-attention dispatch + canonical hook vocabulary.

Builds tiny synthetic models (no Hub access) for each SSM family and verifies:
- SSMMixerProtocol conformance (Mamba-1, Mamba-2, gated-delta-net; not attention),
- cache.ssm_layers / cache.compute_ssm_effective_attention dispatch across families,
- option forwarding (per_state_coord Mamba-1 only; include_dt_scaling not on GDN),
- the additive canonical hook vocabulary resolving cross-family via run_with_cache.
"""
from unittest.mock import MagicMock

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    SSM2MixerBridge,
    SSMMixerBridge,
    SSMMixerProtocol,
    find_ssm_mixer,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)

try:
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    _QWEN3_5_AVAILABLE = True
except ImportError:
    _QWEN3_5_AVAILABLE = False


class _Tok:
    pass


def _boot(hf_model, arch):
    cfg = build_bridge_config_from_hf(hf_model.config, arch, "tiny", torch.float32)
    from transformer_lens.factories.architecture_adapter_factory import (
        ArchitectureAdapterFactory,
    )

    adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
    return TransformerBridge(model=hf_model, adapter=adapter, tokenizer=_Tok())


@pytest.fixture(scope="module")
def mamba1_bridge():
    from transformers import AutoModelForCausalLM
    from transformers.models.mamba import MambaConfig

    torch.manual_seed(0)
    c = MambaConfig(
        vocab_size=128,
        hidden_size=32,
        state_size=16,
        num_hidden_layers=2,
        expand=2,
        time_step_rank=4,
        conv_kernel=4,
    )
    c.architectures = ["MambaForCausalLM"]
    return _boot(AutoModelForCausalLM.from_config(c).eval(), "MambaForCausalLM")


@pytest.fixture(scope="module")
def mamba2_bridge():
    from transformers import AutoModelForCausalLM
    from transformers.models.mamba2 import Mamba2Config

    torch.manual_seed(0)
    c = Mamba2Config(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        state_size=16,
        expand=2,
        head_dim=16,
        num_heads=4,
        n_groups=2,
        conv_kernel=4,
        chunk_size=8,
    )
    c.architectures = ["Mamba2ForCausalLM"]
    return _boot(AutoModelForCausalLM.from_config(c).eval(), "Mamba2ForCausalLM")


@pytest.fixture(scope="module")
def granite_bridge():
    from transformers import AutoModelForCausalLM
    from transformers.models.granitemoehybrid import GraniteMoeHybridConfig

    torch.manual_seed(0)
    c = GraniteMoeHybridConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=32,
        shared_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        layer_types=["mamba", "attention", "mamba"],
        mamba_n_heads=8,
        mamba_n_groups=2,
        mamba_d_state=16,
        mamba_d_head=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_chunk_size=16,
        position_embedding_type="rope",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    c.architectures = ["GraniteMoeHybridForCausalLM"]
    return _boot(AutoModelForCausalLM.from_config(c).eval(), "GraniteMoeHybridForCausalLM")


@pytest.fixture(scope="module")
def nemotronh_bridge():
    """Tiny NemotronH (Mamba-2 + attention + MLP hybrid) — exercises the single
    passthrough ``.mixer`` slot wired on every layer."""
    from transformers import AutoModelForCausalLM
    from transformers.models.nemotron_h import NemotronHConfig

    torch.manual_seed(0)
    c = NemotronHConfig(
        vocab_size=256,
        hidden_size=64,
        layers_block_type=["mamba", "attention", "mamba", "mlp"],
        num_attention_heads=4,
        num_key_value_heads=2,
        ssm_state_size=16,
        mamba_num_heads=4,
        mamba_head_dim=16,
        n_groups=2,
        conv_kernel=4,
        expand=2,
        intermediate_size=128,
        chunk_size=8,
    )
    c.architectures = ["NemotronHForCausalLM"]
    return _boot(AutoModelForCausalLM.from_config(c).eval(), "NemotronHForCausalLM")


@pytest.fixture(scope="module")
def qwen35_bridge():
    from transformer_lens.config.transformer_bridge_config import (
        TransformerBridgeConfig,
    )
    from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
        Qwen3_5ArchitectureAdapter,
    )

    torch.manual_seed(0)
    c = Qwen3_5TextConfig(
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
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        rope_parameters={
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        },
    )
    m = Qwen3_5ForCausalLM(c).eval()
    bcfg = TransformerBridgeConfig(
        d_model=128,
        d_head=32,
        n_heads=4,
        n_layers=8,
        n_ctx=2048,
        d_vocab=512,
        n_key_value_heads=2,
        architecture="Qwen3_5ForCausalLM",
    )
    return TransformerBridge(m, Qwen3_5ArchitectureAdapter(bcfg), tokenizer=MagicMock())


TOKENS = torch.tensor([[1, 2, 3, 4, 5]])


# ---------------------------------------------------------------------------
# SSMMixerProtocol conformance
# ---------------------------------------------------------------------------


class TestSSMMixerProtocol:
    def test_mamba2_conforms(self, mamba2_bridge):
        assert isinstance(mamba2_bridge.blocks[0].mixer, SSMMixerProtocol)

    def test_mamba1_conforms(self, mamba1_bridge):
        assert isinstance(mamba1_bridge.blocks[0].mixer, SSMMixerProtocol)

    def test_attention_does_not_conform(self, granite_bridge):
        # Block 1 is an attention layer — no SSM mixer.
        assert not isinstance(granite_bridge.blocks[1].attn, SSMMixerProtocol)

    def test_find_ssm_mixer_on_attention_layer_is_none(self, granite_bridge):
        assert find_ssm_mixer(granite_bridge.blocks[1]) is None

    def test_find_ssm_mixer_on_ssm_layer(self, granite_bridge):
        assert isinstance(find_ssm_mixer(granite_bridge.blocks[0]), SSM2MixerBridge)

    def test_nemotronh_passthrough_mixer_is_not_found(self, nemotronh_bridge):
        """NemotronH wires a passthrough SSM2MixerBridge .mixer on EVERY layer; it
        conforms to the protocol by method presence, but find_ssm_mixer must reject
        the no-op wrapper on attention / MLP layers (layer_types = mamba/attn/mamba/mlp)."""
        assert isinstance(find_ssm_mixer(nemotronh_bridge.blocks[0]), SSM2MixerBridge)  # mamba
        assert find_ssm_mixer(nemotronh_bridge.blocks[1]) is None  # attention passthrough
        assert find_ssm_mixer(nemotronh_bridge.blocks[3]) is None  # mlp passthrough

    @pytest.mark.skipif(not _QWEN3_5_AVAILABLE, reason="Qwen3_5 not available")
    def test_gated_delta_net_conforms(self, qwen35_bridge):
        assert isinstance(qwen35_bridge.blocks[0].linear_attn, SSMMixerProtocol)


# ---------------------------------------------------------------------------
# cache.ssm_layers
# ---------------------------------------------------------------------------


class TestSSMLayers:
    def test_mamba2_all_layers(self, mamba2_bridge):
        with torch.no_grad():
            _, cache = mamba2_bridge.run_with_cache(TOKENS)
        assert cache.ssm_layers() == [0, 1]

    def test_granite_only_mamba_layers(self, granite_bridge):
        with torch.no_grad():
            _, cache = granite_bridge.run_with_cache(TOKENS)
        assert cache.ssm_layers() == [0, 2]

    def test_nemotronh_excludes_passthrough_layers(self, nemotronh_bridge):
        # Structural detection must drop the passthrough .mixer on attn/mlp layers
        # without relying on cfg.layers_block_type (which is empty in this path).
        with torch.no_grad():
            _, cache = nemotronh_bridge.run_with_cache(TOKENS)
        assert cache.ssm_layers() == [0, 2]

    def test_mixer_type_filter(self, granite_bridge):
        with torch.no_grad():
            _, cache = granite_bridge.run_with_cache(TOKENS)
        assert cache.ssm_layers(mixer_type=SSM2MixerBridge) == [0, 2]
        assert cache.ssm_layers(mixer_type=SSMMixerBridge) == []

    @pytest.mark.skipif(not _QWEN3_5_AVAILABLE, reason="Qwen3_5 not available")
    def test_qwen35_linear_attn_layers(self, qwen35_bridge):
        with torch.no_grad():
            _, cache = qwen35_bridge.run_with_cache(TOKENS, use_cache=False)
        # full_attention_interval=4 -> full-attn at 3 and 7, linear-attn elsewhere
        assert cache.ssm_layers() == [0, 1, 2, 4, 5, 6]


# ---------------------------------------------------------------------------
# cache.compute_ssm_effective_attention dispatch
# ---------------------------------------------------------------------------


class TestComputeSSMEffectiveAttention:
    def test_mamba2_stacks_and_matches_direct(self, mamba2_bridge):
        with torch.no_grad():
            _, cache = mamba2_bridge.run_with_cache(TOKENS)
        M_all = cache.compute_ssm_effective_attention()
        assert torch.is_tensor(M_all)
        assert M_all.shape[0] == mamba2_bridge.cfg.n_layers
        assert torch.equal(M_all[0], cache.compute_ssm_effective_attention(layer=0))

    def test_granite_returns_dict(self, granite_bridge):
        with torch.no_grad():
            _, cache = granite_bridge.run_with_cache(TOKENS)
        M_all = cache.compute_ssm_effective_attention()
        assert isinstance(M_all, dict)
        assert sorted(M_all.keys()) == [0, 2]
        direct = granite_bridge.blocks[0].mixer.compute_effective_attention(cache, layer_idx=0)
        assert torch.equal(M_all[0], direct)

    def test_layer_without_mixer_raises_typeerror(self, granite_bridge):
        with torch.no_grad():
            _, cache = granite_bridge.run_with_cache(TOKENS)
        with pytest.raises(TypeError):
            cache.compute_ssm_effective_attention(layer=1)  # attention layer

    def test_nemotronh_dispatch_and_passthrough_contract(self, nemotronh_bridge):
        """All-layers returns only the mamba layers; the single-layer path on a
        passthrough (attention) layer raises the documented TypeError, not the
        mixer's internal "needs ... in cache" RuntimeError."""
        with torch.no_grad():
            _, cache = nemotronh_bridge.run_with_cache(TOKENS)
        M_all = cache.compute_ssm_effective_attention()
        assert isinstance(M_all, dict)
        assert sorted(M_all.keys()) == [0, 2]
        direct = nemotronh_bridge.blocks[0].mixer.compute_effective_attention(cache, layer_idx=0)
        assert torch.equal(M_all[0], direct)
        with pytest.raises(TypeError):
            cache.compute_ssm_effective_attention(layer=1)  # passthrough attention layer

    def test_per_state_coord_mamba1(self, mamba1_bridge):
        with torch.no_grad():
            _, cache = mamba1_bridge.run_with_cache(TOKENS)
        M = cache.compute_ssm_effective_attention(layer=0, per_state_coord=True)
        assert M.ndim == 5  # [batch, channels, state, seq, seq]

    def test_per_state_coord_unsupported_raises(self, mamba2_bridge):
        with torch.no_grad():
            _, cache = mamba2_bridge.run_with_cache(TOKENS)
        with pytest.raises(ValueError, match="per_state_coord"):
            cache.compute_ssm_effective_attention(per_state_coord=True)

    @pytest.mark.skipif(not _QWEN3_5_AVAILABLE, reason="Qwen3_5 not available")
    def test_qwen35_dict_and_include_dt_scaling_unsupported(self, qwen35_bridge):
        with torch.no_grad():
            _, cache = qwen35_bridge.run_with_cache(TOKENS, use_cache=False)
        M_all = cache.compute_ssm_effective_attention()
        assert isinstance(M_all, dict)
        assert sorted(M_all.keys()) == [0, 1, 2, 4, 5, 6]
        with pytest.raises(ValueError, match="include_dt_scaling"):
            cache.compute_ssm_effective_attention(include_dt_scaling=True)


# ---------------------------------------------------------------------------
# Canonical hook vocabulary — survives the hybrid alias resolution
# ---------------------------------------------------------------------------


class TestCanonicalHookVocabulary:
    @pytest.mark.skipif(not _QWEN3_5_AVAILABLE, reason="Qwen3_5 not available")
    def test_hook_ssm_out_resolves_cross_family(self, granite_bridge, qwen35_bridge):
        """The critical test: one canonical name resolves on a Mamba-2 hybrid SSM
        layer AND a gated-delta-net layer via run_with_cache (different slots)."""
        with torch.no_grad():
            _, cg = granite_bridge.run_with_cache(TOKENS)
            _, cq = qwen35_bridge.run_with_cache(TOKENS, use_cache=False)
        assert "blocks.0.mixer.hook_ssm_out" in cg
        assert "blocks.0.linear_attn.hook_ssm_out" in cq

    @pytest.mark.skipif(not _QWEN3_5_AVAILABLE, reason="Qwen3_5 not available")
    def test_gated_delta_net_canonical_aliases(self, qwen35_bridge):
        with torch.no_grad():
            _, cq = qwen35_bridge.run_with_cache(TOKENS, use_cache=False)
        p = "blocks.0.linear_attn"
        assert torch.equal(cq[f"{p}.hook_ssm_decay"], cq[f"{p}.hook_log_decay"])
        assert torch.equal(cq[f"{p}.hook_ssm_C"], cq[f"{p}.hook_q"])
        assert torch.equal(cq[f"{p}.hook_ssm_B"], cq[f"{p}.hook_k"])
        assert torch.equal(cq[f"{p}.hook_ssm_write"], cq[f"{p}.hook_beta"])

    def test_mamba1_hook_ssm_dt_alias(self, mamba1_bridge):
        with torch.no_grad():
            _, cache = mamba1_bridge.run_with_cache(TOKENS)
        assert torch.equal(
            cache["blocks.0.mixer.hook_ssm_dt"], cache["blocks.0.mixer.dt_proj.hook_out"]
        )


# ---------------------------------------------------------------------------
# cache.compute_ssm_state — family-agnostic over Mamba-1 / Mamba-2
# and gated-delta-net (each reconstructs its own recurrent state).
# ---------------------------------------------------------------------------


class TestComputeSsmStateDispatch:
    def test_mamba1_reachable_via_cache(self, mamba1_bridge):
        with torch.no_grad():
            _, cache = mamba1_bridge.run_with_cache(TOKENS)
        S = cache.compute_ssm_state()  # pure Mamba-1 → stacked
        assert torch.is_tensor(S)
        assert S.shape[0] == mamba1_bridge.cfg.n_layers
        assert torch.equal(S[0], cache.compute_ssm_state(layer=0))

    def test_mamba2_reachable_via_cache(self, mamba2_bridge):
        with torch.no_grad():
            _, cache = mamba2_bridge.run_with_cache(TOKENS)
        S = cache.compute_ssm_state()
        assert torch.is_tensor(S)
        assert S.shape[0] == mamba2_bridge.cfg.n_layers

    def test_granite_returns_per_ssm_layer_dict(self, granite_bridge):
        with torch.no_grad():
            _, cache = granite_bridge.run_with_cache(TOKENS)
        S = cache.compute_ssm_state()
        assert isinstance(S, dict)
        assert sorted(S.keys()) == [0, 2]

    @pytest.mark.skipif(not _QWEN3_5_AVAILABLE, reason="Qwen3_5 not available")
    def test_gated_delta_net_reachable_via_cache(self, qwen35_bridge):
        with torch.no_grad():
            _, cache = qwen35_bridge.run_with_cache(TOKENS, use_cache=False)
        # Hybrid (full-attn layers interleaved) → dict over the linear-attn layers.
        S = cache.compute_ssm_state()
        assert isinstance(S, dict)
        assert cache.ssm_layers() == sorted(S.keys())
        layer = cache.ssm_layers()[0]
        # [batch, seq, n_v_heads, head_k_dim, head_v_dim]
        assert S[layer].ndim == 5
        assert torch.equal(S[layer], cache.compute_ssm_state(layer=layer))

    @pytest.mark.skipif(not _QWEN3_5_AVAILABLE, reason="Qwen3_5 not available")
    def test_full_attention_layer_raises_typeerror(self, qwen35_bridge):
        with torch.no_grad():
            _, cache = qwen35_bridge.run_with_cache(TOKENS, use_cache=False)
        attn_layer = next(
            i for i in range(qwen35_bridge.cfg.n_layers) if i not in cache.ssm_layers()
        )
        with pytest.raises(TypeError, match="gated-delta-net"):
            cache.compute_ssm_state(layer=attn_layer)
