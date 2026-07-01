"""Integration tests for the GraniteMoeHybrid architecture adapter.

GraniteMoeHybrid is a hybrid Mamba-2 + Attention + shared-MLP + sparse-MoE
model: each layer is *either* a Mamba SSM mixer *or* GQA attention, and every
layer additionally has a shared MLP and an optional MoE. Unlike NemotronH it is
a multi-slot transformer block (ln1/ln2/attn/mixer/shared_mlp/moe), not a
single-mixer SSMBlockBridge.

Uses a tiny programmatic config with real (random) CPU weights — no network
access or weight downloads (mirrors tests/unit/model_bridge/test_gpt_oss_moe.py
and test_qwen3_moe_bridge.py for the from_config + direct-constructor pattern).

Coverage focus (Phase 0, SSM mixer access normalization):
- The Mamba-2 mixer is reachable at the canonical ``.mixer`` slot on SSM layers.
- compute_effective_attention runs on a Granite SSM layer and reconstructs the
  SSM output (proves dims are sourced from the wrapped HF mixer, not the shared
  cfg whose n_heads/state_size hold the *attention* dims on a hybrid).
- The all-layers helper returns a per-SSM-layer dict on a heterogeneous hybrid.
- Attention / shared-MLP / MoE hooks still fire (rename did not disturb them).
"""

import contextlib

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.granitemoehybrid import GraniteMoeHybridConfig

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    GatedRMSNormBridge,
    SSM2MixerBridge,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.supported_architectures.granite_moe_hybrid import (
    GraniteMoeHybridArchitectureAdapter,
)

LAYER_TYPES = ["mamba", "attention", "mamba"]
MAMBA_LAYERS = [0, 2]
ATTN_LAYER = 1


class _MockTokenizer:
    """Stand-in to satisfy TransformerBridge(tokenizer=...) without a Hub call."""

    pass


@pytest.fixture(scope="module")
def hf_model():
    cfg = GraniteMoeHybridConfig(
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
        layer_types=LAYER_TYPES,
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
    cfg.architectures = ["GraniteMoeHybridForCausalLM"]
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()


@pytest.fixture(scope="module")
def bridge(hf_model):
    bridge_config = build_bridge_config_from_hf(
        hf_model.config, "GraniteMoeHybridForCausalLM", "granitemoehybrid-tiny", torch.float32
    )
    adapter = GraniteMoeHybridArchitectureAdapter(bridge_config)
    return TransformerBridge(model=hf_model, adapter=adapter, tokenizer=_MockTokenizer())


@pytest.fixture(scope="module")
def tokens():
    return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])


@pytest.fixture(scope="module")
def cache(bridge, tokens):
    with torch.no_grad():
        _, c = bridge.run_with_cache(tokens)
    return c


# ---------------------------------------------------------------------------
# Structure: Mamba mixer is reachable at the canonical .mixer slot
# ---------------------------------------------------------------------------


class TestGraniteMoeHybridStructure:
    def test_block_count(self, bridge: TransformerBridge) -> None:
        assert len(bridge.blocks) == len(LAYER_TYPES)

    def test_layers_block_type_surfaced(self, bridge: TransformerBridge) -> None:
        assert getattr(bridge.cfg, "layers_block_type", None) == LAYER_TYPES

    def test_mamba_layers_expose_ssm2_mixer(self, bridge: TransformerBridge) -> None:
        for i in MAMBA_LAYERS:
            mixer = getattr(bridge.blocks[i], "mixer", None)
            assert isinstance(mixer, SSM2MixerBridge), f"block {i} missing .mixer"

    def test_attention_layer_has_no_mixer_slot(self, bridge: TransformerBridge) -> None:
        # The mamba bridge is optional and skipped on attention layers, so the
        # structural `.mixer` slot must be absent there (blocks_with relies on this).
        assert "mixer" not in bridge.blocks[ATTN_LAYER]._modules
        assert "attn" in bridge.blocks[ATTN_LAYER]._modules

    def test_inner_norm_is_gated(self, bridge: TransformerBridge) -> None:
        mixer = bridge.blocks[MAMBA_LAYERS[0]].mixer
        assert isinstance(mixer.inner_norm, GatedRMSNormBridge)


# ---------------------------------------------------------------------------
# Forward parity: bridge delegates fully, so logits match HF exactly
# ---------------------------------------------------------------------------


class TestGraniteMoeHybridForwardPass:
    def test_forward_matches_hf_exactly(
        self, bridge: TransformerBridge, hf_model, tokens: torch.Tensor
    ) -> None:
        with torch.no_grad():
            bridge_out = bridge(tokens)
            hf_out = hf_model(tokens).logits
        max_diff = (bridge_out.float() - hf_out.float()).abs().max().item()
        assert max_diff == 0.0, (
            f"Bridge vs HF forward max diff = {max_diff:.2e}. Expected 0 because "
            "the bridge delegates the full forward to HF."
        )


# ---------------------------------------------------------------------------
# Hook coverage: rename to .mixer did not disturb attn / shared_mlp / moe hooks
# ---------------------------------------------------------------------------


class TestGraniteMoeHybridHookCoverage:
    def test_mamba_mixer_submodule_hooks_fire(self, cache) -> None:
        for i in MAMBA_LAYERS:
            for submod in ("in_proj", "conv1d"):
                assert f"blocks.{i}.mixer.{submod}.hook_out" in cache
            assert f"blocks.{i}.mixer.inner_norm.hook_in" in cache

    def test_attention_hooks_fire_on_attention_layer(self, cache) -> None:
        assert f"blocks.{ATTN_LAYER}.attn.hook_out" in cache

    def test_shared_mlp_and_moe_hooks_fire(self, cache) -> None:
        # Every layer has a shared MLP and a sparse MoE; both must still hook.
        assert f"blocks.{MAMBA_LAYERS[0]}.shared_mlp.hook_out" in cache
        assert f"blocks.{MAMBA_LAYERS[0]}.moe.hook_out" in cache


# ---------------------------------------------------------------------------
# Effective attention: the Phase 0 acceptance surface
# ---------------------------------------------------------------------------


class TestGraniteMoeHybridEffectiveAttention:
    def test_single_layer_shape(self, bridge: TransformerBridge, cache, tokens) -> None:
        seq_len = tokens.shape[1]
        mixer = bridge.blocks[MAMBA_LAYERS[0]].mixer
        n_heads = mixer.original_component.num_heads
        M = cache.compute_ssm_effective_attention(layer=MAMBA_LAYERS[0])
        assert isinstance(M, torch.Tensor)
        assert M.shape == (1, n_heads, seq_len, seq_len)
        assert torch.isfinite(M).all()

    def test_single_layer_is_causal(self, bridge: TransformerBridge, cache) -> None:
        M = cache.compute_ssm_effective_attention(layer=MAMBA_LAYERS[0])
        seq_len = M.shape[-1]
        upper = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        assert torch.all(M[..., upper] == 0), "effective attention must be lower-triangular"

    def test_all_layers_returns_per_ssm_layer_dict(self, bridge: TransformerBridge, cache) -> None:
        # Heterogeneous hybrid -> dict keyed by SSM layer index (attention layer skipped).
        M_all = cache.compute_ssm_effective_attention()
        assert isinstance(M_all, dict)
        assert sorted(M_all.keys()) == MAMBA_LAYERS
        for idx, M in M_all.items():
            assert torch.equal(M, cache.compute_ssm_effective_attention(layer=idx))

    def test_attention_layer_index_raises_typeerror(self, bridge: TransformerBridge, cache) -> None:
        with pytest.raises(TypeError):
            cache.compute_ssm_effective_attention(layer=ATTN_LAYER)

    def test_reconstruction_matches_ssm_output(
        self, bridge: TransformerBridge, cache, tokens
    ) -> None:
        """M @ x + D*x must reconstruct the SSM output (inner_norm.hook_in).

        Non-tautological: HF computes the SSM output via an independent chunked
        path. Agreement to float32 precision proves compute_effective_attention
        sourced the correct *Mamba* dims from the wrapped module — on a hybrid the
        shared cfg holds attention dims, so a cfg-based read would produce garbage.
        """
        layer = MAMBA_LAYERS[0]
        seq_len = tokens.shape[1]
        mixer = bridge.blocks[layer].mixer
        oc = mixer.original_component
        num_heads, head_dim = oc.num_heads, oc.head_dim
        intermediate_size, n_groups, state_size = (
            oc.intermediate_size,
            oc.n_groups,
            oc.ssm_state_size,
        )

        M_full = cache.compute_ssm_effective_attention(layer=layer, include_dt_scaling=True)

        conv_out = cache[f"blocks.{layer}.mixer.conv1d.hook_out"][..., :seq_len].float()
        conv_activated = torch.nn.functional.silu(conv_out).transpose(1, 2)
        hidden_x, _, _ = conv_activated.split(
            [intermediate_size, n_groups * state_size, n_groups * state_size], dim=-1
        )
        batch = hidden_x.shape[0]
        x_per_head = hidden_x.view(batch, seq_len, num_heads, head_dim)

        D = mixer.D.float()
        y_pred = torch.einsum("bhij,bjhd->bihd", M_full, x_per_head)
        y_pred = y_pred + D[None, None, :, None] * x_per_head
        y_pred_flat = y_pred.reshape(batch, seq_len, -1)

        y_actual = cache[f"blocks.{layer}.mixer.inner_norm.hook_in"].float()
        max_diff = (y_actual - y_pred_flat).abs().max().item()
        scale = max(y_actual.abs().max().item(), 1e-8)
        assert max_diff / scale < 1e-4, (
            f"Reconstruction mismatch: rel {max_diff / scale:.2e}. The effective "
            "attention dims are inconsistent with HF's SSM output (likely reading "
            "attention dims off the shared cfg instead of the mamba module)."
        )


class TestGraniteMoeHybridSSMState:
    """compute_ssm_state works across the hybrid: per-SSM-layer dict, correct dims."""

    def test_single_layer_shape(self, bridge: TransformerBridge, cache, tokens) -> None:
        seq_len = tokens.shape[1]
        oc = bridge.blocks[MAMBA_LAYERS[0]].mixer.original_component
        S = cache.compute_ssm_state(layer=MAMBA_LAYERS[0])
        assert isinstance(S, torch.Tensor)
        assert S.shape == (1, oc.num_heads, seq_len, oc.head_dim, oc.ssm_state_size)
        assert torch.isfinite(S).all()

    def test_all_layers_returns_per_ssm_layer_dict(self, bridge: TransformerBridge, cache) -> None:
        S_all = cache.compute_ssm_state()
        assert isinstance(S_all, dict)
        assert sorted(S_all.keys()) == MAMBA_LAYERS

    def test_attention_layer_raises_typeerror(self, bridge: TransformerBridge, cache) -> None:
        with pytest.raises(TypeError):
            cache.compute_ssm_state(layer=ATTN_LAYER)

    def test_reconstructs_ssm_output(self, bridge: TransformerBridge, cache, tokens) -> None:
        """y = C·S + D·x reconstructs HF's SSM output — proves hybrid dims are right."""
        layer = MAMBA_LAYERS[0]
        seq_len = tokens.shape[1]
        mixer = bridge.blocks[layer].mixer
        oc = mixer.original_component
        nh, hd, ns, ng = oc.num_heads, oc.head_dim, oc.ssm_state_size, oc.n_groups

        S = cache.compute_ssm_state(layer=layer)
        conv = cache[f"blocks.{layer}.mixer.conv1d.hook_out"][..., :seq_len].float()
        conv_act = torch.nn.functional.silu(conv).transpose(1, 2)
        x_flat, _, C_flat = conv_act.split([oc.intermediate_size, ng * ns, ng * ns], dim=-1)
        x = x_flat.view(1, seq_len, nh, hd)
        C_h = C_flat.view(1, seq_len, ng, ns).repeat_interleave(nh // ng, dim=2)
        D = mixer.D.float()

        y = torch.einsum("bthn,bhtpn->bthp", C_h, S) + D[None, None, :, None] * x
        y_actual = cache[f"blocks.{layer}.mixer.inner_norm.hook_in"].float()
        max_diff = (y_actual - y.reshape(1, seq_len, -1)).abs().max().item()
        scale = max(y_actual.abs().max().item(), 1e-8)
        assert max_diff / scale < 1e-4, f"state reconstruction rel diff {max_diff / scale:.2e}"


@contextlib.contextmanager
def _granite_eager(bridge):
    """Enable the eager-scan intervention path on Granite's Mamba-2 mixer layers."""
    for i in MAMBA_LAYERS:
        bridge.blocks[i].mixer.eager_scan = True
    try:
        yield bridge
    finally:
        for i in MAMBA_LAYERS:
            bridge.blocks[i].mixer.eager_scan = False


def _build_granite_bridge(device="cpu"):
    """Fresh tiny Granite bridge on the given device (for device-parametrized tests)."""
    cfg = GraniteMoeHybridConfig(
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
        layer_types=LAYER_TYPES,
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
    cfg.architectures = ["GraniteMoeHybridForCausalLM"]
    torch.manual_seed(0)
    hf = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
    bridge_config = build_bridge_config_from_hf(
        hf.config, "GraniteMoeHybridForCausalLM", "granitemoehybrid-tiny", torch.float32
    )
    bridge = TransformerBridge(
        model=hf,
        adapter=GraniteMoeHybridArchitectureAdapter(bridge_config),
        tokenizer=_MockTokenizer(),
    )
    if device != "cpu":
        hf.to(device)
    return bridge


def _available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


class TestGraniteMoeHybridEagerScanIntervention:
    """Phase 4 eager-scan intervention on Granite's Mamba-2 mixer layers (hybrid):
    hooks fire on the Mamba layers only, interventions propagate to logits, and the
    default path is untouched. Eager scan needs use_cache=False (prefill)."""

    def test_default_path_has_no_eager_hooks(self, bridge, tokens):
        with torch.no_grad():
            _, c = bridge.run_with_cache(tokens)
        assert "blocks.0.mixer.hook_ssm_write" not in c

    def test_eager_hooks_fire_on_mamba_layers_only(self, bridge, tokens):
        with _granite_eager(bridge), torch.no_grad():
            _, c = bridge.run_with_cache(tokens, use_cache=False)
        for i in MAMBA_LAYERS:
            assert f"blocks.{i}.mixer.hook_ssm_write" in c
            assert f"blocks.{i}.mixer.hook_ssm_state" in c
        assert f"blocks.{ATTN_LAYER}.mixer.hook_ssm_write" not in c  # attention layer: no mixer

    def test_eager_scan_matches_fused(self, bridge, tokens):
        with torch.no_grad():
            fused = bridge(tokens)
        with _granite_eager(bridge), torch.no_grad():
            eager = bridge.run_with_hooks(tokens, use_cache=False, fwd_hooks=[])
        rel = (eager - fused).abs().max().item() / max(fused.abs().max().item(), 1e-8)
        assert rel < 1e-4, f"Granite eager vs fused rel diff {rel:.2e}"

    def test_write_knockout_changes_logits(self, bridge, tokens):
        def knockout(writes, hook):
            writes = writes.clone()
            writes[:, 2] = 0.0
            return writes

        with _granite_eager(bridge), torch.no_grad():
            base = bridge.run_with_hooks(tokens, use_cache=False, fwd_hooks=[])
            patched = bridge.run_with_hooks(
                tokens, use_cache=False, fwd_hooks=[("blocks.0.mixer.hook_ssm_write", knockout)]
            )
        assert (patched - base).abs().max().item() > 1e-6


@pytest.mark.parametrize("device", _available_devices())
def test_granite_eager_scan_device_correctness(device):
    """Regression for the eager-scan device fix: ssm_state must be created on the
    input's device. Parametrized over every available device (cpu + cuda/mps)."""
    bridge = _build_granite_bridge(device)
    toks = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    with torch.no_grad():
        fused = bridge(toks)
    for i in MAMBA_LAYERS:
        bridge.blocks[i].mixer.eager_scan = True
    with torch.no_grad():
        _, cache = bridge.run_with_cache(toks, use_cache=False)
        eager = bridge.run_with_hooks(toks, use_cache=False, fwd_hooks=[])

    dev_type = torch.device(device).type
    assert eager.device.type == dev_type
    assert cache["blocks.0.mixer.hook_ssm_write"].device.type == dev_type
    rel = (eager.cpu() - fused.cpu()).abs().max().item() / max(fused.abs().max().item(), 1e-8)
    assert rel < 1e-4, f"eager vs fused rel diff on {device}: {rel:.2e}"
