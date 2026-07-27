"""Integration tests for the Jamba architecture adapter.

Jamba interleaves Mamba-1 SSM layers with GQA attention (no RoPE). Every layer
has a dual-norm residual skeleton and either a dense SwiGLU MLP or a sparse MoE
FFN. Uses a tiny programmatic config with random CPU weights — no Hub download
(mirrors ``test_granite_moe_hybrid_adapter.py``).

Coverage:
- Bridge boots with ``SSMMixerBridge`` under the canonical ``.mixer`` slot
- Forward logits match HF exactly (passthrough mixers / native attention)
- Attention and Mamba streams are independently present/absent per layer type
- Mamba-1 projection hooks and residual hooks fire
- ``find_ssm_mixer`` discovers realized Mamba layers only
- PR #1481 surface: ``compute_ssm_*`` / ``eager_scan`` honor Jamba's dt/B/C LNs
- Greedy generation matches HF (unified ``past_key_values`` DynamicCache)
"""

import contextlib

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.jamba import JambaConfig

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    SSMMixerBridge,
)
from transformer_lens.model_bridge.generalized_components.ssm_protocol import (
    find_ssm_mixer,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.supported_architectures.jamba import (
    JambaArchitectureAdapter,
)

LAYER_TYPES = ["mamba", "attention", "mamba", "attention"]
MAMBA_LAYERS = [0, 2]
ATTN_LAYERS = [1, 3]


class _MockTokenizer:
    """Stand-in to satisfy TransformerBridge(tokenizer=...) without a Hub call."""

    eos_token_id = 2
    pad_token_id = 0
    bos_token_id = 1


@pytest.fixture(scope="module")
def hf_model():
    cfg = JambaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        expert_layer_period=2,
        expert_layer_offset=1,
        attn_layer_period=2,
        attn_layer_offset=1,
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank=4,
        max_position_embeddings=64,
        use_mamba_kernels=False,
    )
    cfg.architectures = ["JambaForCausalLM"]
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()


@pytest.fixture(scope="module")
def bridge(hf_model):
    bridge_config = build_bridge_config_from_hf(
        hf_model.config, "JambaForCausalLM", "jamba-tiny", torch.float32
    )
    adapter = JambaArchitectureAdapter(bridge_config)
    return TransformerBridge(model=hf_model, adapter=adapter, tokenizer=_MockTokenizer())


@pytest.fixture(scope="module")
def tokens():
    return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])


@pytest.fixture(scope="module")
def cache(bridge, tokens):
    with torch.no_grad():
        _, c = bridge.run_with_cache(tokens)
    return c


class TestJambaBridgeCreation:
    def test_block_count(self, bridge: TransformerBridge) -> None:
        assert len(bridge.blocks) == len(LAYER_TYPES)

    def test_blocks_are_block_bridge(self, bridge: TransformerBridge) -> None:
        assert isinstance(bridge.blocks[0], BlockBridge)

    def test_config_flags(self, bridge: TransformerBridge) -> None:
        assert bridge.cfg.normalization_type == "RMS"
        assert bridge.cfg.positional_embedding_type == "none"
        assert bridge.cfg.gated_mlp is True
        assert bridge.cfg.is_stateful is False
        assert bridge.cfg.final_rms is True

    def test_layers_block_type_surfaced(self, bridge: TransformerBridge) -> None:
        assert getattr(bridge.cfg, "layers_block_type", None) == LAYER_TYPES

    def test_mamba_layers_expose_ssm_mixer(self, bridge: TransformerBridge) -> None:
        for i in MAMBA_LAYERS:
            mixer = getattr(bridge.blocks[i], "mixer", None)
            assert isinstance(mixer, SSMMixerBridge), f"block {i} missing SSMMixerBridge"

    def test_attention_layers_expose_attn(self, bridge: TransformerBridge) -> None:
        for i in ATTN_LAYERS:
            attn = getattr(bridge.blocks[i], "attn", None)
            assert isinstance(attn, AttentionBridge), f"block {i} missing AttentionBridge"

    def test_attention_layer_has_no_realized_mixer(self, bridge: TransformerBridge) -> None:
        for i in ATTN_LAYERS:
            assert find_ssm_mixer(bridge.blocks[i]) is None

    def test_find_ssm_mixer_on_mamba_layers(self, bridge: TransformerBridge) -> None:
        for i in MAMBA_LAYERS:
            mixer = find_ssm_mixer(bridge.blocks[i])
            assert isinstance(mixer, SSMMixerBridge)


class TestJambaForwardEquivalence:
    def test_forward_returns_logits(self, bridge: TransformerBridge, tokens: torch.Tensor) -> None:
        with torch.no_grad():
            out = bridge(tokens)
        assert out.shape == (1, tokens.shape[1], bridge.cfg.d_vocab)
        assert not torch.isnan(out).any()

    def test_forward_matches_hf_exactly(
        self, bridge: TransformerBridge, hf_model, tokens: torch.Tensor
    ) -> None:
        with torch.no_grad():
            bridge_out = bridge(tokens)
            hf_out = hf_model(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff == 0.0, f"Bridge vs HF max diff = {max_diff}"


class TestJambaHFDelegation:
    def test_mamba_in_proj_is_live_linear(self, bridge: TransformerBridge) -> None:
        """Bridge wraps a real nn.Linear; replace_remote_component mutates HF in place."""
        i = MAMBA_LAYERS[0]
        oc = bridge.blocks[i].mixer.in_proj.original_component
        assert isinstance(oc, torch.nn.Linear)
        assert oc.in_features == bridge.cfg.d_model

    def test_attn_q_is_live_linear(self, bridge: TransformerBridge) -> None:
        i = ATTN_LAYERS[0]
        oc = bridge.blocks[i].attn.q.original_component
        assert isinstance(oc, torch.nn.Linear)
        assert oc.in_features == bridge.cfg.d_model

    def test_hf_layer_slots_are_bridged(self, bridge: TransformerBridge, hf_model) -> None:
        """After setup, HF layer attributes hold the bridge wrappers."""
        i = MAMBA_LAYERS[0]
        assert isinstance(hf_model.model.layers[i].mamba, SSMMixerBridge)
        j = ATTN_LAYERS[0]
        assert isinstance(hf_model.model.layers[j].self_attn, AttentionBridge)


class TestJambaHookShapes:
    def test_block_residual_hooks(self, cache, bridge: TransformerBridge) -> None:
        for i in range(len(bridge.blocks)):
            assert f"blocks.{i}.hook_in" in cache
            assert f"blocks.{i}.hook_out" in cache

    def test_mamba_projection_hooks_fire(self, cache) -> None:
        for i in MAMBA_LAYERS:
            for submod in ("in_proj", "conv1d", "x_proj", "dt_proj", "out_proj"):
                assert f"blocks.{i}.mixer.{submod}.hook_in" in cache
                assert f"blocks.{i}.mixer.{submod}.hook_out" in cache

    def test_attn_projection_hooks_fire(self, cache) -> None:
        for i in ATTN_LAYERS:
            for submod in ("q", "k", "v", "o"):
                assert f"blocks.{i}.attn.{submod}.hook_out" in cache

    def test_resid_mid_alias_via_ln2(self, cache) -> None:
        # BlockBridge aliases hook_resid_mid -> ln2.hook_in
        assert "blocks.0.ln2.hook_in" in cache


class TestJambaMoEDenseMix:
    def test_dense_and_moe_feed_forward_types(self, bridge: TransformerBridge) -> None:
        from transformers.models.jamba.modeling_jamba import (
            JambaMLP,
            JambaSparseMoeBlock,
        )

        # Bridge wraps feed_forward; the underlying HF module is on original_component.
        dense = bridge.blocks[0].mlp.original_component
        moe = bridge.blocks[1].mlp.original_component
        assert isinstance(dense, JambaMLP)
        assert isinstance(moe, JambaSparseMoeBlock)


class TestJambaAblationSurface:
    def test_mamba_and_attn_streams_ablate_independently(
        self, bridge: TransformerBridge, tokens: torch.Tensor
    ) -> None:
        def zero(t: torch.Tensor, hook: object) -> torch.Tensor:
            return torch.zeros_like(t)

        with torch.no_grad():
            base = bridge(tokens)
            ab_m = bridge.run_with_hooks(
                tokens, fwd_hooks=[(f"blocks.{MAMBA_LAYERS[0]}.mixer.hook_out", zero)]
            )
            ab_a = bridge.run_with_hooks(
                tokens, fwd_hooks=[(f"blocks.{ATTN_LAYERS[0]}.attn.hook_out", zero)]
            )
        assert not torch.allclose(base, ab_m)
        assert not torch.allclose(base, ab_a)
        assert not torch.allclose(ab_m, ab_a)


class TestJambaSSMInterpSurface:
    """PR #1481 surface must use post-norm B/C (Jamba's selective-param LNs)."""

    def test_ssm_layers_skip_attention(self, cache) -> None:
        assert cache.ssm_layers() == MAMBA_LAYERS

    def test_effective_attention_shape_and_causal(self, bridge: TransformerBridge, cache, tokens):
        layer = MAMBA_LAYERS[0]
        oc = bridge.blocks[layer].mixer.original_component
        seq = tokens.shape[1]
        M = cache.compute_ssm_effective_attention(layer=layer)
        assert isinstance(M, torch.Tensor)
        assert M.shape == (1, oc.intermediate_size, seq, seq)
        upper = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        assert torch.all(M[..., upper] == 0)

    def test_uses_post_norm_B_C(self, bridge: TransformerBridge, cache) -> None:
        layer = MAMBA_LAYERS[0]
        mixer = bridge.blocks[layer].mixer
        t = mixer._s6_terms(cache, layer)
        B_ln = cache[f"blocks.{layer}.mixer.b_layernorm.hook_out"].float()
        C_ln = cache[f"blocks.{layer}.mixer.c_layernorm.hook_out"].float()
        assert torch.equal(t.B, B_ln)
        assert torch.equal(t.C, C_ln)

    def test_ssm_state_matches_jamba_correct_recurrence(
        self, bridge: TransformerBridge, cache, tokens
    ) -> None:
        """Independent fp32 recurrence with LNs must match compute_ssm_state."""
        layer = MAMBA_LAYERS[0]
        mixer = bridge.blocks[layer].mixer
        oc = mixer.original_component
        hidden = cache[f"blocks.{layer}.mixer.hook_in"]
        seq = tokens.shape[1]

        with torch.no_grad():
            projected = oc.in_proj(hidden).transpose(1, 2)
            x, _gate = projected.chunk(2, dim=1)
            x = oc.act(oc.conv1d(x)[..., :seq])
            ssm_params = oc.x_proj(x.transpose(1, 2))
            dt_rank, state = oc.time_step_rank, oc.ssm_state_size
            time_step, B, C = ssm_params.split([dt_rank, state, state], dim=-1)
            time_step = oc.dt_layernorm(time_step)
            B = oc.b_layernorm(B)
            C = oc.c_layernorm(C)
            dt = torch.nn.functional.softplus(oc.dt_proj(time_step)).transpose(1, 2).float()
            A = -torch.exp(oc.A_log.float())
            x_f, B_f = x.float(), B.float()
            writes = (dt * x_f)[:, :, :, None] * B_f[:, None, :, :]
            ssm_state = torch.zeros(
                hidden.shape[0],
                oc.intermediate_size,
                state,
                dtype=writes.dtype,
                device=writes.device,
            )
            states = []
            for t in range(seq):
                decay = torch.exp(A[None] * dt[:, :, t, None])
                ssm_state = decay * ssm_state + writes[:, :, t]
                states.append(ssm_state.clone())
            S_true = torch.stack(states, dim=2)

        S_cache = cache.compute_ssm_state(layer=layer)
        max_diff = (S_cache - S_true).abs().max().item()
        scale = max(S_true.abs().max().item(), 1e-8)
        assert max_diff / scale < 1e-4, f"state rel diff {max_diff / scale:.2e}"

    def test_state_reconstructs_pre_gate_scan_output(
        self, bridge: TransformerBridge, cache, tokens
    ) -> None:
        """y = C·S + D·x matches HF slow_forward pre-gate scan (post-LN C)."""
        layer = MAMBA_LAYERS[0]
        mixer = bridge.blocks[layer].mixer
        oc = mixer.original_component
        seq = tokens.shape[1]
        S = cache.compute_ssm_state(layer=layer)
        C = cache[f"blocks.{layer}.mixer.c_layernorm.hook_out"].float()
        conv = cache[f"blocks.{layer}.mixer.conv1d.hook_out"].float()[..., :seq]
        x = oc.act(conv)
        D = mixer.D.float()
        y_pred = torch.einsum("bcts,bts->bct", S, C) + D[None, :, None] * x

        # Rebuild HF pre-gate y via the same selective path for a non-tautological check.
        hidden = cache[f"blocks.{layer}.mixer.hook_in"]
        with torch.no_grad():
            projected = oc.in_proj(hidden).transpose(1, 2)
            x_hf, gate = projected.chunk(2, dim=1)
            x_hf = oc.act(oc.conv1d(x_hf)[..., :seq])
            ssm_params = oc.x_proj(x_hf.transpose(1, 2))
            dt_rank, state = oc.time_step_rank, oc.ssm_state_size
            time_step, B, C_hf = ssm_params.split([dt_rank, state, state], dim=-1)
            time_step = oc.dt_layernorm(time_step)
            B = oc.b_layernorm(B)
            C_hf = oc.c_layernorm(C_hf)
            dt = torch.nn.functional.softplus(oc.dt_proj(time_step)).transpose(1, 2).float()
            A = -torch.exp(oc.A_log.float())
            x_f, B_f, C_f = x_hf.float(), B.float(), C_hf.float()
            writes = (dt * x_f)[:, :, :, None] * B_f[:, None, :, :]
            ssm_state = torch.zeros(
                hidden.shape[0],
                oc.intermediate_size,
                state,
                dtype=writes.dtype,
                device=writes.device,
            )
            ys = []
            for t in range(seq):
                decay = torch.exp(A[None] * dt[:, :, t, None])
                ssm_state = decay * ssm_state + writes[:, :, t]
                ys.append(
                    torch.einsum("bcs,bs->bc", ssm_state, C_f[:, t]) + D[None, :] * x_f[:, :, t]
                )
            y_true = torch.stack(ys, dim=-1)

        max_diff = (y_pred - y_true).abs().max().item()
        scale = max(y_true.abs().max().item(), 1e-8)
        assert max_diff / scale < 1e-4, f"y reconstruction rel diff {max_diff / scale:.2e}"


@contextlib.contextmanager
def _jamba_eager(bridge: TransformerBridge):
    mixers: list[SSMMixerBridge] = []
    for block in bridge.blocks:
        mixer = find_ssm_mixer(block)
        if isinstance(mixer, SSMMixerBridge):
            mixers.append(mixer)
    for m in mixers:
        m.eager_scan = True
    try:
        yield bridge
    finally:
        for m in mixers:
            m.eager_scan = False


class TestJambaEagerScan:
    def test_eager_scan_matches_fused_logits(
        self, bridge: TransformerBridge, tokens: torch.Tensor
    ) -> None:
        with torch.no_grad():
            fused = bridge(tokens)
            with _jamba_eager(bridge):
                eager = bridge(tokens, use_cache=False)
        max_diff = (fused - eager).abs().max().item()
        scale = max(fused.abs().max().item(), 1e-8)
        assert max_diff / scale < 1e-4, f"eager vs fused rel diff {max_diff / scale:.2e}"

    def test_eager_scan_fires_ssm_state_hook(
        self, bridge: TransformerBridge, tokens: torch.Tensor
    ) -> None:
        captured: dict[str, torch.Tensor] = {}

        def capture(t: torch.Tensor, hook: object) -> torch.Tensor:
            captured["S"] = t.detach().clone()
            return t

        with _jamba_eager(bridge), torch.no_grad():
            bridge.run_with_hooks(
                tokens,
                use_cache=False,
                fwd_hooks=[(f"blocks.{MAMBA_LAYERS[0]}.mixer.hook_ssm_state", capture)],
            )
        assert "S" in captured
        oc = bridge.blocks[MAMBA_LAYERS[0]].mixer.original_component
        assert captured["S"].shape == (
            1,
            oc.intermediate_size,
            tokens.shape[1],
            oc.ssm_state_size,
        )


class TestJambaGeneration:
    def test_greedy_matches_hf(
        self, bridge: TransformerBridge, hf_model, tokens: torch.Tensor
    ) -> None:
        prompt = tokens[:, :4]
        with torch.no_grad():
            bridge_out = bridge.generate(prompt, max_new_tokens=4, do_sample=False)
            hf_out = hf_model.generate(prompt, max_new_tokens=4, do_sample=False, pad_token_id=0)
        assert isinstance(bridge_out, torch.Tensor)
        assert isinstance(hf_out, torch.Tensor)
        assert torch.equal(bridge_out, hf_out), f"bridge={bridge_out.tolist()} hf={hf_out.tolist()}"
