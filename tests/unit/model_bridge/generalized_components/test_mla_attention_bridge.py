"""Unit tests for MLAAttentionBridge (DeepSeek Multi-Head Latent Attention)."""

import pytest
import torch
from transformers import (
    DeepseekV2Config,
    DeepseekV2ForCausalLM,
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
)

from transformer_lens.model_bridge.generalized_components.mla_attention import (
    MLAAttentionBridge,
    _apply_rotary_complex,
)


@pytest.fixture(scope="module")
def tiny_config():
    return DeepseekV3Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        vocab_size=1000,
        first_k_dense_replace=1,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        max_position_embeddings=128,
        moe_intermediate_size=256,
        # The mask=None parity tests hold only under sdpa's is_causal; eager with
        # mask=None is bidirectional and adds a boolean mask as 0/1.
        attn_implementation="sdpa",
    )


@pytest.fixture(scope="module")
def tiny_model(tiny_config):
    return DeepseekV3ForCausalLM(tiny_config)


@pytest.fixture(scope="module")
def hf_attn(tiny_model):
    return tiny_model.model.layers[0].self_attn


@pytest.fixture(scope="module")
def mla_bridge(tiny_config, hf_attn, tiny_model):
    bridge = MLAAttentionBridge(name="self_attn", config=tiny_config, submodules={})
    bridge.set_original_component(hf_attn)
    bridge.set_rotary_emb(tiny_model.model.rotary_emb)
    return bridge


class TestMLAAttentionBridgeHooks:
    def test_all_expected_hooks_exist(self, mla_bridge):
        for hook_name in [
            "hook_in",
            "hook_out",
            "hook_q_latent",
            "hook_kv_latent",
            "hook_q",
            "hook_k",
            "hook_v",
            "hook_rot_q",
            "hook_rot_k",
            "hook_attn_scores",
            "hook_pattern",
            "hook_cos",
            "hook_sin",
        ]:
            assert hasattr(mla_bridge, hook_name), f"Missing hook: {hook_name}"

    def test_W_Q_raises_not_implemented(self, mla_bridge):
        with pytest.raises(NotImplementedError, match="not available on MLA"):
            _ = mla_bridge.W_Q

    def test_W_K_raises_not_implemented(self, mla_bridge):
        with pytest.raises(NotImplementedError, match="not available on MLA"):
            _ = mla_bridge.W_K


class TestMLAAttentionBridgeForward:
    @pytest.fixture
    def sample_inputs(self, tiny_config, tiny_model):
        batch, seq = 2, 8
        hidden_states = torch.randn(batch, seq, tiny_config.hidden_size)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = tiny_model.model.rotary_emb(hidden_states, position_ids)
        return hidden_states, (cos, sin)

    def test_output_matches_hf(self, mla_bridge, hf_attn, sample_inputs, tiny_model):
        """HF's SDPA path passes attention_mask=None and relies on is_causal."""
        hidden_states, position_embeddings = sample_inputs

        with torch.no_grad():
            hf_attn_out = hf_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
            )[0]
            bridge_attn_out = mla_bridge(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
            )[0]

        max_diff = (hf_attn_out - bridge_attn_out).abs().max().item()
        assert max_diff < 1e-5, f"Output too different: max diff = {max_diff}"

    def test_boolean_padding_mask_matches_hf(self, mla_bridge, hf_attn, sample_inputs):
        """With padding, HF's SDPA path passes a boolean 4D mask (True = attend)."""
        hidden_states, position_embeddings = sample_inputs
        seq = hidden_states.shape[1]
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        attention_mask = causal.expand(2, 1, seq, seq).clone()
        attention_mask[0, :, :, :2] = False  # left padding on the first sequence

        with torch.no_grad():
            hf_attn_out = hf_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )[0]
            bridge_attn_out = mla_bridge(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )[0]

        # Padded query rows are fully masked (undefined output); compare real positions.
        diff = (hf_attn_out - bridge_attn_out)[:, 2:].abs().max().item()
        assert diff < 1e-5, f"Output too different: max diff = {diff}"

    def test_hooks_fire_and_have_correct_shapes(self, mla_bridge, sample_inputs, tiny_config):
        hidden_states, position_embeddings = sample_inputs
        batch, seq = hidden_states.shape[:2]
        captured = {}

        hooks_to_check = [
            "hook_q_latent",
            "hook_kv_latent",
            "hook_q",
            "hook_k",
            "hook_v",
            "hook_rot_q",
            "hook_rot_k",
            "hook_attn_scores",
            "hook_pattern",
        ]

        handles = []
        for name in hooks_to_check:

            def make_capture(n):
                def hook_fn(module, input, output):
                    captured[n] = output.shape

                return hook_fn

            handles.append(getattr(mla_bridge, name).register_forward_hook(make_capture(name)))

        try:
            with torch.no_grad():
                mla_bridge(
                    hidden_states, position_embeddings=position_embeddings, attention_mask=None
                )
        finally:
            for h in handles:
                h.remove()

        n_heads = tiny_config.num_attention_heads
        qk_head_dim = tiny_config.qk_nope_head_dim + tiny_config.qk_rope_head_dim

        for name in hooks_to_check:
            assert name in captured, f"Hook {name} did not fire"

        assert captured["hook_q_latent"] == (batch, seq, tiny_config.q_lora_rank)
        assert captured["hook_kv_latent"] == (batch, seq, tiny_config.kv_lora_rank)
        assert captured["hook_q"] == (batch, n_heads, seq, qk_head_dim)
        assert captured["hook_k"] == (batch, n_heads, seq, qk_head_dim)
        assert captured["hook_v"] == (batch, n_heads, seq, tiny_config.v_head_dim)
        assert captured["hook_attn_scores"] == (batch, n_heads, seq, seq)
        assert captured["hook_pattern"] == (batch, n_heads, seq, seq)

    def test_hook_q_is_post_rope(self, mla_bridge, sample_inputs):
        """hook_q's rope portion should match hook_rot_q."""
        hidden_states, position_embeddings = sample_inputs
        q_values: list[torch.Tensor] = []
        rot_q_values: list[torch.Tensor] = []

        h1 = mla_bridge.hook_q.register_forward_hook(lambda m, i, o: q_values.append(o.clone()))
        h2 = mla_bridge.hook_rot_q.register_forward_hook(
            lambda m, i, o: rot_q_values.append(o.clone())
        )

        try:
            with torch.no_grad():
                mla_bridge(
                    hidden_states, position_embeddings=position_embeddings, attention_mask=None
                )
        finally:
            h1.remove()
            h2.remove()

        qk_rope_dim = mla_bridge._qk_rope_head_dim
        assert torch.allclose(q_values[0][..., -qk_rope_dim:], rot_q_values[0], atol=1e-5)


class TestMLAAttentionBridgeScaling:
    def test_bridge_tracks_hf_module_scaling(self, tiny_config, tiny_model):
        """Yarn sets hf_attn.scaling to qk_head_dim^-0.5 * mscale^2; the bridge must
        score with the module's scaling. Eager + explicit 4D mask keeps parity ~1e-6,
        so the mscale^2 effect is decisive.
        """
        import copy

        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3Attention,
        )

        # Private eager config/module: the module-scoped fixtures stay pristine.
        eager_config = copy.deepcopy(tiny_config)
        eager_config._attn_implementation = "eager"
        batch, seq = 2, 8
        # fork_rng: deterministic setup without mutating the global RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            hf_attn = DeepseekV3Attention(eager_config, layer_idx=0)
            hidden_states = torch.randn(batch, seq, eager_config.hidden_size)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        position_embeddings = tiny_model.model.rotary_emb(hidden_states, position_ids)
        # 4D additive causal mask — authoritative for both HF eager and the bridge.
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        attention_mask = torch.zeros(batch, 1, seq, seq).masked_fill(
            ~causal, torch.finfo(torch.float32).min
        )

        # mscale^2 for DeepSeek-V3/R1's yarn factor=40, mscale_all_dim=1
        original_scaling = hf_attn.scaling
        hf_attn.scaling = original_scaling * 1.87
        bridge = MLAAttentionBridge(name="self_attn", config=eager_config, submodules={})
        bridge.set_original_component(hf_attn)

        with torch.no_grad():
            hf_out = hf_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )[0]
            bridge_out = bridge(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )[0]
            hf_attn.scaling = original_scaling
            base_out = hf_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )[0]

        torch.testing.assert_close(bridge_out, hf_out, atol=1e-5, rtol=1e-4)
        # A reverted bridge scores with original_scaling and so produces base_out;
        # base_out must violate the parity tolerance or the assertion above is vacuous.
        assert not torch.allclose(base_out, hf_out, atol=1e-5, rtol=1e-4)

    def test_softmax_scale_fallback_for_remote_code_modules(self, tiny_config, tiny_model):
        """trust_remote_code DeepSeek-V2 stores the resolved scale (base × yarn
        mscale²) as softmax_scale, not scaling; the bridge must honor it rather
        than silently recomputing the base scale."""
        import copy

        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3Attention,
        )

        eager_config = copy.deepcopy(tiny_config)
        eager_config._attn_implementation = "eager"
        batch, seq = 2, 8
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            hf_attn = DeepseekV3Attention(eager_config, layer_idx=0)
            hidden_states = torch.randn(batch, seq, eager_config.hidden_size)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        position_embeddings = tiny_model.model.rotary_emb(hidden_states, position_ids)

        bridge = MLAAttentionBridge(name="self_attn", config=eager_config, submodules={})
        bridge.set_original_component(hf_attn)
        modified = hf_attn.scaling * 1.87

        def run() -> torch.Tensor:
            with torch.no_grad():
                return bridge(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                )[0]

        hf_attn.scaling = modified
        out_via_scaling = run()

        del hf_attn.scaling  # remote-code module shape
        hf_attn.softmax_scale = modified
        out_via_softmax_scale = run()

        del hf_attn.softmax_scale  # neither attr -> base-scale fallback
        out_base = run()

        torch.testing.assert_close(out_via_softmax_scale, out_via_scaling)
        assert not torch.allclose(out_base, out_via_scaling, atol=1e-5, rtol=1e-4)


class TestMLAAttentionHookResult:
    """Per-head hook_result (z_h @ W_O_h, pre-sum) gated by use_attn_result."""

    @staticmethod
    def _build_bridge(tiny_config, tiny_model, use_attn_result: bool) -> MLAAttentionBridge:
        import copy

        from transformer_lens.model_bridge.generalized_components import LinearBridge

        cfg = copy.deepcopy(tiny_config)
        cfg.use_attn_result = use_attn_result
        hf_attn = tiny_model.model.layers[0].self_attn
        o_bridge = LinearBridge(name="o_proj")
        o_bridge.set_original_component(hf_attn.o_proj)
        bridge = MLAAttentionBridge(name="self_attn", config=cfg, submodules={"o": o_bridge})
        bridge.add_module("o", o_bridge)
        bridge.set_original_component(hf_attn)
        bridge.set_rotary_emb(tiny_model.model.rotary_emb)
        return bridge

    @staticmethod
    def _sample_inputs(tiny_config, tiny_model):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            hidden_states = torch.randn(2, 8, tiny_config.hidden_size)
        position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
        return hidden_states, tiny_model.model.rotary_emb(hidden_states, position_ids)

    def test_hook_result_is_not_the_summed_output_hookpoint(self, tiny_config, tiny_model):
        """Alias registration must not rebind hook_result onto hook_out: cached
        'hook_result' would then silently hold already-summed [batch, pos, d_model]."""
        bridge = self._build_bridge(tiny_config, tiny_model, use_attn_result=True)
        bridge._register_aliases()
        assert bridge.hook_result is not bridge.hook_out

    def test_hook_result_fires_per_head_and_sums_to_hook_out(self, tiny_config, tiny_model):
        bridge = self._build_bridge(tiny_config, tiny_model, use_attn_result=True)
        hidden_states, position_embeddings = self._sample_inputs(tiny_config, tiny_model)
        captured: dict[str, torch.Tensor] = {}
        handle = bridge.hook_result.register_forward_hook(
            lambda m, i, o: captured.setdefault("result", o.detach().clone())
        )
        try:
            with torch.no_grad():
                out = bridge(
                    hidden_states, position_embeddings=position_embeddings, attention_mask=None
                )[0]
        finally:
            handle.remove()
        n_heads = tiny_config.num_attention_heads
        assert captured["result"].shape == (2, 8, n_heads, tiny_config.hidden_size)
        summed = captured["result"].sum(dim=-2)
        bias = tiny_model.model.layers[0].self_attn.o_proj.bias
        if bias is not None:
            summed = summed + bias
        torch.testing.assert_close(summed, out, atol=1e-5, rtol=1e-4)

    def test_per_head_path_matches_default_projection(self, tiny_config, tiny_model):
        """sum_h z_h @ W_O_h must reproduce o_proj exactly, so enabling the flag
        cannot drift the model's output."""
        hidden_states, position_embeddings = self._sample_inputs(tiny_config, tiny_model)
        outs = []
        for flag in (False, True):
            bridge = self._build_bridge(tiny_config, tiny_model, use_attn_result=flag)
            with torch.no_grad():
                outs.append(
                    bridge(
                        hidden_states, position_embeddings=position_embeddings, attention_mask=None
                    )[0]
                )
        torch.testing.assert_close(outs[0], outs[1], atol=1e-5, rtol=1e-4)

    def test_hook_result_does_not_fire_without_flag(self, tiny_config, tiny_model):
        bridge = self._build_bridge(tiny_config, tiny_model, use_attn_result=False)
        hidden_states, position_embeddings = self._sample_inputs(tiny_config, tiny_model)
        fired: list[torch.Tensor] = []
        handle = bridge.hook_result.register_forward_hook(lambda m, i, o: fired.append(o))
        try:
            with torch.no_grad():
                bridge(hidden_states, position_embeddings=position_embeddings, attention_mask=None)
        finally:
            handle.remove()
        assert fired == []


class TestMLACompatibilityMaskSentinel:
    """Compatibility mode must report masked scores as -inf, not HF's finfo.min."""

    @staticmethod
    def _captured_scores(tiny_config, tiny_model, compatibility_mode: bool) -> torch.Tensor:
        import copy

        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3Attention,
        )

        # Private module: the module-scoped bridge fixture must stay in its default mode.
        eager_config = copy.deepcopy(tiny_config)
        eager_config._attn_implementation = "eager"
        batch, seq = 2, 8
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            hf_attn = DeepseekV3Attention(eager_config, layer_idx=0)
            hidden_states = torch.randn(batch, seq, eager_config.hidden_size)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        position_embeddings = tiny_model.model.rotary_emb(hidden_states, position_ids)
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        attention_mask = torch.zeros(batch, 1, seq, seq).masked_fill(
            ~causal, torch.finfo(torch.float32).min
        )

        bridge = MLAAttentionBridge(name="self_attn", config=eager_config, submodules={})
        bridge.set_original_component(hf_attn)
        bridge.compatibility_mode = compatibility_mode

        captured: dict[str, torch.Tensor] = {}
        handle = bridge.hook_attn_scores.register_forward_hook(
            lambda m, i, o: captured.setdefault("scores", o.detach().clone())
        )
        try:
            with torch.no_grad():
                bridge(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                )
        finally:
            handle.remove()
        return captured["scores"]

    def test_masked_scores_are_negative_infinity(self, tiny_config, tiny_model):
        """`torch.isinf(cache[...hook_attn_scores])` is the documented way to find
        masked positions; a finfo.min sentinel makes it silently return all-False."""
        scores = self._captured_scores(tiny_config, tiny_model, compatibility_mode=True)
        seq = scores.shape[-1]
        masked = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1).expand_as(scores)
        assert masked.any()
        assert torch.isneginf(scores[masked]).all()
        assert torch.isfinite(scores[~masked]).all()

    def test_non_compatibility_mode_keeps_the_hf_sentinel(self, tiny_config, tiny_model):
        """Outside compatibility mode the bridge must leave HF's finfo.min alone."""
        scores = self._captured_scores(tiny_config, tiny_model, compatibility_mode=False)
        seq = scores.shape[-1]
        masked = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1).expand_as(scores)
        assert torch.isfinite(scores[masked]).all()


class TestMLAComplexRope:
    """DeepSeek-V2 complex RoPE on rope slices that an odd ``qk_nope_head_dim`` or
    ``kv_lora_rank`` (e.g. after pruning) leaves at odd storage offsets and strides."""

    ROPE_DIM = 4

    @staticmethod
    def _rotate(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Reference rotation of adjacent pairs, written without complex numbers."""
        cos, sin = angles.cos()[:, None], angles.sin()[:, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(3)

    @pytest.mark.parametrize(
        "nope_dim, kv_lora_rank",
        [(4, 8), (3, 8), (4, 7), (3, 7)],
        ids=["aligned", "odd-nope", "odd-kv-rank", "odd-both"],
    )
    def test_rotation_does_not_depend_on_storage_layout(self, nope_dim, kv_lora_rank):
        batch, heads, seq = 2, 2, 5
        torch.manual_seed(0)
        angles = torch.randn(batch, seq, self.ROPE_DIM // 2)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)

        # Same view/transpose/split sequence as MLAAttentionBridge.forward
        q_states = torch.randn(batch, seq, heads * (nope_dim + self.ROPE_DIM))
        q_states = q_states.view(batch, seq, heads, -1).transpose(1, 2)
        _, q_rot = torch.split(q_states, [nope_dim, self.ROPE_DIM], dim=-1)
        compressed_kv = torch.randn(batch, seq, kv_lora_rank + self.ROPE_DIM)
        _, k_rot = torch.split(compressed_kv, [kv_lora_rank, self.ROPE_DIM], dim=-1)
        k_rot = k_rot.view(batch, 1, seq, self.ROPE_DIM)

        q_out, k_out = _apply_rotary_complex(q_rot, k_rot, freqs_cis)

        assert torch.allclose(q_out, self._rotate(q_rot, angles), atol=1e-6)
        assert torch.allclose(k_out, self._rotate(k_rot, angles), atol=1e-6)

    def test_contiguous_slice_at_odd_offset(self):
        """A slice can be contiguous and still start at an odd storage offset."""
        angles = torch.randn(1, 1, self.ROPE_DIM // 2)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        q_rot = torch.randn(3 + self.ROPE_DIM)[3:].view(1, 1, 1, self.ROPE_DIM)
        assert q_rot.is_contiguous() and q_rot.storage_offset() % 2 == 1

        q_out, k_out = _apply_rotary_complex(q_rot, q_rot, freqs_cis)

        assert torch.allclose(q_out, self._rotate(q_rot, angles), atol=1e-6)
        assert torch.allclose(k_out, q_out)

    def test_bridge_forward_with_odd_nope_dim_and_kv_rank(self):
        """Bridge output must match HF DeepseekV2Attention when the rope slices are misaligned."""
        from unittest import mock

        from transformers.models.deepseek_v2 import modeling_deepseek_v2

        config = DeepseekV2Config(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=8,
            kv_lora_rank=7,
            qk_nope_head_dim=3,
            qk_rope_head_dim=self.ROPE_DIM,
            v_head_dim=8,
            first_k_dense_replace=1,
            n_routed_experts=2,
            n_shared_experts=1,
            num_experts_per_tok=1,
            # At the default 0.02 init, skipping RoPE moves the output by only ~5e-5.
            initializer_range=0.5,
        )
        config._attn_implementation = "eager"
        batch, seq = 2, 6
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = DeepseekV2ForCausalLM(config)
            hidden_states = torch.randn(batch, seq, config.hidden_size)
        hf_attn = model.model.layers[0].self_attn
        bridge = MLAAttentionBridge(name="self_attn", config=config, submodules={})
        bridge.set_original_component(hf_attn)

        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        freqs_cis = model.model.rotary_emb(hidden_states, position_ids)
        assert freqs_cis.is_complex()
        # HF eager with mask=None is bidirectional; the bridge would apply a causal mask.
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        attention_mask = torch.zeros(batch, 1, seq, seq).masked_fill(
            ~causal, torch.finfo(torch.float32).min
        )

        def hf_forward(rotary_fn) -> torch.Tensor:
            # HF's own apply_rotary_emb raises on these slices via view_as_complex.
            with mock.patch.object(modeling_deepseek_v2, "apply_rotary_emb", rotary_fn):
                return hf_attn(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=freqs_cis,
                )[0]

        with torch.no_grad():
            hf_out = hf_forward(_apply_rotary_complex)
            unrotated_out = hf_forward(lambda q, k, freqs_cis: (q, k))
            bridge_out = bridge(
                hidden_states, position_embeddings=freqs_cis, attention_mask=attention_mask
            )[0]

        torch.testing.assert_close(bridge_out, hf_out)
        assert not torch.allclose(unrotated_out, hf_out, atol=1e-3)
