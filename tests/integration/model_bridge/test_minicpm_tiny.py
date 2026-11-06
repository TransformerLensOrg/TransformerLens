"""Tiny MiniCPM integration tests — CI coverage without the checkpoint weights.

MiniCPM has no transformers-native class, so this fetches only the pinned repos'
config and modeling code (no weights) and builds shrunk models with random CPU
weights. That puts the remote-code path — the transformers-v5 patches in
``prepare_loading`` and the rotary repair in ``prepare_model`` — under CI, which
the slow real-checkpoint suite (test_minicpm_adapter.py) cannot. Both code
generations run: MiniCPM4 (LongRoPE, raises on a missing Cache) and MiniCPM 1/2
(plain RoPE, legacy-cache calls).

The muP scales are non-trivial and pairwise distinct, so dropping or swapping any
of them breaks parity. The reference is a deep copy taken before the bridge wraps
the model, never ``bridge.original_model``.
"""

import copy
from typing import Any, NamedTuple

import pytest
import torch
from transformers import AutoConfig

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.supported_architectures._remote_code_compat import (
    force_import_remote_class,
)
from transformer_lens.model_bridge.supported_architectures.minicpm import (
    MiniCPMArchitectureAdapter,
)

N_LAYERS = 3
HEAD_DIM = 16
SCALE_EMB = 7.0
SCALE_DEPTH = 2.3
DIM_MODEL_BASE = 24  # 64 / 24 is not a power of two, so the logit divide is not exact
TOL = 1e-5

# Cache length 64 > original 32: LongRoPE uses long_factor and a
# sqrt(1 + log 2 / log 32) magnitude scale, both off the identity.
_LONG_ROPE = {
    "rope_type": "longrope",
    "short_factor": [1.0 + 0.1 * i for i in range(HEAD_DIM // 2)],
    "long_factor": [1.0 + 0.5 * i for i in range(HEAD_DIM // 2)],
    "original_max_position_embeddings": 32,
}


class _Source(NamedTuple):
    repo: str
    revision: str
    rope_scaling: Any
    rotary_class: str


SOURCES = [
    pytest.param(
        _Source(
            "openbmb/MiniCPM4-0.5B",
            "5253c7fcc5e29e1cf3eacb59a58adf1ba4df8630",
            _LONG_ROPE,
            "MiniCPMLongRoPE",
        ),
        id="minicpm4",
    ),
    pytest.param(
        _Source(
            "openbmb/MiniCPM-1B-sft-bf16",
            "85252935bfc9acb3ee93dcea4b793fb87a0b43e8",
            None,
            "MiniCPMRotaryEmbedding",
        ),
        id="minicpm1",
    ),
]


class _Tok:
    pass


def _tiny_config(source: _Source) -> Any:
    config = AutoConfig.from_pretrained(
        source.repo, trust_remote_code=True, revision=source.revision
    )
    config.update(
        {
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": N_LAYERS,
            "num_attention_heads": 64 // HEAD_DIM,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "scale_emb": SCALE_EMB,
            "scale_depth": SCALE_DEPTH,
            "dim_model_base": DIM_MODEL_BASE,
            "rope_scaling": source.rope_scaling,
        }
    )
    config._attn_implementation = "eager"
    return config


def _build(source: _Source) -> tuple[TransformerBridge, torch.nn.Module]:
    """Return (bridge, independent HF reference) sharing identical weights."""
    config = _tiny_config(source)
    bridge_cfg = build_bridge_config_from_hf(
        config, "MiniCPMForCausalLM", "minicpm-tiny", torch.float32
    )
    adapter = MiniCPMArchitectureAdapter(bridge_cfg)
    adapter.prepare_loading(source.repo, {"config": config, "revision": source.revision})
    model_cls = force_import_remote_class(
        source.repo, "modeling_minicpm.MiniCPMForCausalLM", revision=source.revision
    )
    assert model_cls is not None

    torch.manual_seed(0)
    hf = model_cls(config).to(torch.float32).eval()
    reference = copy.deepcopy(hf)

    # Reproduce the meta-device loading defect so prepare_model must repair it.
    for layer in hf.model.layers:
        rotary = layer.self_attn.rotary_emb
        rotary.inv_freq = torch.zeros_like(rotary.inv_freq)
        rotary.cos_cached = torch.full_like(rotary.cos_cached, float("nan"))
        rotary.sin_cached = torch.full_like(rotary.sin_cached, float("nan"))
    adapter.prepare_model(hf)

    return TransformerBridge(hf, adapter, tokenizer=_Tok()), reference


@pytest.fixture(scope="module", params=SOURCES)
def source(request: pytest.FixtureRequest) -> _Source:
    return request.param


@pytest.fixture(scope="module")
def models(source: _Source) -> tuple[TransformerBridge, torch.nn.Module]:
    return _build(source)


@pytest.fixture(scope="module")
def bridge(models: tuple[TransformerBridge, torch.nn.Module]) -> TransformerBridge:
    return models[0]


@pytest.fixture(scope="module")
def reference(models: tuple[TransformerBridge, torch.nn.Module]) -> torch.nn.Module:
    return models[1]


@pytest.fixture(scope="module")
def tokens() -> torch.Tensor:
    return torch.tensor([[1, 17, 42, 5, 99, 63, 8, 120]])


@pytest.fixture(scope="module")
def cache(bridge: TransformerBridge, tokens: torch.Tensor) -> Any:
    with torch.no_grad():
        _, activation_cache = bridge.run_with_cache(tokens)
    return activation_cache


class TestMiniCPMTinyStructure:
    def test_block_count(self, bridge: TransformerBridge) -> None:
        assert len(bridge.blocks) == N_LAYERS

    def test_scales_reach_the_bridge(self, bridge: TransformerBridge) -> None:
        assert bridge.cfg.scale_emb == SCALE_EMB
        assert bridge.cfg.scale_depth == SCALE_DEPTH
        assert bridge.cfg.dim_model_base == DIM_MODEL_BASE
        assert bridge.blocks[0].residual_contribution_scale == pytest.approx(
            SCALE_DEPTH / N_LAYERS**0.5
        )

    def test_rotary_tables_repaired(
        self, bridge: TransformerBridge, reference: torch.nn.Module, source: _Source
    ) -> None:
        for got, want in zip(bridge.original_model.model.layers, reference.model.layers):
            got_rotary, want_rotary = got.self_attn.rotary_emb, want.self_attn.rotary_emb
            assert type(got_rotary).__name__ == source.rotary_class
            assert torch.allclose(got_rotary.inv_freq, want_rotary.inv_freq)
            assert torch.allclose(got_rotary.cos_cached, want_rotary.cos_cached)
            assert torch.allclose(got_rotary.sin_cached, want_rotary.sin_cached)


class TestMiniCPMTinyForwardParity:
    def test_logits_match_hf(
        self, bridge: TransformerBridge, reference: torch.nn.Module, tokens: torch.Tensor
    ) -> None:
        with torch.no_grad():
            got = bridge(tokens)
            want = reference(tokens).logits
        assert torch.allclose(got, want, atol=TOL), (got - want).abs().max()

    def test_residual_stream_matches_hf_per_layer(
        self, cache: Any, reference: torch.nn.Module, tokens: torch.Tensor
    ) -> None:
        """HF hidden_states[i] is layer i's input, already scaled by scale_emb at i=0."""
        with torch.no_grad():
            hidden_states = reference(tokens, output_hidden_states=True).hidden_states
        for i in range(N_LAYERS):
            got = cache[f"blocks.{i}.hook_resid_pre"]
            assert torch.allclose(got, hidden_states[i], atol=TOL), i


class TestMiniCPMTinyEmbeddingScale:
    def test_hook_embed_is_the_scaled_embedding(
        self, cache: Any, reference: torch.nn.Module, tokens: torch.Tensor
    ) -> None:
        with torch.no_grad():
            want = reference.model.embed_tokens(tokens) * SCALE_EMB
        assert torch.allclose(cache["hook_embed"], want, atol=TOL)
        assert torch.allclose(cache["hook_embed"], cache["blocks.0.hook_resid_pre"], atol=TOL)


class TestMiniCPMTinyResidualScale:
    def test_contributions_sum_to_the_residual_stream(self, cache: Any) -> None:
        for i in range(N_LAYERS):
            resid_mid = cache[f"blocks.{i}.hook_resid_pre"] + cache[f"blocks.{i}.hook_attn_out"]
            assert torch.allclose(resid_mid, cache[f"blocks.{i}.hook_resid_mid"], atol=TOL), i
            resid_post = resid_mid + cache[f"blocks.{i}.hook_mlp_out"]
            assert torch.allclose(resid_post, cache[f"blocks.{i}.hook_resid_post"], atol=TOL), i

    def test_hook_attn_out_is_the_scaled_module_output(self, cache: Any) -> None:
        scale = SCALE_DEPTH / N_LAYERS**0.5
        raw = cache["blocks.0.attn.hook_out"]
        assert torch.allclose(cache["blocks.0.hook_attn_out"], raw * scale, atol=TOL)


class TestMiniCPMTinyLogitScale:
    def test_transform_reproduces_the_logits(
        self, bridge: TransformerBridge, cache: Any, tokens: torch.Tensor
    ) -> None:
        with torch.no_grad():
            logits = bridge(tokens)
            unembedded = bridge.unembed(cache["ln_final.hook_out"])
        recomputed = bridge.adapter.apply_output_logits_transform(unembedded)
        assert torch.allclose(recomputed, logits, atol=TOL)
        assert not torch.allclose(unembedded, logits, atol=TOL)


class TestMiniCPMTinyGeneration:
    def test_cached_generation_matches_uncached_hf_greedy(
        self, bridge: TransformerBridge, reference: torch.nn.Module, tokens: torch.Tensor
    ) -> None:
        """Exercises the injected DynamicCache and the past_key_value rename."""
        want = tokens
        with torch.no_grad():
            for _ in range(5):
                next_token = reference(want, use_cache=False).logits[:, -1].argmax(-1)
                want = torch.cat([want, next_token[:, None]], dim=-1)
        for use_past_kv_cache in (True, False):
            got = bridge.generate(
                tokens,
                max_new_tokens=5,
                do_sample=False,
                stop_at_eos=False,
                use_past_kv_cache=use_past_kv_cache,
                verbose=False,
            )
            assert torch.equal(got, want), use_past_kv_cache


class TestMiniCPMTinyCompatibilityMode:
    def test_processed_weights_preserve_the_distribution(
        self, reference: torch.nn.Module, tokens: torch.Tensor, source: _Source
    ) -> None:
        """fold_ln + centering commute with all three scales: no weight fold needed."""
        compat_bridge, _ = _build(source)
        compat_bridge.enable_compatibility_mode()
        with torch.no_grad():
            got = compat_bridge(tokens).log_softmax(-1)
            want = reference(tokens).logits.log_softmax(-1)
        assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()
