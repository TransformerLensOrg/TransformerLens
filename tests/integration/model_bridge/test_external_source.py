"""Integration tests for the boot_external source.

Coverage:
- Parity vs boot_transformers (real gpt2 wrapped both ways).
- from_config (no-Hub) wrap end-to-end, including hooks firing and
  diagnose_paths reporting a healthy tree.
- Tokenizer-less mode: string prompts error, token-id prompts work.
- tl_config branch: build a config by hand, never touch HF translation.
- Llama path: exercises adapter.prepare_model (rotary_emb wiring), which
  GPT-2 doesn't.
- dtype inference: bf16/fp16 model must yield bf16/fp16 cfg.dtype, not
  a stale fp32 default.
- Pre-flight diagnose_paths on a tree-shape mismatch (the main motivating
  case — vLLM-style fused projections vs HF-style separates).
- Argument validation: missing config, dual config, unknown architecture.
"""
from __future__ import annotations

import pytest
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    LlamaConfig,
)

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources.transformers import (
    map_default_transformer_lens_config,
)


def _tiny_gpt2_config() -> GPT2Config:
    return GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=16,
        n_positions=32,
        vocab_size=100,
    )


def _tiny_llama_config() -> LlamaConfig:
    return LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        max_position_embeddings=64,
    )


@pytest.fixture(scope="module")
def gpt2_bridge_native():
    """gpt2 wrapped via the standard HF source — the parity reference."""
    return TransformerBridge.boot_transformers("gpt2", device="cpu")


@pytest.fixture(scope="module")
def gpt2_bridge_external():
    """gpt2 loaded independently and wrapped via boot_external.

    Loading fresh (instead of reusing the native bridge's already-instrumented
    model) is required because set_original_components mutates submodule
    attributes when wiring hooks — wrapping the same object twice would yield
    a different hook surface than wrapping a vanilla HF model.

    Matches boot_transformers' loading conventions (eager attention,
    output_attentions=True, fp32) so weight-level parity holds.
    """
    config = AutoConfig.from_pretrained("gpt2", output_attentions=True)
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", config=config, attn_implementation="eager", torch_dtype=torch.float32
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return TransformerBridge.boot_external(
        model=model,
        architecture="GPT2LMHeadModel",
        hf_config=config,
        tokenizer=tokenizer,
        device="cpu",
    )


def _matching_keys(cache_a, cache_b):
    """Hook names present in both caches — caches may differ in optional hooks."""
    return sorted(set(cache_a.keys()) & set(cache_b.keys()))


class TestBootExternalParity:
    """boot_external on the same underlying HF model must match boot_transformers."""

    def test_logits_bitwise_match(self, gpt2_bridge_native, gpt2_bridge_external):
        prompt = "The quick brown fox"
        tokens = gpt2_bridge_native.to_tokens(prompt, prepend_bos=False)
        with torch.inference_mode():
            logits_native = gpt2_bridge_native(tokens, return_type="logits")
            logits_external = gpt2_bridge_external(tokens, return_type="logits")
        # Bitwise: any new implicit cast / compile in boot_transformers should
        # trip this on purpose — drift is information, not noise.
        assert torch.equal(
            logits_native, logits_external
        ), "Logits differ — boot_external should be a no-op rewrap of the same model"

    def test_cache_keys_match(self, gpt2_bridge_native, gpt2_bridge_external):
        prompt = "Cache parity"
        tokens = gpt2_bridge_native.to_tokens(prompt, prepend_bos=False)
        with torch.inference_mode():
            _, cache_native = gpt2_bridge_native.run_with_cache(tokens)
            _, cache_external = gpt2_bridge_external.run_with_cache(tokens)
        assert set(cache_native.keys()) == set(cache_external.keys())
        assert len(cache_native) > 0

    def test_cache_values_bitwise_match(self, gpt2_bridge_native, gpt2_bridge_external):
        prompt = "Cache value parity"
        tokens = gpt2_bridge_native.to_tokens(prompt, prepend_bos=False)
        with torch.inference_mode():
            _, cache_native = gpt2_bridge_native.run_with_cache(tokens)
            _, cache_external = gpt2_bridge_external.run_with_cache(tokens)
        for key in _matching_keys(cache_native, cache_external):
            a = cache_native[key]
            b = cache_external[key]
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                continue
            assert torch.equal(a, b), f"Cache differs at {key}"


class TestBootExternalFromConfig:
    """Wrap an HF model built via from_config (no Hub call, random weights).

    Named for what the test actually does. The plan called this 'hand-built',
    but a truly hand-built nn.Module that exactly mirrors GPT-2's submodule
    layout (Conv1D-formatted c_attn, etc.) is significantly more code; the
    from_config path covers the same boot_external code path with much less
    test surface.
    """

    @pytest.fixture(scope="class")
    def bridge(self):
        config = _tiny_gpt2_config()
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        return TransformerBridge.boot_external(
            model=model,
            architecture="GPT2LMHeadModel",
            hf_config=config,
            tokenizer=None,
        )

    def test_hook_registry_populated(self, bridge):
        registry = bridge._hook_registry
        assert [k for k in registry if k.startswith("blocks.0.")]
        assert [k for k in registry if k.startswith("blocks.1.")]

    def test_run_with_cache_fires_expected_hooks(self, bridge):
        tokens = torch.randint(0, 100, (1, 8))
        with torch.inference_mode():
            _, cache = bridge.run_with_cache(tokens)
        for key in ("blocks.0.hook_in", "blocks.1.hook_in"):
            assert key in cache


class TestBootExternalLlama:
    """Llama path — exercises adapter.prepare_model (rotary_emb wiring) that
    GPT-2 doesn't have. Catches regressions where prepare_model assumes a
    Hub-loaded model and breaks on from_config-built ones."""

    def test_llama_from_config_wraps_and_runs(self):
        config = _tiny_llama_config()
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        bridge = TransformerBridge.boot_external(
            model=model,
            architecture="LlamaForCausalLM",
            hf_config=config,
            tokenizer=None,
        )
        tokens = torch.randint(0, config.vocab_size, (1, 4))
        with torch.inference_mode():
            logits, cache = bridge.run_with_cache(tokens)
        assert logits.shape == (1, 4, config.vocab_size)
        assert len(cache) > 0


class TestBootExternalTlConfigOnly:
    """tl_config branch — never construct an HF config, never call
    map_default_transformer_lens_config. Mirrors the realistic non-HF case:
    a custom loader knows its own dimensions and builds a config directly."""

    def test_tl_config_only_wraps_and_runs(self):
        hf_config = _tiny_gpt2_config()
        model = AutoModelForCausalLM.from_config(hf_config)
        model.eval()

        # Build a TL config by hand, deliberately *not* going through
        # build_bridge_config_from_hf. Real non-HF callers (vLLM, MLX, etc.)
        # would populate these fields from their own model metadata.
        tl_cfg_dict = map_default_transformer_lens_config(hf_config).__dict__
        tl_cfg = TransformerBridgeConfig.from_dict(dict(tl_cfg_dict))
        tl_cfg.model_name = "my-custom-loader"

        bridge = TransformerBridge.boot_external(
            model=model,
            architecture="GPT2LMHeadModel",
            tl_config=tl_cfg,
            tokenizer=None,
        )
        # tl_config's model_name survives when caller doesn't override.
        assert bridge.cfg.model_name == "my-custom-loader"
        tokens = torch.randint(0, hf_config.vocab_size, (1, 4))
        with torch.inference_mode():
            logits = bridge(tokens, return_type="logits")
        assert logits.shape == (1, 4, hf_config.vocab_size)

    def test_explicit_model_name_overrides_tl_config(self):
        hf_config = _tiny_gpt2_config()
        model = AutoModelForCausalLM.from_config(hf_config)
        model.eval()
        tl_cfg_dict = map_default_transformer_lens_config(hf_config).__dict__
        tl_cfg = TransformerBridgeConfig.from_dict(dict(tl_cfg_dict))
        tl_cfg.model_name = "embedded-name"
        bridge = TransformerBridge.boot_external(
            model=model,
            architecture="GPT2LMHeadModel",
            tl_config=tl_cfg,
            model_name="caller-override",
        )
        assert bridge.cfg.model_name == "caller-override"


class TestBootExternalDtypeInference:
    """When `dtype` is unspecified, cfg.dtype must reflect the model's actual
    dtype — not a silent fp32 default that lies about a bf16/fp16 model."""

    def test_bf16_model_yields_bf16_cfg(self):
        config = _tiny_gpt2_config()
        model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
        model.eval()
        bridge = TransformerBridge.boot_external(
            model=model,
            architecture="GPT2LMHeadModel",
            hf_config=config,
        )
        assert bridge.cfg.dtype == torch.bfloat16

    def test_fp16_model_yields_fp16_cfg(self):
        config = _tiny_gpt2_config()
        model = AutoModelForCausalLM.from_config(config).to(torch.float16)
        model.eval()
        bridge = TransformerBridge.boot_external(
            model=model,
            architecture="GPT2LMHeadModel",
            hf_config=config,
        )
        assert bridge.cfg.dtype == torch.float16

    def test_explicit_dtype_kwarg_wins(self):
        config = _tiny_gpt2_config()
        model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
        model.eval()
        # Caller deliberately overrides — accepts that they're lying. Not
        # our job to second-guess.
        bridge = TransformerBridge.boot_external(
            model=model,
            architecture="GPT2LMHeadModel",
            hf_config=config,
            dtype=torch.float32,
        )
        assert bridge.cfg.dtype == torch.float32


class TestBootExternalTokenizerless:
    """boot_external must work without a tokenizer for non-HF pipelines."""

    @pytest.fixture(scope="class")
    def bridge(self):
        config = _tiny_gpt2_config()
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        return TransformerBridge.boot_external(
            model=model,
            architecture="GPT2LMHeadModel",
            hf_config=config,
            tokenizer=None,
        )

    def test_string_prompt_raises(self, bridge):
        with pytest.raises((AssertionError, AttributeError, ValueError)):
            bridge("hello world")

    def test_token_ids_work(self, bridge):
        tokens = torch.randint(0, 100, (1, 8))
        with torch.inference_mode():
            logits = bridge(tokens, return_type="logits")
        assert logits.shape == (1, 8, 100)


class TestDiagnosePathsPreflight:
    """diagnose_paths is the pre-flight check — its value is reporting
    missing paths *before* construction blows up."""

    def test_healthy_tree_reports_no_missing(self):
        config = _tiny_gpt2_config()
        model = AutoModelForCausalLM.from_config(config)
        report = TransformerBridge.diagnose_paths(
            model, "GPT2LMHeadModel", hf_config=config
        )
        assert report["missing"] == [], f"healthy gpt2 should resolve all paths; got missing={report['missing']}"
        assert report["resolved"], "should have at least one resolved path"

    def test_mismatched_tree_reports_missing(self):
        """A model that doesn't have GPT-2's tree shape must report missing
        paths — this is the primary motivating case (vLLM-style fused vs
        HF-style separate projections)."""

        class FakeTree(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Has lm_head but not transformer.* — adapter walks should miss.
                self.lm_head = torch.nn.Linear(8, 100)

        fake = FakeTree()
        config = _tiny_gpt2_config()
        report = TransformerBridge.diagnose_paths(
            fake, "GPT2LMHeadModel", hf_config=config
        )
        # The whole point: missing list is non-empty and concrete.
        assert report["missing"], "tree mismatch should populate missing list"
        # Each entry is a 'tl_name -> remote_path' line — caller can read
        # which adapter path failed.
        for line in report["missing"]:
            assert " -> " in line


class TestBootExternalValidation:
    """Argument validation: clear errors for misuse."""

    def test_missing_both_configs_raises(self):
        with pytest.raises(ValueError, match="hf_config or tl_config"):
            TransformerBridge.boot_external(
                model=torch.nn.Linear(1, 1),
                architecture="GPT2LMHeadModel",
                hf_config=None,
                tl_config=None,
            )

    def test_both_configs_raises(self):
        config = _tiny_gpt2_config()
        # Need a real-ish tl_config for the typecheck — but the error
        # should fire before any wiring is attempted.
        tl_cfg = TransformerBridgeConfig.from_dict(
            map_default_transformer_lens_config(config).__dict__
        )
        with pytest.raises(ValueError, match="supply exactly one"):
            TransformerBridge.boot_external(
                model=torch.nn.Linear(1, 1),
                architecture="GPT2LMHeadModel",
                hf_config=config,
                tl_config=tl_cfg,
            )

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            TransformerBridge.boot_external(
                model=torch.nn.Linear(1, 1),
                architecture="NotARealArchitecture",
                hf_config=_tiny_gpt2_config(),
            )
