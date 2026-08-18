from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.config import HookedTransformerConfig
from transformer_lens.loading_from_pretrained import (
    _mxfp4_dequantize_config,
    fill_missing_keys,
    get_pretrained_model_config,
)


def get_default_config():
    return HookedTransformerConfig(
        d_model=128, d_head=8, n_heads=16, n_ctx=128, n_layers=1, d_vocab=50257, attn_only=True
    )


def _config_with_architecture(
    architecture: str, quantization_method: str | None = None
) -> HookedTransformerConfig:
    return HookedTransformerConfig(
        d_model=128,
        d_head=8,
        n_heads=16,
        n_ctx=128,
        n_layers=1,
        d_vocab=50257,
        attn_only=True,
        original_architecture=architecture,
        quantization_method=quantization_method,
    )


class TestMxfp4DequantizeConfig:
    """Packed-MXFP4 checkpoints must load dequantized so the weight converter
    sees plain torch.Tensors instead of triton-kernels wrappers (#1619)."""

    def test_mxfp4_gets_dequantize_config(self):
        result = _mxfp4_dequantize_config(_config_with_architecture("GptOssForCausalLM", "mxfp4"))
        assert result is not None
        assert result.dequantize is True

    def test_applies_beyond_gpt_oss(self):
        """The architecture gate this used to carry existed only to dodge a
        second AutoConfig fetch. MXFP4 Qwen3-MoE checkpoints are in the registry
        and pack their experts the same way, so they must dequantize too."""
        result = _mxfp4_dequantize_config(_config_with_architecture("Qwen3MoeForCausalLM", "mxfp4"))
        assert result is not None
        assert result.dequantize is True

    def test_unquantized_finetune_untouched(self):
        assert _mxfp4_dequantize_config(_config_with_architecture("GptOssForCausalLM")) is None

    def test_other_quantizations_are_not_dequantized_here(self):
        """Only MXFP4 has a dequantize-on-load path; the rest are refused later
        by the converter guards, which give a better-targeted message."""
        for method in ("bitsandbytes", "gptq", "awq", "fp8"):
            assert (
                _mxfp4_dequantize_config(_config_with_architecture("LlamaForCausalLM", method))
                is None
            )

    @mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
    def test_costs_no_hub_round_trip(self, mock_auto_config: mock.MagicMock):
        """The whole point of carrying the method on the cfg: this runs on every
        load, and a fetch here would be a Hub HEAD request per model load."""
        _mxfp4_dequantize_config(_config_with_architecture("GptOssForCausalLM", "mxfp4"))
        mock_auto_config.from_pretrained.assert_not_called()


class TestUnsupportedQuantizationRefusal:
    """The ordinary from_pretrained("<name>") path must refuse quantized weights:
    the old refusal lived under `if hf_model is not None`, and same-shape
    int8/FP8 survives converter AND load_state_dict (cast to fp32) silently.
    """

    @staticmethod
    def _model(dtype=None, quant_method="gptq"):
        class _Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = torch.nn.Linear(4, 4, bias=False)
                if dtype is not None:
                    self.q_proj.weight = torch.nn.Parameter(
                        torch.zeros(4, 4, dtype=dtype), requires_grad=False
                    )

        model = _Tiny()
        quantization_config = {"quant_method": quant_method} if quant_method is not None else None
        model.config = SimpleNamespace(quantization_config=quantization_config)
        return model

    @pytest.mark.parametrize(
        "dtype",
        [torch.int8, torch.uint8, torch.float8_e4m3fn],
    )
    def test_quantized_storage_is_refused(self, dtype):
        from transformer_lens.loading_from_pretrained import (
            _refuse_unsupported_quantization,
        )

        cfg = _config_with_architecture("LlamaForCausalLM")
        with pytest.raises(NotImplementedError, match="gptq"):
            _refuse_unsupported_quantization(cfg, self._model(dtype))

    def test_unquantized_model_passes(self):
        """The positive control — every ordinary load goes through this."""
        from transformer_lens.loading_from_pretrained import (
            _refuse_unsupported_quantization,
        )

        cfg = _config_with_architecture("LlamaForCausalLM")
        _refuse_unsupported_quantization(cfg, self._model(quant_method=None))

    def test_dequantized_checkpoint_still_loads(self):
        """A model loaded with dequantize=True keeps advertising its original
        quant_method while holding real bf16 tensors. Refusing on the
        declaration alone would break the MXFP4 auto-dequantize path."""
        from transformer_lens.loading_from_pretrained import (
            _refuse_unsupported_quantization,
        )

        cfg = _config_with_architecture("GptOssForCausalLM", "mxfp4")
        _refuse_unsupported_quantization(cfg, self._model(quant_method="mxfp4"))

    def test_bitsandbytes_4bit_llama_flow_is_preserved(self):
        """The one supported quantized HT flow: weight conversion and
        abstract_attention's matmul_4bit both depend on it."""
        from transformer_lens.loading_from_pretrained import (
            _refuse_unsupported_quantization,
        )

        cfg = _config_with_architecture("LlamaForCausalLM")
        cfg.load_in_4bit = True
        _refuse_unsupported_quantization(cfg, self._model(torch.uint8, quant_method="bitsandbytes"))

    @mock.patch("transformer_lens.loading_from_pretrained.AutoModelForCausalLM")
    def test_refusal_is_wired_into_the_internal_load_path(self, mock_auto_model: mock.MagicMock):
        """The wiring, not just the helper: the refusal must fire where TL loads the
        model itself and the caller never sees an hf_model.
        """
        from transformer_lens.loading_from_pretrained import get_pretrained_state_dict

        mock_auto_model.from_pretrained.return_value = self._model(torch.int8)

        cfg = _config_with_architecture("LlamaForCausalLM")
        with pytest.raises(NotImplementedError, match="gptq"):
            get_pretrained_state_dict("meta-llama/Llama-2-7b-hf", cfg)


class TestQuantizationMethodCapture:
    """`convert_hf_model_config` must record the checkpoint's quant_method while
    the HF config is in hand — that capture is what lets the loader act on the
    quantization without refetching."""

    @pytest.mark.parametrize(
        "quantization_config",
        [
            {"quant_method": "mxfp4"},
            SimpleNamespace(quant_method="mxfp4"),
        ],
        ids=["dict-style", "object-style"],
    )
    def test_capture_handles_both_config_shapes(self, quantization_config):
        from transformer_lens.utilities.quantization import quantization_method

        assert (
            quantization_method(SimpleNamespace(quantization_config=quantization_config)) == "mxfp4"
        )

    def test_unquantized_config_captures_none(self):
        from transformer_lens.utilities.quantization import quantization_method

        assert quantization_method(SimpleNamespace()) is None
        assert quantization_method(None) is None

    @staticmethod
    def _gpt2_shaped_config(**extra):
        """The attributes convert_hf_model_config's GPT2 branch reads, so the
        real function can run without touching the Hub."""
        return SimpleNamespace(
            architectures=["GPT2LMHeadModel"],
            n_embd=128,
            n_head=4,
            n_layer=2,
            n_ctx=1024,
            layer_norm_epsilon=1e-5,
            vocab_size=50257,
            activation_function="gelu_new",
            scale_attn_by_inverse_layer_idx=False,
            **extra,
        )

    @mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
    def test_convert_hf_model_config_records_the_method(self, mock_auto_config: mock.MagicMock):
        """The wiring, not just the extractor: without this the loader would
        silently stop dequantizing MXFP4 and the converter would raise instead."""
        from transformer_lens.loading_from_pretrained import convert_hf_model_config

        mock_auto_config.from_pretrained.return_value = self._gpt2_shaped_config(
            quantization_config={"quant_method": "mxfp4"}
        )
        assert convert_hf_model_config("gpt2")["quantization_method"] == "mxfp4"

    @mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
    def test_convert_hf_model_config_records_none_when_unquantized(
        self, mock_auto_config: mock.MagicMock
    ):
        from transformer_lens.loading_from_pretrained import convert_hf_model_config

        mock_auto_config.from_pretrained.return_value = self._gpt2_shaped_config()
        assert convert_hf_model_config("gpt2")["quantization_method"] is None

    @mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
    def test_user_supplied_hf_model_config_wins(self, mock_auto_config: mock.MagicMock):
        """A user passing hf_model= says how the weights in hand are stored,
        which can differ from the Hub repo's declaration (they may have loaded
        it dequantized, or quantized one that ships in bf16)."""
        mock_auto_config.from_pretrained.return_value = self._gpt2_shaped_config()

        cfg = get_pretrained_model_config(
            "gpt2", hf_cfg={"quantization_config": {"quant_method": "bitsandbytes"}}
        )
        assert cfg.quantization_method == "bitsandbytes"

    @mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
    def test_unquantized_hf_model_does_not_erase_the_repo_method(
        self, mock_auto_config: mock.MagicMock
    ):
        """hf_cfg is often present but carries no quantization_config at all;
        that absence must not wipe what the repo config declared."""
        mock_auto_config.from_pretrained.return_value = self._gpt2_shaped_config(
            quantization_config={"quant_method": "mxfp4"}
        )

        cfg = get_pretrained_model_config("gpt2", hf_cfg={"vocab_size": 50257})
        assert cfg.quantization_method == "mxfp4"

    @mock.patch("transformer_lens.loading_from_pretrained.convert_neel_model_config")
    def test_config_builders_that_never_see_an_hf_config(self, mock_neel: mock.MagicMock):
        """convert_neel_model_config builds cfg_dict from the name alone — reading
        quantization_method with [] broke every NeelNanda load.
        """
        mock_neel.return_value = {
            "d_model": 128,
            "d_head": 8,
            "n_heads": 16,
            "n_ctx": 128,
            "n_layers": 1,
            "d_vocab": 50257,
            "attn_only": True,
            "original_architecture": "neel",
        }

        cfg = get_pretrained_model_config("NeelNanda/SoLU_2L512W_C4_Code", hf_cfg={})
        assert cfg.quantization_method is None

    def test_name_based_architecture_branches_capture_none(self):
        """Llama/gemma names never fetch a config, so nothing is there to read.
        Pinned because the field must be present-and-None rather than missing —
        HookedTransformerConfig.from_dict passes cfg_dict through unfiltered."""
        from transformer_lens.loading_from_pretrained import convert_hf_model_config

        cfg_dict = convert_hf_model_config("llama-7b-hf")
        assert cfg_dict["quantization_method"] is None


# Successes


def test_fill_missing_keys_raises_on_missing_attention_weights():
    """A missing attention W means converter/component naming mismatch (#1620) —
    zero-filling it silently breaks the model, so it must raise instead.
    (The warn-and-fill behavior is covered by the bridge-based tests in
    tests/unit/model_bridge/compatibility/test_loading_from_pretrained_utilities.py.)"""
    cfg = HookedTransformerConfig(
        d_model=128, d_head=8, n_heads=16, n_ctx=128, n_layers=1, d_vocab=50257, attn_only=True
    )
    model = HookedTransformer(cfg)
    default_state_dict = model.state_dict()

    incomplete_state_dict = {
        k: v for k, v in default_state_dict.items() if not k.endswith("attn.W_K")
    }

    with pytest.raises(ValueError, match="missing weight matrices"):
        fill_missing_keys(model, incomplete_state_dict)


def test_n_ctx_override_reduces_context():
    """
    n_ctx override should work when reducing below the model default.
    Uses the config loader directly — no model loading needed.
    Fixes #1006.
    """
    from transformer_lens.loading_from_pretrained import get_pretrained_model_config

    cfg = get_pretrained_model_config("gpt2", n_ctx=256)
    assert cfg.n_ctx == 256, f"Expected n_ctx=256, got {cfg.n_ctx}"


@mock.patch("logging.warning")
def test_n_ctx_override_larger_than_default_warns(mock_warning: mock.MagicMock):
    """
    A warning should be issued when n_ctx exceeds the model's default.
    GPT-2 default n_ctx is 1024 — requesting 2048 should trigger the warning.
    Fixes #1006.
    """
    from transformer_lens.loading_from_pretrained import get_pretrained_model_config

    cfg = get_pretrained_model_config("gpt2", n_ctx=2048)
    assert cfg.n_ctx == 2048, f"Expected n_ctx=2048, got {cfg.n_ctx}"
    mock_warning.assert_any_call(
        "You are setting n_ctx=2048 which is larger than this model's "
        "default context length of 1024. The model was not "
        "trained on sequences this long and may produce unreliable results. "
        "Ensure you have sufficient memory for this context length."
    )


def _minimal_qwen_config(architecture: str, **overrides: object) -> SimpleNamespace:
    values = {
        "architectures": [architecture],
        "hidden_size": 128,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "layer_norm_epsilon": 1e-6,
        "rms_norm_eps": 1e-6,
        "vocab_size": 1000,
        "scale_attn_weights": True,
        "initializer_range": 0.02,
        "kv_channels": 32,
        "num_key_value_heads": 4,
        "hidden_act": "silu",
        "rope_theta": 1000000.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@mock.patch("transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained")
def test_qwen_uses_hf_seq_length_for_n_ctx(mock_from_pretrained: mock.MagicMock):
    mock_from_pretrained.return_value = _minimal_qwen_config(
        "QWenLMHeadModel",
        seq_length=8192,
        max_position_embeddings=32768,
    )

    cfg = get_pretrained_model_config("Qwen/Qwen-7B")

    assert cfg.n_ctx == 8192


@mock.patch("transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained")
@pytest.mark.parametrize(
    ("model_name", "max_position_embeddings"),
    [
        ("Qwen/Qwen1.5-0.5B", 32768),
        ("Qwen/Qwen2-0.5B", 131072),
    ],
)
def test_qwen2_uses_hf_max_position_embeddings_for_n_ctx(
    mock_from_pretrained: mock.MagicMock,
    model_name: str,
    max_position_embeddings: int,
):
    mock_from_pretrained.return_value = _minimal_qwen_config(
        "Qwen2ForCausalLM",
        max_position_embeddings=max_position_embeddings,
    )

    cfg = get_pretrained_model_config(model_name)

    assert cfg.n_ctx == max_position_embeddings


# --- Architecture config tests ---


class TestArchitectureConfigs:
    """Verify that convert_hf_model_config produces correct configs for new architectures."""

    def test_apertus_config(self):
        try:
            cfg = get_pretrained_model_config("apertus-8b")
        except ValueError as e:
            if "does not recognize this architecture" in str(e):
                pytest.skip(f"transformers version too old: {e}")
            raise
        assert cfg.original_architecture == "ApertusForCausalLM"
        assert cfg.normalization_type == "RMS"
        assert cfg.positional_embedding_type == "rotary"
        assert cfg.gated_mlp is False
        assert cfg.final_rms is True
        assert cfg.act_fn == "xielu"
        assert cfg.use_qk_norm is True
        assert cfg.n_key_value_heads is not None
        assert cfg.d_model > 0
        assert cfg.n_heads > 0

    def test_gpt_oss_config(self):
        try:
            cfg = get_pretrained_model_config("gpt-oss-20b")
        except ValueError as e:
            if "does not recognize this architecture" in str(e):
                pytest.skip(f"transformers version too old: {e}")
            raise
        assert cfg.original_architecture == "GptOssForCausalLM"
        assert cfg.normalization_type == "RMS"
        assert cfg.positional_embedding_type == "rotary"
        assert cfg.gated_mlp is True
        assert cfg.final_rms is True
        assert cfg.num_experts is not None
        assert cfg.num_experts > 0
        assert cfg.experts_per_token is not None
        assert cfg.n_key_value_heads is not None

    def test_apertus_instruct_config(self):
        try:
            cfg = get_pretrained_model_config("apertus-8b-instruct")
        except ValueError as e:
            if "does not recognize this architecture" in str(e):
                pytest.skip(f"transformers version too old: {e}")
            raise
        assert cfg.original_architecture == "ApertusForCausalLM"
        assert cfg.act_fn == "xielu"

    @pytest.mark.parametrize("use_parallel_residual", [True, False])
    def test_gpt_neox_parallel_residual_follows_hf_config(
        self, tmp_path, use_parallel_residual: bool
    ):
        """GPTNeoX ships sequential variants; parallel_attn_mlp must not be hardcoded."""
        from transformers import GPTNeoXConfig

        from transformer_lens.loading_from_pretrained import convert_hf_model_config

        hf_config = GPTNeoXConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=128,
            use_parallel_residual=use_parallel_residual,
        )
        hf_config.architectures = ["GPTNeoXForCausalLM"]
        hf_config.save_pretrained(tmp_path)

        cfg_dict = convert_hf_model_config(str(tmp_path))

        assert cfg_dict["parallel_attn_mlp"] is use_parallel_residual


class TestHetConfigThroughConvertHfModelConfig:
    """convert_hf_model_config had zero het protection: transformers>=5.15
    per-layer fields raise (not AttributeError) on global reads, so the branch
    chain's own hasattr probes were crash sites."""

    @mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
    def test_per_layer_field_resolves_to_majority(self, mock_auto_config: mock.MagicMock):
        from tests.unit.model_bridge.test_config_mapping import _HeterogeneousConfig
        from transformer_lens.loading_from_pretrained import convert_hf_model_config

        mock_auto_config.from_pretrained.return_value = _HeterogeneousConfig(
            {"scale_attn_by_inverse_layer_idx": [False, False]},
            architectures=["GPT2LMHeadModel"],
            n_embd=128,
            n_head=4,
            n_layer=2,
            num_hidden_layers=2,
            n_ctx=1024,
            layer_norm_epsilon=1e-5,
            vocab_size=50257,
            activation_function="gelu_new",
        )
        cfg_dict = convert_hf_model_config("gpt2")
        assert cfg_dict["scale_attn_by_inverse_layer_idx"] is False


class TestFillMissingKeysFailLoud:
    """Zero-filling an MLP weight matrix is the same silent-sublayer-death as
    the attention case; norm fills are frequently right but must be named."""

    @staticmethod
    def _model_and_state(missing_leaf):
        cfg = get_default_config()
        cfg.attn_only = False
        model = HookedTransformer(
            HookedTransformerConfig(
                d_model=128,
                d_head=8,
                n_heads=16,
                n_ctx=128,
                n_layers=1,
                d_vocab=50257,
                d_mlp=256,
                act_fn="relu",
            )
        )
        state_dict = model.state_dict()
        removed = [key for key in state_dict if key.rsplit(".", 1)[-1] == missing_leaf]
        assert removed, f"fixture produced no {missing_leaf} keys"
        for key in removed:
            del state_dict[key]
        return model, state_dict

    @pytest.mark.parametrize("leaf", ["W_in", "W_out"])
    def test_missing_mlp_matrices_raise(self, leaf):
        model, state_dict = self._model_and_state(leaf)
        with pytest.raises(ValueError, match=leaf):
            fill_missing_keys(model, state_dict)

    @mock.patch("logging.warning")
    def test_missing_norm_keys_warn_by_name(self, mock_warning: mock.MagicMock):
        model, state_dict = self._model_and_state("w")
        fill_missing_keys(model, state_dict)
        norm_warnings = [
            call for call in mock_warning.call_args_list if "normalization" in str(call.args[0])
        ]
        assert norm_warnings, "norm fills must be named, not silent"
