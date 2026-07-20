"""Shared HF-format utilities used by every source whose backend produces an
HF-shaped config or ``PreTrainedTokenizerBase`` tokenizer.

The transformers source loads HF objects directly; the vLLM source extracts the
same HF-shaped config via ``llm.llm_engine.model_config.hf_config`` because vLLM
re-uses the ``transformers`` config and tokenizer libraries internally. This
module is loader-agnostic — it speaks HF format, not HF loading.
"""
from __future__ import annotations

import copy

from transformers import PreTrainedTokenizerBase

from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.utilities import get_tokenizer_with_bos


def get_effective_text_config(hf_config):
    """Return the config that owns the language-model forward path."""
    if getattr(hf_config, "text_config", None) is not None:
        return hf_config.text_config
    decoder = getattr(hf_config, "decoder", None)
    if decoder is not None and hasattr(decoder, "hidden_size"):
        return decoder
    return hf_config


def map_default_transformer_lens_config(hf_config):
    """Map HuggingFace config fields to TransformerLens config format.

    Standardized mapping from various HuggingFace config field names to the
    consistent TransformerLens naming convention. For multimodal models (LLaVA,
    Gemma3ForConditionalGeneration), the language model dimensions are nested
    under ``text_config``; we extract from there first.

    Args:
        hf_config: The HuggingFace config object

    Returns:
        A copy of hf_config with additional TransformerLens fields
    """
    # Extract language model config from text_config for multimodal models
    source_config = get_effective_text_config(hf_config)

    tl_config = copy.deepcopy(hf_config)
    if hasattr(source_config, "n_embd"):
        tl_config.d_model = source_config.n_embd
    elif hasattr(source_config, "hidden_size"):
        tl_config.d_model = source_config.hidden_size
    elif hasattr(source_config, "model_dim"):
        tl_config.d_model = source_config.model_dim
    elif hasattr(source_config, "d_model"):
        tl_config.d_model = source_config.d_model
    elif hasattr(source_config, "hidden_dim"):
        tl_config.d_model = source_config.hidden_dim
    if hasattr(source_config, "n_head"):
        tl_config.n_heads = source_config.n_head
    elif hasattr(source_config, "num_attention_heads"):
        n_heads = source_config.num_attention_heads
        if isinstance(n_heads, list):
            n_heads = max(n_heads)
        tl_config.n_heads = n_heads
    elif hasattr(source_config, "num_heads"):
        tl_config.n_heads = source_config.num_heads
    elif hasattr(source_config, "n_heads"):
        tl_config.n_heads = source_config.n_heads
    elif hasattr(source_config, "num_query_heads") and isinstance(
        source_config.num_query_heads, list
    ):
        tl_config.n_heads = max(source_config.num_query_heads)
    if (
        hasattr(source_config, "num_key_value_heads")
        and source_config.num_key_value_heads is not None
    ):
        try:
            num_kv_heads = source_config.num_key_value_heads
            # Per-layer lists (e.g., OpenELM) collapse to the max.
            if isinstance(num_kv_heads, list):
                num_kv_heads = max(num_kv_heads)
            if hasattr(num_kv_heads, "item"):
                num_kv_heads = num_kv_heads.item()
            num_kv_heads = int(num_kv_heads)
            num_heads = tl_config.n_heads
            if hasattr(num_heads, "item"):
                num_heads = num_heads.item()
            num_heads = int(num_heads)
            if num_kv_heads != num_heads:
                tl_config.n_key_value_heads = num_kv_heads
        except (TypeError, ValueError, AttributeError):
            pass
    elif hasattr(source_config, "num_kv_heads") and source_config.num_kv_heads is not None:
        try:
            num_kv_heads = source_config.num_kv_heads
            if isinstance(num_kv_heads, list):
                num_kv_heads = max(num_kv_heads)
            if hasattr(num_kv_heads, "item"):
                num_kv_heads = num_kv_heads.item()
            num_kv_heads = int(num_kv_heads)
            num_heads = tl_config.n_heads
            if hasattr(num_heads, "item"):
                num_heads = num_heads.item()
            num_heads = int(num_heads)
            if num_kv_heads != num_heads:
                tl_config.n_key_value_heads = num_kv_heads
        except (TypeError, ValueError, AttributeError):
            pass
    elif hasattr(source_config, "n_kv_heads") and source_config.n_kv_heads is not None:
        try:
            num_kv_heads = int(source_config.n_kv_heads)
            if num_kv_heads != getattr(tl_config, "n_heads", None):
                tl_config.n_key_value_heads = num_kv_heads
        except (TypeError, ValueError, AttributeError):
            pass
    if hasattr(source_config, "n_layer"):
        tl_config.n_layers = source_config.n_layer
    elif hasattr(source_config, "num_hidden_layers"):
        tl_config.n_layers = source_config.num_hidden_layers
    elif hasattr(source_config, "num_transformer_layers"):
        tl_config.n_layers = source_config.num_transformer_layers
    elif hasattr(source_config, "num_layers"):
        tl_config.n_layers = source_config.num_layers
    elif hasattr(source_config, "n_layers"):
        tl_config.n_layers = source_config.n_layers
    elif hasattr(source_config, "n_blocks"):
        tl_config.n_layers = source_config.n_blocks
    if hasattr(source_config, "vocab_size") and isinstance(source_config.vocab_size, int):
        tl_config.d_vocab = source_config.vocab_size
    if hasattr(source_config, "n_positions"):
        tl_config.n_ctx = source_config.n_positions
    elif hasattr(source_config, "max_position_embeddings"):
        tl_config.n_ctx = source_config.max_position_embeddings
    elif hasattr(source_config, "max_context_length"):
        tl_config.n_ctx = source_config.max_context_length
    elif hasattr(source_config, "max_length"):
        tl_config.n_ctx = source_config.max_length
    elif hasattr(source_config, "seq_length"):
        tl_config.n_ctx = source_config.seq_length
    elif hasattr(source_config, "max_sequence_length"):
        tl_config.n_ctx = source_config.max_sequence_length
    else:
        # ALiBi models (Bloom) have no context length field; 2048 is a safe fallback.
        tl_config.n_ctx = 2048
    if hasattr(source_config, "n_inner"):
        tl_config.d_mlp = source_config.n_inner
    elif hasattr(source_config, "intermediate_size"):
        intermediate_size = source_config.intermediate_size
        # Gemma 3n exposes a per-layer intermediate_size list (the MatFormer design permits
        # variation). All released checkpoints (E2B/E4B) are uniform, and d_mlp is scalar
        # metadata (the bridge defers MLP math to HF), so collapse to max — the shared value
        # when uniform, an upper bound otherwise.
        if isinstance(intermediate_size, (list, tuple)):
            intermediate_size = max(intermediate_size) if intermediate_size else None
        tl_config.d_mlp = intermediate_size
    elif hasattr(source_config, "mlp_hidden_size"):
        tl_config.d_mlp = source_config.mlp_hidden_size
    elif hasattr(tl_config, "d_model"):
        tl_config.d_mlp = getattr(source_config, "n_inner", 4 * tl_config.d_model)
    if hasattr(source_config, "head_dim") and source_config.head_dim is not None:
        tl_config.d_head = source_config.head_dim
    elif hasattr(tl_config, "d_model") and hasattr(tl_config, "n_heads"):
        tl_config.d_head = tl_config.d_model // tl_config.n_heads
    elif hasattr(tl_config, "d_model"):
        # Attention-less architectures (Mamba SSMs): set d_head = d_model so
        # __post_init__ computes n_heads = 1. Values are nominal.
        tl_config.d_head = tl_config.d_model
    if hasattr(source_config, "activation_function"):
        tl_config.act_fn = source_config.activation_function
    elif hasattr(source_config, "hidden_act"):
        tl_config.act_fn = source_config.hidden_act
    elif hasattr(source_config, "activation_type"):
        activation_type = source_config.activation_type
        tl_config.act_fn = getattr(activation_type, "value", activation_type)
    if hasattr(source_config, "rope_theta"):
        tl_config.rotary_base = source_config.rope_theta
    if hasattr(source_config, "weight_tying"):
        tl_config.tie_word_embeddings = bool(source_config.weight_tying)
    # LayerNorm / RMSNorm epsilon — HF uses 3 different field names.
    if hasattr(source_config, "rms_norm_eps"):
        tl_config.eps = source_config.rms_norm_eps
    elif hasattr(source_config, "layer_norm_eps"):
        tl_config.eps = source_config.layer_norm_eps
    elif hasattr(source_config, "layer_norm_epsilon"):
        tl_config.eps = source_config.layer_norm_epsilon
    elif hasattr(source_config, "norm_eps"):
        tl_config.eps = source_config.norm_eps
    if hasattr(source_config, "num_experts"):
        tl_config.num_experts = source_config.num_experts
    elif hasattr(source_config, "num_local_experts"):
        tl_config.num_experts = source_config.num_local_experts
    if hasattr(source_config, "num_experts_per_tok"):
        tl_config.experts_per_token = source_config.num_experts_per_tok
    if hasattr(source_config, "sliding_window") and source_config.sliding_window is not None:
        tl_config.sliding_window = source_config.sliding_window
    if getattr(hf_config, "use_parallel_residual", False):
        tl_config.parallel_attn_mlp = True
    # GPT-J and CodeGen run parallel attn+MLP but don't set use_parallel_residual.
    arch_classes = getattr(hf_config, "architectures", []) or []
    if any(a in ("GPTJForCausalLM", "CodeGenForCausalLM") for a in arch_classes):
        tl_config.parallel_attn_mlp = True
    tl_config.default_prepend_bos = True
    return tl_config


def determine_architecture_from_hf_config(hf_config):
    """Determine the architecture name from HuggingFace config.

    Returns:
        str: The architecture name (e.g., "GPT2LMHeadModel", "LlamaForCausalLM")

    Raises:
        ValueError: If architecture cannot be determined
    """
    architectures = []
    if hasattr(hf_config, "original_architecture"):
        architectures.append(hf_config.original_architecture)
    if hasattr(hf_config, "architectures") and hf_config.architectures:
        architectures.extend(hf_config.architectures)
    if hasattr(hf_config, "model_type"):
        model_type = hf_config.model_type
        model_type_mappings = {
            "apertus": "ApertusForCausalLM",
            "gpt2": "GPT2LMHeadModel",
            "hubert": "HubertModel",
            "bart": "BartForConditionalGeneration",
            "llama": "LlamaForCausalLM",
            "llada": "LLaDAModelLM",
            "mamba": "MambaForCausalLM",
            "mamba2": "Mamba2ForCausalLM",
            "mistral": "MistralForCausalLM",
            "mixtral": "MixtralForCausalLM",
            "gemma": "GemmaForCausalLM",
            "gemma2": "Gemma2ForCausalLM",
            "gemma3": "Gemma3ForCausalLM",
            # gemma3n is tri-modal; the text path loads as the full ForConditionalGeneration
            # (vision/audio referenced but unbridged in the text-only adapter).
            "gemma3n": "Gemma3nForConditionalGeneration",
            # gemma4 is multimodal-only; all released checkpoints load as the full
            # ForConditionalGeneration (vision/audio referenced but unbridged).
            "gemma4": "Gemma4ForConditionalGeneration",
            "gemma4_unified": "Gemma4UnifiedForConditionalGeneration",
            "glm4_moe": "Glm4MoeForCausalLM",
            "glm_moe_dsa": "GlmMoeDsaForCausalLM",
            "t5gemma": "T5GemmaForConditionalGeneration",
            "t5gemma2": "T5Gemma2ForConditionalGeneration",
            "bert": "BertForMaskedLM",
            "bloom": "BloomForCausalLM",
            "codegen": "CodeGenForCausalLM",
            "cohere2": "Cohere2ForCausalLM",
            "gptj": "GPTJForCausalLM",
            "gpt_neo": "GPTNeoForCausalLM",
            "gpt_neox": "GPTNeoXForCausalLM",
            "opt": "OPTForCausalLM",
            "phi": "PhiForCausalLM",
            "phi3": "Phi3ForCausalLM",
            "qwen": "QwenForCausalLM",
            "qwen2": "Qwen2ForCausalLM",
            "qwen2_moe": "Qwen2MoeForCausalLM",
            "qwen3": "Qwen3ForCausalLM",
            # qwen3_5 is the top-level multimodal config type; qwen3_5_text is
            # the text-only sub-config. Both map to the text-only adapter so
            # Qwen3.5 checkpoints (which report qwen3_5 even when loaded as
            # text-only) are routed to Qwen3_5ForCausalLM.
            "qwen3_5": "Qwen3_5ForCausalLM",
            "qwen3_5_text": "Qwen3_5ForCausalLM",
            "smollm3": "SmolLM3ForCausalLM",
            "openelm": "OpenELMForCausalLM",
            "ouro": "OuroForCausalLM",
            "stablelm": "StableLmForCausalLM",
            "t5": "T5ForConditionalGeneration",
            "mt5": "MT5ForConditionalGeneration",
        }
        if model_type in model_type_mappings:
            architectures.append(model_type_mappings[model_type])

    for arch in architectures:
        if arch in SUPPORTED_ARCHITECTURES:
            return arch
    raise ValueError(
        f"Could not determine supported architecture from config. Available architectures: "
        f"{list(SUPPORTED_ARCHITECTURES.keys())}, Config architectures: {architectures}, "
        f"Model type: {getattr(hf_config, 'model_type', None)}"
    )


def setup_tokenizer(tokenizer, default_padding_side=None):
    """Normalize a HuggingFace tokenizer for use with the bridge.

    Args:
        tokenizer: A ``PreTrainedTokenizer`` or ``PreTrainedTokenizerFast``.
        default_padding_side: ``"right"`` or ``"left"``; sets ``tokenizer.padding_side``.
    """
    assert isinstance(
        tokenizer, PreTrainedTokenizerBase
    ), f"{type(tokenizer)} is not a supported tokenizer; use PreTrainedTokenizer or PreTrainedTokenizerFast"
    assert default_padding_side in [
        "right",
        "left",
        None,
    ], f"padding_side must be 'right', 'left' or None, got {default_padding_side}"
    tokenizer = get_tokenizer_with_bos(tokenizer)
    assert tokenizer is not None
    if default_padding_side is not None:
        tokenizer.padding_side = default_padding_side
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "right"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    # Some vocabularies lack default IDs for these tokens; register them.
    if tokenizer.pad_token is not None and tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.pad_token})
    if tokenizer.eos_token is not None and tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
    if tokenizer.bos_token is not None and tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.bos_token})

    return tokenizer
