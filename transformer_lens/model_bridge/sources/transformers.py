"""Transformers module for TransformerLens.

This module provides functionality to load and convert models from HuggingFace to TransformerLens format.
"""
import contextlib
import copy
import logging
import os
import warnings
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.supported_models import MODEL_ALIASES
from transformer_lens.utils import get_device, get_tokenizer_with_bos

# Suppress transformers warnings that go to stderr
# This prevents notebook tests from failing due to unexpected stderr output
warnings.filterwarnings("ignore", message=".*generation flags.*not valid.*")
logging.getLogger("transformers").setLevel(logging.ERROR)


def map_default_transformer_lens_config(hf_config):
    """Map HuggingFace config fields to TransformerLens config format.

    This function provides a standardized mapping from various HuggingFace config
    field names to the consistent TransformerLens naming convention.

    For multimodal models (LLaVA, Gemma3ForConditionalGeneration), the language
    model dimensions are nested under text_config. We extract from text_config
    first, then apply the standard mapping.

    Args:
        hf_config: The HuggingFace config object

    Returns:
        A copy of hf_config with additional TransformerLens fields
    """
    # For multimodal models, extract language model config from text_config.
    # The text_config contains hidden_size, num_attention_heads, etc. that
    # TransformerLens needs for the language backbone.
    source_config = hf_config
    if hasattr(hf_config, "text_config") and hf_config.text_config is not None:
        source_config = hf_config.text_config

    tl_config = copy.deepcopy(hf_config)
    if hasattr(source_config, "n_embd"):
        tl_config.d_model = source_config.n_embd
    elif hasattr(source_config, "hidden_size"):
        tl_config.d_model = source_config.hidden_size
    elif hasattr(source_config, "model_dim"):
        tl_config.d_model = source_config.model_dim
    elif hasattr(source_config, "d_model"):
        tl_config.d_model = source_config.d_model
    if hasattr(source_config, "n_head"):
        tl_config.n_heads = source_config.n_head
    elif hasattr(source_config, "num_attention_heads"):
        n_heads = source_config.num_attention_heads
        if isinstance(n_heads, list):
            n_heads = max(n_heads)
        tl_config.n_heads = n_heads
    elif hasattr(source_config, "num_heads"):
        tl_config.n_heads = source_config.num_heads
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
            # Handle per-layer lists (e.g., OpenELM) by taking the max
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
    if hasattr(source_config, "n_layer"):
        tl_config.n_layers = source_config.n_layer
    elif hasattr(source_config, "num_hidden_layers"):
        tl_config.n_layers = source_config.num_hidden_layers
    elif hasattr(source_config, "num_transformer_layers"):
        tl_config.n_layers = source_config.num_transformer_layers
    elif hasattr(source_config, "num_layers"):
        tl_config.n_layers = source_config.num_layers
    if hasattr(source_config, "vocab_size"):
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
    else:
        # Models like Bloom use ALiBi (no positional embeddings) and have no
        # context length field. Default to 2048 as a reasonable fallback.
        tl_config.n_ctx = 2048
    if hasattr(source_config, "n_inner"):
        tl_config.d_mlp = source_config.n_inner
    elif hasattr(source_config, "intermediate_size"):
        tl_config.d_mlp = source_config.intermediate_size
    elif hasattr(tl_config, "d_model"):
        tl_config.d_mlp = getattr(source_config, "n_inner", 4 * tl_config.d_model)
    if hasattr(source_config, "head_dim") and source_config.head_dim is not None:
        tl_config.d_head = source_config.head_dim
    elif hasattr(tl_config, "d_model") and hasattr(tl_config, "n_heads"):
        tl_config.d_head = tl_config.d_model // tl_config.n_heads
    if hasattr(source_config, "activation_function"):
        tl_config.act_fn = source_config.activation_function
    if hasattr(source_config, "num_local_experts"):
        tl_config.num_experts = source_config.num_local_experts
    if hasattr(source_config, "num_experts_per_tok"):
        tl_config.experts_per_token = source_config.num_experts_per_tok
    if hasattr(source_config, "sliding_window") and source_config.sliding_window is not None:
        tl_config.sliding_window = source_config.sliding_window
    if getattr(hf_config, "use_parallel_residual", False):
        tl_config.parallel_attn_mlp = True
    # GPT-J has a parallel attention+MLP architecture (both read from same ln_1
    # output) but doesn't set use_parallel_residual in its HF config. Detect it
    # by architecture class so fold_ln correctly folds ln1 into BOTH attn and MLP.
    arch_classes = getattr(hf_config, "architectures", []) or []
    if any(a in ("GPTJForCausalLM",) for a in arch_classes):
        tl_config.parallel_attn_mlp = True
    tl_config.default_prepend_bos = True
    return tl_config


def determine_architecture_from_hf_config(hf_config):
    """Determine the architecture name from HuggingFace config.

    Args:
        hf_config: The HuggingFace config object

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
            "gpt2": "GPT2LMHeadModel",
            "llama": "LlamaForCausalLM",
            "mistral": "MistralForCausalLM",
            "mixtral": "MixtralForCausalLM",
            "gemma": "GemmaForCausalLM",
            "gemma2": "Gemma2ForCausalLM",
            "gemma3": "Gemma3ForCausalLM",
            "bert": "BertForMaskedLM",
            "bloom": "BloomForCausalLM",
            "gptj": "GPTJForCausalLM",
            "gpt_neo": "GPTNeoForCausalLM",
            "gpt_neox": "GPTNeoXForCausalLM",
            "opt": "OPTForCausalLM",
            "phi": "PhiForCausalLM",
            "phi3": "Phi3ForCausalLM",
            "qwen": "QwenForCausalLM",
            "qwen2": "Qwen2ForCausalLM",
            "qwen3": "Qwen3ForCausalLM",
            "openelm": "OpenELMForCausalLM",
            "stablelm": "StableLmForCausalLM",
            "t5": "T5ForConditionalGeneration",
        }
        if model_type in model_type_mappings:
            architectures.append(model_type_mappings[model_type])

    for arch in architectures:
        if arch in SUPPORTED_ARCHITECTURES:
            return arch
    raise ValueError(
        f"Could not determine supported architecture from config. Available architectures: {list(SUPPORTED_ARCHITECTURES.keys())}, Config architectures: {architectures}, Model type: {getattr(hf_config, 'model_type', None)}"
    )


def get_hf_model_class_for_architecture(architecture: str):
    """Determine the correct HuggingFace AutoModel class to use for loading.

    Args:
        architecture: The architecture name (e.g., "GPT2LMHeadModel", "T5ForConditionalGeneration")

    Returns:
        The appropriate HuggingFace AutoModel class to use

    Raises:
        ValueError: If architecture is not recognized
    """
    seq2seq_architectures = {
        "T5ForConditionalGeneration",
        "BartForConditionalGeneration",
        "MBartForConditionalGeneration",
        "MarianMTModel",
        "PegasusForConditionalGeneration",
        "BlenderbotForConditionalGeneration",
        "BlenderbotSmallForConditionalGeneration",
    }
    masked_lm_architectures = {
        "BertForMaskedLM",
        "RobertaForMaskedLM",
        "DistilBertForMaskedLM",
        "AlbertForMaskedLM",
        "ElectraForMaskedLM",
    }
    multimodal_architectures = {
        "LlavaForConditionalGeneration",
        "Gemma3ForConditionalGeneration",
    }
    if architecture in seq2seq_architectures:
        return AutoModelForSeq2SeqLM
    elif architecture in masked_lm_architectures:
        return AutoModelForMaskedLM
    elif architecture in multimodal_architectures:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    else:
        return AutoModelForCausalLM


def boot(
    model_name: str,
    hf_config_overrides: dict | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    tokenizer: PreTrainedTokenizerBase | None = None,
    load_weights: bool = True,
    trust_remote_code: bool = False,
    model_class: Any | None = None,
) -> TransformerBridge:
    """Boot a model from HuggingFace.

    Args:
        model_name: The name of the model to load.
        hf_config_overrides: Optional overrides applied to the HuggingFace config before model load.
        device: The device to use. If None, will be determined automatically.
        dtype: The dtype to use for the model.
        tokenizer: Optional pre-initialized tokenizer to use; if not provided one will be created.
        load_weights: If False, load model without weights (on meta device) for config inspection only.
        model_class: Optional HuggingFace model class to use instead of the default auto-detected
            class. When the class name matches a key in SUPPORTED_ARCHITECTURES, the corresponding
            adapter is selected automatically (e.g., BertForNextSentencePrediction).

    Returns:
        The bridge to the loaded model.
    """
    for official_name, aliases in MODEL_ALIASES.items():
        if model_name in aliases:
            logging.warning(
                f"DEPRECATED: You are using a deprecated, model_name alias '{model_name}'. TransformerLens will now load the official transformers model name, '{official_name}' instead.\n Please update your code to use the official name by changing model_name from '{model_name}' to '{official_name}'.\nSince TransformerLens v3, all model names should be the official transformers model names.\nThe aliases will be removed in the next version of TransformerLens, so please do the update now."
            )
            model_name = official_name
            break
    # Pass HF token for gated model access (e.g. meta-llama/*)
    _hf_token = os.environ.get("HF_TOKEN", "") or None
    hf_config = AutoConfig.from_pretrained(
        model_name,
        output_attentions=True,
        trust_remote_code=trust_remote_code,
        token=_hf_token,
    )
    if hf_config_overrides:
        hf_config.__dict__.update(hf_config_overrides)
    tl_config = map_default_transformer_lens_config(hf_config)
    architecture = determine_architecture_from_hf_config(hf_config)
    config_dict = dict(tl_config.__dict__)
    # HF configs may remap attribute names via attribute_map (e.g., MixtralConfig maps
    # `num_experts` -> `num_local_experts`). Explicitly restore the TL name so that
    # TransformerBridgeConfig.from_dict receives the expected key.
    if "num_local_experts" in config_dict and "num_experts" not in config_dict:
        config_dict["num_experts"] = config_dict["num_local_experts"]
    bridge_config = TransformerBridgeConfig.from_dict(config_dict)
    bridge_config.architecture = architecture
    bridge_config.model_name = model_name
    bridge_config.dtype = dtype
    # Preserve HF-specific config attributes that adapters may need
    if getattr(hf_config, "is_gated_act", False):
        bridge_config.is_gated_act = True
    # OPT-350m: word_embed_proj_dim != hidden_size means the model uses
    # project_in/project_out instead of final_layer_norm.
    word_embed_proj_dim = getattr(hf_config, "word_embed_proj_dim", None)
    if word_embed_proj_dim is not None:
        bridge_config.word_embed_proj_dim = word_embed_proj_dim
    # OPT post-norm breaks fold_ln assumptions (pre-norm only).
    do_layer_norm_before = getattr(hf_config, "do_layer_norm_before", None)
    if do_layer_norm_before is not None:
        bridge_config.do_layer_norm_before = do_layer_norm_before
    # Propagate Gemma2 logit/attn softcapping config from HF to TL fields.
    final_logit_softcapping = getattr(hf_config, "final_logit_softcapping", None)
    if final_logit_softcapping is not None:
        bridge_config.output_logits_soft_cap = float(final_logit_softcapping)
    attn_logit_softcapping = getattr(hf_config, "attn_logit_softcapping", None)
    if attn_logit_softcapping is not None:
        bridge_config.attn_scores_soft_cap = float(attn_logit_softcapping)
    # Propagate vision config for multimodal models so the adapter can
    # select the correct vision encoder bridge (CLIP vs SigLIP).
    if hasattr(hf_config, "vision_config") and hf_config.vision_config is not None:
        bridge_config.vision_config = hf_config.vision_config
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)
    if device is None:
        device = get_device()
    adapter.cfg.device = str(device)
    if model_class is None:
        model_class = get_hf_model_class_for_architecture(architecture)
    # Ensure pad_token_id exists on HF config. Transformers v5 raises AttributeError
    # for missing config attributes (instead of returning None), which crashes models
    # like Phi-1 that access config.pad_token_id during __init__.
    if not hasattr(hf_config, "pad_token_id") or "pad_token_id" not in hf_config.__dict__:
        fallback_pad = getattr(hf_config, "eos_token_id", None)
        # eos_token_id can be a list (e.g., Gemma3 uses [1, 106]); take the first.
        if isinstance(fallback_pad, list):
            fallback_pad = fallback_pad[0] if fallback_pad else None
        hf_config.pad_token_id = fallback_pad
    model_kwargs = {"config": hf_config, "torch_dtype": dtype}
    if _hf_token:
        model_kwargs["token"] = _hf_token
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if hasattr(adapter.cfg, "attn_implementation") and adapter.cfg.attn_implementation is not None:
        model_kwargs["attn_implementation"] = adapter.cfg.attn_implementation
    else:
        # Default to "eager" — the Bridge uses output_attentions for hooks,
        # which requires eager attention.  This also ensures numerical parity
        # with benchmarks that compare Bridge vs HF reference (both use eager).
        model_kwargs["attn_implementation"] = "eager"
    adapter.prepare_loading(model_name, model_kwargs)
    if not load_weights:
        from_config_kwargs = {}
        if trust_remote_code:
            from_config_kwargs["trust_remote_code"] = True
        with contextlib.redirect_stdout(None):
            hf_model = model_class.from_config(hf_config, **from_config_kwargs)
    else:
        hf_model = model_class.from_pretrained(model_name, **model_kwargs)
        if device is not None:
            hf_model = hf_model.to(device)
        # Ensure all parameters match the requested dtype. Some architectures
        # (e.g., MoE models) retain native bfloat16 weights even when
        # torch_dtype is specified during from_pretrained().
        hf_model = hf_model.to(dtype=dtype)
    adapter.prepare_model(hf_model)
    tokenizer = tokenizer
    default_padding_side = getattr(adapter.cfg, "default_padding_side", None)
    use_fast = getattr(adapter.cfg, "use_fast", True)
    if tokenizer is not None:
        tokenizer = setup_tokenizer(tokenizer, default_padding_side=default_padding_side)
    else:
        huggingface_token = os.environ.get("HF_TOKEN", "")
        token_arg = huggingface_token if len(huggingface_token) > 0 else None
        # Determine tokenizer source: use adapter's tokenizer_name if the model
        # doesn't ship its own tokenizer (e.g., OpenELM uses LLaMA tokenizer)
        tokenizer_source = model_name
        if hasattr(adapter.cfg, "tokenizer_name") and adapter.cfg.tokenizer_name is not None:
            tokenizer_source = adapter.cfg.tokenizer_name
        # Try to load tokenizer with add_bos_token=True first
        # (encoder-decoder models like T5 don't have BOS tokens and will raise ValueError)
        try:
            base_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                add_bos_token=True,
                use_fast=use_fast,
                token=token_arg,
                trust_remote_code=trust_remote_code,
            )
        except ValueError:
            # Model doesn't have a BOS token, load without add_bos_token
            base_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                use_fast=use_fast,
                token=token_arg,
                trust_remote_code=trust_remote_code,
            )
        tokenizer = setup_tokenizer(
            base_tokenizer,
            default_padding_side=default_padding_side,
        )
    if tokenizer is not None:
        # Detect whether the tokenizer auto-prepends BOS or auto-appends EOS.
        # We encode a non-empty test string and check the first/last tokens.
        # Using encode("") is unreliable because setup_tokenizer may set
        # bos_token = eos_token, making them indistinguishable.
        encoded_test = tokenizer.encode("a")
        adapter.cfg.tokenizer_prepends_bos = (
            len(encoded_test) > 1
            and tokenizer.bos_token_id is not None
            and encoded_test[0] == tokenizer.bos_token_id
        )
        adapter.cfg.tokenizer_appends_eos = (
            len(encoded_test) > 1
            and tokenizer.eos_token_id is not None
            and encoded_test[-1] == tokenizer.eos_token_id
        )
    bridge = TransformerBridge(hf_model, adapter, tokenizer)

    # Load processor for multimodal models (needed for image preprocessing)
    if getattr(adapter.cfg, "is_multimodal", False):
        try:
            from transformers import AutoProcessor

            huggingface_token = os.environ.get("HF_TOKEN", "")
            token_arg = huggingface_token if len(huggingface_token) > 0 else None
            bridge.processor = AutoProcessor.from_pretrained(
                model_name,
                token=token_arg,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            pass  # Processor not available; user can set bridge.processor manually

    return bridge


def setup_tokenizer(tokenizer, default_padding_side=None):
    """Set's up the tokenizer.

    Args:
        tokenizer (PreTrainedTokenizer): a pretrained HuggingFace tokenizer.
        default_padding_side (str): "right" or "left", which side to pad on.

    """
    assert isinstance(
        tokenizer, PreTrainedTokenizerBase
    ), f"{type(tokenizer)} is not a supported tokenizer, please use PreTrainedTokenizer or PreTrainedTokenizerFast"
    assert default_padding_side in [
        "right",
        "left",
        None,
    ], f"padding_side must be 'right', 'left' or 'None', got {default_padding_side}"
    tokenizer_with_bos = get_tokenizer_with_bos(tokenizer)
    tokenizer = tokenizer_with_bos
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

    # Ensure special token strings resolve to valid IDs.  Some tokenizers
    # (e.g. ChemGPT's SMILES vocabulary) don't contain the default fallback
    # strings, leaving pad_token_id as None.  HF's padding logic then crashes
    # with "TypeError: '<' not supported between instances of 'NoneType' and 'int'".
    if tokenizer.pad_token is not None and tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.pad_token})
    if tokenizer.eos_token is not None and tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
    if tokenizer.bos_token is not None and tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.bos_token})

    return tokenizer


def list_supported_models(
    architecture: str | None = None,
    verified_only: bool = False,
) -> list[str]:
    """List all models supported by TransformerLens.

    This function provides convenient access to the model registry API
    for discovering which HuggingFace models can be loaded.

    Args:
        architecture: Filter by architecture ID (e.g., "GPT2LMHeadModel").
            If None, returns all supported models.
        verified_only: If True, only return models that have been verified
            to work with TransformerLens.

    Returns:
        List of model IDs (e.g., ["gpt2", "gpt2-medium", ...])

    Example:
        >>> from transformer_lens.model_bridge.sources.transformers import list_supported_models
        >>> models = list_supported_models()
        >>> gpt2_models = list_supported_models(architecture="GPT2LMHeadModel")
    """
    try:
        from transformer_lens.tools.model_registry import api

        models = api.get_supported_models(architecture=architecture, verified_only=verified_only)
        return [m.model_id for m in models]
    except ImportError:
        return []
    except Exception:
        return []


def check_model_support(model_id: str) -> dict:
    """Check if a model is supported and get detailed support info.

    This function provides detailed information about a model's compatibility
    with TransformerLens, including architecture type and verification status.

    Args:
        model_id: The HuggingFace model ID to check (e.g., "gpt2")

    Returns:
        Dictionary with support information:
        - is_supported: bool - Whether the model is supported
        - architecture_id: str | None - The architecture type if supported
        - verified: bool - Whether the model has been verified to work
        - suggestion: str | None - Suggested alternative if not supported

    Example:
        >>> from transformer_lens.model_bridge.sources.transformers import check_model_support  # doctest: +SKIP
        >>> info = check_model_support("openai-community/gpt2")  # doctest: +SKIP
        >>> info["is_supported"]  # doctest: +SKIP
        True
    """
    try:
        from transformer_lens.tools.model_registry import api

        is_supported = api.is_model_supported(model_id)

        if is_supported:
            model_info = api.get_model_info(model_id)
            return {
                "is_supported": True,
                "architecture_id": model_info.architecture_id,
                "status": model_info.status,
                "verified_date": (
                    model_info.verified_date.isoformat() if model_info.verified_date else None
                ),
                "suggestion": None,
            }
        else:
            suggestion = api.suggest_similar_model(model_id)
            return {
                "is_supported": False,
                "architecture_id": None,
                "verified": False,
                "verified_date": None,
                "suggestion": suggestion,
            }
    except ImportError:
        return {
            "is_supported": None,
            "architecture_id": None,
            "verified": False,
            "verified_date": None,
            "suggestion": None,
            "error": "Model registry not available",
        }
    except Exception as e:
        return {
            "is_supported": None,
            "architecture_id": None,
            "verified": False,
            "verified_date": None,
            "suggestion": None,
            "error": str(e),
        }


# Attach functions to TransformerBridge as static methods
setattr(TransformerBridge, "boot_transformers", staticmethod(boot))
setattr(TransformerBridge, "list_supported_models", staticmethod(list_supported_models))
setattr(TransformerBridge, "check_model_support", staticmethod(check_model_support))
