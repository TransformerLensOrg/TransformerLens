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
from transformer_lens.utilities import get_device, get_tokenizer_with_bos

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
    # Extract language model config from text_config for multimodal models
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
    elif hasattr(tl_config, "d_model"):
        # Models without attention (e.g., Mamba SSMs) have no n_heads or head_dim.
        # Set d_head = d_model so TransformerLensConfig.__post_init__ computes
        # n_heads = 1. These values are nominal and have no functional meaning
        # for attention-less architectures.
        tl_config.d_head = tl_config.d_model
    if hasattr(source_config, "activation_function"):
        tl_config.act_fn = source_config.activation_function
    elif hasattr(source_config, "hidden_act"):
        tl_config.act_fn = source_config.hidden_act
    # Layer norm / RMS norm epsilon — HF uses 3 different field names
    if hasattr(source_config, "rms_norm_eps"):
        tl_config.eps = source_config.rms_norm_eps
    elif hasattr(source_config, "layer_norm_eps"):
        tl_config.eps = source_config.layer_norm_eps
    elif hasattr(source_config, "layer_norm_epsilon"):
        tl_config.eps = source_config.layer_norm_epsilon
    if hasattr(source_config, "num_local_experts"):
        tl_config.num_experts = source_config.num_local_experts
    if hasattr(source_config, "num_experts_per_tok"):
        tl_config.experts_per_token = source_config.num_experts_per_tok
    if hasattr(source_config, "sliding_window") and source_config.sliding_window is not None:
        tl_config.sliding_window = source_config.sliding_window
    if getattr(hf_config, "use_parallel_residual", False):
        tl_config.parallel_attn_mlp = True
    # GPT-J and CodeGen: parallel attn+MLP but missing use_parallel_residual in HF config
    arch_classes = getattr(hf_config, "architectures", []) or []
    if any(a in ("GPTJForCausalLM", "CodeGenForCausalLM") for a in arch_classes):
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
            "apertus": "ApertusForCausalLM",
            "gpt2": "GPT2LMHeadModel",
            "hubert": "HubertModel",
            "llama": "LlamaForCausalLM",
            "mamba": "MambaForCausalLM",
            "mamba2": "Mamba2ForCausalLM",
            "mistral": "MistralForCausalLM",
            "mixtral": "MixtralForCausalLM",
            "gemma": "GemmaForCausalLM",
            "gemma2": "Gemma2ForCausalLM",
            "gemma3": "Gemma3ForCausalLM",
            "bert": "BertForMaskedLM",
            "bloom": "BloomForCausalLM",
            "codegen": "CodeGenForCausalLM",
            "gptj": "GPTJForCausalLM",
            "gpt_neo": "GPTNeoForCausalLM",
            "gpt_neox": "GPTNeoXForCausalLM",
            "opt": "OPTForCausalLM",
            "phi": "PhiForCausalLM",
            "phi3": "Phi3ForCausalLM",
            "qwen": "QwenForCausalLM",
            "qwen2": "Qwen2ForCausalLM",
            "qwen3": "Qwen3ForCausalLM",
            # qwen3_5 is the top-level multimodal config type; qwen3_5_text is
            # the text-only sub-config. Both map to the text-only adapter so
            # Qwen3.5 checkpoints (which report qwen3_5 even when loaded as
            # text-only) are routed to Qwen3_5ForCausalLM.
            "qwen3_5": "Qwen3_5ForCausalLM",
            "qwen3_5_text": "Qwen3_5ForCausalLM",
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
    """Determine the correct HuggingFace AutoModel class for loading.

    Uses centralized architecture sets from utilities.architectures.
    """
    from transformer_lens.utilities.architectures import (
        AUDIO_ARCHITECTURES,
        MASKED_LM_ARCHITECTURES,
        MULTIMODAL_ARCHITECTURES,
        SEQ2SEQ_ARCHITECTURES,
    )

    if architecture in SEQ2SEQ_ARCHITECTURES:
        return AutoModelForSeq2SeqLM
    elif architecture in MASKED_LM_ARCHITECTURES:
        return AutoModelForMaskedLM
    elif architecture in MULTIMODAL_ARCHITECTURES:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    elif architecture in AUDIO_ARCHITECTURES:
        if "ForCTC" in architecture:
            from transformers import AutoModelForCTC

            return AutoModelForCTC
        from transformers import AutoModel

        return AutoModel
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
    hf_model: Any | None = None,
    n_ctx: int | None = None,
    # Experimental – Have not been fully tested on multi-gpu devices
    # Use at your own risk, report any issues here: https://github.com/TransformerLensOrg/TransformerLens/issues
    device_map: str | dict[str, str | int] | None = None,
    n_devices: int | None = None,
    max_memory: dict[str | int, str] | None = None,
) -> TransformerBridge:
    """Boot a model from HuggingFace.

    Args:
        model_name: The name of the model to load.
        hf_config_overrides: Optional overrides applied to the HuggingFace config before model load.
        device: The device to use. If None, will be determined automatically. Mutually exclusive
            with ``device_map``.
        dtype: The dtype to use for the model.
        tokenizer: Optional pre-initialized tokenizer to use; if not provided one will be created.
        load_weights: If False, load model without weights (on meta device) for config inspection only.
        model_class: Optional HuggingFace model class to use instead of the default auto-detected
            class. When the class name matches a key in SUPPORTED_ARCHITECTURES, the corresponding
            adapter is selected automatically (e.g., BertForNextSentencePrediction).
        hf_model: Optional pre-loaded HuggingFace model to use instead of loading one. Useful for
            models loaded with custom configurations (e.g., quantization via BitsAndBytesConfig).
            When provided, load_weights is ignored.
        device_map: HuggingFace-style device map (``"auto"``, ``"balanced"``, dict, etc.) for
            multi-GPU inference. Passed straight to ``from_pretrained``. Mutually exclusive
            with ``device``.
        n_devices: Convenience: split the model across this many CUDA devices (translated to a
            ``max_memory`` dict internally). Requires CUDA with at least this many visible devices.
        max_memory: Optional per-device memory budget for HF's dispatcher.
        n_ctx: Optional context length override. The bridge normally uses the model's documented
            max context from the HF config. Setting this writes to whichever HF field the model
            uses (n_positions / max_position_embeddings / etc.), so callers don't need to know
            the field name. If larger than the model's default, a warning is emitted — quality
            may degrade past the trained length for rotary models.

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
    from transformer_lens.utilities.hf_utils import get_hf_token

    _hf_token = get_hf_token()
    if hf_model is not None:
        # Reuse the pre-loaded model's config to avoid a Hub call when model_name
        # is a Hub repo ID, but the model is already loaded locally.
        hf_config = copy.deepcopy(hf_model.config)
    else:
        hf_config = AutoConfig.from_pretrained(
            model_name,
            output_attentions=True,
            trust_remote_code=trust_remote_code,
            token=_hf_token,
        )
    _n_ctx_field: str | None = None
    if n_ctx is not None:
        # Validation (#2): reject non-positive values before doing anything else.
        if n_ctx <= 0:
            raise ValueError(f"n_ctx must be a positive integer, got n_ctx={n_ctx}.")
        # Resolve n_ctx to whichever HF config field this model uses. Mirrors
        # the order in map_default_transformer_lens_config so the TL config
        # derivation picks up the override.
        for _field in (
            "n_positions",
            "max_position_embeddings",
            "max_context_length",
            "max_length",
            "seq_length",
        ):
            if hasattr(hf_config, _field):
                _n_ctx_field = _field
                break
        if _n_ctx_field is None:
            raise ValueError(
                f"Cannot apply n_ctx={n_ctx}: no recognized context-length field on "
                f"HF config for {model_name}. Use hf_config_overrides instead."
            )
        _default_n_ctx = getattr(hf_config, _n_ctx_field)
        if _default_n_ctx is not None and n_ctx > _default_n_ctx:
            logging.warning(
                "Setting n_ctx=%d which is larger than the model's default "
                "context length of %d. The model was not trained on sequences "
                "this long and may produce unreliable results (especially for "
                "rotary models without RoPE scaling).",
                n_ctx,
                _default_n_ctx,
            )
        # Conflict detection (#4): warn if the caller also set the same field
        # via hf_config_overrides — explicit n_ctx wins but users should know.
        if hf_config_overrides and _n_ctx_field in hf_config_overrides:
            _conflicting_value = hf_config_overrides[_n_ctx_field]
            if _conflicting_value != n_ctx:
                logging.warning(
                    "Both n_ctx=%d and hf_config_overrides['%s']=%s were provided. "
                    "The explicit n_ctx takes precedence.",
                    n_ctx,
                    _n_ctx_field,
                    _conflicting_value,
                )
        # Explicit n_ctx wins over hf_config_overrides for the resolved field.
        hf_config_overrides = dict(hf_config_overrides or {})
        hf_config_overrides[_n_ctx_field] = n_ctx
    if hf_config_overrides:
        hf_config.__dict__.update(hf_config_overrides)
    tl_config = map_default_transformer_lens_config(hf_config)
    architecture = determine_architecture_from_hf_config(hf_config)
    config_dict = dict(tl_config.__dict__)
    # Restore TL attribute names that HF remaps via attribute_map
    if "num_local_experts" in config_dict and "num_experts" not in config_dict:
        config_dict["num_experts"] = config_dict["num_local_experts"]
    bridge_config = TransformerBridgeConfig.from_dict(config_dict)
    bridge_config.architecture = architecture
    bridge_config.model_name = model_name
    bridge_config.dtype = dtype
    # Propagate HF-specific config attributes that adapters may need.
    # Any attribute present on the HF config and not None is copied to bridge_config.
    # This is architecture-agnostic — new architectures don't need changes here.
    _HF_PASSTHROUGH_ATTRS = [
        # OPT
        "is_gated_act",
        "word_embed_proj_dim",
        "do_layer_norm_before",
        # Granite
        "position_embedding_type",
        # Falcon
        "parallel_attn",
        "multi_query",
        "new_decoder_architecture",
        "alibi",
        "num_ln_in_parallel_attn",
        # Mamba (SSM config)
        "state_size",
        "conv_kernel",
        "expand",
        "time_step_rank",
        "intermediate_size",
        # Mamba-2 (additional SSM config)
        "n_groups",
        "chunk_size",
        # Multimodal
        "vision_config",
    ]
    for attr in _HF_PASSTHROUGH_ATTRS:
        val = getattr(hf_config, attr, None)
        if val is not None:
            setattr(bridge_config, attr, val)

    # Gemma2 softcapping: HF names differ from TL names, need explicit mapping
    final_logit_softcapping = getattr(hf_config, "final_logit_softcapping", None)
    if final_logit_softcapping is not None:
        bridge_config.output_logits_soft_cap = float(final_logit_softcapping)
    attn_logit_softcapping = getattr(hf_config, "attn_logit_softcapping", None)
    if attn_logit_softcapping is not None:
        bridge_config.attn_scores_soft_cap = float(attn_logit_softcapping)
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)
    # Pre-loaded models carry their own weight placement (possibly set by the caller via
    # device_map). Passing device_map / n_devices / max_memory alongside hf_model= is
    # ambiguous and would silently be ignored, so fail loudly.
    if hf_model is not None and (
        device_map is not None or n_devices is not None or max_memory is not None
    ):
        raise ValueError(
            "device_map / n_devices / max_memory are only supported when the bridge loads "
            "the HF model itself. When passing hf_model=..., apply device_map via "
            "AutoModel.from_pretrained before handing the model to the bridge."
        )
    # Stateful/SSM (e.g. Mamba) models keep a per-layer recurrent cache that must live on
    # that layer's device. The bridge currently allocates the stateful cache on a single
    # cfg.device, so cross-device splits would silently misplace the cache. Block this
    # combination until a v2 addresses per-layer stateful cache placement.
    if (n_devices is not None and n_devices > 1) or device_map is not None:
        if getattr(bridge_config, "is_stateful", False):
            raise ValueError(
                "Multi-device splits are not yet supported for stateful (SSM / Mamba) "
                "architectures: the stateful cache allocation is single-device. "
                "Load on one device, or wait for v2 support."
            )
    # Resolve device_map before defaulting `device` — the two are mutually exclusive, and
    # the resolver raises on conflict. If n_devices>1 is passed, it's translated into a
    # device_map + max_memory pair here so downstream code only needs to check the
    # resolved values.
    from transformer_lens.utilities.multi_gpu import (
        count_unique_devices,
        find_embedding_device,
        resolve_device_map,
    )

    resolved_device_map, resolved_max_memory = resolve_device_map(
        n_devices, device_map, device, max_memory
    )
    if resolved_device_map is None:
        if device is None:
            device = get_device()
        adapter.cfg.device = str(device)
    else:
        # cfg.device will be set from hf_device_map after the model is loaded.
        # Provisionally keep it None; find_embedding_device fills it in below.
        adapter.cfg.device = None
    if model_class is None:
        model_class = get_hf_model_class_for_architecture(architecture)
    # Ensure pad_token_id exists (v5 raises AttributeError if missing)
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
    if resolved_device_map is not None:
        model_kwargs["device_map"] = resolved_device_map
    if resolved_max_memory is not None:
        model_kwargs["max_memory"] = resolved_max_memory
    if hasattr(adapter.cfg, "attn_implementation") and adapter.cfg.attn_implementation is not None:
        model_kwargs["attn_implementation"] = adapter.cfg.attn_implementation
    else:
        # Default to eager (required for output_attentions hooks)
        model_kwargs["attn_implementation"] = "eager"
    adapter.prepare_loading(model_name, model_kwargs)
    if hf_model is not None:
        # Use the pre-loaded model as-is (e.g., quantized models with custom device_map)
        pass
    elif not load_weights:
        from_config_kwargs = {}
        if trust_remote_code:
            from_config_kwargs["trust_remote_code"] = True
        with contextlib.redirect_stdout(None):
            hf_model = model_class.from_config(hf_config, **from_config_kwargs)
    else:
        try:
            hf_model = model_class.from_pretrained(model_name, **model_kwargs)
        except RuntimeError as e:
            # #5: HF refuses to load when positional-weight shapes don't match.
            # If the user requested an n_ctx that conflicts with the saved weights
            # (common for learned-pos-embed models like GPT-2), re-raise with a
            # clearer message pointing them at the likely cause.
            if n_ctx is not None and "ignore_mismatched_sizes" in str(e):
                raise RuntimeError(
                    f"Failed to load {model_name} with n_ctx={n_ctx}: the pretrained "
                    f"weights' positional-embedding shape does not match the requested "
                    f"context length. This affects models with learned positional "
                    f"embeddings (e.g. GPT-2, OPT). Options: (1) use the model's "
                    f"default n_ctx, (2) pass load_weights=False if you only need "
                    f"config inspection, or (3) choose a rotary-embedding model "
                    f"(e.g. Llama, Mistral) which supports n_ctx changes without "
                    f"weight mismatch."
                ) from e
            raise
        # Skip explicit .to(device) when accelerate has placed weights via device_map.
        if resolved_device_map is None and device is not None:
            hf_model = hf_model.to(device)
        # Cast params to dtype; preserve float32 buffers (e.g., RotaryEmbedding.inv_freq)
        for param in hf_model.parameters():
            if param.is_floating_point() and param.dtype != dtype:
                param.data = param.data.to(dtype=dtype)
    # Derive cfg.device / cfg.n_devices from hf_device_map when present. This covers:
    #   - fresh loads with a resolved device_map (set above)
    #   - pre-loaded hf_model that the caller dispatched themselves (e.g., device_map="auto")
    hf_device_map_post = getattr(hf_model, "hf_device_map", None)
    if hf_device_map_post:
        # Pre-loaded path can still smuggle CPU/disk offload in; validate here too.
        offload_values = {str(v).lower() for v in hf_device_map_post.values() if isinstance(v, str)}
        forbidden = offload_values & {"cpu", "disk", "meta"}
        if forbidden and ((n_devices is not None and n_devices > 1) or device_map is not None):
            # Fresh-load path: we set the device_map ourselves, so this shouldn't happen —
            # but if the user asked for n_devices>1 and somehow got CPU offload, surface it.
            raise ValueError(
                f"hf_device_map contains unsupported offload targets: {sorted(forbidden)}. "
                "v1 multi-device support is GPU-only."
            )
    embedding_device = find_embedding_device(hf_model)
    if embedding_device is not None:
        adapter.cfg.device = str(embedding_device)
        adapter.cfg.n_devices = count_unique_devices(hf_model)
    elif adapter.cfg.device is None:
        # Pre-loaded single-device model with no hf_device_map — fall back to first param.
        try:
            adapter.cfg.device = str(next(hf_model.parameters()).device)
        except StopIteration:
            adapter.cfg.device = "cpu"
    # #7: Verify the n_ctx override actually took effect on the loaded model.
    # If HF's config class silently dropped or normalized the value, warn so
    # the user doesn't get misled into thinking longer sequences are supported.
    if n_ctx is not None and _n_ctx_field is not None and hf_model is not None:
        _actual = getattr(hf_model.config, _n_ctx_field, None)
        if _actual != n_ctx:
            logging.warning(
                "n_ctx=%d was requested but hf_model.config.%s=%s after load. "
                "The override may not have taken effect; the model may not "
                "accept sequences longer than %s.",
                n_ctx,
                _n_ctx_field,
                _actual,
                _actual,
            )
    adapter.prepare_model(hf_model)
    tokenizer = tokenizer
    default_padding_side = getattr(adapter.cfg, "default_padding_side", None)
    use_fast = getattr(adapter.cfg, "use_fast", True)
    # Audio models use feature extractors, not text tokenizers
    _is_audio = getattr(adapter.cfg, "is_audio_model", False)
    if _is_audio and tokenizer is None:
        tokenizer = None  # Skip tokenizer loading for audio models
    elif tokenizer is not None:
        tokenizer = setup_tokenizer(tokenizer, default_padding_side=default_padding_side)
    else:
        token_arg = get_hf_token()
        # Use adapter's tokenizer_name if model lacks one (e.g., OpenELM)
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
        # Detect BOS/EOS behavior (use non-empty string; empty is unreliable with token aliasing)
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
            # Some processors need torchvision (e.g., LlavaOnevision); install if needed
            _torchvision_available = False
            try:
                import torchvision  # noqa: F401

                _torchvision_available = True
            except Exception:
                # Install/reinstall torchvision if missing or broken
                import shutil
                import subprocess
                import sys

                try:
                    if shutil.which("uv"):
                        subprocess.check_call(
                            ["uv", "pip", "install", "torchvision", "-q"],
                        )
                    else:
                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", "torchvision", "-q"],
                        )
                    import importlib

                    importlib.invalidate_caches()
                    _torchvision_available = True
                except Exception:
                    pass  # torchvision install failed; processor will be unavailable

            if _torchvision_available:
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

    # Load feature extractor for audio models (needed for audio preprocessing)
    if getattr(adapter.cfg, "is_audio_model", False):
        try:
            from transformers import AutoFeatureExtractor

            huggingface_token = os.environ.get("HF_TOKEN", "")
            token_arg = huggingface_token if len(huggingface_token) > 0 else None
            bridge.processor = AutoFeatureExtractor.from_pretrained(
                model_name,
                token=token_arg,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            pass  # Feature extractor not available; user can set bridge.processor manually

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

    # Ensure special tokens resolve to valid IDs (some vocabularies lack defaults)
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
