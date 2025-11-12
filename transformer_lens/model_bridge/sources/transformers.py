"""Transformers module for TransformerLens.

This module provides functionality to load and convert models from HuggingFace to TransformerLens format.
"""

import copy
import logging
import os

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.supported_models import MODEL_ALIASES
from transformer_lens.utils import get_device, get_tokenizer_with_bos


def map_default_transformer_lens_config(hf_config):
    """Map HuggingFace config fields to TransformerLens config format.

    This function provides a standardized mapping from various HuggingFace config
    field names to the consistent TransformerLens naming convention.

    Args:
        hf_config: The HuggingFace config object

    Returns:
        A copy of hf_config with additional TransformerLens fields
    """
    tl_config = copy.deepcopy(hf_config)

    # Map d_model (hidden dimension)
    if hasattr(hf_config, "n_embd"):  # GPT2-style
        tl_config.d_model = hf_config.n_embd
    elif hasattr(hf_config, "hidden_size"):  # LLaMA/BERT-style
        tl_config.d_model = hf_config.hidden_size
    elif hasattr(hf_config, "d_model"):  # T5-style
        tl_config.d_model = hf_config.d_model

    # Map number of attention heads
    if hasattr(hf_config, "n_head"):  # GPT2-style
        tl_config.n_heads = hf_config.n_head
    elif hasattr(hf_config, "num_attention_heads"):  # LLaMA/BERT-style
        tl_config.n_heads = hf_config.num_attention_heads
    elif hasattr(hf_config, "num_heads"):  # T5-style
        tl_config.n_heads = hf_config.num_heads

    # Map number of key-value heads for GQA models
    if hasattr(hf_config, "num_key_value_heads") and hf_config.num_key_value_heads is not None:
        # Only set if different from n_heads (i.e., actually using GQA)
        # Convert to Python int/float to avoid Tensor boolean comparison issues
        try:
            # Handle both Tensor and numeric types
            num_kv_heads = hf_config.num_key_value_heads
            if hasattr(num_kv_heads, "item"):  # Tensor
                num_kv_heads = num_kv_heads.item()
            num_kv_heads = int(num_kv_heads)

            num_heads = tl_config.n_heads
            if hasattr(num_heads, "item"):  # Tensor
                num_heads = num_heads.item()
            num_heads = int(num_heads)

            if num_kv_heads != num_heads:
                tl_config.n_key_value_heads = num_kv_heads
        except (TypeError, ValueError, AttributeError):
            # If conversion fails, skip GQA mapping
            pass

    # Map number of layers
    if hasattr(hf_config, "n_layer"):  # GPT2-style
        tl_config.n_layers = hf_config.n_layer
    elif hasattr(hf_config, "num_hidden_layers"):  # LLaMA/BERT-style
        tl_config.n_layers = hf_config.num_hidden_layers
    elif hasattr(hf_config, "num_layers"):  # T5-style
        tl_config.n_layers = hf_config.num_layers

    # Map vocabulary size
    if hasattr(hf_config, "vocab_size"):
        tl_config.d_vocab = hf_config.vocab_size

    # Map context length
    if hasattr(hf_config, "n_positions"):  # GPT2-style
        tl_config.n_ctx = hf_config.n_positions
    elif hasattr(hf_config, "max_position_embeddings"):  # LLaMA/BERT-style
        tl_config.n_ctx = hf_config.max_position_embeddings
    elif hasattr(hf_config, "max_length"):  # Some models
        tl_config.n_ctx = hf_config.max_length

    # Map MLP dimension
    if hasattr(hf_config, "n_inner"):  # GPT2 explicit
        tl_config.d_mlp = hf_config.n_inner
    elif hasattr(hf_config, "intermediate_size"):  # BERT/LLaMA-style
        tl_config.d_mlp = hf_config.intermediate_size
    elif hasattr(tl_config, "d_model"):  # Default to 4x for GPT2-style
        tl_config.d_mlp = getattr(hf_config, "n_inner", 4 * tl_config.d_model)

    # Map head dimension (prefer calculation from d_model // n_heads over explicit head_dim)
    # This ensures correctness for models with incorrect head_dim in HF config (e.g., Gemma-3)
    if hasattr(tl_config, "d_model") and hasattr(tl_config, "n_heads"):
        # Calculate d_head from d_model and n_heads
        tl_config.d_head = tl_config.d_model // tl_config.n_heads
    elif hasattr(hf_config, "head_dim") and hf_config.head_dim is not None:
        # Fallback to explicit head_dim from HF config if calculation not possible
        tl_config.d_head = hf_config.head_dim

    # Set activation function
    if hasattr(hf_config, "activation_function"):
        tl_config.act_fn = hf_config.activation_function

    # Set number of experts
    if hasattr(hf_config, "num_local_experts"):
        tl_config.num_experts = hf_config.num_local_experts

    # Set number of experts per token
    if hasattr(hf_config, "num_experts_per_tok"):
        tl_config.experts_per_token = hf_config.num_experts_per_tok

    # Set sliding window size for models with local attention
    if hasattr(hf_config, "sliding_window") and hf_config.sliding_window is not None:
        tl_config.sliding_window = hf_config.sliding_window

    # Set common defaults for transformer models
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
    # Try to get architecture from various config fields
    architectures = []

    # Check original_architecture first (highest priority)
    if hasattr(hf_config, "original_architecture"):
        architectures.append(hf_config.original_architecture)

    # Check architectures list
    if hasattr(hf_config, "architectures") and hf_config.architectures:
        architectures.extend(hf_config.architectures)

    # Try to map model_type to architecture name
    if hasattr(hf_config, "model_type"):
        model_type = hf_config.model_type

        # Common mappings from model_type to architecture
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
            "t5": "T5ForConditionalGeneration",
        }

        if model_type in model_type_mappings:
            architectures.append(model_type_mappings[model_type])

    # Import here to avoid circular import
    from transformer_lens.factories.architecture_adapter_factory import (
        SUPPORTED_ARCHITECTURES,
    )

    # Return the first supported architecture
    for arch in architectures:
        if arch in SUPPORTED_ARCHITECTURES:
            return arch

    raise ValueError(
        f"Could not determine supported architecture from config. "
        f"Available architectures: {list(SUPPORTED_ARCHITECTURES.keys())}, "
        f"Config architectures: {architectures}, "
        f"Model type: {getattr(hf_config, 'model_type', None)}"
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
    # Encoder-decoder models (seq2seq)
    seq2seq_architectures = {
        "T5ForConditionalGeneration",
        "BartForConditionalGeneration",
        "MBartForConditionalGeneration",
        "MarianMTModel",
        "PegasusForConditionalGeneration",
        "BlenderbotForConditionalGeneration",
        "BlenderbotSmallForConditionalGeneration",
    }

    # Masked language models (BERT-style)
    masked_lm_architectures = {
        "BertForMaskedLM",
        "RobertaForMaskedLM",
        "DistilBertForMaskedLM",
        "AlbertForMaskedLM",
        "ElectraForMaskedLM",
    }

    # Causal language models (GPT-style) - this is the default
    # Includes: GPT2, LLaMA, Mistral, Mixtral, Gemma, etc.

    if architecture in seq2seq_architectures:
        return AutoModelForSeq2SeqLM
    elif architecture in masked_lm_architectures:
        # For now, use AutoModel for masked LM since they may need special handling
        return AutoModel
    else:
        # Default to causal LM for GPT-style models
        return AutoModelForCausalLM


def boot(
    model_name: str,
    hf_config_overrides: dict | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> TransformerBridge:
    """Boot a model from HuggingFace.

    Args:
        model_name: The name of the model to load.
        hf_config_overrides: Optional overrides applied to the HuggingFace config before model load.
        device: The device to use. If None, will be determined automatically.
        dtype: The dtype to use for the model.
        tokenizer: Optional pre-initialized tokenizer to use; if not provided one will be created.

    Returns:
        The bridge to the loaded model.
    """
    # Lazy import to avoid circular import
    from transformer_lens.factories.architecture_adapter_factory import (
        ArchitectureAdapterFactory,
    )

    # MODEL_ALIASES is a dict of {official_name: [alias1, alias2, ...]}
    # Check if model_name that the user passed is an alias, and if so, use the official name
    for official_name, aliases in MODEL_ALIASES.items():
        if model_name in aliases:
            logging.warning(
                f"DEPRECATED: You are using a deprecated, model_name alias '{model_name}'. TransformerLens will now load the official transformers model name, '{official_name}' instead.\n Please update your code to use the official name by changing model_name from '{model_name}' to '{official_name}'.\nSince TransformerLens v3, all model names should be the official transformers model names.\nThe aliases will be removed in the next version of TransformerLens, so please do the update now."
            )
            model_name = official_name
            break

    hf_config = AutoConfig.from_pretrained(model_name, output_attentions=True)

    # Apply config variables to hf_config before selecting adapter
    if hf_config_overrides:
        hf_config.__dict__.update(hf_config_overrides)

    # Apply HuggingFace to TransformerLens config mapping
    tl_config = map_default_transformer_lens_config(hf_config)

    architecture = determine_architecture_from_hf_config(hf_config)

    # Convert to TransformerBridgeConfig with unified architecture
    bridge_config = TransformerBridgeConfig.from_dict(tl_config.__dict__)
    bridge_config.architecture = architecture
    bridge_config.model_name = model_name  # Set the actual model name instead of default "custom"
    bridge_config.dtype = dtype  # Set the dtype from the boot parameter

    adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)

    # No device specified by user, use the best available device for the current system
    if device is None:
        device = get_device()

    # Add device information to the config
    adapter.cfg.device = str(device)

    # Determine the correct HuggingFace model class based on architecture
    model_class = get_hf_model_class_for_architecture(architecture)

    # Prepare model loading kwargs
    model_kwargs = {
        "config": hf_config,
        "torch_dtype": dtype,
    }

    # Add attn_implementation if specified in config
    # This allows architectures to specify their preferred attention implementation
    # (e.g., 'sdpa' for Scaled Dot Product Attention, 'eager' for basic implementation)
    if hasattr(adapter.cfg, "attn_implementation") and adapter.cfg.attn_implementation is not None:
        model_kwargs["attn_implementation"] = adapter.cfg.attn_implementation

    # Load the model from HuggingFace using the appropriate model class
    hf_model = model_class.from_pretrained(model_name, **model_kwargs)

    # Move model to device
    if device is not None:
        hf_model = hf_model.to(device)  # type: ignore

    # Load the tokenizer
    tokenizer = tokenizer
    default_padding_side = getattr(adapter.cfg, "default_padding_side", None)
    use_fast = getattr(adapter.cfg, "use_fast", True)

    if tokenizer is not None:
        tokenizer = setup_tokenizer(tokenizer, default_padding_side=default_padding_side)
    else:
        huggingface_token = os.environ.get("HF_TOKEN", "")
        tokenizer = setup_tokenizer(
            AutoTokenizer.from_pretrained(
                model_name,
                add_bos_token=True,
                use_fast=use_fast,
                token=huggingface_token if len(huggingface_token) > 0 else None,
            ),
            default_padding_side=default_padding_side,
        )

    # Set tokenizer_prepends_bos configuration
    if tokenizer is not None:
        adapter.cfg.tokenizer_prepends_bos = len(tokenizer.encode("")) > 0

    # Create the bridge
    bridge = TransformerBridge(
        hf_model,
        adapter,
        tokenizer,
    )

    # Note: enable_compatibility_mode() is available to apply folding if needed
    # This should be called manually when folding is desired

    return bridge


def setup_tokenizer(
    tokenizer,
    default_padding_side=None,
):
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

    # Use a tokenizer that is initialized with add_bos_token=True as the default tokenizer.
    # Such a tokenizer should be set as the default tokenizer because the tokenization of some
    # tokenizers like LlamaTokenizer are different when bos token is automatically/manually
    # prepended, and add_bos_token cannot be dynamically controlled after initialization
    # (https://github.com/huggingface/transformers/issues/25886).
    tokenizer_with_bos = get_tokenizer_with_bos(tokenizer)
    tokenizer = tokenizer_with_bos
    assert tokenizer is not None  # keep mypy happy

    # If user passes default_padding_side explicitly, use that value
    if default_padding_side is not None:
        tokenizer.padding_side = default_padding_side
    # If not, then use the tokenizer's default padding side
    # If the tokenizer doesn't have a default padding side, use the global default "right"
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "right"

    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    return tokenizer


setattr(TransformerBridge, "boot_transformers", staticmethod(boot))
