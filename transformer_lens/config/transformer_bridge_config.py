"""Configuration class for TransformerBridge."""

import warnings
import weakref
from typing import Any, Optional

import numpy as np
import torch

from transformer_lens.utilities.activation_functions import SOFTCAP_DISABLED

from .transformer_lens_config import TransformerLensConfig


class TransformerBridgeConfig(TransformerLensConfig):
    """
    Configuration for TransformerBridge.

    This extends TransformerLensConfig with bridge-specific properties,
    particularly architecture information needed for adapter selection.
    Also includes all HookedTransformerConfig fields for compatibility.
    """

    __slots__ = ("_bridge_ref",)

    _BRIDGE_MANAGED_HOOK_FLAGS = frozenset(
        {
            "use_attn_result",
            "use_attn_in",
            "use_hook_mlp_in",
            "use_split_qkv_input",
        }
    )

    def __init__(
        self,
        d_model: int,
        d_head: int,
        n_layers: int,
        n_ctx: int,
        n_heads: int = -1,  # Add n_heads to signature so it's not filtered out by from_dict
        d_vocab: int = -1,
        architecture: Optional[str] = None,
        tokenizer_prepends_bos: bool = True,
        tokenizer_appends_eos: bool = False,
        default_padding_side: Optional[str] = None,
        # HookedTransformerConfig compatibility fields
        model_name: str = "custom",
        act_fn: str = "relu",
        eps: float = 1e-5,
        use_attn_scale: bool = True,
        attn_scale: float = -1.0,
        use_hook_mlp_in: bool = False,
        use_attn_in: bool = False,
        use_qk_norm: bool = False,
        use_local_attn: bool = False,
        original_architecture: Optional[str] = None,
        from_checkpoint: bool = False,
        checkpoint_index: Optional[int] = None,
        checkpoint_label_type: Optional[str] = None,
        checkpoint_value: Optional[int] = None,
        tokenizer_name: Optional[str] = None,
        window_size: Optional[int] = None,
        attn_types: Optional[list] = None,
        init_mode: str = "gpt2",
        normalization_type: Optional[str] = "LN",
        n_devices: int = 1,
        attention_dir: str = "causal",
        attn_only: bool = False,
        seed: Optional[int] = None,
        initializer_range: float = -1.0,
        init_weights: bool = True,
        scale_attn_by_inverse_layer_idx: bool = False,
        final_rms: bool = False,
        d_vocab_out: int = -1,
        parallel_attn_mlp: bool = False,
        rotary_dim: Optional[int] = None,
        n_params: Optional[int] = None,
        use_hook_tokens: bool = False,
        gated_mlp: bool = False,
        dtype: Optional[torch.dtype] = torch.float32,
        post_embedding_ln: bool = False,
        rotary_base: int | float = 10000,
        trust_remote_code: bool = False,
        rotary_adjacent_pairs: bool = False,
        load_in_4bit: bool = False,
        num_experts: Optional[int] = None,
        experts_per_token: Optional[int] = None,
        n_key_value_heads: Optional[int] = None,
        # Heterogeneous attention geometry (e.g. Gemma 4): per-layer values when
        # they vary across layers; d_head / n_key_value_heads then hold the
        # majority-layer scalar and attention math is delegated to HF.
        per_layer_head_dim: Optional[list] = None,
        per_layer_num_key_value_heads: Optional[list] = None,
        relative_attention_max_distance: Optional[int] = None,
        relative_attention_num_buckets: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        scale_embedding: Optional[bool] = None,
        tie_word_embeddings: bool = False,
        use_normalization_before_and_after: bool = False,
        attn_scores_soft_cap: float = SOFTCAP_DISABLED,
        output_logits_soft_cap: float = SOFTCAP_DISABLED,
        use_NTK_by_parts_rope: bool = False,
        NTK_by_parts_low_freq_factor: float = 1.0,
        NTK_by_parts_high_freq_factor: float = 4.0,
        NTK_by_parts_factor: float = 8.0,
        rmsnorm_uses_offset: bool = False,
        attn_implementation: Optional[str] = None,
        # Audio model configuration
        is_audio_model: bool = False,
        # Vision model (ViT, DeiT) configuration
        is_visual_model: bool = False,
        # Stateful model configuration (e.g., Mamba SSMs use cache_params,
        # not past_key_values, so generation delegates to hf_generate)
        is_stateful: bool = False,
        # Multimodal configuration
        is_multimodal: bool = False,
        vision_hidden_size: Optional[int] = None,
        vision_num_layers: Optional[int] = None,
        vision_num_heads: Optional[int] = None,
        mm_tokens_per_image: Optional[int] = None,
        **kwargs,
    ):
        """Initialize TransformerBridgeConfig."""
        object.__setattr__(self, "_bridge_ref", None)
        super().__init__(
            d_model=d_model,
            d_head=d_head,
            n_layers=n_layers,
            n_ctx=n_ctx,
            d_vocab=d_vocab,
            n_heads=n_heads,
            **kwargs,
        )

        # Architecture information for adapter selection
        self.architecture = architecture

        # Tokenizer configuration
        self.tokenizer_prepends_bos = tokenizer_prepends_bos
        self.tokenizer_appends_eos = tokenizer_appends_eos
        self.default_padding_side = default_padding_side

        # Attention weight processing configuration
        self.split_attention_weights = False

        # HookedTransformerConfig compatibility fields
        self.model_name = model_name
        self.act_fn = act_fn
        self.eps = eps
        self.use_attn_scale = use_attn_scale
        self.attn_scale = attn_scale
        self.use_hook_mlp_in = use_hook_mlp_in
        self.use_attn_in = use_attn_in
        self.use_qk_norm = use_qk_norm
        self.use_local_attn = use_local_attn
        self.original_architecture = original_architecture
        self.from_checkpoint = from_checkpoint
        self.checkpoint_index = checkpoint_index
        self.checkpoint_label_type = checkpoint_label_type
        self.checkpoint_value = checkpoint_value
        self.tokenizer_name = tokenizer_name
        self.window_size = window_size
        self.attn_types = attn_types
        self.init_mode = init_mode
        self.normalization_type = normalization_type
        self.n_devices = n_devices
        self.attention_dir = attention_dir
        self.attn_only = attn_only
        self.seed = seed
        self.initializer_range = initializer_range
        self.init_weights = init_weights
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.final_rms = final_rms
        self.d_vocab_out = d_vocab_out
        self.parallel_attn_mlp = parallel_attn_mlp
        self.rotary_dim = rotary_dim
        self.n_params = n_params
        self.use_hook_tokens = use_hook_tokens
        self.gated_mlp = gated_mlp
        self.dtype = dtype if dtype is not None else torch.float32
        self.post_embedding_ln = post_embedding_ln
        self.rotary_base = int(rotary_base)
        self.trust_remote_code = trust_remote_code
        self.rotary_adjacent_pairs = rotary_adjacent_pairs
        self.load_in_4bit = load_in_4bit
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.n_key_value_heads = n_key_value_heads
        self.per_layer_head_dim = per_layer_head_dim
        self.per_layer_num_key_value_heads = per_layer_num_key_value_heads
        self.relative_attention_max_distance = relative_attention_max_distance
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.decoder_start_token_id = decoder_start_token_id
        # Seq2seq families (Bart, Marian, Pegasus, ...) scale token embeddings
        # by sqrt(d_model) in the HF forward when set.
        self.scale_embedding = scale_embedding
        self.tie_word_embeddings = tie_word_embeddings
        self.use_normalization_before_and_after = use_normalization_before_and_after
        self.attn_scores_soft_cap = attn_scores_soft_cap
        self.output_logits_soft_cap = output_logits_soft_cap
        self.use_NTK_by_parts_rope = use_NTK_by_parts_rope
        self.NTK_by_parts_low_freq_factor = NTK_by_parts_low_freq_factor
        self.NTK_by_parts_high_freq_factor = NTK_by_parts_high_freq_factor
        self.NTK_by_parts_factor = NTK_by_parts_factor
        self.rmsnorm_uses_offset = rmsnorm_uses_offset
        self.attn_implementation = attn_implementation
        # Audio model configuration
        self.is_audio_model = is_audio_model
        # Vision model (ViT, DeiT) configuration
        self.is_visual_model = is_visual_model
        # Stateful model configuration
        self.is_stateful = is_stateful
        # Multimodal configuration
        self.is_multimodal = is_multimodal
        self.vision_hidden_size = vision_hidden_size
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.mm_tokens_per_image = mm_tokens_per_image
        self.__post_init__()

    def __setattr__(self, name: str, value: Any) -> None:
        """Route live Bridge hook-flag assignments through their public setters."""
        if name in self._BRIDGE_MANAGED_HOOK_FLAGS:
            bridge_ref = getattr(self, "_bridge_ref", None)
            bridge = bridge_ref() if bridge_ref is not None else None
            if bridge is not None:
                getattr(bridge, f"set_{name}")(value)
                return
        super().__setattr__(name, value)

    def __getstate__(self) -> dict[str, Any]:
        """Serialize config data without retaining its live Bridge binding."""
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore an unbound config copy."""
        self.__dict__.update(state)
        object.__setattr__(self, "_bridge_ref", None)

    def _bind_bridge(self, bridge: Any) -> None:
        """Bind runtime hook-flag assignments to a constructed Bridge."""
        bridge_ref = getattr(self, "_bridge_ref", None)
        bound_bridge = bridge_ref() if bridge_ref is not None else None
        if bound_bridge is None:
            object.__setattr__(self, "_bridge_ref", weakref.ref(bridge))
        elif bound_bridge is not bridge:
            warnings.warn(
                "TransformerBridgeConfig is already bound to another live "
                "TransformerBridge; declining to bind it to this instance. "
                "Direct assignments to Bridge-managed hook flags will continue "
                "to configure the existing TransformerBridge.",
                stacklevel=3,
            )

    def _set_bridge_managed_hook_flag(self, name: str, value: bool) -> None:
        """Set a managed flag without re-entering the Bridge setter."""
        if name not in self._BRIDGE_MANAGED_HOOK_FLAGS:
            raise ValueError(f"Unknown Bridge-managed hook flag: {name}")
        object.__setattr__(self, name, value)

    def __post_init__(self):
        """Post-initialization processing."""
        # dtype is guaranteed to be set at this point

        # Validate architecture if provided before calling super()
        if (
            hasattr(self, "architecture")
            and self.architecture is not None
            and not isinstance(self.architecture, str)
        ):
            raise ValueError(f"architecture must be a string, got {type(self.architecture)}")

        # Resolve the initializer_range sentinel (-1.0 means "not set by the user").
        # Mirrors HookedTransformerConfig.__post_init__ (hooked_transformer_config.py).
        # Guarded with getattr: this method also runs once from the dataclass
        # parent's __init__, before self.initializer_range is assigned below.
        if getattr(self, "initializer_range", None) is not None:
            if self.initializer_range < 0 and self.init_mode == "gpt2":
                # Roughly copy the GPT-2 value, but proportional to sqrt(1/d_model)
                self.initializer_range = 0.8 / np.sqrt(self.d_model)
            if self.initializer_range < 0 and self.init_mode != "gpt2":
                # This is the gain parameter for the weight initialisation
                self.initializer_range = 1.0

        # Call parent's __post_init__ after our validation
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    @property
    def head_dim(self) -> int:
        """Alias for d_head to match HuggingFace config naming convention."""
        return self.d_head
