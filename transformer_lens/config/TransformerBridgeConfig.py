"""Configuration class for TransformerBridge."""

from typing import Optional

import torch

from .TransformerLensConfig import TransformerLensConfig


class TransformerBridgeConfig(TransformerLensConfig):
    """
    Configuration for TransformerBridge.

    This extends TransformerLensConfig with bridge-specific properties,
    particularly architecture information needed for adapter selection.
    Also includes all HookedTransformerConfig fields for compatibility.
    """

    def __init__(
        self,
        d_model: int,
        d_head: int,
        n_layers: int,
        n_ctx: int,
        architecture: Optional[str] = None,
        tokenizer_prepends_bos: bool = True,
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
        ungroup_grouped_query_attention: bool = False,
        original_architecture: Optional[str] = None,
        from_checkpoint: bool = False,
        checkpoint_index: Optional[int] = None,
        checkpoint_label_type: Optional[str] = None,
        checkpoint_value: Optional[int] = None,
        tokenizer_name: Optional[str] = None,
        window_size: Optional[int] = None,
        attn_types: Optional[list] = None,
        init_mode: str = "gpt2",
        normalization_type: str = "LN",
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
        rotary_base: int = 10000,
        trust_remote_code: bool = False,
        rotary_adjacent_pairs: bool = False,
        load_in_4bit: bool = False,
        num_experts: Optional[int] = None,
        experts_per_token: Optional[int] = None,
        relative_attention_max_distance: Optional[int] = None,
        relative_attention_num_buckets: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        tie_word_embeddings: bool = False,
        use_normalization_before_and_after: bool = False,
        attn_scores_soft_cap: float = -1.0,
        output_logits_soft_cap: float = -1.0,
        use_NTK_by_parts_rope: bool = False,
        NTK_by_parts_low_freq_factor: float = 1.0,
        NTK_by_parts_high_freq_factor: float = 4.0,
        NTK_by_parts_factor: float = 8.0,
        **kwargs,
    ):
        """Initialize TransformerBridgeConfig."""
        super().__init__(d_model=d_model, d_head=d_head, n_layers=n_layers, n_ctx=n_ctx, **kwargs)

        # Architecture information for adapter selection
        self.architecture = architecture

        # Tokenizer configuration
        self.tokenizer_prepends_bos = tokenizer_prepends_bos
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
        self.ungroup_grouped_query_attention = ungroup_grouped_query_attention
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
        self.rotary_base = rotary_base
        self.trust_remote_code = trust_remote_code
        self.rotary_adjacent_pairs = rotary_adjacent_pairs
        self.load_in_4bit = load_in_4bit
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.relative_attention_max_distance = relative_attention_max_distance
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.decoder_start_token_id = decoder_start_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.use_normalization_before_and_after = use_normalization_before_and_after
        self.attn_scores_soft_cap = attn_scores_soft_cap
        self.output_logits_soft_cap = output_logits_soft_cap
        self.use_NTK_by_parts_rope = use_NTK_by_parts_rope
        self.NTK_by_parts_low_freq_factor = NTK_by_parts_low_freq_factor
        self.NTK_by_parts_high_freq_factor = NTK_by_parts_high_freq_factor
        self.NTK_by_parts_factor = NTK_by_parts_factor

        self.__post_init__()

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

        # Call parent's __post_init__ after our validation
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
