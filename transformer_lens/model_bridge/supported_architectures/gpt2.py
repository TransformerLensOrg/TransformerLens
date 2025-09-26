"""GPT2 architecture adapter."""

from typing import Any

import torch

from transformer_lens.conversion_utils.conversion_steps import (
    BaseHookConversion,
    HookConversionSet,
    RearrangeHookConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    JointQKVAttentionBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    UnembeddingBridge,
)


class QKVSplitRearrangeConversion(BaseHookConversion):
    """Custom conversion that splits QKV tensor and then rearranges."""

    def __init__(self, qkv_index: int, rearrange_pattern: str, **axes_lengths):
        """Initialize the conversion.

        Args:
            qkv_index: Index of Q (0), K (1), or V (2) in the QKV tensor
            rearrange_pattern: Einops pattern for rearrangement
            **axes_lengths: Additional axes lengths for einops
        """
        super().__init__()
        self.qkv_index = qkv_index
        self.rearrange_pattern = rearrange_pattern
        self.axes_lengths = axes_lengths

    def handle_conversion(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        """Split QKV tensor and rearrange the selected part."""
        # Determine the split dimension based on tensor shape
        if len(input_value.shape) == 2:
            # Weight tensor: [d_model, 3*d_model] -> split along dim=1
            split_dim = 1
        elif len(input_value.shape) == 1:
            # Bias tensor: [3*n_heads*d_head] -> split along dim=0
            split_dim = 0
        else:
            raise ValueError(f"Unexpected tensor shape: {input_value.shape}")

        # Split the QKV tensor
        qkv_parts = torch.chunk(input_value, 3, dim=split_dim)
        selected_part = qkv_parts[self.qkv_index]

        # Apply rearrangement
        import einops

        return einops.rearrange(selected_part, self.rearrange_pattern, **self.axes_lengths)

    def revert(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        """Revert the conversion by reconstructing the QKV tensor from Q, K, V components."""
        # This method expects to be called with all three QKV components available
        # in the full_context or needs to be coordinated with other conversions

        # For now, reverse the rearrangement first
        import einops

        # Reverse the rearrange operation
        left, right = self.rearrange_pattern.split("->")
        reverse_pattern = f"{right.strip()} -> {left.strip()}"
        reversed_tensor = einops.rearrange(input_value, reverse_pattern, **self.axes_lengths)

        # Note: The full QKV reconstruction needs to be handled at a higher level
        # where all Q, K, V components are available together
        return reversed_tensor

    def __repr__(self):
        return f'QKVSplitRearrangeConversion(qkv_index={self.qkv_index}, pattern="{self.rearrange_pattern}")'


class QKVBiasConversion(BaseHookConversion):
    """Custom conversion for QKV biases that matches the original GPT-2 logic."""

    def __init__(self, qkv_index: int, n_heads: int, d_head: int):
        """Initialize the conversion.

        Args:
            qkv_index: Index of Q (0), K (1), or V (2) in the QKV tensor
            n_heads: Number of attention heads
            d_head: Dimension of each head
        """
        super().__init__()
        self.qkv_index = qkv_index
        self.n_heads = n_heads
        self.d_head = d_head

    def handle_conversion(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        """Convert QKV bias following the original GPT-2 logic."""
        import einops

        # Original logic: rearrange the entire bias tensor first, then split by QKV
        qkv_bias = einops.rearrange(
            input_value,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=self.n_heads,
            head=self.d_head,
        )
        # Return the selected QKV part
        return qkv_bias[self.qkv_index]

    def revert(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        """Revert the conversion (not fully implemented for QKV case)."""
        # This is complex for QKV case since we need to reconstruct the full tensor
        # For now, just return the input
        return input_value

    def __repr__(self):
        return f"QKVBiasConversion(qkv_index={self.qkv_index}, n_heads={self.n_heads}, d_head={self.d_head})"


class GPT2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GPT2 models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the GPT2 architecture adapter."""
        super().__init__(cfg)

        # Set default config for GPT2 models
        self.default_cfg = {
            "default_prepend_bos": True,  # Default for GPT-2 style models
            "uses_split_attention": True,  # GPT-2 uses combined QKV attention that needs splitting
        }

        # GPT-2 uses combined QKV weights in HuggingFace format
        self.uses_combined_qkv = True

        # Set config variable to indicate that attention weights are split (use TransformerLens format processing)
        self.cfg.split_attention_weights = True

        self.conversion_rules = HookConversionSet(
            {
                # Original parameter names (for compatibility)
                "pos_embed.pos": "transformer.wpe.weight",
                "embed.e": "transformer.wte.weight",
                "blocks.{i}.ln1.weight": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.bias": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.attn.q.weight": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeHookConversion(
                        "(n h) m-> n m h",
                        n=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.k.weight": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeHookConversion(
                        "(n h) m-> n m h",
                        n=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.v.weight": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeHookConversion(
                        "(n h) m-> n m h",
                        n=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.o.weight": (
                    "transformer.h.{i}.attn.c_proj.weight",
                    RearrangeHookConversion("(n h) m -> n h m", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.q.bias": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeHookConversion("(n d_head) -> n d_head", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.k.bias": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeHookConversion("(n d_head) -> n d_head", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.v.bias": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeHookConversion("(n d_head) -> n d_head", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.o.bias": "transformer.h.{i}.attn.c_proj.bias",
                "blocks.{i}.ln2.weight": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.ln2.bias": "transformer.h.{i}.ln_2.bias",
                "blocks.{i}.mlp.input.weight": "transformer.h.{i}.mlp.c_fc.weight",
                "blocks.{i}.mlp.input.bias": "transformer.h.{i}.mlp.c_fc.bias",
                "blocks.{i}.mlp.out": "transformer.h.{i}.mlp.c_proj.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
                "ln_final.weight": "transformer.ln_f.weight",
                "ln_final.bias": "transformer.ln_f.bias",
                "unembed.weight": (
                    "lm_head.weight",
                    RearrangeHookConversion("d_model d_vocab -> d_vocab d_model"),
                ),
                "unembed.bias": "lm_head.bias",
                # TransformerLens parameter names (for weight processing functions)
                "embed.W_E": "transformer.wte.weight",
                "pos_embed.W_pos": "transformer.wpe.weight",
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    QKVSplitRearrangeConversion(
                        qkv_index=0,  # Q is the first part
                        rearrange_pattern="m (i h) -> i m h",
                        i=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    QKVSplitRearrangeConversion(
                        qkv_index=1,  # K is the second part
                        rearrange_pattern="m (i h) -> i m h",
                        i=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    QKVSplitRearrangeConversion(
                        qkv_index=2,  # V is the third part
                        rearrange_pattern="m (i h) -> i m h",
                        i=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.W_O": (
                    "transformer.h.{i}.attn.c_proj.weight",
                    RearrangeHookConversion("(i h) m -> i h m", i=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.b_Q": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    QKVBiasConversion(
                        qkv_index=0,  # Q bias is the first part
                        n_heads=self.cfg.n_heads,
                        d_head=self.cfg.d_head,
                    ),
                ),
                "blocks.{i}.attn.b_K": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    QKVBiasConversion(
                        qkv_index=1,  # K bias is the second part
                        n_heads=self.cfg.n_heads,
                        d_head=self.cfg.d_head,
                    ),
                ),
                "blocks.{i}.attn.b_V": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    QKVBiasConversion(
                        qkv_index=2,  # V bias is the third part
                        n_heads=self.cfg.n_heads,
                        d_head=self.cfg.d_head,
                    ),
                ),
                "blocks.{i}.attn.b_O": "transformer.h.{i}.attn.c_proj.bias",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.ln2.b": "transformer.h.{i}.ln_2.bias",
                "blocks.{i}.mlp.W_in": "transformer.h.{i}.mlp.c_fc.weight",
                "blocks.{i}.mlp.W_out": "transformer.h.{i}.mlp.c_proj.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.c_fc.bias",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
                "unembed.W_U": (
                    "lm_head.weight",
                    RearrangeHookConversion("d_model d_vocab -> d_vocab d_model"),
                ),
                "unembed.b_U": "lm_head.bias",
            }
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            "pos_embed": EmbeddingBridge(name="transformer.wpe"),
            "blocks": BlockBridge(
                name="transformer.h",
                config=self.cfg,
                submodules={
                    "ln1": self._create_normalization_bridge("ln_1"),
                    "attn": JointQKVAttentionBridge(
                        name="attn",
                        config=self.cfg,
                        split_qkv_matrix=self.split_qkv_matrix,
                        submodules={
                            "qkv": LinearBridge(name="c_attn"),
                            "o": LinearBridge(name="c_proj"),
                        },
                    ),
                    "ln2": self._create_normalization_bridge("ln_2"),
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "input": LinearBridge(name="c_fc"),
                            "out": LinearBridge(name="c_proj"),
                        },
                    ),
                },
            ),
            "ln_final": self._create_normalization_bridge("transformer.ln_f"),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def _create_normalization_bridge(self, name: str) -> NormalizationBridge:
        """Create a normalization bridge with appropriate identity settings based on configuration.

        Args:
            name: The name of the normalization component

        Returns:
            NormalizationBridge configured for the current layer norm folding setting
        """
        # Check if layer norm folding is enabled
        if hasattr(self.cfg, 'layer_norm_folding') and self.cfg.layer_norm_folding:
            # Use identity normalization when layer norm folding is enabled
            return NormalizationBridge(
                name=name,
                config=self.cfg,
                force_identity=True,  # Use normalized identity for folded weights
            )
        else:
            # Standard normalization when folding is disabled
            return NormalizationBridge(name=name, config=self.cfg)

    def split_qkv_matrix(
        self, original_attention_component: Any
    ) -> tuple[torch.nn.Linear, torch.nn.Linear, torch.nn.Linear]:
        """Split the QKV matrix into separate linear transformations.
        Args:
            attention_component: The original attention layer component
        Returns:
            Tuple of nn.Linear modules for Q, K, and V transformations
        """

        # Keep mypy happy
        assert original_attention_component is not None
        assert original_attention_component.c_attn is not None

        qkv_weights = original_attention_component.c_attn.weight

        # Keep mypy happy
        assert isinstance(qkv_weights, torch.Tensor)

        # Original qkv_weights shape: [d_model, 3 * d_model]
        # Split into three equal parts along dimension 1 to get Q, K, V weights
        W_Q, W_K, W_V = torch.tensor_split(qkv_weights, 3, dim=1)

        qkv_bias = original_attention_component.c_attn.bias

        # Keep mypy happy
        assert isinstance(qkv_bias, torch.Tensor)

        # Original qkv_bias shape: [3 * n_head * d_head]
        # Reshape to [3, n_head * d_head] to split by Q, K, V
        qkv_bias = qkv_bias.reshape(3, self.cfg.n_heads * self.cfg.d_head)
        b_Q, b_K, b_V = qkv_bias[0, :], qkv_bias[1, :], qkv_bias[2, :]

        # Create nn.Linear modules
        # After tensor_split, W_Q, W_K, W_V shapes are [d_model, d_model] ([in_features, out_features])
        # nn.Linear expects weight shape [out_features, in_features]
        # So we need to transpose the weights
        W_Q_transformation = torch.nn.Linear(W_Q.shape[0], W_Q.shape[1], bias=True)
        W_Q_transformation.weight = torch.nn.Parameter(W_Q.T)
        W_Q_transformation.bias = torch.nn.Parameter(b_Q)

        W_K_transformation = torch.nn.Linear(W_K.shape[0], W_K.shape[1], bias=True)
        W_K_transformation.weight = torch.nn.Parameter(W_K.T)
        W_K_transformation.bias = torch.nn.Parameter(b_K)

        W_V_transformation = torch.nn.Linear(W_V.shape[0], W_V.shape[1], bias=True)
        W_V_transformation.weight = torch.nn.Parameter(W_V.T)
        W_V_transformation.bias = torch.nn.Parameter(b_V)

        return W_Q_transformation, W_K_transformation, W_V_transformation

    def process_weights_and_create_components(self, state_dict, tl_cfg,
                                             fold_ln=True, center_writing_weights=True,
                                             center_unembed=True, fold_value_biases=True,
                                             refactor_factored_attn_matrices=False):
        """Process weights and create components directly with integrated layer norm folding.

        This method handles the complete weight processing pipeline for GPT-2 models.
        """
        print("GPT-2 adapter: Processing weights with integrated folding...")

        # Process weights first using our proven method
        from transformer_lens.loading_from_pretrained import fill_missing_keys
        from transformer_lens.weight_processing import ProcessWeights

        # Fill missing keys using a temporary minimal structure for compatibility
        temp_structure = self._create_minimal_structure_for_filling_keys(tl_cfg)
        complete_state_dict = fill_missing_keys(temp_structure, state_dict)

        # Process weights with exact same parameters
        processed_weights = ProcessWeights.process_weights(
            complete_state_dict,
            tl_cfg,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            fold_value_biases=fold_value_biases,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
        )

        # Create components directly based on the processing that was applied
        components_dict = self._create_folded_components_directly(tl_cfg, processed_weights, fold_ln)

        print(f"GPT-2 adapter: Created {len(components_dict['blocks'])} transformer blocks with integrated folding")
        return components_dict

    def _create_minimal_structure_for_filling_keys(self, tl_cfg):
        """Create minimal structure needed for fill_missing_keys compatibility."""
        import torch.nn as nn
        from transformer_lens.components import (
            Embed, LayerNorm, LayerNormPre, PosEmbed, RMSNorm, RMSNormPre,
            TransformerBlock, Unembed
        )

        # Create minimal structure that matches what fill_missing_keys expects
        temp_structure = nn.Module()
        temp_structure.cfg = tl_cfg

        # Create components without processing - just for fill_missing_keys
        temp_structure.embed = Embed(tl_cfg)

        if tl_cfg.positional_embedding_type != "rotary":
            temp_structure.pos_embed = PosEmbed(tl_cfg)

        temp_structure.blocks = nn.ModuleList([
            TransformerBlock(tl_cfg, block_index) for block_index in range(tl_cfg.n_layers)
        ])

        # Create ln_final based on original config (before folding)
        if tl_cfg.normalization_type in ["RMS", "RMSPre"]:
            temp_structure.ln_final = RMSNorm(tl_cfg) if tl_cfg.normalization_type == "RMS" else RMSNormPre(tl_cfg)
        elif tl_cfg.normalization_type in ["LN", "LNPre"]:
            if tl_cfg.final_rms:
                temp_structure.ln_final = RMSNorm(tl_cfg)
            else:
                temp_structure.ln_final = LayerNorm(tl_cfg) if tl_cfg.normalization_type == "LN" else LayerNormPre(tl_cfg)

        temp_structure.unembed = Unembed(tl_cfg)
        return temp_structure

    def _create_folded_components_directly(self, tl_cfg, processed_weights, fold_ln):
        """Create components directly with processed weights, respecting folding."""
        import torch.nn as nn
        from transformer_lens.components import (
            Embed, LayerNorm, LayerNormPre, PosEmbed, RMSNorm, RMSNormPre,
            TransformerBlock, Unembed
        )
        from transformer_lens.hook_points import HookPoint

        print("GPT-2 adapter: Creating components with folded configuration...")

        # Create embed component
        embed_component = Embed(tl_cfg)
        hook_embed = HookPoint()

        # Create pos_embed if needed
        pos_embed_component = None
        hook_pos_embed = None
        if tl_cfg.positional_embedding_type != "rotary":
            pos_embed_component = PosEmbed(tl_cfg)
            hook_pos_embed = HookPoint()

        # Create transformer blocks
        blocks = nn.ModuleList([
            TransformerBlock(tl_cfg, block_index) for block_index in range(tl_cfg.n_layers)
        ])

        # Create final layer norm - if folded, use LayerNormPre
        ln_final = None
        if fold_ln and tl_cfg.normalization_type == "LN":
            # When folded, LN becomes LNPre
            ln_final = LayerNormPre(tl_cfg)
        elif tl_cfg.normalization_type == "RMS":
            ln_final = RMSNorm(tl_cfg)
        elif tl_cfg.normalization_type == "RMSPre":
            ln_final = RMSNormPre(tl_cfg)
        elif tl_cfg.normalization_type == "LN":
            if tl_cfg.final_rms:
                ln_final = RMSNorm(tl_cfg)
            else:
                ln_final = LayerNorm(tl_cfg)
        elif tl_cfg.normalization_type == "LNPre":
            if tl_cfg.final_rms:
                ln_final = RMSNormPre(tl_cfg)
            else:
                ln_final = LayerNormPre(tl_cfg)

        # Create unembed
        unembed_component = Unembed(tl_cfg)

        # Load processed weights directly into components
        self._load_processed_weights_into_components(
            processed_weights, embed_component, pos_embed_component, blocks, ln_final, unembed_component
        )

        # Return components dictionary
        components_dict = {
            'embed': embed_component,
            'hook_embed': hook_embed,
            'blocks': blocks,
            'unembed': unembed_component,
        }

        if pos_embed_component is not None:
            components_dict['pos_embed'] = pos_embed_component
            components_dict['hook_pos_embed'] = hook_pos_embed

        if ln_final is not None:
            components_dict['ln_final'] = ln_final

        return components_dict

    def _load_processed_weights_into_components(self, processed_weights, embed_component,
                                              pos_embed_component, blocks, ln_final, unembed_component):
        """Load processed weights directly into components."""
        print("GPT-2 adapter: Loading processed weights into components...")

        # Load embed weights
        if 'embed.W_E' in processed_weights:
            embed_component.W_E.data = processed_weights['embed.W_E']

        # Load pos_embed weights
        if pos_embed_component is not None and 'pos_embed.W_pos' in processed_weights:
            pos_embed_component.W_pos.data = processed_weights['pos_embed.W_pos']

        # Load block weights
        for i, block in enumerate(blocks):
            prefix = f"blocks.{i}"

            # Attention weights
            if f"{prefix}.attn.W_Q" in processed_weights:
                block.attn.W_Q.data = processed_weights[f"{prefix}.attn.W_Q"]
            if f"{prefix}.attn.W_K" in processed_weights:
                block.attn.W_K.data = processed_weights[f"{prefix}.attn.W_K"]
            if f"{prefix}.attn.W_V" in processed_weights:
                block.attn.W_V.data = processed_weights[f"{prefix}.attn.W_V"]
            if f"{prefix}.attn.W_O" in processed_weights:
                block.attn.W_O.data = processed_weights[f"{prefix}.attn.W_O"]

            # Attention biases (if they exist)
            if hasattr(block.attn, 'b_Q') and f"{prefix}.attn.b_Q" in processed_weights:
                block.attn.b_Q.data = processed_weights[f"{prefix}.attn.b_Q"]
            if hasattr(block.attn, 'b_K') and f"{prefix}.attn.b_K" in processed_weights:
                block.attn.b_K.data = processed_weights[f"{prefix}.attn.b_K"]
            if hasattr(block.attn, 'b_V') and f"{prefix}.attn.b_V" in processed_weights:
                block.attn.b_V.data = processed_weights[f"{prefix}.attn.b_V"]
            if hasattr(block.attn, 'b_O') and f"{prefix}.attn.b_O" in processed_weights:
                block.attn.b_O.data = processed_weights[f"{prefix}.attn.b_O"]

            # MLP weights
            if f"{prefix}.mlp.W_in" in processed_weights:
                block.mlp.W_in.data = processed_weights[f"{prefix}.mlp.W_in"]
            if f"{prefix}.mlp.W_out" in processed_weights:
                block.mlp.W_out.data = processed_weights[f"{prefix}.mlp.W_out"]
            if hasattr(block.mlp, 'b_in') and f"{prefix}.mlp.b_in" in processed_weights:
                block.mlp.b_in.data = processed_weights[f"{prefix}.mlp.b_in"]
            if hasattr(block.mlp, 'b_out') and f"{prefix}.mlp.b_out" in processed_weights:
                block.mlp.b_out.data = processed_weights[f"{prefix}.mlp.b_out"]

        # Load final layer norm weights
        if ln_final is not None:
            if hasattr(ln_final, 'w') and 'ln_final.w' in processed_weights:
                ln_final.w.data = processed_weights['ln_final.w']
            if hasattr(ln_final, 'b') and 'ln_final.b' in processed_weights:
                ln_final.b.data = processed_weights['ln_final.b']

        # Load unembed weights
        if 'unembed.W_U' in processed_weights:
            unembed_component.W_U.data = processed_weights['unembed.W_U']
        if hasattr(unembed_component, 'b_U') and 'unembed.b_U' in processed_weights:
            unembed_component.b_U.data = processed_weights['unembed.b_U']

    def extract_hooks_from_components(self, components_dict, hook_registry):
        """Extract hooks from created components and populate the hook registry."""
        print("GPT-2 adapter: Extracting hooks from created components...")

        # Extract hooks from main components
        if 'hook_embed' in components_dict:
            hook_registry['hook_embed'] = components_dict['hook_embed']
        if 'hook_pos_embed' in components_dict:
            hook_registry['hook_pos_embed'] = components_dict['hook_pos_embed']

        # Extract hooks from all components using scan method
        # Note: This requires access to the bridge's _scan_existing_hooks method
        # For now, we'll return the components and let the bridge handle hook extraction

        print(f"GPT-2 adapter: Ready for hook extraction from components")
