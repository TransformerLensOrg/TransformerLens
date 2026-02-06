"""Pythia architecture adapter."""

from typing import Any

import torch

from transformer_lens.conversion_utils.conversion_steps import (
    RearrangeTensorConversion,
    SplitTensorConversion,
)
from transformer_lens.conversion_utils.conversion_steps.chain_tensor_conversion import (
    ChainTensorConversion,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    JointQKVPositionEmbeddingsAttentionBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class PythiaArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Pythia models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Pythia architecture adapter.

        Args:
            cfg: The configuration object.
        """
        super().__init__(cfg)
        self.cfg.positional_embedding_type = "rotary"
        # Pythia wasn't trained with BOS tokens, so match HuggingFace behavior
        self.cfg.default_prepend_bos = False

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q": ParamProcessingConversion(
                tensor_conversion=ChainTensorConversion(
                    [
                        SplitTensorConversion(0, 3),
                        RearrangeTensorConversion(
                            "(head d_head) d_model -> head d_model d_head",
                            head=self.cfg.n_heads,
                            d_head=self.cfg.d_model // self.cfg.n_heads,
                        ),
                    ]
                ),
                source_key="gpt_neox.layers.{i}.attention.query_key_value.weight",
            ),
            "blocks.{i}.attn.k": ParamProcessingConversion(
                tensor_conversion=ChainTensorConversion(
                    [
                        SplitTensorConversion(1, 3),
                        RearrangeTensorConversion(
                            "(head d_head) d_model -> head d_model d_head",
                            head=self.cfg.n_heads,
                            d_head=self.cfg.d_model // self.cfg.n_heads,
                        ),
                    ]
                ),
                source_key="gpt_neox.layers.{i}.attention.query_key_value.weight",
            ),
            "blocks.{i}.attn.v": ParamProcessingConversion(
                tensor_conversion=ChainTensorConversion(
                    [
                        SplitTensorConversion(2, 3),
                        RearrangeTensorConversion(
                            "(head d_head) d_model -> head d_model d_head",
                            head=self.cfg.n_heads,
                            d_head=self.cfg.d_model // self.cfg.n_heads,
                        ),
                    ]
                ),
                source_key="gpt_neox.layers.{i}.attention.query_key_value.weight",
            ),
            "blocks.{i}.attn.b_Q": ParamProcessingConversion(
                tensor_conversion=ChainTensorConversion(
                    [
                        SplitTensorConversion(0, 3),
                        RearrangeTensorConversion(
                            "(head d_head) -> head d_head",
                            head=self.cfg.n_heads,
                        ),
                    ]
                ),
                source_key="gpt_neox.layers.{i}.attention.query_key_value.bias",
            ),
            "blocks.{i}.attn.b_K": ParamProcessingConversion(
                tensor_conversion=ChainTensorConversion(
                    [
                        SplitTensorConversion(1, 3),
                        RearrangeTensorConversion(
                            "(head d_head) -> head d_head",
                            head=self.cfg.n_heads,
                        ),
                    ]
                ),
                source_key="gpt_neox.layers.{i}.attention.query_key_value.bias",
            ),
            "blocks.{i}.attn.b_V": ParamProcessingConversion(
                tensor_conversion=ChainTensorConversion(
                    [
                        SplitTensorConversion(2, 3),
                        RearrangeTensorConversion(
                            "(head d_head) -> head d_head",
                            head=self.cfg.n_heads,
                        ),
                    ]
                ),
                source_key="gpt_neox.layers.{i}.attention.query_key_value.bias",
            ),
            "blocks.{i}.attn.o": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (head d_head) -> head d_head d_model"
                ),
                source_key="gpt_neox.layers.{i}.attention.dense.weight",
            ),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="gpt_neox.embed_in"),
            "rotary_emb": RotaryEmbeddingBridge(name="gpt_neox.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="gpt_neox.layers",
                submodules={
                    "ln1": NormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": NormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": JointQKVPositionEmbeddingsAttentionBridge(
                        name="attention",
                        config=self.cfg,
                        split_qkv_matrix=self.split_qkv_matrix,
                        requires_attention_mask=True,  # GPTNeoX/Pythia requires attention_mask
                        submodules={
                            "qkv": LinearBridge(name="query_key_value"),
                            "o": LinearBridge(name="dense"),
                        },
                    ),
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "in": LinearBridge(name="dense_h_to_4h"),
                            "out": LinearBridge(name="dense_4h_to_h"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="gpt_neox.final_layer_norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="embed_out"),
        }

    def split_qkv_matrix(
        self, original_attention_component: Any
    ) -> tuple[torch.nn.Linear, torch.nn.Linear, torch.nn.Linear]:
        """Split the QKV matrix into separate linear transformations.

        GPT-NeoX/Pythia uses an interleaved QKV format where the weights are stored as
        [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...] - i.e., Q, K, V are interleaved per head.

        The weight shape is [n_heads * 3 * d_head, d_model] and the output is reshaped
        by HuggingFace as [batch, seq, n_heads, 3*d_head] then split on the last dim.

        Args:
            original_attention_component: The original attention layer component

        Returns:
            Tuple of nn.Linear modules for Q, K, and V transformations
        """
        assert original_attention_component is not None
        assert original_attention_component.query_key_value is not None

        qkv_weights = original_attention_component.query_key_value.weight
        assert isinstance(qkv_weights, torch.Tensor)

        n_heads = self.cfg.n_heads
        d_head = self.cfg.d_head
        d_model = self.cfg.d_model

        # Weight shape: [n_heads * 3 * d_head, d_model]
        # Reshape to [n_heads, 3 * d_head, d_model] to access Q, K, V per head
        W_reshaped = qkv_weights.view(n_heads, 3 * d_head, d_model)

        # Extract Q, K, V weights for all heads and flatten back
        W_Q = W_reshaped[:, :d_head, :].reshape(n_heads * d_head, d_model)
        W_K = W_reshaped[:, d_head : 2 * d_head, :].reshape(n_heads * d_head, d_model)
        W_V = W_reshaped[:, 2 * d_head :, :].reshape(n_heads * d_head, d_model)

        # Handle bias - same interleaved format
        qkv_bias = original_attention_component.query_key_value.bias
        assert isinstance(qkv_bias, torch.Tensor)

        # Bias shape: [n_heads * 3 * d_head]
        # Reshape to [n_heads, 3 * d_head] to access Q, K, V per head
        b_reshaped = qkv_bias.view(n_heads, 3 * d_head)
        b_Q = b_reshaped[:, :d_head].reshape(n_heads * d_head)
        b_K = b_reshaped[:, d_head : 2 * d_head].reshape(n_heads * d_head)
        b_V = b_reshaped[:, 2 * d_head :].reshape(n_heads * d_head)

        # Create nn.Linear modules
        # Weight shape for nn.Linear is [out_features, in_features]
        W_Q_transformation = torch.nn.Linear(d_model, n_heads * d_head, bias=True)
        W_Q_transformation.weight = torch.nn.Parameter(W_Q)
        W_Q_transformation.bias = torch.nn.Parameter(b_Q)

        W_K_transformation = torch.nn.Linear(d_model, n_heads * d_head, bias=True)
        W_K_transformation.weight = torch.nn.Parameter(W_K)
        W_K_transformation.bias = torch.nn.Parameter(b_K)

        W_V_transformation = torch.nn.Linear(d_model, n_heads * d_head, bias=True)
        W_V_transformation.weight = torch.nn.Parameter(W_V)
        W_V_transformation.bias = torch.nn.Parameter(b_V)

        return W_Q_transformation, W_K_transformation, W_V_transformation

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for Pythia component testing.

        Pythia uses RoPE (Rotary Position Embeddings) in the GPT-NeoX architecture.
        We need to set the rotary_emb reference on all attention bridge instances
        for component testing.

        Args:
            hf_model: The HuggingFace Pythia model instance
            bridge_model: The TransformerBridge model (if available, set rotary_emb on actual instances)
        """
        # Get rotary embedding instance from model level
        # In GPT-NeoX/Pythia, rotary_emb is at the model level
        rotary_emb = hf_model.gpt_neox.rotary_emb

        # Set rotary_emb on actual bridge instances in bridge_model if available
        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            # Set on each layer's actual attention bridge instance
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        # Also set on the template for get_generalized_component() calls
        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
