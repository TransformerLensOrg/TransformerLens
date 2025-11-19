"""Bloom architecture adapter."""

from typing import Any

import torch

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
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


class BloomArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Bloom models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Bloom architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "alibi"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False

        self.cfg.default_prepend_bos = False
        self.weight_processing_conversions = {
            "embed.e": "transformer.word_embeddings.weight",
            "blocks.{i}.ln1.w": "transformer.h.{i}.input_layernorm.weight",
            "blocks.{i}.ln1.b": "transformer.h.{i}.input_layernorm.bias",
            "blocks.{i}.attn.q": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(three n h) m -> three n m h",
                    three=3,
                    n=self.cfg.n_heads,
                ),
                source_key="transformer.h.{i}.self_attention.query_key_value.weight",
            ),
            "blocks.{i}.attn.k": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(three n h) m -> three n m h",
                    three=3,
                    n=self.cfg.n_heads,
                ),
                source_key="transformer.h.{i}.self_attention.query_key_value.weight",
            ),
            "blocks.{i}.attn.v": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(three n h) m -> three n m h",
                    three=3,
                    n=self.cfg.n_heads,
                ),
                source_key="transformer.h.{i}.self_attention.query_key_value.weight",
            ),
            "blocks.{i}.attn.o": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                source_key="transformer.h.{i}.self_attention.dense.weight",
            ),
            "blocks.{i}.attn.b_Q": "transformer.h.{i}.self_attention.query_key_value.bias",
            "blocks.{i}.attn.b_K": "transformer.h.{i}.self_attention.query_key_value.bias",
            "blocks.{i}.attn.b_V": "transformer.h.{i}.self_attention.query_key_value.bias",
            "blocks.{i}.attn.b_O": "transformer.h.{i}.self_attention.dense.bias",
            "blocks.{i}.ln2.w": "transformer.h.{i}.post_attention_layernorm.weight",
            "blocks.{i}.ln2.b": "transformer.h.{i}.post_attention_layernorm.bias",
            "blocks.{i}.mlp.in": "transformer.h.{i}.mlp.dense_h_to_4h.weight",
            "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.dense_h_to_4h.bias",
            "blocks.{i}.mlp.out": "transformer.h.{i}.mlp.dense_4h_to_h.weight",
            "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.dense_4h_to_h.bias",
            "ln_final.w": "transformer.ln_f.weight",
            "ln_final.b": "transformer.ln_f.bias",
            "unembed.u": "lm_head.weight",
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.word_embeddings"),
            "embed_ln": NormalizationBridge(
                name="transformer.word_embeddings_layernorm", config=self.cfg
            ),
            "blocks": BlockBridge(
                name="transformer.h",
                submodules={
                    "ln1": NormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": NormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": JointQKVAttentionBridge(
                        name="self_attention",
                        config=self.cfg,
                        split_qkv_matrix=self.split_qkv_matrix,
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
            "ln_final": NormalizationBridge(name="transformer.ln_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

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
        assert original_attention_component.query_key_value is not None

        qkv_weights = original_attention_component.query_key_value.weight

        # Keep mypy happy
        assert isinstance(qkv_weights, torch.Tensor)

        # We want to split weights into [d_model, n_heads * d_head] for each of Q, K, V
        W_split = qkv_weights.T.reshape(self.cfg.d_model, 3, self.cfg.n_heads * self.cfg.d_head)

        W_Q, W_K, W_V = W_split[:, 0, :], W_split[:, 1, :], W_split[:, 2, :]

        qkv_bias = original_attention_component.query_key_value.bias

        # Keep mypy happy
        assert isinstance(qkv_bias, torch.Tensor)

        # Reshape to [3, n_heads * d_head] to split by Q, K, V
        qkv_bias = qkv_bias.reshape(3, self.cfg.n_heads * self.cfg.d_head)

        b_Q, b_K, b_V = qkv_bias[0, :], qkv_bias[1, :], qkv_bias[2, :]

        # Create nn.Linear modules
        # W_Q, W_K, W_V shapes are [d_model, n_heads * d_head]
        # nn.Linear expects weight shape [out_features, in_features]
        # So for Linear(d_model, n_heads * d_head), weight should be [n_heads * d_head, d_model]
        W_Q_transformation = torch.nn.Linear(W_Q.shape[0], W_Q.shape[1], bias=True)
        W_Q_transformation.weight = torch.nn.Parameter(
            W_Q.T
        )  # Transpose to [n_heads * d_head, d_model]
        W_Q_transformation.bias = torch.nn.Parameter(b_Q)

        W_K_transformation = torch.nn.Linear(W_K.shape[0], W_K.shape[1], bias=True)
        W_K_transformation.weight = torch.nn.Parameter(
            W_K.T
        )  # Transpose to [n_heads * d_head, d_model]
        W_K_transformation.bias = torch.nn.Parameter(b_K)

        W_V_transformation = torch.nn.Linear(W_V.shape[0], W_V.shape[1], bias=True)
        W_V_transformation.weight = torch.nn.Parameter(
            W_V.T
        )  # Transpose to [n_heads * d_head, d_model]
        W_V_transformation.bias = torch.nn.Parameter(b_V)

        return W_Q_transformation, W_K_transformation, W_V_transformation
