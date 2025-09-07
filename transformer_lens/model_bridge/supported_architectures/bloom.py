"""Bloom architecture adapter."""

from typing import Any

import torch

from transformer_lens.conversion_utils.conversion_steps import (
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


class BloomArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Bloom models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Bloom architecture adapter."""
        super().__init__(cfg)

        self.cfg.default_prepend_bos = False
        self.conversion_rules = HookConversionSet(
            {
                "embed.e": "transformer.word_embeddings.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.input_layernorm.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.input_layernorm.bias",
                "blocks.{i}.attn.q": (
                    "transformer.h.{i}.self_attention.query_key_value.weight",
                    RearrangeHookConversion(
                        "(three n h) m -> three n m h",
                        three=3,
                        n=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.k": (
                    "transformer.h.{i}.self_attention.query_key_value.weight",
                    RearrangeHookConversion(
                        "(three n h) m -> three n m h",
                        three=3,
                        n=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.v": (
                    "transformer.h.{i}.self_attention.query_key_value.weight",
                    RearrangeHookConversion(
                        "(three n h) m -> three n m h",
                        three=3,
                        n=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.o": (
                    "transformer.h.{i}.self_attention.dense.weight",
                    RearrangeHookConversion("m (n h) -> n h m", n=self.cfg.n_heads),
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
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.word_embeddings"),
            "embed_ln": NormalizationBridge(name="transformer.word_embeddings_layernorm"),
            "blocks": BlockBridge(
                name="transformer.h",
                submodules={
                    "ln1": NormalizationBridge(name="input_layernorm"),
                    "ln2": NormalizationBridge(name="post_attention_layernorm"),
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
            "ln_final": NormalizationBridge(name="transformer.ln_f"),
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
