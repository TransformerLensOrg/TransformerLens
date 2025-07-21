"""GPT2 architecture adapter."""

from typing import Any

import einops
import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    JointQKVAttentionBridge,
    LayerNormBridge,
    LinearBridge,
    MLPBridge,
    UnembeddingBridge,
)


class GPT2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GPT2 models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the GPT2 architecture adapter."""
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "pos_embed.W_pos": "transformer.wpe.weight",
                "embed.W_E": "transformer.wte.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion(
                        "m (three n h) -> three n m h",
                        three=3,
                        n=self.cfg.num_attention_heads,
                    ),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion(
                        "m (three n h) -> three n m h",
                        three=3,
                        n=self.cfg.num_attention_heads,
                    ),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion(
                        "m (three n h) -> three n m h",
                        three=3,
                        n=self.cfg.num_attention_heads,
                    ),
                ),
                "blocks.{i}.attn.W_O": (
                    "transformer.h.{i}.attn.c_proj.weight",
                    RearrangeWeightConversion("(n h) m -> n h m", n=self.cfg.num_attention_heads),
                ),
                "blocks.{i}.attn.b_Q": "transformer.h.{i}.attn.c_attn.bias",
                "blocks.{i}.attn.b_K": "transformer.h.{i}.attn.c_attn.bias",
                "blocks.{i}.attn.b_V": "transformer.h.{i}.attn.c_attn.bias",
                "blocks.{i}.attn.b_O": "transformer.h.{i}.attn.c_proj.bias",
                "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.ln2.b": "transformer.h.{i}.ln_2.bias",
                "blocks.{i}.mlp.W_in": "transformer.h.{i}.mlp.c_fc.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.c_fc.bias",
                "blocks.{i}.mlp.W_out": "transformer.h.{i}.mlp.c_proj.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
                "unembed.W_U": "lm_head.weight",
            }
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            "pos_embed": EmbeddingBridge(name="transformer.wpe"),
            "blocks": BlockBridge(
                name="transformer.h",
                submodules={
                    "ln1": LayerNormBridge(name="ln_1"),
                    "attn": JointQKVAttentionBridge(
                        name="attn",
                        submodules={
                            "W_QKV": LinearBridge(name="c_attn"),
                            "W_O": LinearBridge(name="c_proj"),
                        },
                        config={
                            "d_model": self.cfg.n_embd,
                            "n_head": self.cfg.n_head,
                            "d_head": self.cfg.n_embd // self.cfg.n_head,
                            "split_qkv_matrix": self.split_qkv_matrix,
                        },
                    ),
                    "ln2": LayerNormBridge(name="ln_2"),
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "W_in": LinearBridge(name="c_fc"),
                            "W_out": LinearBridge(name="c_proj"),
                        },
                    ),
                },
            ),
            "ln_final": LayerNormBridge(name="transformer.ln_f"),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def split_qkv_matrix(
        self, attention_component: Any
    ) -> tuple[torch.nn.Linear, torch.nn.Linear, torch.nn.Linear]:
        """Split the QKV matrix into separate linear transformations.
        Args:
            attention_component: The original attention layer component
        Returns:
            Tuple of nn.Linear modules for Q, K, and V transformations
        """

        qkv_weights = attention_component.c_attn.original_component.weight
        W_Q, W_K, W_V = torch.tensor_split(qkv_weights, 3, dim=1)
        print(type(attention_component))
        qkv_bias = einops.rearrange(
            attention_component.c_attn.original_component.bias,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=self.cfg.n_head,
            head=self.cfg.n_embd // self.cfg.n_head,
        )

        # Create nn.Linear module
        W_Q_transformation = torch.nn.Linear(W_Q.shape[0], W_Q.shape[1], bias=True)

        # Set the weight and bias
        W_Q_transformation.weight = torch.nn.Parameter(W_Q.T)
        W_Q_transformation.bias = torch.nn.Parameter(qkv_bias[0].flatten())

        # Create nn.Linear module for K
        W_K_transformation = torch.nn.Linear(W_K.shape[0], W_K.shape[1], bias=True)

        # Set the weight and bias
        W_K_transformation.weight = torch.nn.Parameter(W_K.T)
        W_K_transformation.bias = torch.nn.Parameter(qkv_bias[1].flatten())

        # Create nn.Linear module for V
        W_V_transformation = torch.nn.Linear(W_V.shape[0], W_V.shape[1], bias=True)

        # Set the weight and bias
        W_V_transformation.weight = torch.nn.Parameter(W_V.T)
        W_V_transformation.bias = torch.nn.Parameter(qkv_bias[2].flatten())

        return W_Q_transformation, W_K_transformation, W_V_transformation
