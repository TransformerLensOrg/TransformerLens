"""Qwen architecture adapter."""

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
    GatedMLPBridge,
    JointQKVAttentionBridge,
    LinearBridge,
    NormalizationBridge,
    UnembeddingBridge,
)


class QwenArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Qwen models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Qwen architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.k": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.v": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.o": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                source_key="transformer.h.{i}.attn.c_proj.weight",
            ),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            "blocks": BlockBridge(
                name="transformer.h",
                submodules={
                    "ln1": NormalizationBridge(name="ln_1", config=self.cfg),
                    "attn": JointQKVAttentionBridge(
                        name="attn",
                        config=self.cfg,
                        split_qkv_matrix=self._split_qkv_matrix,
                        submodules={
                            "qkv": LinearBridge(name="c_attn"),
                            "o": LinearBridge(name="c_proj"),
                        },
                    ),
                    "ln2": NormalizationBridge(name="ln_2", config=self.cfg),
                    "mlp": GatedMLPBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": LinearBridge(name="w1"),
                            "in": LinearBridge(name="w2"),
                            "out": LinearBridge(name="c_proj"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="transformer.ln_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def _split_qkv_matrix(
        self, original_attention_component: Any
    ) -> tuple[torch.nn.Linear, torch.nn.Linear, torch.nn.Linear]:
        """Split Qwen's fused c_attn linear layer into q, k, v projections."""

        assert original_attention_component is not None
        assert hasattr(original_attention_component, "c_attn")

        c_attn = original_attention_component.c_attn
        assert isinstance(c_attn, torch.nn.Linear)

        d_model = self.cfg.d_model
        qkv_weights = c_attn.weight.detach().clone()

        if qkv_weights.shape == (d_model, 3 * d_model):
            # Weight stored as [in_features, 3*out_features] (Conv1D style)
            W_Q, W_K, W_V = torch.tensor_split(qkv_weights, 3, dim=1)
            W_Q, W_K, W_V = W_Q.T.contiguous(), W_K.T.contiguous(), W_V.T.contiguous()
        elif qkv_weights.shape == (3 * d_model, d_model):
            # Standard Linear layout [3*out_features, in_features]
            W_Q, W_K, W_V = torch.tensor_split(qkv_weights, 3, dim=0)
        else:
            raise ValueError(
                f"Unexpected c_attn weight shape {qkv_weights.shape} for Qwen attention "
                f"(expected ({d_model}, {3*d_model}) or ({3*d_model}, {d_model}))"
            )

        if c_attn.bias is not None:
            qkv_bias = c_attn.bias.detach().clone()
            if qkv_bias.shape[0] != 3 * d_model:
                raise ValueError(
                    f"Unexpected c_attn bias shape {qkv_bias.shape} for Qwen attention "
                    f"(expected ({3*d_model},))"
                )
            b_Q, b_K, b_V = torch.tensor_split(qkv_bias, 3, dim=0)
        else:
            device = qkv_weights.device
            dtype = qkv_weights.dtype
            b_Q = torch.zeros(d_model, device=device, dtype=dtype)
            b_K = torch.zeros_like(b_Q)
            b_V = torch.zeros_like(b_Q)

        def build_linear(weight: torch.Tensor, bias: torch.Tensor) -> torch.nn.Linear:
            linear = torch.nn.Linear(
                d_model, d_model, bias=True, device=weight.device, dtype=weight.dtype
            )
            linear.weight = torch.nn.Parameter(weight.contiguous())
            linear.bias = torch.nn.Parameter(bias.contiguous())
            return linear

        q_proj = build_linear(W_Q, b_Q)
        k_proj = build_linear(W_K, b_K)
        v_proj = build_linear(W_V, b_V)

        return q_proj, k_proj, v_proj
