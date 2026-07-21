"""GLM-4.5 MoE architecture adapter.

Supports GLM-4.5/4.6/4.7 mixture-of-experts families (`Glm4MoeForCausalLM`).

Key features:
- RMSNorm with partial pre-norm layout.
- RoPE-style rotary embeddings (partial RoPE supported by Hugging Face model logic).
- Q/K normalization blocks (`q_norm`, `k_norm`) and GQA / MQA handling.
- Sparse MoE block in `model.layers[i].mlp`, with optional dense-prefix layers.
- QKVO rearrangements for bridge-side attention hooks.

Optional Parameters (may not exist in state_dict):
-------------------------------------------------
- blocks.{i}.mlp.gate - absent on dense-prefix layers before sparse MoE starts.
"""

from typing import Any

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MoEBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class Glm4MoeRouterBridge(LinearBridge):
    """Bridge GLM-4 MoE router logits while preserving HF's tuple return.

    ``Glm4MoeTopkRouter.forward()`` returns a 3-tuple
    ``(router_logits, topk_weights, topk_indices)``.  The base
    ``LinearBridge.forward`` would pass the tuple to ``hook_out``, which expects
    a single ``torch.Tensor``, causing a ``beartype`` runtime error.
    """

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        input = self.hook_in(input)
        output = self.original_component(input, *args, **kwargs)
        if not isinstance(output, tuple) or len(output) == 0:
            return self.hook_out(output)
        router_logits = self.hook_out(output[0])
        return (router_logits,) + output[1:]


class Glm4MoeArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GLM-4.5 / 4.6 / 4.7 MoE decoder models.

    GLM-4x MoE families use RMSNorm, RoPE and sparse routing, with early
    dense-MLP layers in some checkpoints. The dense layers are represented by
    a present-but-slightly-thinner `mlp` sub-module where routing is absent.
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the GLM-4 MoE architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()
        # Force eager attention for output_attentions / compatibility-path parity.
        self.cfg.attn_implementation = "eager"
        # GLM-4 defaults do not prepend BOS in current tiny checkpoints.
        self.cfg.default_prepend_bos = False

        # QKVO rearrangements; MoE experts and gate are passed through unchanged.
        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": PositionEmbeddingsAttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                            "q_norm": RMSNormalizationBridge(name="q_norm", config=self.cfg),
                            "k_norm": RMSNormalizationBridge(name="k_norm", config=self.cfg),
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    # Dense prefix layers expose `mlp` but no router; mark gate optional
                    # for the dense-MoE boundary.
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": Glm4MoeRouterBridge(name="gate", optional=True),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Force eager attention and wire the shared rotary onto attention bridges."""
        self._wire_rotary_for_testing(hf_model, bridge_model)
