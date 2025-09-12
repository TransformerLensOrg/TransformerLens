"""GPT-OSS architecture adapter."""

from typing import Any

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    JointGateUpMLPBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    UnembeddingBridge,
)


class GPTOSSArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GPT-OSS model."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the GPT-OSS architecture adapter."""
        super().__init__(cfg)

        self.cfg.gated_mlp = True

        self.cfg.uses_rms_norm = True

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": EmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": NormalizationBridge(name="input_layernorm", config=self.cfg),
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                    ),
                    "ln2": NormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "router": LinearBridge(name="router"),
                            "experts": BlockBridge(
                                name="experts",
                                submodules={
                                    "gate_up": JointGateUpMLPBridge(
                                        name="gate_up_proj",
                                        gate_up_config={
                                            "split_gate_up_matrix": self.split_gate_up_matrix
                                        },
                                    ),
                                    "down": LinearBridge(name="down_proj"),
                                },
                            ),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def split_gate_up_matrix(
        self, original_mlp_component: Any
    ) -> tuple[torch.nn.Linear, torch.nn.Linear]:
        gate_up_weight = original_mlp_component.gate_up_proj
        gate_up_bias = original_mlp_component.gate_up_proj_bias

        # In GPT-OSS, all the gate projection weights lie at even indices,
        # all the up projection weights lie at odd indices
        gate_weight = gate_up_weight[..., ::2]
        up_weight = gate_up_weight[..., 1::2]

        gate_bias = gate_up_bias[..., ::2]
        up_bias = gate_up_bias[..., 1::2]

        gate_projection = torch.nn.Linear(gate_weight.shape[0], gate_weight.shape[1], bias=True)

        gate_projection.weight = torch.nn.Parameter(gate_weight)
        gate_projection.bias = torch.nn.Parameter(gate_bias)

        up_projection = torch.nn.Linear(up_weight.shape[0], up_weight.shape[1])

        up_projection.weight = torch.nn.Parameter(up_weight)
        up_projection.bias = torch.nn.Parameter(up_bias)

        return gate_projection, up_projection
