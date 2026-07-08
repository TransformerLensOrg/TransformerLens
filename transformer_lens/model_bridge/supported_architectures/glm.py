"""GLM architecture adapter.

Z.ai's dense GLM-4 family (``GlmForCausalLM``: glm-4-9b-*-hf, glm-edge):
llama-shaped GQA decoder with GLM's adjacent-pair interleaved RoPE at a
partial rotary factor, attention biases, and a Phi3-style combined
``gate_up_proj`` MLP.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    JointGateUpMLPBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.phi3 import (
    Phi3ArchitectureAdapter,
)


class GlmArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GlmForCausalLM models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the GLM architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()
        # GLM rotates adjacent element pairs (interleaved RoPE), like ERNIE.
        self.cfg.rotary_adjacent_pairs = True
        # GLM tokenizers carry no BOS token.
        self.cfg.default_prepend_bos = False

        # Joint gate_up_proj cannot be folded by the standard LN machinery.
        self.supports_fold_ln = False
        self.weight_processing_conversions = {}

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
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    # GlmMLP chunks gate_up_proj output in half — same layout
                    # Phi3 uses, so its splitter applies unchanged.
                    "mlp": JointGateUpMLPBridge(
                        name="mlp",
                        config=self.cfg,
                        split_gate_up_matrix=Phi3ArchitectureAdapter._split_gate_up,
                        submodules={
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Force eager attention and wire the shared rotary onto attention bridges."""
        self._wire_rotary_for_testing(hf_model, bridge_model, eager="config")
