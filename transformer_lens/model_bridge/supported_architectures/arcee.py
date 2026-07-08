"""Arcee architecture adapter."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class ArceeArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Arcee models (ArceeForCausalLM / AFM-4.5B).

    Arcee is a Llama-style dense decoder: pre-norm RMSNorm, rotary position
    embeddings (RoPE), grouped query attention (GQA), and no biases on any
    projection. The single distinguishing feature is the MLP: an *ungated*
    feed-forward block (``up_proj -> ReLU^2 -> down_proj``) using the squared-ReLU
    activation (HF ``hidden_act = "relu2"``) instead of the gated SiLU/GeLU used by
    Llama. The post-activation neurons are exposed via the MLP bridge's
    ``hook_post`` (``mlp.out.hook_in``), which is useful for inspecting the sparse
    activation structure ReLU^2 produces.

    Structurally identical to Llama except for the ungated ReLU^2 MLP; unlike
    Apertus it uses standard ``input_layernorm`` / ``post_attention_layernorm``
    names and has no Q/K normalization.

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    Arcee models do NOT have biases on attention or MLP projections
    (``attention_bias = false``, ``mlp_bias = false``):

    - blocks.{i}.attn.b_Q / b_K / b_V / b_O - No bias on attention projections
    - blocks.{i}.mlp.b_in - No bias on MLP input (up_proj)
    - blocks.{i}.mlp.b_out - No bias on MLP output (down_proj)
    - blocks.{i}.ln1.b / ln2.b / ln_final.b - RMSNorm has no bias

    Weight processing handles these missing biases gracefully via
    ProcessWeights._safe_get_tensor().
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the Arcee architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = False  # ungated ReLU^2 MLP (up_proj -> act -> down_proj)
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True

        # Use eager attention so output_attentions works for hook_attn_scores /
        # hook_pattern; SDPA does not support output_attentions.
        self.cfg.attn_implementation = "eager"

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
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
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "in": LinearBridge(name="up_proj"),
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Wire the shared rotary onto attention bridges (attn implementation untouched)."""
        self._wire_rotary_for_testing(hf_model, bridge_model, eager=None)
