"""VaultGemma architecture adapter.

Google's VaultGemma (``VaultGemmaForCausalLM``, native in transformers):
the only fully DP-SGD-pretrained open LLM — a Gemma-2 recipe trained under
differential privacy. Structurally it is Gemma 2 with the two post-norms
removed (blocks keep only input_layernorm and pre_feedforward_layernorm);
the scaled word embedding and attention logit soft-cap carry over, so this
subclasses the Gemma 2 adapter and rebuilds the block entry without
ln1_post/ln2_post.
"""

from typing import Any

from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
)
from transformer_lens.model_bridge.supported_architectures.gemma2 import (
    Gemma2ArchitectureAdapter,
)


class VaultGemmaArchitectureAdapter(Gemma2ArchitectureAdapter):
    """Architecture adapter for VaultGemmaForCausalLM models."""

    # Compatibility mode's stored-processed-weights forward diverges for this
    # offset-RMS variant even with every weight step disabled (vaultgemma-1b:
    # logits shift 9.6 with zero state-dict changes; 19.56 with defaults) —
    # the pipeline's gemma path assumes the post-norm sandwich this variant
    # removed. P3 excluded until the compat path handles offset-RMS without
    # post-norms; P1/P2/P4 verify clean.
    applicable_phases: list[int] = [1, 2, 4]

    def __init__(self, cfg: Any) -> None:
        """Initialize the VaultGemma architecture adapter."""
        super().__init__(cfg)

        # Gemma 2 minus the post-norms: only input_layernorm and
        # pre_feedforward_layernorm exist.
        self.components["blocks"] = BlockBridge(
            name="model.layers",
            config=self.cfg,
            submodules={
                "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                "ln2": RMSNormalizationBridge(name="pre_feedforward_layernorm", config=self.cfg),
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
                "mlp": self._gated_mlp(),
            },
        )
