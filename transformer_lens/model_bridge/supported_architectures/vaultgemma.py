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

from transformer_lens.model_bridge.generalized_components import RMSNormalizationBridge
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
    # post-norms; P1/P2/P4 verify clean. Gated at runtime so
    # enable_compatibility_mode() raises instead of silently diverging.
    applicable_phases: list[int] = [1, 2, 4]
    supports_compatibility_mode: bool = False

    def __init__(self, cfg: Any) -> None:
        """Initialize the VaultGemma architecture adapter."""
        super().__init__(cfg)

        # Gemma 2 minus the post-norms: drop the inherited RMS-offset
        # conversions for the ln1_post/ln2_post norms this variant removes.
        if self.weight_processing_conversions is not None:
            for dead in ("blocks.{i}.ln1_post.weight", "blocks.{i}.ln2_post.weight"):
                self.weight_processing_conversions.pop(dead, None)

    def _block_norms(self):
        """Gemma 2 minus the post-norms: only input and pre-feedforward norms."""
        return {
            "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
            "ln2": RMSNormalizationBridge(name="pre_feedforward_layernorm", config=self.cfg),
        }
