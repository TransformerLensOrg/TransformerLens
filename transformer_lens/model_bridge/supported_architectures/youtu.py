"""Youtu architecture adapter.

Tencent's Youtu-LLM (``YoutuForCausalLM``, native in transformers): the
only laptop-scale densely-MLP'd Multi-head Latent Attention checkpoint —
DeepSeek-V2's MLA (q LoRA + compressed KV with decoupled rope) with every
layer dense. Module names match DeepSeek-V2 exactly and the MLA bridge
already handles both q-projection variants, so this is a pure subclass:
the MoE router/shared-expert submodules are optional and simply never
bind on the dense MLPs.
"""

from transformer_lens.model_bridge.supported_architectures.deepseek_v2 import (
    DeepSeekV2ArchitectureAdapter,
)


class YoutuArchitectureAdapter(DeepSeekV2ArchitectureAdapter):
    """Architecture adapter for YoutuForCausalLM models."""

    def _build_mlp_bridge(self):
        """Every layer is a plain gated MLP — map it so weight processing
        sees mlp.in/gate/out (the parent's MoE wrapper leaves them opaque)."""
        return self._gated_mlp()
