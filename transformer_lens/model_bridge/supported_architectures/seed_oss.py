"""Seed-OSS architecture adapter.

ByteDance's Seed-OSS (``SeedOssForCausalLM``) is a Llama-layout decoder —
RMSNorm + RoPE + GQA + gated MLP under identical module paths — with
config-gated attention/MLP biases (handled by the shared weight machinery)
and no BOS prepending.
"""

from typing import Any

from transformer_lens.model_bridge.supported_architectures.llama import (
    LlamaArchitectureAdapter,
)


class SeedOssArchitectureAdapter(LlamaArchitectureAdapter):
    """Architecture adapter for SeedOssForCausalLM models."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        # Verified against ByteDance-Seed/Seed-OSS-36B-Instruct's tokenizer.
        self.cfg.default_prepend_bos = False
        # Seed-OSS ships attention_bias=True with GQA; the Llama parent omits
        # bias reshapes, so K/V biases would keep the flat (n_kv*d_head,) layout.
        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(include_biases=True),
        }
