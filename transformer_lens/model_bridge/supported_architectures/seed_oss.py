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
