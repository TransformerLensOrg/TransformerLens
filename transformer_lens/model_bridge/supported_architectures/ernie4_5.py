"""ERNIE 4.5 architecture adapter.

Baidu's dense ERNIE 4.5 (``Ernie4_5ForCausalLM``) is a Llama-layout decoder —
RMSNorm + RoPE + GQA + gated MLP under identical module paths — with
config-gated biases (``use_bias``) and no BOS prepending.
"""

from typing import Any

from transformer_lens.model_bridge.supported_architectures.llama import (
    LlamaArchitectureAdapter,
)


class Ernie4_5ArchitectureAdapter(LlamaArchitectureAdapter):
    """Architecture adapter for Ernie4_5ForCausalLM models."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        # Verified against baidu/ERNIE-4.5-0.3B-PT's tokenizer.
        self.cfg.default_prepend_bos = False
        # ERNIE rotates adjacent element pairs (GLM-style interleaved RoPE),
        # unlike llama's half-split convention.
        self.cfg.rotary_adjacent_pairs = True
