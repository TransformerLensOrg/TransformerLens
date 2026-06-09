"""Re-export of the vLLM intervention vocabulary — one place to add an op."""
from __future__ import annotations

from transformer_lens.model_bridge.sources.vllm.intervention_specs import SUPPORTED_OPS

__all__ = ["SUPPORTED_OPS"]
