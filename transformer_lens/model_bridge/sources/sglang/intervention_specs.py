"""Re-export of the vLLM intervention vocabulary — shared chokepoint so adding
an op only requires editing the vLLM module + both worker extensions."""
from __future__ import annotations

from transformer_lens.model_bridge.sources.vllm.intervention_specs import SUPPORTED_OPS

__all__ = ["SUPPORTED_OPS"]
