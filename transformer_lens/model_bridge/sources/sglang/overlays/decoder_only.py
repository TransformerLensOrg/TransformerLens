"""Generic overlay for SGLang decoder-only models (Llama, Qwen, Qwen3, Mistral,
Phi3, Gemma, GLM, Kimi). All share vLLM's dotted-path layout. DeepSeek-V2/V3
needs its own overlay (MLA — see :mod:`overlays.deepseek`)."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .base import AdapterOverlay


class DecoderOnlyOverlay(AdapterOverlay):
    """Default overlay for SGLang decoder-only models."""

    def capture_specs(self, hf_config: Any) -> Dict[str, Tuple[str, int]]:
        d_model = hf_config.hidden_size
        n_layers = hf_config.num_hidden_layers
        # unembed.hook_out NOT captured: sampler bypasses lm_head.__call__, driver
        # synthesizes logits from sampler output.
        specs: Dict[str, Tuple[str, int]] = {
            "embed.hook_out": ("model.embed_tokens", d_model),
            "ln_final.hook_normalized": ("model.norm", d_model),
        }
        for i in range(n_layers):
            specs[f"blocks.{i}.hook_out"] = (f"model.layers.{i}", d_model)
            specs[f"blocks.{i}.attn.hook_out"] = (f"model.layers.{i}.self_attn", d_model)
            specs[f"blocks.{i}.mlp.hook_out"] = (f"model.layers.{i}.mlp", d_model)
        return specs

    def nonfiring_hooks(self) -> List[str]:
        # RadixAttention fuses QK^T → softmax → attn-weight; rotary lives inside
        # the backend; sampler bypasses lm_head.
        return [
            "blocks.{i}.attn.hook_pattern",
            "blocks.{i}.attn.hook_attn_scores",
            "blocks.{i}.attn.hook_rot_q",
            "blocks.{i}.attn.hook_rot_k",
            "unembed.hook_out",
        ]
