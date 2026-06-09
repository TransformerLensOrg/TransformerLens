"""Overlay for DeepSeek-V2/V3/V4 (MLA attention). Per-layer dotted paths match the
decoder-only overlay; MLA's internals (latent K/V split) live below the
``self_attn`` boundary and aren't individually hookable. Per-expert MoE routing
fires below ``mlp.__call__`` for DeepseekV2MoE models — same restriction."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .base import AdapterOverlay


class DeepseekOverlay(AdapterOverlay):
    """Overlay for DeepSeek-V2 / V3 / V4 (MLA attention)."""

    def capture_specs(self, hf_config: Any) -> Dict[str, Tuple[str, int]]:
        d_model = hf_config.hidden_size
        n_layers = hf_config.num_hidden_layers
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
        return [
            "blocks.{i}.attn.hook_pattern",
            "blocks.{i}.attn.hook_attn_scores",
            "blocks.{i}.attn.hook_rot_q",
            "blocks.{i}.attn.hook_rot_k",
            "blocks.{i}.attn.hook_latent_kv",
            "blocks.{i}.mlp.hook_expert_out",
            "unembed.hook_out",
        ]
