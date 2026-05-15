"""Generic overlay for any decoder-only model vLLM supports.

vLLM's decoder-only models all share the same internal structure:
``model.embed_tokens`` / ``model.layers.{i}`` (each with ``self_attn`` and
``mlp`` submodules) / ``model.norm`` / ``lm_head``. This overlay hooks that
shared abstraction, so one file works for Llama, Qwen, Mistral, Gemma, Phi3,
Qwen3, Kimi, GLM, and every other model that inherits the standard shape.

Non-decoder-only architectures (Mamba SSM, T5 encoder-decoder, BERT, MoE
per-expert) break the convention and would need their own overlays.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .base import AdapterOverlay


class DecoderOnlyOverlay(AdapterOverlay):
    """Default overlay for vLLM decoder-only models."""

    def capture_specs(self, hf_config: Any) -> Dict[str, Tuple[str, int]]:
        d_model = hf_config.hidden_size
        d_vocab = hf_config.vocab_size
        n_layers = hf_config.num_hidden_layers
        specs: Dict[str, Tuple[str, int]] = {
            "embed.hook_out": ("model.embed_tokens", d_model),
            "ln_final.hook_normalized": ("model.norm", d_model),
            "unembed.hook_out": ("lm_head", d_vocab),
        }
        for i in range(n_layers):
            specs[f"blocks.{i}.hook_out"] = (f"model.layers.{i}", d_model)
            specs[f"blocks.{i}.attn.hook_out"] = (f"model.layers.{i}.self_attn", d_model)
            specs[f"blocks.{i}.mlp.hook_out"] = (f"model.layers.{i}.mlp", d_model)
        return specs

    def nonfiring_hooks(self) -> List[str]:
        # vLLM's universal fused-kernel limitations — PagedAttention fuses the
        # QK^T → softmax → attn-weight path; QKVParallelLinear fuses RoPE inside
        # the projection. Same restriction on every decoder-only model.
        return [
            "blocks.{i}.attn.hook_pattern",
            "blocks.{i}.attn.hook_attn_scores",
            "blocks.{i}.attn.hook_rot_q",
            "blocks.{i}.attn.hook_rot_k",
        ]
