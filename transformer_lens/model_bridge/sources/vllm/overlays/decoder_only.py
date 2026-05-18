"""Generic overlay for any decoder-only model vLLM supports.

vLLM's decoder-only models all share the same internal structure:
``model.embed_tokens`` / ``model.layers.{i}`` (each with ``self_attn`` and
``mlp`` submodules) / ``model.norm`` / ``lm_head``. This overlay hooks that
shared abstraction, so one file works for Llama, Qwen, Mistral, Gemma, Phi3,
Qwen3, Kimi, GLM, and every other model that inherits the standard shape.

Non-decoder-only architectures (Mamba SSM, T5 encoder-decoder, BERT, MoE
per-expert) break the convention and would need their own overlays.

Caveat: ``model.layers.{i}`` returns ``(mlp_output, residual)`` separately
(vLLM's fused-residual pattern) — our hook on the layer captures ``output[0]``
which is the MLP delta, not the accumulated residual stream. HF's equivalent
hook captures the post-MLP residual. Per-layer ``blocks.{i}.hook_out`` values
will therefore diverge between sources; ``attn.hook_out`` and ``mlp.hook_out``
match across sources.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .base import AdapterOverlay


class DecoderOnlyOverlay(AdapterOverlay):
    """Default overlay for vLLM decoder-only models."""

    def capture_specs(self, hf_config: Any) -> Dict[str, Tuple[str, int]]:
        d_model = hf_config.hidden_size
        n_layers = hf_config.num_hidden_layers
        # unembed.hook_out is intentionally NOT captured here — vLLM's sampler
        # computes logits via a direct matmul on the final hidden state and
        # never invokes lm_head.__call__, so register_forward_hook on lm_head
        # would install but never fire. See nonfiring_hooks() below; the
        # driver synthesizes the next-token logits from vLLM's sampler output.
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
        # vLLM's universal fused-kernel limitations — PagedAttention fuses the
        # QK^T → softmax → attn-weight path; QKVParallelLinear fuses RoPE inside
        # the projection. Same restriction on every decoder-only model.
        return [
            "blocks.{i}.attn.hook_pattern",
            "blocks.{i}.attn.hook_attn_scores",
            "blocks.{i}.attn.hook_rot_q",
            "blocks.{i}.attn.hook_rot_k",
            # vLLM's sampler bypasses lm_head.__call__ — capture-via-forward-hook
            # never fires; driver synthesizes argmax-matching logits from the
            # sampler's returned token.
            "unembed.hook_out",
        ]
