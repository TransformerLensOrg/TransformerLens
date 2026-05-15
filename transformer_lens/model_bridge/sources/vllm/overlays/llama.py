"""Llama-family vLLM overlay.

vLLM's Llama model uses ``qkv_proj`` (fused QKV) and ``gate_up_proj`` (fused gate+up);
the canonical TL Llama adapter expects split ``q_proj``/``k_proj``/``v_proj`` and
``gate_proj``/``up_proj``. This overlay swaps in :class:`JointQKVPositionEmbeddingsAttentionBridge`
and :class:`JointGateUpMLPBridge`, mirroring the Phi-3 adapter which already handles
the same fused-projection layout on HF.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from transformer_lens.model_bridge.generalized_components import (
    JointGateUpMLPBridge,
    JointQKVPositionEmbeddingsAttentionBridge,
    LinearBridge,
)

from .base import AdapterOverlay


def _split_llama_qkv_factory(cfg: Any):
    """Return a closure that splits vLLM's fused qkv_proj into Q/K/V Linear modules.

    GQA-aware: K and V share ``n_key_value_heads * d_head`` width while Q uses
    ``n_heads * d_head``.
    """

    def _split(
        original_attention_component: Any,
    ) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
        qkv_weight = original_attention_component.qkv_proj.weight
        d_model = qkv_weight.shape[1]

        d_head = cfg.d_model // cfg.n_heads
        n_kv_heads = cfg.n_key_value_heads or cfg.n_heads
        q_size = cfg.n_heads * d_head
        kv_size = n_kv_heads * d_head

        q_w, k_w, v_w = torch.split(qkv_weight, [q_size, kv_size, kv_size], dim=0)

        has_bias = (
            hasattr(original_attention_component.qkv_proj, "bias")
            and original_attention_component.qkv_proj.bias is not None
        )
        q_b: torch.Tensor | None
        k_b: torch.Tensor | None
        v_b: torch.Tensor | None
        if has_bias:
            q_b, k_b, v_b = torch.split(
                original_attention_component.qkv_proj.bias, [q_size, kv_size, kv_size], dim=0
            )
        else:
            q_b = k_b = v_b = None

        def _make_linear(weight: torch.Tensor, bias: torch.Tensor | None) -> torch.nn.Linear:
            linear = torch.nn.Linear(d_model, weight.shape[0], bias=bias is not None)
            linear.weight = torch.nn.Parameter(weight)
            if bias is not None:
                linear.bias = torch.nn.Parameter(bias)
            return linear

        return _make_linear(q_w, q_b), _make_linear(k_w, k_b), _make_linear(v_w, v_b)

    return _split


def _split_gate_up(original_mlp_component: Any) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Split vLLM's fused gate_up_proj into gate and up Linear modules."""
    fused_weight = original_mlp_component.gate_up_proj.weight
    gate_w, up_w = torch.tensor_split(fused_weight, 2, dim=0)
    d_model = fused_weight.shape[1]
    d_mlp = gate_w.shape[0]

    has_bias = (
        hasattr(original_mlp_component.gate_up_proj, "bias")
        and original_mlp_component.gate_up_proj.bias is not None
    )
    gate_b: torch.Tensor | None
    up_b: torch.Tensor | None
    if has_bias:
        gate_b, up_b = torch.tensor_split(original_mlp_component.gate_up_proj.bias, 2, dim=0)
    else:
        gate_b = up_b = None

    def _make_linear(weight: torch.Tensor, bias: torch.Tensor | None) -> torch.nn.Linear:
        linear = torch.nn.Linear(d_model, weight.shape[0], bias=bias is not None)
        linear.weight = torch.nn.Parameter(weight)
        if bias is not None:
            linear.bias = torch.nn.Parameter(bias)
        return linear

    return _make_linear(gate_w, gate_b), _make_linear(up_w, up_b)


class LlamaVLLMOverlay(AdapterOverlay):
    """Llama / Mistral / Qwen2 share this overlay shape (all use qkv_proj + gate_up_proj)."""

    def capture_specs(self, hf_config: Any) -> Dict[str, Tuple[str, int]]:
        # Minimal v1 capture surface: module-boundary outputs, all d_model-wide
        # except the unembed which is vocab-wide.
        d_model = hf_config.hidden_size
        d_vocab = hf_config.vocab_size
        n_layers = hf_config.num_hidden_layers
        specs: Dict[str, Tuple[str, int]] = {
            "hook_embed": ("model.embed_tokens", d_model),
            "ln_final.hook_normalized": ("model.norm", d_model),
            "unembed.hook_out": ("lm_head", d_vocab),
        }
        for i in range(n_layers):
            specs[f"blocks.{i}.hook_resid_post"] = (f"model.layers.{i}", d_model)
            specs[f"blocks.{i}.attn.hook_attn_out"] = (f"model.layers.{i}.self_attn", d_model)
            specs[f"blocks.{i}.mlp.hook_post"] = (f"model.layers.{i}.mlp", d_model)
        return specs

    def apply(self, adapter: Any) -> None:
        block = adapter.component_mapping["blocks"]
        block.submodules["attn"] = JointQKVPositionEmbeddingsAttentionBridge(
            name="self_attn",
            config=adapter.cfg,
            split_qkv_matrix=_split_llama_qkv_factory(adapter.cfg),
            submodules={
                "qkv": LinearBridge(name="qkv_proj"),
                "o": LinearBridge(name="o_proj"),
            },
            requires_attention_mask=True,
            requires_position_embeddings=True,
        )
        block.submodules["mlp"] = JointGateUpMLPBridge(
            name="mlp",
            config=adapter.cfg,
            split_gate_up_matrix=_split_gate_up,
            submodules={
                "out": LinearBridge(name="down_proj"),
            },
        )

    def nonfiring_hooks(self) -> List[str]:
        # vLLM fuses these inside PagedAttention / QKVParallelLinear; never materialized.
        return [
            "blocks.{i}.attn.hook_pattern",
            "blocks.{i}.attn.hook_attn_scores",
            "blocks.{i}.attn.hook_rot_q",
            "blocks.{i}.attn.hook_rot_k",
        ]
