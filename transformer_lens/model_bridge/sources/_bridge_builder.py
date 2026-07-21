"""Loader-agnostic helpers for building a TransformerBridge around a pre-loaded model."""
from __future__ import annotations

import copy
from typing import Any, Callable, Optional

import torch
from torch import nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.bridge import TransformerBridge

# Architecture-agnostic; do not extend per-architecture.
_HF_PASSTHROUGH_ATTRS = [
    # OPT
    "is_gated_act",
    "word_embed_proj_dim",
    "do_layer_norm_before",
    # BART
    "encoder_layers",
    "decoder_layers",
    "encoder_attention_heads",
    "decoder_attention_heads",
    "encoder_ffn_dim",
    "decoder_ffn_dim",
    # Granite
    "position_embedding_type",
    "logits_scaling",
    # Falcon
    "parallel_attn",
    "multi_query",
    "new_decoder_architecture",
    "alibi",
    "num_ln_in_parallel_attn",
    # Mamba (SSM config)
    "state_size",
    "conv_kernel",
    "expand",
    "time_step_rank",
    "intermediate_size",
    # Mamba-2 (additional SSM config)
    "n_groups",
    "chunk_size",
    # Falcon-H1 (parallel attn + Mamba-2 hybrid SSM config)
    "mamba_d_ssm",
    "mamba_n_heads",
    "mamba_d_head",
    "mamba_d_state",
    "mamba_n_groups",
    "mamba_d_conv",
    "mamba_chunk_size",
    "lm_head_multiplier",
    # Multimodal
    "vision_config",
    # Cohere
    "logit_scale",
    "rope_parameters",
    # HRM-Text
    "H_cycles",
    "L_cycles",
    "L_bp_cycles",
    "embedding_scale",
    "prefix_lm",
    "num_layers_per_stack",
    "sliding_window_pattern",
    "_sliding_window_pattern",
    # Hybrid/MoE architectures
    "layer_types",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "norm_eps",
    "attention_bias",
    "lm_head_bias",
    "router_jitter_noise",
    "input_jitter_noise",
    "eos_token_id",
    # LLaDA remote-code model contract and tokenizer metadata
    "block_type",
    "block_group_size",
    "rope",
    "rope_full_precision",
    "attention_layer_norm",
    "include_bias",
    "include_qkv_bias",
    "scale_logits",
    "input_emb_norm",
    "layer_norm_type",
    "embedding_size",
    "mask_token_id",
    "pad_token_id",
    "bos_token_id",
    # BD3LM
    "model_length",
    "block_size",
    "cond_dim",
    "adaln",
    "cross_attn",
    # Zamba2 (Mamba-2 + shared-attention hybrid)
    "mamba_expand",
    "mamba_ngroups",
    "num_mem_blocks",
    "layers_block_type",
    "use_shared_attention_adapter",
    # Ouro (LoopLM)
    "total_ut_steps",
    "early_exit_threshold",
    # DeepSeek V4 (mHC + compressed attention)
    "compress_rates",
    "compress_rope_theta",
    "hc_mult",
    "hc_sinkhorn_iters",
    "hc_eps",
    "mlp_layer_types",
    "swiglu_limit",
    "o_groups",
    "o_lora_rank",
    "index_n_heads",
    "index_head_dim",
    "index_topk",
    "q_lora_rank",
    # Raven / Huginn (depth-recurrent)
    "mean_recurrence",
    "mean_backprop_depth",
    "n_layers_in_prelude",
    "n_layers_in_recurrent_block",
    "n_layers_in_coda",
    "injection_type",
    "qk_bias",
    # RWKV-7 (attention-free recurrent, generalized delta-rule time-mixing).
    # head_dim is intentionally omitted: it is a read-only alias of d_head on
    # TransformerBridgeConfig, so a passthrough setattr would raise.
    "num_heads",
    "value_dim",
    "decay_low_rank_dim",
    "gate_low_rank_dim",
    "a_low_rank_dim",
    "v_low_rank_dim",
    "norm_first",
    "norm_bias",
    "fuse_norm",
    "attn_mode",
    "hidden_act",
]


def build_bridge_config_from_hf(
    hf_config: Any,
    architecture: str,
    model_name: str,
    dtype: torch.dtype,
) -> TransformerBridgeConfig:
    """Translate an HF config into a :class:`TransformerBridgeConfig`."""
    from transformer_lens.model_bridge.sources.transformers import (
        get_effective_text_config,
        map_default_transformer_lens_config,
    )

    tl_config = map_default_transformer_lens_config(hf_config)
    config_dict = dict(tl_config.__dict__)
    # HF's attribute_map remaps num_experts → num_local_experts; restore the TL name.
    if "num_local_experts" in config_dict and "num_experts" not in config_dict:
        config_dict["num_experts"] = config_dict["num_local_experts"]
    bridge_config = TransformerBridgeConfig.from_dict(config_dict)
    bridge_config.architecture = architecture
    bridge_config.model_name = model_name
    bridge_config.dtype = dtype

    effective_config = get_effective_text_config(hf_config)
    for attr in _HF_PASSTHROUGH_ATTRS:
        val = getattr(effective_config, attr, None)
        if val is None and effective_config is not hf_config:
            val = getattr(hf_config, attr, None)
        if val is not None:
            setattr(bridge_config, attr, val)

    # Gemma2: HF softcap field names differ from TL's.
    final_logit_softcapping = getattr(effective_config, "final_logit_softcapping", None)
    if final_logit_softcapping is not None:
        bridge_config.output_logits_soft_cap = float(final_logit_softcapping)
    logits_soft_cap = getattr(effective_config, "logits_soft_cap", None)
    if logits_soft_cap is not None:
        bridge_config.output_logits_soft_cap = float(logits_soft_cap)
    attn_logit_softcapping = getattr(effective_config, "attn_logit_softcapping", None)
    if attn_logit_softcapping is not None:
        bridge_config.attn_scores_soft_cap = float(attn_logit_softcapping)

    return bridge_config


def detect_tokenizer_bos_eos(tokenizer: Any) -> tuple[bool, bool]:
    """Detect whether the tokenizer prepends BOS and/or appends EOS.

    Non-empty test string — "" is unreliable with token aliasing.
    """
    encoded_test = tokenizer.encode("a")
    prepends_bos = (
        len(encoded_test) > 1
        and tokenizer.bos_token_id is not None
        and encoded_test[0] == tokenizer.bos_token_id
    )
    appends_eos = (
        len(encoded_test) > 1
        and tokenizer.eos_token_id is not None
        and encoded_test[-1] == tokenizer.eos_token_id
    )
    return prepends_bos, appends_eos


def build_bridge_from_module(
    model: nn.Module,
    architecture: str,
    *,
    hf_config: Optional[Any] = None,
    tl_config: Optional[TransformerBridgeConfig] = None,
    tokenizer: Optional[Any] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Any] = None,
    model_name: str = "external",
    post_adapter_hook: Optional[Callable[[ArchitectureAdapter], None]] = None,
) -> TransformerBridge:
    """Build a :class:`TransformerBridge` around a pre-loaded model.

    The bridge never moves, casts, or mutates the supplied model.

    Args:
        model: Any ``nn.Module`` whose submodule tree matches the adapter's
            expected dot-paths for ``architecture``.
        architecture: Architecture identifier registered in the
            ``ArchitectureAdapterFactory`` (e.g. ``"LlamaForCausalLM"``,
            ``"TransformerLensNative"``).
        hf_config: Optional HF-style config; translated via
            :func:`build_bridge_config_from_hf`. Mutually exclusive with ``tl_config``.
        tl_config: Optional pre-built :class:`TransformerBridgeConfig`; bypasses
            HF translation. Mutually exclusive with ``hf_config``.
        tokenizer: Optional tokenizer. If supplied, passes through
            ``setup_tokenizer`` and detects BOS/EOS behavior.
        dtype: Recorded on ``cfg.dtype``. Default ``None`` reads from the model's
            first parameter; explicit values override.
        device: Recorded on ``cfg.device``. Default ``None`` reads from the
            model's first parameter.
        model_name: Recorded on ``cfg.model_name``.
        post_adapter_hook: Optional callback invoked after adapter selection and
            before :meth:`adapter.prepare_model`. Source-specific overlays mutate
            ``component_mapping`` here.

    Returns:
        A :class:`TransformerBridge` wrapping the supplied model.
    """
    if hf_config is None and tl_config is None:
        raise ValueError(
            "build_bridge_from_module requires exactly one of hf_config or "
            "tl_config — the bridge needs config fields (d_model, n_heads, "
            "n_layers, ...) that can't be inferred from the model alone."
        )
    if hf_config is not None and tl_config is not None:
        raise ValueError(
            "build_bridge_from_module got both hf_config and tl_config; supply "
            "exactly one. hf_config triggers HF→bridge translation; tl_config "
            "bypasses it."
        )

    # Reading dtype from the model avoids silently lying about a bf16 model.
    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            dtype = torch.float32

    if tl_config is not None:
        # Defensive copy so adapter-init mutations (normalization_type, device,
        # ...) don't leak between bridges built from the same config.
        bridge_config = copy.deepcopy(tl_config)
        bridge_config.architecture = architecture
        if model_name != "external" or not getattr(bridge_config, "model_name", None):
            bridge_config.model_name = model_name
        bridge_config.dtype = dtype
    else:
        bridge_config = build_bridge_config_from_hf(hf_config, architecture, model_name, dtype)

    adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)

    if post_adapter_hook is not None:
        post_adapter_hook(adapter)

    if device is not None:
        adapter.cfg.device = str(device)
    else:
        try:
            adapter.cfg.device = str(next(model.parameters()).device)
        except StopIteration:
            adapter.cfg.device = "cpu"

    adapter.prepare_model(model)

    if tokenizer is not None:
        from transformer_lens.model_bridge.sources.transformers import setup_tokenizer

        default_padding_side = getattr(adapter.cfg, "default_padding_side", None)
        tokenizer = setup_tokenizer(tokenizer, default_padding_side=default_padding_side)
        (
            adapter.cfg.tokenizer_prepends_bos,
            adapter.cfg.tokenizer_appends_eos,
        ) = detect_tokenizer_bos_eos(tokenizer)

    return TransformerBridge(model, adapter, tokenizer)
