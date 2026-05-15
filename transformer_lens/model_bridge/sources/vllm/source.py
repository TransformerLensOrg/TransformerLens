"""``boot_vllm`` — construct an LLM, wire its inner module to a TransformerBridge."""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)

from . import plugin
from .internals import extract_hf_config, extract_inner_model
from .overlays import VLLM_OVERLAYS

# Forced LLM(...) kwargs locked by the v1 plan. Caller override → ValueError.
_LOCKED_KWARGS = {
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "skip_tokenizer_init": True,
    "disable_log_stats": True,
}


def boot_vllm(
    model_name: str,
    tokenizer: Optional[Any] = None,
    dtype: Optional[torch.dtype] = None,
    gpu_memory_utilization: float = 0.5,
    max_model_len: Optional[int] = None,
    max_num_batched_tokens: int = 2048,
    **vllm_kwargs: Any,
) -> TransformerBridge:
    """Boot a model via vLLM and wrap its inner ``nn.Module`` in a TransformerBridge.

    vLLM drives the forward pass (PagedAttention + ``torch.compile`` + CUDA graphs).
    The bridge's activation cache is populated from GPU buffers the plugin's
    pre-compile hooks write into.

    Args:
        model_name: HF Hub repo id; e.g. ``"meta-llama/Llama-3.2-1B"``.
        tokenizer: Optional pre-built tokenizer. If ``None``, an HF tokenizer is loaded.
        dtype: Model dtype; default ``None`` lets vLLM pick (``"auto"``).
        gpu_memory_utilization: vLLM's fraction of GPU memory. Default 0.5 leaves
            room for SAEs / probes alongside.
        max_model_len: vLLM context length cap. ``None`` uses the model default.
        max_num_batched_tokens: Capture-buffer length. Caps the longest single
            ``generate`` call supported under capture. Lower → less GPU memory for
            capture buffers; higher → supports longer prompts.
        **vllm_kwargs: Passthrough to ``vllm.LLM(...)``. Locked kwargs raise
            :class:`ValueError` on override.

    Returns:
        A wired :class:`TransformerBridge`.
    """
    _reject_locked_overrides(vllm_kwargs)

    from transformers import AutoConfig, AutoTokenizer

    # Resolve architecture WITHOUT loading weights so we can tell the plugin
    # which dot-paths to hook before LLM(...) constructs the worker.
    hf_config_preview = AutoConfig.from_pretrained(model_name)
    architecture = hf_config_preview.architectures[0]
    overlay = VLLM_OVERLAYS.get(architecture)
    if overlay is None:
        raise NotImplementedError(
            f"No vLLM overlay registered for {architecture}. "
            f"Supported: {sorted(VLLM_OVERLAYS)}."
        )

    # Resolve dtype now so the plugin pre-allocates buffers at the right precision.
    resolved_dtype = dtype or _dtype_from_hf_config(hf_config_preview)
    plugin.reset()
    plugin.configure(
        capture_specs=overlay.capture_specs(hf_config_preview),
        max_num_batched_tokens=max_num_batched_tokens,
        dtype=resolved_dtype,
    )
    plugin.register()

    # In-process engine core. The plugin's _config singleton must be visible to
    # the worker, which only holds with single-process execution. Multi-process /
    # multi-GPU is a v2 concern — the plugin would marshal _config differently then.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm import LLM

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=str(resolved_dtype).replace("torch.", "") if dtype is not None else "auto",
        **_LOCKED_KWARGS,
        **vllm_kwargs,
    )

    inner = extract_inner_model(llm)
    hf_config = extract_hf_config(llm)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    bridge = build_bridge_from_module(
        model=inner,
        architecture=architecture,
        hf_config=hf_config,
        tokenizer=tokenizer,
        dtype=next(inner.parameters()).dtype,
        device=str(next(inner.parameters()).device),
        model_name=model_name,
        post_adapter_hook=overlay.apply,
    )

    # Lifetime anchor — bypass nn.Module registration so vLLM's parameter trees
    # aren't double-registered (mirrors original_model at bridge.py:144).
    bridge.__dict__["_vllm_engine"] = llm
    bridge._forward_impl = _make_vllm_forward(llm, bridge)
    _warn_nonfiring_hooks(model_name, architecture, overlay, hf_config)
    return bridge


def _reject_locked_overrides(vllm_kwargs: Dict[str, Any]) -> None:
    for key, locked in _LOCKED_KWARGS.items():
        if key in vllm_kwargs and vllm_kwargs[key] != locked:
            raise ValueError(
                f"boot_vllm forces {key}={locked}; caller passed {key}={vllm_kwargs[key]}. "
                f"Multi-device / continuous batching / vLLM-owned tokenizer are v2 scope."
            )


def _dtype_from_hf_config(hf_config: Any) -> torch.dtype:
    raw = getattr(hf_config, "torch_dtype", None)
    if isinstance(raw, torch.dtype):
        return raw
    if isinstance(raw, str):
        return getattr(torch, raw, torch.float16)
    return torch.float16


def _make_vllm_forward(llm: Any, bridge: TransformerBridge) -> Callable:
    """Build the closure assigned to ``bridge._forward_impl``.

    Captures vLLM-side activations via collective_rpc, replays them through the
    bridge's HookPoint registry so ``run_with_cache`` populates correctly, and
    returns the logits as a tensor.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    hook_registry = bridge._hook_registry

    def forward(input_ids: Any, **kwargs: Any) -> torch.Tensor:
        if isinstance(input_ids, torch.Tensor):
            ids_list = input_ids.tolist()
        else:
            ids_list = list(input_ids)
        if ids_list and isinstance(ids_list[0], list):
            if len(ids_list) != 1:
                raise NotImplementedError("boot_vllm v1 supports batch_size=1 only.")
            ids_list = ids_list[0]

        max_new = int(kwargs.get("max_new_tokens", 1))
        llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=ids_list)],
            sampling_params=SamplingParams(max_tokens=max_new, temperature=0.0),
        )

        n_tokens = len(ids_list)
        captures = llm.collective_rpc("tl_read_captures", args=([n_tokens],))[0]

        device = next(bridge.original_model.parameters()).device
        for canonical_name, tensor in captures.items():
            hookpoint = hook_registry.get(canonical_name)
            if hookpoint is None:
                continue
            hookpoint(tensor.to(device).unsqueeze(0))

        logits = captures.get("unembed.hook_out")
        if logits is None:
            raise RuntimeError(
                "vLLM source captured no unembed.hook_out; overlay capture_specs "
                "must include it."
            )
        logits = logits.to(device).unsqueeze(0)  # (1, n_tokens, d_vocab)
        return logits

    return forward


def _warn_nonfiring_hooks(model_name: str, architecture: str, overlay: Any, hf_config: Any) -> None:
    non_firing = overlay.nonfiring_hooks()
    if not non_firing:
        return
    n_layers = getattr(hf_config, "num_hidden_layers", None)
    if isinstance(n_layers, int) and n_layers > 0:
        expanded = [name.replace("{i}", f"0..{n_layers - 1}") for name in non_firing]
    else:
        expanded = list(non_firing)
    logging.getLogger("transformer_lens.vllm").info(
        "vLLM source on %s (%s): the following hooks will not fire (vLLM fuses these): %s. "
        "Use boot_transformers() if you need them.",
        model_name,
        architecture,
        ", ".join(expanded),
    )


__all__ = ["boot_vllm"]
