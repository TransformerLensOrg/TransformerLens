"""``boot_vllm`` — construct a vLLM LLM, wrap it in a RemoteBridge via VLLMDriver."""
from __future__ import annotations

import logging
import os
import threading
import warnings
from typing import Any, Dict, Optional

import torch

from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.remote_bridge import RemoteBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
    configure_tokenizer,
)
from transformer_lens.utilities.hf_utils import get_hf_token

from . import plugin
from .driver import VLLMDriver
from .internals import extract_hf_config
from .overlays import get_overlay

# Forced LLM(...) kwargs that the capture-hook design depends on. Caller override → ValueError.
# enable_prefix_caching MUST stay off: a prefix-cache hit makes the prefill compute only the
# uncached suffix, so hooks fire for a subset of positions — captures land row-misaligned,
# interventions skip cached positions, and an intervened forward writes poisoned K/V that a
# later clean forward on the same prompt would silently reuse.
_LOCKED_KWARGS = {
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "skip_tokenizer_init": True,
    "disable_log_stats": True,
    "enable_prefix_caching": False,
}

# Serializes the configure() → LLM(...) → clear() handoff: the spec channel is a module
# global, so interleaved boots would cross-wire capture specs between engines.
_BOOT_LOCK = threading.Lock()


def boot_vllm(
    model_name: str,
    tokenizer: Optional[Any] = None,
    dtype: Optional[torch.dtype] = None,
    gpu_memory_utilization: float = 0.5,
    max_model_len: Optional[int] = None,
    max_num_batched_tokens: int = 2048,
    enable_batching: bool = False,
    enable_position_interventions: bool = False,
    **vllm_kwargs: Any,
) -> RemoteBridge:
    """Boot a model via vLLM and wrap it in a :class:`RemoteBridge` via :class:`VLLMDriver`.

    vLLM drives the forward pass (PagedAttention + ``torch.compile`` + CUDA graphs).
    Capture buffers are populated by hooks the plugin installs pre-compile inside
    the worker; they come back via ``collective_rpc`` and replay through the
    bridge's HookPoint tree.

    **Scope vs vllm-lens:** vllm-lens is observation-only. This source extends to
    observation + spec-vocabulary *mutation* — each capture hook also applies an
    affine transform ``output = output * scale + bias`` (default identity), so
    interventions (``suppress`` / ``scale`` / ``add`` / ``set``) propagate to
    downstream layers. The hook's return value replaces the module output per
    PyTorch ``register_forward_hook`` semantics. The mutation path under
    torch.compile + CUDA graphs is exercised end-to-end by
    ``demos/vLLM_Bridge_Integration_Test.ipynb`` (a manual GPU run, not CI);
    unit tests cover the dispatch protocol only.

    Some captures use vLLM-native conventions that differ from HF/HT; see
    :mod:`transformer_lens.model_bridge.sources.vllm.overlays.decoder_only` for
    which hooks diverge and the conversion to apply for HT-equivalent values.

    **Returned logits are reconstructed full-sequence logits.** vLLM's sampler
    bypasses ``lm_head``, so the driver rebuilds real logits host-side as
    ``ln_final @ lm_head.weight.T`` (+ bias, + Gemma soft-cap) from the captured
    final-norm activation — valid at every position, so ``return_type`` in
    ``{"loss", "both"}`` works. If the unembedding weight is unreachable the
    driver falls back to the sampler's final-position log-probs (earlier
    positions ``-inf``), declares ``provides_sequence_logits=False``, and the
    bridge then rejects loss.

    GPU memory cost: each capture buffer is ``max_num_batched_tokens × width`` at
    the model's dtype. For Llama-3.2-1B at fp16 with ``max_num_batched_tokens=2048``,
    the unembed buffer alone is ~525 MB (2048 × 128256 × 2 bytes); residual-stream
    buffers add ~8 MB per hook. The affine intervention hook also allocates a
    transient output-shape tensor per forward (even in identity mode), so peak
    forward memory is ~1.5× the capture buffers' resident size.

    KV-cache footprint: vLLM reserves KV cache sized for ``max_model_len`` × layers
    × heads × head_dim. If ``max_model_len`` is left as ``None``, vLLM uses the
    model's native context (e.g. 131072 for Llama-3.2-1B) — easily 4+ GiB even
    on a 1B model. Pass an explicit ``max_model_len`` (e.g. ``2048`` for typical
    mech-interp prompts) to keep the budget on smaller GPUs.

    ``enable_batching`` switches to the eager batched path (``enforce_eager``,
    ``batch_size > 1``) — the throughput path for SAE/probe data collection.
    Default ``False`` keeps the compile-validated single-prompt path. Batched
    caches are right-padded with zeros to the longest sequence.

    ``enable_position_interventions`` widens each hook's affine scale/bias buffers
    from ``(width,)`` to ``(max_num_batched_tokens, width)`` so an intervention spec
    can carry a ``pos`` field (int or list[int]) that scopes the edit to specific
    sequence positions — position-scoped activation patching / tensor injection.
    Costs ~2× extra resident GPU memory across all hooks (the scale and bias buffers
    join the already-``(max_n, width)`` capture buffer), so it is opt-in and defaults
    ``False``. Compiled-path only — incompatible with ``enable_batching``.
    """
    if enable_position_interventions and enable_batching:
        raise ValueError(
            "enable_position_interventions requires the compiled path and is incompatible "
            "with enable_batching=True (the batched/eager path has no affine buffers)."
        )
    _reject_locked_overrides(vllm_kwargs)
    # Import-check first: fail with an actionable message before any network I/O
    # or plugin state mutation.
    try:
        from vllm import LLM
    except ImportError as exc:
        raise ImportError(
            "boot_vllm requires vLLM (Linux + CUDA). Install with "
            'pip install "transformer-lens[vllm]" or uv sync --extra vllm; '
            "the driver is validated against vllm 0.20.x."
        ) from exc

    from transformers import AutoConfig, AutoTokenizer

    # Resolve architecture WITHOUT loading weights so we can tell the plugin
    # which dot-paths to hook before LLM(...) constructs the worker.
    hf_token = get_hf_token()
    hf_config_preview = AutoConfig.from_pretrained(model_name, token=hf_token)
    architecture = hf_config_preview.architectures[0]
    overlay = get_overlay(architecture)

    resolved_dtype = dtype or _dtype_from_hf_config(hf_config_preview)

    # The plugin's _config singleton must be visible to the worker, which only
    # holds with single-process execution. Multi-GPU is unsupported. Override
    # any user setting — silent capture failure otherwise.
    existing_mp = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    if existing_mp not in (None, "0"):
        warnings.warn(
            f"VLLM_ENABLE_V1_MULTIPROCESSING={existing_mp!r} overridden to '0' — "
            "boot_vllm needs single-process execution for capture hooks to install. "
            "Multi-GPU vLLM is unsupported.",
            UserWarning,
            stacklevel=2,
        )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Batched capture reads query_start_loc from the forward context, untraceable
    # under torch.compile — so the batched path must run eager.
    eager_kwargs: Dict[str, Any] = {"enforce_eager": True} if enable_batching else {}

    with _BOOT_LOCK:
        plugin.configure(
            capture_specs=overlay.capture_specs(hf_config_preview),
            max_num_batched_tokens=max_num_batched_tokens,
            dtype=resolved_dtype,
            enable_batching=enable_batching,
            enable_position_interventions=enable_position_interventions,
        )
        plugin.register()
        try:
            llm = LLM(
                model=model_name,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                # Critical: vLLM defaults max_num_batched_tokens to 8192 for chunked prefill.
                # We size our capture buffer to this same value, so vLLM must compile for
                # the matching dynamic-shape range — otherwise Dynamo's symbolic-shape
                # hint exceeds the buffer dim and narrow() fails at compile time.
                max_num_batched_tokens=max_num_batched_tokens,
                # Register TLWorkerExtension so its tl_* methods are reachable via
                # collective_rpc. Passed as a dotted path; vLLM imports the class at
                # worker construction and mixes it into the Worker via multiple
                # inheritance (asserts no attribute name collisions — hence the tl_ prefix).
                worker_extension_cls="transformer_lens.model_bridge.sources.vllm.worker_extension.TLWorkerExtension",
                # Allow full-vocab logprobs so the fallback path can synthesize logits when
                # host-side reconstruction is unavailable (vLLM caps logprobs to this value;
                # default 20 is too small for mech-interp).
                max_logprobs=hf_config_preview.vocab_size,
                # Always explicit — vLLM's "auto" downcasts fp32 checkpoints to fp16, which
                # would leave the capture/affine buffers (allocated at resolved_dtype) at a
                # different dtype than the engine's activations.
                dtype=str(resolved_dtype).replace("torch.", ""),
                **eager_kwargs,
                **_LOCKED_KWARGS,
                **vllm_kwargs,
            )
        finally:
            # Always clear, even on a failed boot: a stale populated _config would make the
            # next in-process vllm.LLM(...) walk our dot-paths on a foreign model.
            plugin._config.clear()

    # Capture-path validity is enforced inside patched_load_model during
    # LLM(...) above — any missing dot-path raises AttributeError there. By
    # the time we reach this line every spec has already been walked successfully.
    hf_config = extract_hf_config(llm)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    # Build the adapter (RemoteBridge skips adapter.prepare_model — there's no
    # local model tree to walk). Use the shared HF→TL config builder so
    # bridge_config is a real TransformerBridgeConfig with all dataclass
    # defaults (d_vocab_out=-1, etc.) — not a deep-copied HF config with
    # extra fields, which is missing the TL-only attributes BridgeCore reads.
    bridge_config = build_bridge_config_from_hf(hf_config, architecture, model_name, resolved_dtype)
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)

    # Match boot_transformers' tokenizer setup so to_tokens(str) is token-identical.
    tokenizer = configure_tokenizer(tokenizer, adapter.cfg)

    driver = VLLMDriver(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        overlay=overlay,
        hf_config=hf_config,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_batching=enable_batching,
        enable_position_interventions=enable_position_interventions,
    )
    # One-time unembedding fetch: caches the weight for per-forward reconstruction and
    # downgrades provides_sequence_logits honestly if no unembedding is reachable.
    driver.probe_logit_reconstruction()
    bridge = RemoteBridge(adapter=adapter, tokenizer=tokenizer, driver=driver)
    _log_hook_summary(model_name, architecture, driver)
    return bridge


def _reject_locked_overrides(vllm_kwargs: Dict[str, Any]) -> None:
    for key, locked in _LOCKED_KWARGS.items():
        if key in vllm_kwargs and vllm_kwargs[key] != locked:
            raise ValueError(
                f"boot_vllm forces {key}={locked}; caller passed {key}={vllm_kwargs[key]}. "
                "Multi-device, prefix caching, continuous batching, and vLLM-owned "
                "tokenizers are unsupported — each breaks the row=position capture invariant."
            )


def _dtype_from_hf_config(hf_config: Any) -> torch.dtype:
    raw = getattr(hf_config, "torch_dtype", None)
    if isinstance(raw, torch.dtype):
        return raw
    if isinstance(raw, str):
        return getattr(torch, raw, torch.float16)
    return torch.float16


def _log_hook_summary(model_name: str, architecture: str, driver: VLLMDriver) -> None:
    """Log the fireable and non-fireable hook sets so users don't have to grep."""
    log = logging.getLogger("transformer_lens.vllm")
    fireable = sorted(driver.supported_hook_points)
    log.info(
        "vLLM source on %s (%s) captures %d hook(s): %s",
        model_name,
        architecture,
        len(fireable),
        ", ".join(fireable),
    )
    nonfiring = sorted(driver.non_fireable_hook_points)
    if nonfiring:
        log.info(
            "vLLM source on %s (%s) cannot fire %d hook(s) (vLLM fuses these): %s. "
            "Use boot_transformers() if you need them.",
            model_name,
            architecture,
            len(nonfiring),
            ", ".join(nonfiring),
        )


__all__ = ["boot_vllm"]
