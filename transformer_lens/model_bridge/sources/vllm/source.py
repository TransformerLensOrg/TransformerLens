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
from .internals import extract_hf_config, verify_hook_coverage
from .overlays import get_overlay
from .worker_extension import dtype_name

# Forced LLM(...) kwargs that the capture-hook design depends on. Caller override → ValueError.
# enable_prefix_caching MUST stay off: a prefix-cache hit makes the prefill compute only the
# uncached suffix, so hooks fire for a subset of positions — captures land row-misaligned,
# interventions skip cached positions, and an intervened forward writes poisoned K/V that a
# later clean forward on the same prompt would silently reuse.
_LOCKED_KWARGS = {
    "skip_tokenizer_init": True,
    "disable_log_stats": True,
    "enable_prefix_caching": False,
}

# Serializes the configure() → LLM(...) → clear() handoff: the spec channel is a
# process-wide env var, so interleaved boots would cross-wire capture specs between engines.
_BOOT_LOCK = threading.Lock()

_WORKER_EXTENSION_CLS = (
    "transformer_lens.model_bridge.sources.vllm.worker_extension.TLWorkerExtension"
)


def construct_instrumented_llm(
    model_name: str,
    *,
    capture_specs: Dict[str, Any],
    max_num_batched_tokens: int,
    dtype: torch.dtype,
    enable_batching: bool = False,
    enable_position_interventions: bool = False,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    llm_kwargs: Dict[str, Any],
) -> Any:
    """The one place a TL-instrumented ``vllm.LLM`` is constructed — shared by
    ``boot_vllm`` and the Inspect vLLM provider so the capture contract can't drift
    between them. Owns everything hook correctness depends on:

    - ``VLLM_DISABLE_COMPILE_CACHE=1``: our hooks are traced INTO vLLM's compiled
      graph, but its compile cache is keyed only on its own config — a cached
      artifact from a differently-instrumented process either crashes at AOT load
      (bytecode binds the hook closures) or silently serves a hookless graph.
    - ``VLLM_ENABLE_V1_MULTIPROCESSING``: forced ``"0"`` single-rank (in-process
      worker, the historical GPU-validated path); parallel boots spawn workers,
      which read the specs via the plugin's env channel — and must not inherit a
      stale ``"0"`` that would force the uni-process executor.
    - ``enable_prefix_caching=False``: a prefix-cache hit computes only the uncached
      suffix, so captures land row-misaligned, interventions skip cached positions,
      and an intervened forward writes poisoned K/V a later clean forward reuses.
    - configure → register → ``LLM(...)`` → clear under ``_BOOT_LOCK``, then a
      hook-coverage check so a spec that landed on no rank fails here, not as zeros.

    ``llm_kwargs`` carries the caller's remaining ``LLM(...)`` arguments; restating a
    contract kwarg is allowed only at the same value (callers validate overrides).
    """
    from vllm import LLM

    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    if tensor_parallel_size == 1 and pipeline_parallel_size == 1:
        existing_mp = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
        if existing_mp not in (None, "0"):
            warnings.warn(
                f"VLLM_ENABLE_V1_MULTIPROCESSING={existing_mp!r} overridden to '0' — "
                "single-rank TL boots keep the worker in-process.",
                UserWarning,
                stacklevel=3,
            )
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    elif os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0":
        os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING")

    with _BOOT_LOCK:
        plugin.configure(
            capture_specs=capture_specs,
            max_num_batched_tokens=max_num_batched_tokens,
            dtype=dtype,
            enable_batching=enable_batching,
            enable_position_interventions=enable_position_interventions,
        )
        plugin.register()
        try:
            llm = LLM(
                **{
                    "model": model_name,
                    # vLLM defaults this to 8192 for chunked prefill; the capture buffers
                    # are sized to it, so the compiled dynamic-shape range must match —
                    # otherwise Dynamo's symbolic hint exceeds the buffer dim at compile.
                    "max_num_batched_tokens": max_num_batched_tokens,
                    # Mixes tl_* methods into the Worker for collective_rpc (multiple
                    # inheritance; the tl_ prefix avoids attribute collisions).
                    "worker_extension_cls": _WORKER_EXTENSION_CLS,
                    "enable_prefix_caching": False,
                    "tensor_parallel_size": tensor_parallel_size,
                    "pipeline_parallel_size": pipeline_parallel_size,
                    **llm_kwargs,
                }
            )
        finally:
            # Always clear, even on a failed boot: stale specs would make the next
            # in-process vllm.LLM(...) walk our dot-paths on a foreign model.
            plugin.clear_config()

    # Hook installation skips modules absent on a rank (PP shards); a spec that
    # landed on NO rank is a broken dot-path and must fail here, not read zeros.
    verify_hook_coverage(llm)
    return llm


def boot_vllm(
    model_name: str,
    tokenizer: Optional[Any] = None,
    dtype: Optional[torch.dtype] = None,
    gpu_memory_utilization: float = 0.5,
    max_model_len: Optional[int] = None,
    max_num_batched_tokens: int = 2048,
    enable_batching: bool = False,
    enable_position_interventions: bool = False,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
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

    ``tensor_parallel_size`` > 1 enables single-node tensor parallelism. Every
    capture point the overlay hooks is post-all-reduce and therefore replicated
    across a stage's TP ranks; ``pipeline_parallel_size`` > 1 shards layers across
    stages, each owning the hooks that actually fire on it.
    Capture reads merge across ranks with a first-forward layout check that fails
    loud on installation drift or non-replicated hook points; sharded/stage-local
    weights (``lm_head``, final norm) are gathered for logit reconstruction and the
    ln_final un-fold. Single-node only (the spec channel is an env var, which never
    reaches Ray remote workers); both are incompatible with ``enable_batching``
    (per-rank chunk boundaries are unvalidated).
    """
    if enable_position_interventions and enable_batching:
        raise ValueError(
            "enable_position_interventions requires the compiled path and is incompatible "
            "with enable_batching=True (the batched/eager path has no affine buffers)."
        )
    for kwarg_name, size in (
        ("tensor_parallel_size", tensor_parallel_size),
        ("pipeline_parallel_size", pipeline_parallel_size),
    ):
        if not isinstance(size, int) or size < 1:
            raise ValueError(f"{kwarg_name} must be a positive int; got {size!r}.")
    if (tensor_parallel_size > 1 or pipeline_parallel_size > 1) and enable_batching:
        raise ValueError(
            "enable_batching=True with tensor/pipeline parallelism is unsupported: the "
            "eager batched path's per-rank chunk boundaries are unvalidated. "
            "Use the compiled single-prompt path."
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

    # Batched capture reads query_start_loc from the forward context, untraceable
    # under torch.compile — so the batched path must run eager.
    eager_kwargs: Dict[str, Any] = {"enforce_eager": True} if enable_batching else {}

    llm = construct_instrumented_llm(
        model_name,
        capture_specs=overlay.capture_specs(hf_config_preview),
        max_num_batched_tokens=max_num_batched_tokens,
        dtype=resolved_dtype,
        enable_batching=enable_batching,
        enable_position_interventions=enable_position_interventions,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        llm_kwargs={
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            # Allow full-vocab logprobs so the fallback path can synthesize logits when
            # host-side reconstruction is unavailable (vLLM caps logprobs to this value;
            # default 20 is too small for mech-interp).
            "max_logprobs": hf_config_preview.vocab_size,
            # Always explicit — vLLM's "auto" downcasts fp32 checkpoints to fp16, which
            # would leave the capture/affine buffers (allocated at resolved_dtype) at a
            # different dtype than the engine's activations.
            "dtype": dtype_name(resolved_dtype),
            **eager_kwargs,
            **_LOCKED_KWARGS,
            **vllm_kwargs,
        },
    )
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
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
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
                "Prefix caching, continuous batching, and vLLM-owned tokenizers are "
                "unsupported — each breaks the capture-read invariants. (TP/PP are "
                "supported via the tensor_parallel_size/pipeline_parallel_size kwargs.)"
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


__all__ = ["boot_vllm", "construct_instrumented_llm"]
