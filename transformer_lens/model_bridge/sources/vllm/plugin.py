"""vLLM plugin entry point.

Monkey-patches ``Worker.load_model`` to install capture hooks after weights
load and before ``compile_or_warm_up_model`` — the only window where hooks
make it into the compiled FX graph (PyTorch #117758).

Two hook flavors, selected by ``configure(enable_batching=...)``:

* Compiled (default): in-place writes to a pre-allocated GPU tensor; no
  ``.cpu()`` (illegal during CUDA-graph capture); SymInt-indexed slicing only
  (Python ``.shape`` access forces specialization). Single prompt.
  Interventions ride the same hook as an affine transform
  ``output = output * scale_buf + bias_buf`` (defaults identity). The driver
  swaps buffer contents between forwards via ``tl_set_interventions`` — the FX
  graph references the buffers, so swaps take effect without recompiling.
  Memory cost: the affine allocates a transient output-shape tensor per hook
  per forward, even at identity — peak forward memory ~1.5× capture-only.
  Branching to skip the affine would defeat the swap trick and break the graph.

* Batched (``enable_batching=True``, runs ``enforce_eager``): the hook reads
  per-request token boundaries via ``segment_by_request`` (only valid inside a
  forward, untraceable under compile — hence eager), slices each request's rows
  to CPU, and appends to per-(req_id, hook) accumulators across chunked-prefill
  forwards. Interventions apply directly to the tensor via ``_apply_op``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import torch

from .internals import segment_by_request
from .worker_extension import _apply_op

# Decoder layers return the fused-residual (mlp_delta, residual) 2-tuple, so
# their hooks materialize the sum (see _make_capture_hook); other modules don't.
_DECODER_LAYER_PATH = re.compile(r"^model\.layers\.\d+$")

# Transient signal driver → worker during LLM construction. Per-Worker buffers
# live on Worker instances so concurrent boot_vllm calls don't collide.
_config: Dict[str, Any] = {}
_install_patched = False
_orig_load_model = None


def configure(
    capture_specs: Dict[str, Tuple[str, int]],
    max_num_batched_tokens: int,
    dtype: torch.dtype,
    enable_batching: bool = False,
) -> None:
    """Set capture specs, buffer length, dtype, and hook flavor before ``LLM(...)``."""
    _config["capture_specs"] = capture_specs
    _config["max_num_batched_tokens"] = max_num_batched_tokens
    _config["dtype"] = dtype
    _config["enable_batching"] = enable_batching


def register() -> None:
    """Idempotent monkey-patch of ``Worker.load_model``.

    vLLM calls ``register()`` once per process at entry-points discovery. Idempotent
    so re-imports (notebook restarts, repeated ``boot_vllm`` in the same process)
    don't double-wrap.

    No ``unregister()`` symmetry: the patch stays for process lifetime. Benign
    because ``patched_load_model`` no-ops when ``_config["capture_specs"]`` is
    absent — and ``boot_vllm`` clears ``_config`` after each ``LLM(...)``, so
    any subsequent non-TL ``LLM(...)`` in the same process hits the no-op path.
    """
    global _install_patched, _orig_load_model
    if _install_patched:
        return
    from vllm.v1.worker.gpu_worker import Worker

    _orig_load_model = Worker.load_model

    def patched_load_model(self):
        _orig_load_model(self)
        specs = _config.get("capture_specs")
        if not specs:
            return  # not a TL-driven LLM; no hooks to install
        max_n = _config["max_num_batched_tokens"]
        dtype = _config["dtype"]
        enable_batching = _config.get("enable_batching", False)
        device = next(self.model_runner.model.parameters()).device

        # Detach prior handles before reassigning — vLLM doesn't double-load
        # today, but unconditional reassignment would orphan hooks if it ever did.
        for handle in getattr(self, "_tl_hook_handles", []):
            handle.remove()
        self._tl_buffers = {}
        self._tl_scale_buffers = {}
        self._tl_bias_buffers = {}
        self._tl_hook_handles = []
        # Batched-mode per-(req_id, hook) accumulators + global spec dict.
        self._tl_accum = {}
        self._tl_intervention_specs = {}
        # Shared counter — surfaces hook double-fire under compile via tl_read_counter.
        self._tl_fire_counter = torch.zeros(1, device=device, dtype=torch.int64)
        for canonical_name, (dot_path, width) in specs.items():
            target = self.model_runner.model
            for seg in dot_path.split("."):
                target = target[int(seg)] if seg.isdigit() else getattr(target, seg)
            # Decoder layers return vLLM's (mlp_delta, residual) tuple; the
            # hook materializes their sum so the capture semantically matches
            # HF's full residual stream. Other modules use the default path.
            materialize = bool(_DECODER_LAYER_PATH.match(dot_path))
            if enable_batching:
                handle = target.register_forward_hook(
                    _make_batched_hook(
                        self,
                        canonical_name,
                        self._tl_fire_counter,
                        materialize=materialize,
                    )
                )
            else:
                capture_buf = torch.zeros(max_n, width, device=device, dtype=dtype)
                # Affine identity at install. Driver swaps via tl_set_interventions
                # to enable suppress/scale/add/set ops between forwards.
                scale_buf = torch.ones(width, device=device, dtype=dtype)
                bias_buf = torch.zeros(width, device=device, dtype=dtype)
                self._tl_buffers[canonical_name] = capture_buf
                self._tl_scale_buffers[canonical_name] = scale_buf
                self._tl_bias_buffers[canonical_name] = bias_buf
                handle = target.register_forward_hook(
                    _make_capture_hook(
                        capture_buf,
                        scale_buf,
                        bias_buf,
                        self._tl_fire_counter,
                        materialize=materialize,
                    )
                )
            self._tl_hook_handles.append(handle)

    Worker.load_model = patched_load_model
    _install_patched = True


def _make_capture_hook(
    capture_buf: torch.Tensor,
    scale_buf: torch.Tensor,
    bias_buf: torch.Tensor,
    fire_counter: torch.Tensor,
    *,
    materialize: bool = False,
):
    """GPU-only, dynamic-shape-safe affine + capture into pre-allocated buffers.

    When ``materialize=True`` (decoder layers), treat the module's output as
    vLLM's fused-residual ``(mlp_delta, residual)`` tuple: capture
    ``mlp_delta + residual`` (the full residual stream, matching HF's
    blocks.{i}.hook_out semantics) and return ``(modified - residual, residual)``
    so the next layer's input_layernorm sees the same fused sum. Mutations
    propagate through both the capture and the downstream graph.

    ``fire_counter`` is incremented per call for the fire-once check.
    """

    @torch.no_grad()
    def hook(_module, _inputs, output):
        fire_counter.add_(1)
        if materialize and isinstance(output, tuple) and len(output) == 2:
            hidden, residual = output
            if isinstance(hidden, torch.Tensor) and isinstance(residual, torch.Tensor):
                t = hidden + residual
                modified = t * scale_buf + bias_buf
                n = t.shape[0]
                capture_buf.narrow(0, 0, n).copy_(modified)
                # Reconstructs ``modified`` in the next layer's fused norm: exact at
                # identity, bounded fp16 error under intervention.
                return (modified - residual, residual)

        tuple_tail: tuple = ()
        if isinstance(output, tuple):
            t = output[0]
            tuple_tail = output[1:]
        else:
            t = output
        if not isinstance(t, torch.Tensor):
            return None
        # Affine transform; default scale=1 / bias=0 means identity. Driver
        # swaps buffer contents to enable interventions.
        modified = t * scale_buf + bias_buf
        # narrow() keeps the dynamic shape; [:n] gets erased under fake-tensor
        # tracing and copy_ then sees the full buffer ("expand s72 -> max_n").
        n = t.shape[0]
        capture_buf.narrow(0, 0, n).copy_(modified)
        # Gate on isinstance, not truthy tuple_tail — a 1-tuple has an empty tail.
        if isinstance(output, tuple):
            return (modified,) + tuple_tail
        return modified

    return hook


def _make_batched_hook(
    worker: Any,
    canonical_name: str,
    fire_counter: torch.Tensor,
    *,
    materialize: bool = False,
):
    """Eager hook: intervene, then append each request's rows to ``worker._tl_accum``.

    Chunked prefill fires this once per chunk; appends are token-order for the
    later cat. ``materialize`` mirrors the compiled hook (fused-residual sum).
    """

    @torch.no_grad()
    def hook(_module, _inputs, output):
        fire_counter.add_(1)
        residual = None
        if materialize and isinstance(output, tuple) and len(output) == 2:
            hidden, residual = output
            if isinstance(hidden, torch.Tensor) and isinstance(residual, torch.Tensor):
                t = hidden + residual
            else:
                return None
            tuple_tail: tuple = ()
        elif isinstance(output, tuple):
            t = output[0]
            tuple_tail = output[1:]
        else:
            t = output
            tuple_tail = ()
        if not isinstance(t, torch.Tensor):
            return None

        modified = t
        for spec in getattr(worker, "_tl_intervention_specs", {}).values():
            modified = _apply_op(modified, spec)

        qsl, req_ids = segment_by_request(worker.model_runner)
        accum: Dict[tuple, list] = worker._tl_accum
        if qsl is None:
            # No per-request boundaries available — treat the batch as one request.
            req_id = req_ids[0] if req_ids else "0"
            accum.setdefault((req_id, canonical_name), []).append(modified.detach().cpu())
        else:
            for i, req_id in enumerate(req_ids):
                start, end = int(qsl[i]), int(qsl[i + 1])
                if end <= start:
                    continue
                chunk = modified[start:end].detach().cpu()
                accum.setdefault((req_id, canonical_name), []).append(chunk)

        if residual is not None:
            return (modified - residual, residual)
        if isinstance(output, tuple):
            return (modified,) + tuple_tail
        return modified

    return hook
