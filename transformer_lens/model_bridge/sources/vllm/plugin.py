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

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

import torch

from .internals import segment_by_request
from .worker_extension import _apply_op

# Decoder layers return the fused-residual (mlp_delta, residual) 2-tuple, so
# their hooks materialize the sum (see _make_capture_hook); other modules don't.
_DECODER_LAYER_PATH = re.compile(r"^model\.layers\.\d+$")

# Transient signal driver → worker during LLM construction. Per-Worker buffers
# live on Worker instances so concurrent boot_vllm calls don't collide.
# ``configure`` mirrors it into the env var so SPAWNED worker processes (TP>1 /
# VLLM_ENABLE_V1_MULTIPROCESSING=1), which re-import this module and see an empty
# global, still receive the specs — vLLM runs ``register()`` in every worker, so
# the patch itself propagates; only this payload needs a cross-process channel.
# Single-node only: env vars don't reach Ray remote workers.
_config: Dict[str, Any] = {}
_ENV_CONFIG_KEY = "TL_VLLM_PLUGIN_CONFIG"
_install_patched = False
_orig_load_model = None


def configure(
    capture_specs: Dict[str, Tuple[str, int]],
    max_num_batched_tokens: int,
    dtype: torch.dtype,
    enable_batching: bool = False,
    enable_position_interventions: bool = False,
) -> None:
    """Set capture specs, buffer length, dtype, and hook flavor before ``LLM(...)``."""
    _config["capture_specs"] = capture_specs
    _config["max_num_batched_tokens"] = max_num_batched_tokens
    _config["dtype"] = dtype
    _config["enable_batching"] = enable_batching
    _config["enable_position_interventions"] = enable_position_interventions
    os.environ[_ENV_CONFIG_KEY] = _serialize_config(_config)


def clear_config() -> None:
    """Clear both spec channels (module global + env var). Boot sites call this in a
    ``finally`` after ``LLM(...)`` so a later non-TL engine can't inherit the specs."""
    _config.clear()
    os.environ.pop(_ENV_CONFIG_KEY, None)


def _serialize_config(config: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "capture_specs": {
                name: [path, width] for name, (path, width) in config["capture_specs"].items()
            },
            "max_num_batched_tokens": config["max_num_batched_tokens"],
            "dtype": str(config["dtype"]).removeprefix("torch."),
            "enable_batching": config["enable_batching"],
            "enable_position_interventions": config["enable_position_interventions"],
        }
    )


def _deserialize_config(raw: str) -> Dict[str, Any]:
    data = json.loads(raw)
    dtype = getattr(torch, data["dtype"], None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{_ENV_CONFIG_KEY}: {data['dtype']!r} is not a torch dtype.")
    return {
        "capture_specs": {
            name: (path, int(width)) for name, (path, width) in data["capture_specs"].items()
        },
        "max_num_batched_tokens": int(data["max_num_batched_tokens"]),
        "dtype": dtype,
        "enable_batching": bool(data["enable_batching"]),
        "enable_position_interventions": bool(data["enable_position_interventions"]),
    }


def _active_config() -> Optional[Dict[str, Any]]:
    """The module global in-process; the env-var fallback in spawned workers."""
    if _config.get("capture_specs"):
        return _config
    raw = os.environ.get(_ENV_CONFIG_KEY)
    if raw:
        return _deserialize_config(raw)
    return None


def _resolve_dot_path(root: Any, dot_path: str) -> Any:
    """Walk a spec's dot-path; ``None`` when any segment is missing. Per-rank absence
    is legal under pipeline parallelism (each rank owns a layer subset) — the boot
    site verifies every hook landed on at least one rank."""
    target = root
    for seg in dot_path.split("."):
        if seg.isdigit():
            try:
                target = target[int(seg)]
            except (IndexError, KeyError, TypeError):
                return None
        else:
            target = getattr(target, seg, None)
        if target is None:
            return None
    return target


def register() -> None:
    """Idempotent monkey-patch of ``Worker.load_model``.

    vLLM calls ``register()`` once per process at entry-points discovery. Idempotent
    so re-imports (notebook restarts, repeated ``boot_vllm`` in the same process)
    don't double-wrap.

    No ``unregister()`` symmetry: the patch stays for process lifetime. Benign
    because ``patched_load_model`` no-ops when no spec channel is populated — and
    boot sites call ``clear_config()`` after each ``LLM(...)``, so any subsequent
    non-TL ``LLM(...)`` in the same process hits the no-op path.
    """
    global _install_patched, _orig_load_model
    if _install_patched:
        return
    from vllm.v1.worker.gpu_worker import Worker

    _orig_load_model = Worker.load_model

    def patched_load_model(self):
        _orig_load_model(self)
        config = _active_config()
        if config is None:
            return  # not a TL-driven LLM; no hooks to install
        specs = config["capture_specs"]
        max_n = config["max_num_batched_tokens"]
        dtype = config["dtype"]
        enable_batching = config.get("enable_batching", False)
        # Per-position affine buffers are (max_n, width) instead of (width,), so each
        # sequence row can carry a distinct scale/bias (position-scoped patching).
        per_position = config.get("enable_position_interventions", False)
        device = next(self.model_runner.model.parameters()).device

        # Detach prior handles before reassigning — vLLM doesn't double-load
        # today, but unconditional reassignment would orphan hooks if it ever did.
        for handle in getattr(self, "_tl_hook_handles", []):
            handle.remove()
        self._tl_buffers = {}
        self._tl_scale_buffers = {}
        self._tl_bias_buffers = {}
        # Per-hook first-write-wins flag (compiled mode). 0 = open (next forward captures),
        # 1 = closed (subsequent forwards self-copy and don't overwrite). Driver opens via
        # tl_reset_capture_flags before any capture-needing call so a multi-token generate's
        # prefill captures cleanly and decode steps don't overwrite row 0.
        self._tl_capture_flags = {}
        self._tl_hook_handles = []
        # Batched-mode per-(req_id, hook) accumulators + global spec dict.
        self._tl_accum = {}
        self._tl_intervention_specs = {}
        # Specs whose module isn't on this rank (PP layer shards); the boot site
        # verifies via tl_absent_hooks that every spec landed somewhere.
        self._tl_absent_hooks = set()
        # Shared counter — surfaces hook double-fire under compile via tl_read_counter.
        self._tl_fire_counter = torch.zeros(1, device=device, dtype=torch.int64)
        for canonical_name, (dot_path, width) in specs.items():
            target = _resolve_dot_path(self.model_runner.model, dot_path)
            if target is None:
                self._tl_absent_hooks.add(canonical_name)
                continue
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
                # to enable suppress/scale/add/set ops between forwards. Shape is
                # (max_n, width) when position interventions are enabled so each row
                # can differ; (width,) otherwise (broadcast across all positions).
                affine_shape = (max_n, width) if per_position else (width,)
                scale_buf = torch.ones(affine_shape, device=device, dtype=dtype)
                bias_buf = torch.zeros(affine_shape, device=device, dtype=dtype)
                # Default closed — opened explicitly by tl_reset_capture_flags for the
                # next forward(s) that need to capture.
                capture_flag = torch.ones(1, device=device, dtype=torch.int64)
                self._tl_buffers[canonical_name] = capture_buf
                self._tl_scale_buffers[canonical_name] = scale_buf
                self._tl_bias_buffers[canonical_name] = bias_buf
                self._tl_capture_flags[canonical_name] = capture_flag
                handle = target.register_forward_hook(
                    _make_capture_hook(
                        capture_buf,
                        scale_buf,
                        bias_buf,
                        self._tl_fire_counter,
                        capture_flag,
                        materialize=materialize,
                        per_position=per_position,
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
    capture_flag: torch.Tensor,
    *,
    materialize: bool = False,
    per_position: bool = False,
):
    """GPU-only, dynamic-shape-safe affine + first-write-wins capture into pre-allocated buffers.

    When ``materialize=True`` (decoder layers), treat the module's output as
    vLLM's fused-residual ``(mlp_delta, residual)`` tuple: capture
    ``mlp_delta + residual`` (the full residual stream, matching HF's
    blocks.{i}.hook_out semantics) and return ``(modified - residual, residual)``
    so the next layer's input_layernorm sees the same fused sum. Mutations
    propagate through both the capture and the downstream graph.

    ``capture_flag`` (0 = open, 1 = closed) gates the buffer write — driver opens it
    via ``tl_reset_capture_flags`` before each capture-needing forward, the hook closes
    it on first fire, so a multi-token generate's prefill captures cleanly and decode
    steps self-copy (no overwrite). Interventions still apply on every forward
    regardless of the flag — the gate only affects the capture write.

    ``per_position`` (compile-time constant): when set, ``scale_buf``/``bias_buf`` are
    ``(max_n, width)`` and the affine is row-scoped (``buf.narrow(0, 0, n)``) so a
    ``pos``-scoped intervention edits only its rows; otherwise they are ``(width,)``
    and broadcast across every position.

    ``fire_counter`` is incremented per call for the fire-once check.
    """

    def _affine(t: torch.Tensor, n: Any) -> torch.Tensor:
        # per_position is a closure constant → torch.compile specializes this branch.
        # narrow(0, 0, n) keeps the SymInt dynamic shape (same trick as _gated_capture);
        # the (width,) path broadcasts across all rows.
        if per_position:
            return t * scale_buf.narrow(0, 0, n) + bias_buf.narrow(0, 0, n)
        return t * scale_buf + bias_buf

    @torch.no_grad()
    def hook(_module, _inputs, output):
        fire_counter.add_(1)
        if materialize and isinstance(output, tuple) and len(output) == 2:
            hidden, residual = output
            if isinstance(hidden, torch.Tensor) and isinstance(residual, torch.Tensor):
                n = hidden.shape[0]
                modified = _affine(hidden + residual, n)
                _gated_capture(capture_buf, n, modified, capture_flag)
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
        n = t.shape[0]
        modified = _affine(t, n)
        _gated_capture(capture_buf, n, modified, capture_flag)
        # Gate on isinstance, not truthy tuple_tail — a 1-tuple has an empty tail.
        if isinstance(output, tuple):
            return (modified,) + tuple_tail
        return modified

    return hook


def _gated_capture(
    capture_buf: torch.Tensor, n: Any, modified: torch.Tensor, capture_flag: torch.Tensor
) -> None:
    """First-write-wins via torch.where, compile-safe (no Python branching).

    When ``capture_flag == 0`` (open), writes ``modified`` to ``capture_buf[:n]``.
    When ``capture_flag == 1`` (closed), self-copies ``capture_buf[:n]`` (no-op).
    Always closes the flag — driver explicitly opens it before each capture forward.
    """
    existing = capture_buf.narrow(0, 0, n)
    to_write = torch.where(capture_flag.bool(), existing, modified)
    capture_buf.narrow(0, 0, n).copy_(to_write)
    capture_flag.fill_(1)


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

        # Only this hook's spec — keyed by name like the compiled per-hook buffers.
        # Iterating all specs would apply every intervention to every hook.
        modified = t
        spec = getattr(worker, "_tl_intervention_specs", {}).get(canonical_name)
        if spec is not None:
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
