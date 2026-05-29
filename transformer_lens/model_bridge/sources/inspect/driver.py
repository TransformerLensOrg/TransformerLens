"""InspectDriver — torch-free consumer of an ``inspect_ai`` provider's output.

Talks to a provider (ours, HF-backed, or a wire-compatible peer like vllm-lens)
through the inspect_ai ``ModelOutput`` envelope. Everything provider-specific —
the request schema, which hooks are served, full vs last-token logits, intervention
translation — lives in a :mod:`profiles` Profile; the driver just drives it. Stays
numpy-only (``to_torch`` runs at the bridge boundary), so this file imports zero
torch symbols (enforced by a unit test). ``inspect_ai`` is imported lazily.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import warnings
from typing import Any, Mapping

import numpy as np

from transformer_lens.model_bridge.driver_protocol import (
    ForwardResult,
    Intervention,
    TensorLike,
)
from transformer_lens.model_bridge.sources._driver_base import DriverBase

from . import hooks, wire
from .profiles import TLBridgeProfile


class InspectDriver(DriverBase):
    """Driver wrapping an ``inspect_ai`` model; capture + interventions via a Profile."""

    # Remote provider — no torch weight/grad surface.
    _supported_features = frozenset()

    def __init__(self, model: Any, adapter: Any, tokenizer: Any, profile: Any = None) -> None:
        super().__init__(adapter.cfg, tokenizer)
        self._model = model
        self._profile = profile if profile is not None else TLBridgeProfile()
        self._n_layers = int(self.bridge_config.n_layers)
        self._d_vocab = int(self.bridge_config.d_vocab)
        # Provider-specific: loss/both allowed only if the provider returns full logits.
        self.provides_sequence_logits = self._profile.provides_sequence_logits
        self.supported_hook_points = self._profile.supported_hooks(self._n_layers)
        full = hooks.supported_hook_points(self._n_layers)
        self.non_fireable_hook_points = hooks.nonfireable_hook_points(self._n_layers) | (
            full - self.supported_hook_points
        )
        # Background event loop, created lazily on first forward.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._warned_missing: set[str] = set()  # hooks we've already warned were absent

    def forward(
        self,
        input_ids: TensorLike | None = None,
        *,
        capture: tuple[str, ...] = (),
        intervene: Mapping[str, Intervention] | None = None,
        max_new_tokens: int = 1,
        return_logits: bool = True,
        **kwargs: Any,
    ) -> ForwardResult:
        if self._model is None:
            raise RuntimeError("InspectDriver is closed.")
        if input_ids is None:
            raise ValueError("InspectDriver requires input_ids")
        if int(max_new_tokens) != 1:
            raise NotImplementedError(
                "InspectDriver supports max_new_tokens=1 only (single-forward capture)."
            )
        ids = self._normalize_input_ids(input_ids)
        # capture is authoritative: the bridge passes exactly the hooks with handlers,
        # so () means "capture nothing" (logits only), not "capture everything".
        names = list(capture)
        wire_keys = self._wire_keys(names)
        interventions = self._profile.translate_interventions(
            intervene or {}, self.supported_hook_points
        )
        prompt, extra_args = self._profile.build_request(
            ids, wire_keys, interventions, return_logits, self.tokenizer
        )

        output = self._run_coro(self._generate(prompt, extra_args))

        captured = self._assemble_captures(output, names)
        logits = (
            self._profile.decode_logits(output, len(ids), self._d_vocab, self.tokenizer)
            if return_logits
            else None
        )
        return ForwardResult(logits=logits, captured=captured, raw_output=output)

    async def _generate(self, prompt: Any, extra_args: dict[str, Any]) -> Any:
        from inspect_ai.model import GenerateConfig

        config = GenerateConfig(
            temperature=0.0, max_tokens=1, extra_body={"extra_args": extra_args}
        )
        return await self._model.generate(prompt, config=config)

    def _assemble_captures(self, output: Any, names: list[str]) -> dict[str, np.ndarray]:
        """Decode the requested boundaries → ``{hook_name: (1, seq, d_model)}``.

        ``wire.decode_activations`` reads both our flat ``<layer>:<kind>`` map and a
        vllm-lens nested ``residual_stream``, so this is provider-agnostic. Hook names
        are TransformerBridge-native (``blocks.{i}.attn.hook_out``, ...).
        """
        metadata = getattr(output, "metadata", None) or {}
        decoded = wire.decode_activations(metadata, self._wire_keys(names))
        captured: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for name in names:
            resolved = hooks.resolve(name)
            if resolved is None:
                continue
            arr = decoded.get(hooks.wire_key(*resolved))
            if arr is None:
                missing.append(name)
                continue
            captured[name] = arr[np.newaxis, ...] if arr.ndim == 2 else arr
        self._warn_missing(missing)
        return captured

    def _warn_missing(self, missing: list[str]) -> None:
        """Warn once per hook the provider was asked for but didn't return — else its
        cache entry is silently absent and surfaces only as a later KeyError."""
        new = [name for name in missing if name not in self._warned_missing]
        if new:
            self._warned_missing.update(new)
            warnings.warn(
                f"InspectDriver: provider returned no activation for {sorted(new)} "
                "(requested and in supported_hook_points); those cache keys will be absent.",
                UserWarning,
                stacklevel=2,
            )

    def close(self) -> None:
        log = logging.getLogger("transformer_lens.inspect")
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
                if self._loop_thread is not None:
                    self._loop_thread.join(timeout=5)
                self._loop.close()
            except Exception as e:
                log.debug("event-loop teardown failed during close(): %s", e)
        self._loop = None
        self._loop_thread = None
        self._model = None  # drop the provider reference; the server owns its own lifecycle

    # ---- helpers ----

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """A private loop on a daemon thread — works even when the caller is already
        inside a running loop (Jupyter), unlike asyncio.run()."""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._loop.run_forever, daemon=True, name="inspect-driver-loop"
            )
            self._loop_thread.start()
        return self._loop

    def _run_coro(self, coro: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._ensure_loop())
        return future.result()  # blocks the sync caller; re-raises provider errors

    def _wire_keys(self, names: list[str]) -> list[str]:
        """Unique ``<layer>:<kind>`` keys for the requested hook names (aliases collapse)."""
        keys = {hooks.wire_key(*r) for r in (hooks.resolve(n) for n in names) if r is not None}
        return sorted(keys)

    @staticmethod
    def _normalize_input_ids(input_ids: Any) -> list[int]:
        """Coerce to a flat list[int] (batch_size=1 only); numpy/list/tensor, no torch import."""
        arr = np.asarray(input_ids)
        if arr.ndim == 2:
            if arr.shape[0] != 1:
                raise NotImplementedError("InspectDriver supports batch_size=1 only.")
            arr = arr[0]
        return [int(x) for x in arr.tolist()]
