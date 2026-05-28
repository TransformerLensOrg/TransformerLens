"""InspectDriver — torch-free consumer of an ``inspect_ai`` provider's output.

Talks to a provider (ours, HF-backed, or a wire-compatible peer like vllm-lens)
through the inspect_ai ``ModelOutput`` envelope: sends token ids + capture/intervention
requests via ``config.extra_body["extra_args"]``, reads residual-stream activations
and last-position logits back out of ``metadata``. Stays numpy-only — ``to_torch``
runs at the bridge boundary — so this file imports zero torch symbols (enforced by
a unit test). ``inspect_ai`` is imported lazily so the rest of the package works
without it installed.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Mapping

import numpy as np

from transformer_lens.model_bridge.driver_protocol import (
    ForwardResult,
    Intervention,
    TensorLike,
)
from transformer_lens.model_bridge.sources._driver_base import DriverBase

from . import intervention, wire


class InspectDriver(DriverBase):
    """Driver wrapping an ``inspect_ai`` model; residual-stream capture, last-token logits."""

    # Remote provider — no torch weight/grad surface.
    _supported_features = frozenset()
    # Logits synthesized for the final position only ⇒ RemoteBridge rejects loss/both.
    provides_sequence_logits = False

    def __init__(self, model: Any, adapter: Any, tokenizer: Any) -> None:
        super().__init__(adapter.cfg, tokenizer)
        self._model = model
        self._n_layers = int(self.bridge_config.n_layers)
        self._d_vocab = int(self.bridge_config.d_vocab)
        self.supported_hook_points = frozenset(
            f"blocks.{i}.hook_resid_post" for i in range(self._n_layers)
        )
        self.non_fireable_hook_points = frozenset(self._nonfireable())
        # Background event loop, created lazily on first forward.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

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
        layers = self._layers_for_capture(capture)
        extra_args: dict[str, Any] = {"input_ids": ids, "output_residual_stream": layers}
        spec_payload = intervention.build_extra_args(
            intervene or {}, self.supported_hook_points, self._hook_to_layer()
        )
        if spec_payload:
            extra_args["interventions"] = spec_payload

        output = self._run_coro(self._generate(extra_args, return_logits))

        captured = self._assemble_captures(output, layers)
        logits = self._synthesize_logits(output, len(ids)) if return_logits else None
        return ForwardResult(logits=logits, captured=captured, raw_output=output)

    async def _generate(self, extra_args: dict[str, Any], return_logits: bool) -> Any:
        from inspect_ai.model import GenerateConfig

        config = GenerateConfig(
            temperature=0.0,
            max_tokens=1,
            logprobs=return_logits or None,
            extra_body={"extra_args": extra_args},
        )
        return await self._model.generate("", config=config)

    def _assemble_captures(self, output: Any, layers: list[int]) -> dict[str, np.ndarray]:
        """Decode residual streams → ``{blocks.{i}.hook_resid_post: (1, seq, d_model)}``."""
        metadata = getattr(output, "metadata", None) or {}
        by_layer = wire.decode_activations(metadata, layers)
        captured: dict[str, np.ndarray] = {}
        for layer, arr in by_layer.items():
            if arr.ndim == 2:  # (seq, d_model) → add batch dim for has_batch_dim caches
                arr = arr[np.newaxis, ...]
            captured[f"blocks.{layer}.hook_resid_post"] = arr
        return captured

    def _synthesize_logits(self, output: Any, n_tokens: int) -> np.ndarray:
        """Place the provider's exact last-position logits at position -1; rest -inf.

        Earlier positions stay -inf (only the next-token row is provided), so any
        argmax there is loud rather than silently wrong — and provides_sequence_logits
        is False so the bridge won't ask for a loss over them.
        """
        logits = np.full((1, n_tokens, self._d_vocab), -np.inf, dtype=np.float32)
        metadata = getattr(output, "metadata", None) or {}
        entry = metadata.get("tl_last_logits")
        if entry is not None:
            logits[0, -1, :] = wire.decode_array(entry)
        return logits

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

    def _hook_to_layer(self) -> dict[str, int]:
        return {f"blocks.{i}.hook_resid_post": i for i in range(self._n_layers)}

    def _layers_for_capture(self, capture: tuple[str, ...]) -> list[int]:
        """Capture-restricted layer indices (default: all)."""
        if not capture:
            return list(range(self._n_layers))
        mapping = self._hook_to_layer()
        return sorted({mapping[name] for name in capture if name in mapping})

    def _nonfireable(self) -> list[str]:
        """Hooks the residual-stream-only provider can't fire."""
        names: list[str] = ["embed.hook_out", "ln_final.hook_normalized", "unembed.hook_out"]
        for i in range(self._n_layers):
            names += [
                f"blocks.{i}.hook_resid_pre",
                f"blocks.{i}.hook_resid_mid",
                f"blocks.{i}.attn.hook_pattern",
                f"blocks.{i}.attn.hook_attn_scores",
                f"blocks.{i}.attn.hook_z",
                f"blocks.{i}.hook_attn_out",
                f"blocks.{i}.hook_mlp_out",
            ]
        return names

    @staticmethod
    def _normalize_input_ids(input_ids: Any) -> list[int]:
        """Coerce to a flat list[int] (batch_size=1 only); numpy/list/tensor, no torch import."""
        arr = np.asarray(input_ids)
        if arr.ndim == 2:
            if arr.shape[0] != 1:
                raise NotImplementedError("InspectDriver supports batch_size=1 only.")
            arr = arr[0]
        return [int(x) for x in arr.tolist()]
