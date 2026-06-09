"""SGLang Driver: forward via ``engine.generate``, captures via :mod:`rpc`.
v1 is single-prompt compiled-mode capture; batched-mode capture follows."""
from __future__ import annotations

import gc
import logging
from typing import Any, Mapping

import torch

from transformer_lens.model_bridge.driver_protocol import (
    ForwardResult,
    Intervention,
    TensorLike,
)
from transformer_lens.model_bridge.sources._driver_base import DriverBase

from . import rpc
from .intervention_specs import SUPPORTED_OPS


class SGLangDriver(DriverBase):
    """Driver wrapping an SGLang ``Engine``; captures via :mod:`rpc`."""

    # SGLang owns the model in a worker process — no torch parameter/state_dict surface.
    _supported_features = frozenset()
    # Sampler bypasses lm_head — only the final position has logits.
    provides_sequence_logits = False

    def __init__(
        self,
        engine: Any,
        adapter: Any,
        tokenizer: Any,
        overlay: Any,
        hf_config: Any,
        max_num_batched_tokens: int,
    ) -> None:
        super().__init__(adapter.cfg, tokenizer)
        self._engine = engine
        self._max_num_batched_tokens = max_num_batched_tokens
        # Full-vocab logprobs per forward → position -1 of the synthesized logits.
        self._n_logprobs = int(getattr(hf_config, "vocab_size", self.bridge_config.d_vocab))

        self.supported_hook_points = frozenset(overlay.capture_specs(hf_config).keys())

        n_layers = getattr(hf_config, "num_hidden_layers", 0)
        nonfiring: list[str] = []
        for tmpl in overlay.nonfiring_hooks():
            if "{i}" in tmpl and isinstance(n_layers, int) and n_layers > 0:
                nonfiring.extend(tmpl.replace("{i}", str(i)) for i in range(n_layers))
            else:
                nonfiring.append(tmpl)
        self.non_fireable_hook_points = frozenset(nonfiring)

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
        if input_ids is None:
            raise ValueError("SGLangDriver requires input_ids")
        if int(max_new_tokens) != 1:
            raise NotImplementedError(
                "SGLangDriver supports max_new_tokens=1 only — decode-step writes "
                "overwrite the prefill buffer; multi-step capture is multi-buffer work."
            )
        intervene_specs = self._validate_interventions(intervene or {})

        # Restrict the GPU→CPU read to these hooks (None = all).
        names = list(capture) or None

        ids_list = self._normalize_input_ids(input_ids)
        if len(ids_list) > self._max_num_batched_tokens:
            raise ValueError(
                f"Prompt length {len(ids_list)} exceeds max_num_batched_tokens="
                f"{self._max_num_batched_tokens}; raise the boot_sglang kwarg or "
                "shorten the prompt."
            )

        # set_interventions also resets any stale state from a prior forward.
        rpc.set_interventions(self._engine, intervene_specs)
        rpc.reset_capture_flags(self._engine)

        sampling_params = self._build_sampling_params(
            max_tokens=int(max_new_tokens),
            return_logits=return_logits,
        )
        outputs = self._engine.generate(input_ids=[ids_list], sampling_params=sampling_params)

        n_tokens = len(ids_list)
        # tl_read_captures returns the single worker's result; SGLang's RPC
        # returns the response payload directly (no per-worker list like vLLM).
        worker_captures = rpc.call_with_prompt_lens(
            self._engine, "tl_read_captures", [n_tokens], names
        )
        # Add batch dim: SGLang hands back (n_tokens, width); bridge expects (1, n_tokens, width).
        captured = {name: t.unsqueeze(0) for name, t in worker_captures.items()}

        logits: torch.Tensor | None = None
        if return_logits:
            logits = self._synthesize_logits(outputs, n_tokens, self.bridge_config.d_vocab)

        return ForwardResult(logits=logits, captured=captured, raw_output=outputs)

    def _build_sampling_params(self, *, max_tokens: int, return_logits: bool) -> dict[str, Any]:
        params: dict[str, Any] = {"max_new_tokens": max_tokens, "temperature": 0.0}
        if return_logits:
            # Full-vocab logprobs so the synthesized logits cover all d_vocab.
            params["top_logprobs_num"] = self._n_logprobs
            params["return_logprob"] = True
        return params

    def get_param(self, dotted_name: str) -> torch.Tensor | None:
        """Fetch a named worker tensor (e.g. ``model.norm.weight``) — ``None`` if
        closed or path missing."""
        if self._engine is None:
            return None
        return rpc.call(self._engine, "tl_get_param", {"dotted_name": dotted_name})

    def close(self) -> None:
        log = logging.getLogger("transformer_lens.sglang")
        if self._engine is not None:
            try:
                rpc.remove_hooks(self._engine)
            except Exception as e:
                log.debug("tl_remove_hooks failed during close(): %s", e)
            for shutdown_attr in ("shutdown", "close"):
                shutdown = getattr(self._engine, shutdown_attr, None)
                if callable(shutdown):
                    try:
                        shutdown()
                    except Exception as e:
                        log.debug("engine.%s() failed during close(): %s", shutdown_attr, e)
                    break
        self._engine = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            log.debug("torch.cuda.empty_cache failed during close(): %s", e)

    @staticmethod
    def _synthesize_logits(outputs: Any, n_tokens: int, d_vocab: int) -> torch.Tensor:
        """``(1, n_tokens, d_vocab)`` log-probs tensor; only position -1 is filled
        (sampler bypasses lm_head). Argmax-correct, absolute scale off."""
        logits = torch.full((1, n_tokens, d_vocab), float("-inf"), dtype=torch.float16)
        first = outputs[0] if isinstance(outputs, list) and outputs else outputs
        if first is None:
            return logits
        # Output shape varies — dict on v1, typed object on newer releases.
        meta = (
            first.get("meta_info") if isinstance(first, dict) else getattr(first, "meta_info", None)
        )
        if meta is None:
            return logits
        top = (
            meta.get("output_top_logprobs")
            if isinstance(meta, dict)
            else getattr(meta, "output_top_logprobs", None)
        )
        if top:
            # top[-1] is the generated step: list[(logprob, token_id, decoded)].
            for entry in top[-1]:
                try:
                    lp, token_id, _ = entry
                except (TypeError, ValueError):
                    continue
                logits[0, -1, int(token_id)] = float(lp)
        else:
            # No top-k logprobs; one-hot the generated token id (argmax-only).
            chosen = (
                first.get("token_ids")
                if isinstance(first, dict)
                else getattr(first, "token_ids", None)
            )
            if chosen:
                logits[0, -1, int(chosen[0])] = 0.0
        return logits

    def _validate_interventions(self, intervene: Mapping[str, Any]) -> dict:
        """Reject callables; validate spec format and hook names; return a plain dict."""
        out: dict = {}
        for hook_name, spec in intervene.items():
            if callable(spec):
                raise NotImplementedError(
                    "SGLangDriver requires intervention specs (dict), not callables. "
                    "Supported ops: suppress, scale (factor: float), "
                    "add (value: scalar or width-shaped), set (value: scalar or width-shaped)."
                )
            if not isinstance(spec, Mapping) or "op" not in spec:
                raise ValueError(
                    f"Intervention spec for {hook_name!r} must be a dict with 'op' key; got {spec!r}"
                )
            op = spec["op"]
            if op not in SUPPORTED_OPS:
                raise ValueError(
                    f"Unsupported intervention op {op!r} for {hook_name!r}. "
                    f"Supported: {sorted(SUPPORTED_OPS)}."
                )
            if op == "scale" and "factor" not in spec:
                raise ValueError(
                    f"Intervention {hook_name!r}: op='scale' requires 'factor' (float)."
                )
            if op in ("add", "set") and "value" not in spec:
                raise ValueError(
                    f"Intervention {hook_name!r}: op={op!r} requires 'value' "
                    "(scalar or width-shaped tensor/list)."
                )
            if hook_name not in self.supported_hook_points:
                raise ValueError(
                    f"Cannot intervene on {hook_name!r}: not in supported_hook_points."
                )
            out[hook_name] = dict(spec)
        return out

    @staticmethod
    def _normalize_input_ids(input_ids: Any) -> list:
        """Coerce input_ids to a flat list[int] for SGLang's generate API; batch_size=1 only."""
        if isinstance(input_ids, torch.Tensor):
            ids_list = input_ids.tolist()
        else:
            ids_list = list(input_ids)
        if ids_list and isinstance(ids_list[0], list):
            if len(ids_list) != 1:
                raise NotImplementedError("SGLangDriver supports batch_size=1 only.")
            ids_list = ids_list[0]
        return ids_list
