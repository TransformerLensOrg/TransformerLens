"""SGLang Driver: forward via ``engine.generate``; control via ``collective_rpc``
(reaches the ``tl_*`` methods :mod:`worker_plugin` patched onto ``Scheduler``);
tensor capture via a ZMQ PULL socket the worker hooks push into.

v1: single-prompt only. Continuous-batching capture needs per-request demarcation
on the PULL socket that vanilla ``register_forward_hook`` can't see today."""
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

from .intervention_specs import SUPPORTED_OPS


class SGLangDriver(DriverBase):
    """Driver wrapping an SGLang ``Engine`` + a ZMQ PULL endpoint for captures."""

    # Model lives in a worker subprocess; no torch parameter surface.
    _supported_features = frozenset()
    # Sampler bypasses lm_head, so only the gen position has logits.
    provides_sequence_logits = False

    def __init__(
        self,
        engine: Any,
        adapter: Any,
        tokenizer: Any,
        overlay: Any,
        hf_config: Any,
        max_num_batched_tokens: int,
        puller: Any,
    ) -> None:
        super().__init__(adapter.cfg, tokenizer)
        self._engine = engine
        self._puller = puller
        self._max_num_batched_tokens = max_num_batched_tokens
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
                "SGLangDriver supports max_new_tokens=1 only — decode-step hook fires "
                "would interleave on the PULL socket without per-step demarcation."
            )
        intervene_specs = self._validate_interventions(intervene or {})

        ids_list = self._normalize_input_ids(input_ids)
        if len(ids_list) > self._max_num_batched_tokens:
            raise ValueError(
                f"Prompt length {len(ids_list)} exceeds max_num_batched_tokens="
                f"{self._max_num_batched_tokens}."
            )

        # Set state before enabling capture so the first forward sees it; drain
        # any straggler from a prior call.
        self._engine.collective_rpc("tl_set_interventions", specs=intervene_specs)
        self._puller.drain(timeout_ms=0)
        self._engine.collective_rpc("tl_set_capture_enabled", enabled=True)

        # top_logprobs_num / return_logprob are top-level generate kwargs, not
        # SamplingParams fields.
        try:
            outputs = self._engine.generate(
                input_ids=[ids_list],
                sampling_params={
                    "max_new_tokens": int(max_new_tokens),
                    "temperature": 0.0,
                },
                **(
                    {"return_logprob": True, "top_logprobs_num": self._n_logprobs}
                    if return_logits
                    else {}
                ),
            )
        finally:
            self._engine.collective_rpc("tl_set_capture_enabled", enabled=False)

        n_tokens = len(ids_list)
        wanted = set(capture) if capture else None
        captured = self._collect_captures(n_tokens, wanted)

        logits: torch.Tensor | None = None
        if return_logits:
            logits = self._synthesize_logits(outputs, n_tokens, self.bridge_config.d_vocab)

        return ForwardResult(logits=logits, captured=captured, raw_output=outputs)

    def _collect_captures(self, n_tokens: int, wanted: set[str] | None) -> dict[str, torch.Tensor]:
        """Drain PULL; first-wins per name; add batch dim."""
        expected = len(wanted) if wanted is not None else len(self.supported_hook_points)
        messages: list[dict] = []
        for _ in range(expected * 4 + 1):
            batch = self._puller.drain(timeout_ms=500)
            messages.extend(batch)
            if len(messages) >= expected:
                break
            if not batch:
                break

        out: dict[str, torch.Tensor] = {}
        for msg in messages:
            name = msg.get("name")
            t = msg.get("tensor")
            if name is None or t is None:
                continue
            if wanted is not None and name not in wanted:
                continue
            if name in out:
                # First-wins: a straggler or decode-step leak shouldn't clobber prefill.
                continue
            out[name] = t.unsqueeze(0)
        return out

    def get_param(self, dotted_name: str) -> torch.Tensor | None:
        """Fetch a named worker tensor (e.g. ``model.norm.weight``) via
        ``tl_get_param``; tensor flows back on the same PUSH/PULL channel as
        captures (SGLang's RpcReqOutput is ack-only). Call outside an active
        generate — drain consumes any leftover capture messages."""
        if self._engine is None:
            return None
        try:
            self._engine.collective_rpc(
                "tl_get_param", dotted_name=dotted_name, channel=self._puller.channel
            )
        except Exception:
            return None
        for msg in self._puller.drain(timeout_ms=1000):
            if msg.get("_param") == dotted_name:
                tensor = msg.get("tensor")
                return tensor if isinstance(tensor, torch.Tensor) else None
        return None

    def close(self) -> None:
        log = logging.getLogger("transformer_lens.sglang")
        if self._engine is not None:
            try:
                self._engine.collective_rpc("tl_clear_state")
            except Exception as e:
                log.debug("tl_clear_state failed during close(): %s", e)
            for shutdown_attr in ("shutdown", "close"):
                fn = getattr(self._engine, shutdown_attr, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception as e:
                        log.debug("engine.%s() failed during close(): %s", shutdown_attr, e)
                    break
        self._engine = None
        try:
            self._puller.close()
        except Exception as e:
            log.debug("puller.close() failed: %s", e)
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            log.debug("torch.cuda.empty_cache failed during close(): %s", e)

    @staticmethod
    def _synthesize_logits(outputs: Any, n_tokens: int, d_vocab: int) -> torch.Tensor:
        """``(1, n_tokens, d_vocab)`` log-probs at the gen position only
        (sampler bypass). Argmax-correct, absolute scale off."""
        logits = torch.full((1, n_tokens, d_vocab), float("-inf"), dtype=torch.float16)
        first = outputs[0] if isinstance(outputs, list) and outputs else outputs
        if first is None:
            return logits
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
            for entry in top[-1]:
                try:
                    lp, token_id, _ = entry
                except (TypeError, ValueError):
                    continue
                logits[0, -1, int(token_id)] = float(lp)
        else:
            chosen = (
                first.get("output_ids")
                if isinstance(first, dict)
                else getattr(first, "output_ids", None)
            ) or []
            if chosen:
                logits[0, -1, int(chosen[0])] = 0.0
        return logits

    def _validate_interventions(self, intervene: Mapping[str, Any]) -> dict:
        """Reject callables, validate spec format and hook names."""
        out: dict = {}
        for hook_name, spec in intervene.items():
            if callable(spec):
                raise NotImplementedError(
                    "SGLangDriver requires intervention specs (dict), not callables."
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
        """Coerce to a flat list[int]; batch_size=1 only."""
        if isinstance(input_ids, torch.Tensor):
            ids_list = input_ids.tolist()
        else:
            ids_list = list(input_ids)
        if ids_list and isinstance(ids_list[0], list):
            if len(ids_list) != 1:
                raise NotImplementedError("SGLangDriver supports batch_size=1 only.")
            ids_list = ids_list[0]
        return ids_list
