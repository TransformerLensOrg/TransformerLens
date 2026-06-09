"""SGLang-backed ``inspect_ai`` provider (``tl_bridge_sglang``). Shares the
:mod:`sources.sglang` boot path — same worker plugin, same Scheduler patches."""
from __future__ import annotations

import gc
from typing import Any, Mapping

import numpy as np
import torch
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    GenerateConfig,
    Logprob,
    Logprobs,
    ModelOutput,
    ModelUsage,
    StopReason,
    TopLogprob,
    modelapi,
)

from . import hooks, wire
from ._provider_base import _InspectModelAPIBase, _parse_tool_calls, _require_served

PROVIDER_NAME = "tl_bridge_sglang"

_LOCKED_SGLANG_KWARGS = {
    "tp_size": 1,
    "dp_size": 1,
    "skip_tokenizer_init": True,
    # Chunked prefill would emit N messages per hook on PULL; collector silently truncates.
    "chunked_prefill_size": -1,
}


def _kinds_from_specs(specs: dict[str, Any]) -> frozenset[str]:
    """Boundary kinds the overlay's ``capture_specs`` serves."""
    kinds = set()
    for name in specs:
        resolved = hooks.resolve(name)
        if resolved is not None:
            kinds.add(resolved[1])
    return frozenset(kinds)


@modelapi(name=PROVIDER_NAME)
def transformer_lens_sglang_provider():
    """Lazy registration hook — returns the provider class on first use."""
    return TransformerLensSGLangModelAPI


class TransformerLensSGLangModelAPI(_InspectModelAPIBase):
    """SGLang-backed Inspect provider. Mirrors :class:`TransformerLensVLLMModelAPI`."""

    # Sampler bypasses lm_head.
    provides_sequence_logits = False

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(model_name, base_url, api_key, [], config)
        from sglang.srt.entrypoints.engine import (  # type: ignore[import-not-found]
            Engine,
        )
        from transformers import AutoConfig, AutoTokenizer

        from transformer_lens.utilities.hf_utils import get_hf_token

        from ..sglang import wire as sglang_wire
        from ..sglang.capture_puller import CapturePuller
        from ..sglang.internals import assert_sglang_supported, extract_hf_config
        from ..sglang.overlays import get_overlay

        sglang_kwargs = model_args.pop("sglang_kwargs", {})
        for key, locked in _LOCKED_SGLANG_KWARGS.items():
            if key in sglang_kwargs and sglang_kwargs[key] != locked:
                raise ValueError(
                    f"tl_bridge_sglang forces {key}={locked}; caller passed "
                    f"{key}={sglang_kwargs[key]}."
                )
        mem_fraction_static = model_args.pop("mem_fraction_static", 0.5)
        max_total_tokens = model_args.pop("max_total_tokens", None)
        max_num_batched_tokens = int(model_args.pop("max_num_batched_tokens", 2048))
        disable_cuda_graph = bool(model_args.pop("disable_cuda_graph", True))
        dtype = model_args.pop("dtype", None)

        # device overridable so mocked CPU unit tests work.
        self._device = model_args.pop("device", "cuda")
        hf_token = get_hf_token()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

        assert_sglang_supported()
        hf_config_preview = AutoConfig.from_pretrained(model_name, token=hf_token)
        architecture = hf_config_preview.architectures[0]
        overlay = get_overlay(architecture)
        resolved_dtype = dtype if dtype is not None else _dtype_from_hf_config(hf_config_preview)
        capture_specs = overlay.capture_specs(hf_config_preview)

        channel = sglang_wire.fresh_channel()
        forward_hooks = sglang_wire.build_forward_hooks(capture_specs, channel)
        self._puller = CapturePuller(channel)

        # max_num_batched_tokens stays provider-internal as the prompt-length gate;
        # ServerArgs uses chunked_prefill_size / max_prefill_tokens instead.
        engine_kwargs: dict[str, Any] = {
            "model_path": model_name,
            "mem_fraction_static": mem_fraction_static,
            "max_total_tokens": max_total_tokens,
            "dtype": str(resolved_dtype).replace("torch.", "") if dtype is not None else "auto",
            "forward_hooks": forward_hooks,
            "disable_cuda_graph": disable_cuda_graph,
        }
        engine_kwargs = {k: v for k, v in engine_kwargs.items() if v is not None}
        try:
            self._engine = Engine(**engine_kwargs, **_LOCKED_SGLANG_KWARGS, **sglang_kwargs)
        except Exception:
            self._puller.close()
            raise

        hf_config = extract_hf_config(self._engine)
        self._d_vocab = int(hf_config.vocab_size)
        self._max_logprobs = int(hf_config_preview.vocab_size)
        self._max_num_batched_tokens = max_num_batched_tokens

        # Fused execution gates resid_pre / resid_mid.
        self._kinds = _kinds_from_specs(capture_specs)
        self._capability_note = (
            "tl_bridge_sglang: SGLang's fused execution gates resid_pre (no block-input hook) "
            "and the derived resid_mid. Use boot_inspect(provider='tl_bridge') for those."
        )
        self._eval_capture = self._parse_eval_capture(model_args)

    def _generate_capture(self, input: Any, extra_args: Mapping[str, Any], config: GenerateConfig):
        """TL-driven single-token capture: push interventions, generate, drain."""
        input_ids = extra_args.get("input_ids")
        if input_ids is None:
            input_ids = self._messages_to_ids(input)[0].tolist()
        n_tokens = len(input_ids)
        if n_tokens > self._max_num_batched_tokens:
            raise ValueError(
                f"Prompt length {n_tokens} exceeds max_num_batched_tokens="
                f"{self._max_num_batched_tokens}."
            )

        capture_keys = list(extra_args.get("capture", []))
        for key in capture_keys:
            _, _, kind = key.partition(":")
            _require_served(kind, self._kinds, self._capability_note, f"capture {key!r}")

        name_by_wire_raw = {wk: hooks.name_from_wire_key(wk) for wk in capture_keys}
        unknown_wire = sorted(wk for wk, name in name_by_wire_raw.items() if name is None)
        if unknown_wire:
            raise ValueError(f"unrecognised wire keys: {unknown_wire}")
        name_by_wire: dict[str, str] = {
            wk: name for wk, name in name_by_wire_raw.items() if name is not None
        }
        wanted_names = set(name_by_wire.values())

        intervention_specs: dict[str, Any] = {}
        for wk, spec in (extra_args.get("interventions") or {}).items():
            name = hooks.name_from_wire_key(wk)
            if name is None:
                raise ValueError(f"intervention wire key {wk!r} is not a fireable hook.")
            intervention_specs[name] = spec

        want_logits = bool(extra_args.get("return_logits", True))

        # Drain any straggler from a prior call; push spec; enable capture; generate.
        self._puller.drain(timeout_ms=0)
        self._engine.collective_rpc("tl_set_interventions", specs=intervention_specs)
        self._engine.collective_rpc("tl_set_capture_enabled", enabled=True)
        # return_logprob / top_logprobs_num are top-level generate kwargs, not SamplingParams.
        try:
            outputs = self._engine.generate(
                input_ids=[list(input_ids)],
                sampling_params={"max_new_tokens": 1, "temperature": 0.0},
                **(
                    {"return_logprob": True, "top_logprobs_num": self._max_logprobs}
                    if want_logits
                    else {}
                ),
            )
        finally:
            self._engine.collective_rpc("tl_set_capture_enabled", enabled=False)

        captures_by_name = self._collect_captures(wanted_names or None)

        captured_wire: dict[str, np.ndarray] = {}
        for wk, name in name_by_wire.items():
            tensor = captures_by_name.get(name)
            if tensor is not None:
                captured_wire[wk] = tensor.detach().float().cpu().numpy()
        metadata: dict[str, Any] = {"activations": wire.encode_activations(captured_wire)}

        if want_logits:
            logits = _synthesize_logits(outputs, n_tokens, self._d_vocab)
            metadata["tl_logits"] = wire.encode_array(logits[0].cpu().numpy())

        next_id = _next_token_id(outputs)
        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=str(self._tokenizer.decode([next_id]))),
                    stop_reason="stop",
                )
            ],
            metadata=metadata,
        )

    def _generate_eval(self, input: Any, config: GenerateConfig, tools: Any):
        """Chat input → completion + Logprobs + usage; opens capture for the prefill
        when ``model_args['capture']`` is set."""
        # Clear stale state; capture starts disabled.
        self._engine.collective_rpc("tl_set_interventions", specs={})
        self._engine.collective_rpc("tl_set_capture_enabled", enabled=False)
        self._puller.drain(timeout_ms=0)

        ids = self._messages_to_ids(input, tools)[0].tolist()
        prompt_len = len(ids)
        if self._eval_capture:
            if prompt_len > self._max_num_batched_tokens:
                raise ValueError(
                    f"Prompt length {prompt_len} exceeds max_num_batched_tokens="
                    f"{self._max_num_batched_tokens}."
                )
            self._engine.collective_rpc("tl_set_capture_enabled", enabled=True)

        max_new = int(config.max_tokens) if config.max_tokens else 16
        temperature = float(config.temperature) if config.temperature is not None else 0.0

        sp: dict[str, Any] = {"max_new_tokens": max_new, "temperature": temperature}
        if temperature > 0:
            if config.top_p is not None:
                sp["top_p"] = float(config.top_p)
            if config.top_k is not None:
                sp["top_k"] = int(config.top_k)
        if config.seed is not None:
            # SamplingParams names this sampling_seed, not seed.
            sp["sampling_seed"] = int(config.seed)
        generate_kwargs: dict[str, Any] = {}
        if config.logprobs:
            generate_kwargs["return_logprob"] = True
            generate_kwargs["top_logprobs_num"] = (
                int(config.top_logprobs) if config.top_logprobs else 1
            )

        try:
            outputs = self._engine.generate(input_ids=[ids], sampling_params=sp, **generate_kwargs)
        finally:
            if self._eval_capture:
                self._engine.collective_rpc("tl_set_capture_enabled", enabled=False)

        result = outputs[0] if isinstance(outputs, list) and outputs else outputs
        new_ids = list(_get(result, "output_ids") or [])
        n_new = len(new_ids)
        completion = str(self._tokenizer.decode(new_ids, skip_special_tokens=True))

        logprobs = None
        if config.logprobs:
            meta = _get(result, "meta_info") or {}
            step_logprobs = (
                _get(meta, "output_top_logprobs") or _get(meta, "output_token_logprobs") or []
            )
            if step_logprobs:
                logprobs = Logprobs(
                    content=[
                        self._logprob_from_step(int(tid), step, config.top_logprobs)
                        for tid, step in zip(new_ids, step_logprobs)
                    ]
                )

        tool_calls = _parse_tool_calls(completion) if len(tools) else None
        finish = str(_get(result, "finish_reason") or "").lower()
        stop_reason: StopReason
        if tool_calls:
            stop_reason = "tool_calls"
        elif finish == "length":
            stop_reason = "max_tokens"
        elif finish == "stop":
            stop_reason = "stop"
        else:
            stop_reason = "unknown"

        # Per-decode-step hooks also fire; first-wins keeps the prefill capture.
        eval_metadata: dict[str, Any] = {}
        if self._eval_capture:
            wanted_names = set(self._eval_capture.values())
            captures_by_name = self._collect_captures(wanted_names)
            captured_wire: dict[str, np.ndarray] = {}
            for wk, name in self._eval_capture.items():
                tensor = captures_by_name.get(name)
                if tensor is not None:
                    captured_wire[wk] = tensor.detach().float().cpu().numpy()
            eval_metadata = {"activations": wire.encode_activations(captured_wire)}

        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=completion, tool_calls=tool_calls),
                    stop_reason=stop_reason,
                    logprobs=logprobs,
                )
            ],
            usage=ModelUsage(
                input_tokens=prompt_len, output_tokens=n_new, total_tokens=prompt_len + n_new
            ),
            metadata=eval_metadata or None,
        )

    def _collect_captures(self, wanted: set[str] | None) -> dict[str, torch.Tensor]:
        """Drain PULL; first-wins per name (prefill, not last decode step)."""
        expected = len(wanted) if wanted is not None else len(self._kinds)
        messages: list[dict] = []
        for _ in range(expected * 4 + 1):
            batch = self._puller.drain(timeout_ms=500)
            messages.extend(batch)
            if len(messages) >= expected and not batch:
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
                continue
            out[name] = t
        return out

    def _logprob_from_step(self, token_id: int, step: Any, top_n: Any) -> Logprob:
        """SGLang per-step top-logprobs entry → :class:`inspect_ai.model.Logprob`."""
        by_tid: dict[int, tuple[float, str]] = {}
        for entry in step or []:
            try:
                lp, tid, decoded = entry
            except (TypeError, ValueError):
                continue
            by_tid[int(tid)] = (float(lp), str(decoded or ""))
        chosen_lp, _ = by_tid.get(token_id, (float("-inf"), ""))

        top: list[TopLogprob] = []
        if top_n:
            ranked = sorted(by_tid.items(), key=lambda kv: -kv[1][0])
            for tid, (lp, decoded) in ranked[: int(top_n)]:
                token = decoded or self._tokenizer.decode([int(tid)])
                top.append(TopLogprob(token=str(token), logprob=float(lp), bytes=None))
        return Logprob(
            token=str(self._tokenizer.decode([int(token_id)])),
            logprob=chosen_lp,
            bytes=None,
            top_logprobs=top,
        )

    def close(self) -> None:
        """Engine shutdown + PULL socket teardown."""
        if self._engine is not None:
            try:
                self._engine.collective_rpc("tl_clear_state")
            except Exception:
                pass
            for shutdown_attr in ("shutdown", "close"):
                fn = getattr(self._engine, shutdown_attr, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
                    break
        self._engine = None
        try:
            self._puller.close()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def _get(obj: Any, name: str) -> Any:
    """Dict-or-object accessor — SGLang's output shape varies by release."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _next_token_id(outputs: Any) -> int:
    """Generated token id from ``generate(max_new_tokens=1)``."""
    first = outputs[0] if isinstance(outputs, list) and outputs else outputs
    if first is None:
        return 0
    ids = _get(first, "output_ids") or []
    return int(ids[0]) if ids else 0


def _dtype_from_hf_config(hf_config: Any) -> torch.dtype:
    raw = getattr(hf_config, "torch_dtype", None)
    if isinstance(raw, torch.dtype):
        return raw
    if isinstance(raw, str):
        return getattr(torch, raw, torch.float16)
    return torch.float16


def _synthesize_logits(outputs: Any, n_tokens: int, d_vocab: int) -> torch.Tensor:
    """``(1, n_tokens, d_vocab)`` at the gen position; same shape as vLLM/HF ``tl_logits``."""
    logits = torch.full((1, n_tokens, d_vocab), float("-inf"), dtype=torch.float32)
    first = outputs[0] if isinstance(outputs, list) and outputs else outputs
    if first is None:
        return logits
    meta = _get(first, "meta_info") or {}
    top = _get(meta, "output_top_logprobs")
    if top:
        for entry in top[-1] or []:
            try:
                lp, token_id, _ = entry
            except (TypeError, ValueError):
                continue
            logits[0, -1, int(token_id)] = float(lp)
        return logits
    chosen = _get(first, "output_ids") or []
    if chosen:
        logits[0, -1, int(chosen[0])] = 0.0
    return logits


__all__ = ["TransformerLensSGLangModelAPI"]
