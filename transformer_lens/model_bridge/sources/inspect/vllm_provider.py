"""vLLM-backed ``inspect_ai`` model provider, registered as ``tl_bridge_vllm``.

A sibling to the HF-backed ``tl_bridge`` provider: instead of running an HF causal LM
locally, it generates via vLLM (PagedAttention + continuous batching) — so it scales to
parallel-sample evals and dataset-scale workloads where the HF provider serializes.

Inherits ``generate()`` dispatch / message-rendering / per-turn-capture validation from
:class:`_InspectModelAPIBase`; this file owns the vLLM ``LLM`` construction (with the
plugin + worker_extension wiring needed for capture), eval-native generation via
``llm.generate(...)``, and the TL-driven capture path via ``collective_rpc`` to the
worker extension. The capture wire format matches the HF provider's, so the existing
``InspectDriver`` consumes it unchanged.

vLLM is GPU-only and imported lazily inside ``__init__`` / ``_generate_*`` so this
module imports cleanly in environments without vLLM (matching the HF provider pattern).
"""
from __future__ import annotations

import gc
import os
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
from ._provider_base import (
    _InspectModelAPIBase,
    _parse_tool_calls,
    _require_interveneable,
    _require_served,
    _warn_unsupported_config,
)

# Distinct from the HF ``tl_bridge`` provider and from inspect_ai's built-in ``vllm``.
PROVIDER_NAME = "tl_bridge_vllm"

# Forced ``LLM(...)`` kwargs that the capture-hook design depends on (matches the same
# set in ``sources/vllm/source.py``). Multi-device / vLLM-owned tokenizer break the wire
# path; we own the tokenizer and the plugin assumes single-process workers.
_LOCKED_VLLM_KWARGS = {
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "skip_tokenizer_init": True,
    "disable_log_stats": True,
}

# Dotted path to the worker extension whose ``tl_*`` methods this provider drives via
# ``collective_rpc`` (capture reads, intervention specs, hook teardown).
_WORKER_EXTENSION_CLS = (
    "transformer_lens.model_bridge.sources.vllm.worker_extension.TLWorkerExtension"
)


def _kinds_from_specs(specs: dict[str, Any]) -> frozenset[str]:
    """Boundary kinds (resid_post/attn_out/...) served by the overlay's ``capture_specs``.

    Non-block hooks (``embed.hook_out``, ``ln_final.hook_normalized``) don't resolve to
    a kind and don't contribute — they're already in the InspectDriver's non-fireable set.
    """
    kinds = set()
    for name in specs:
        resolved = hooks.resolve(name)
        if resolved is not None:
            kinds.add(resolved[1])
    return frozenset(kinds)


@modelapi(name=PROVIDER_NAME)
def transformer_lens_vllm_provider():
    """Lazy registration hook — returns the provider class on first use."""
    return TransformerLensVLLMModelAPI


class TransformerLensVLLMModelAPI(_InspectModelAPIBase):
    """vLLM-backed Inspect provider. See module docstring for scope per increment."""

    # vLLM's sampler bypasses lm_head; _synthesize_logits populates only the gen position,
    # so earlier positions are -inf and loss would be NaN. RemoteBridge.forward must reject
    # return_type ∈ {loss, both} — read by source.py → TLBridgeProfile → InspectDriver.
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
        from transformers import AutoConfig, AutoTokenizer

        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "The tl_bridge_vllm provider requires vLLM (Linux + CUDA). Install with "
                'pip install "transformer-lens[vllm]" or uv sync --extra vllm; '
                "validated against vllm 0.20.x."
            ) from exc

        from transformer_lens.utilities.hf_utils import get_hf_token

        from ..vllm import plugin
        from ..vllm.internals import extract_hf_config, verify_hook_coverage
        from ..vllm.overlays import get_overlay

        # Caller-overridable LLM kwargs go through ``vllm_kwargs``; the locked set above
        # may not be overridden (multi-device / vLLM-owned tokenizer break our wire path).
        vllm_kwargs = model_args.pop("vllm_kwargs", {})
        for key, locked in _LOCKED_VLLM_KWARGS.items():
            if key in vllm_kwargs and vllm_kwargs[key] != locked:
                raise ValueError(
                    f"tl_bridge_vllm forces {key}={locked}; caller passed "
                    f"{key}={vllm_kwargs[key]}."
                )
        gpu_memory_utilization = model_args.pop("gpu_memory_utilization", 0.5)
        max_model_len = model_args.pop("max_model_len", None)
        max_num_batched_tokens = int(model_args.pop("max_num_batched_tokens", 2048))
        dtype = model_args.pop("dtype", None)

        # vLLM is GPU-only in production; "device" stays caller-overridable mainly so
        # mocked unit tests on CPU machines can build the prompt tensor without CUDA
        # (the base's _messages_to_ids honors self._device; we drop to a list before
        # handing the prompt to vLLM, so the device only governs the intermediate tensor).
        self._device = model_args.pop("device", "cuda")
        # skip_tokenizer_init=True ⇒ vLLM has no tokenizer; we own one for prompt rendering
        # (via the base) and for decoding generated token ids back to strings.
        hf_token = get_hf_token()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

        # Pre-LLM: resolve architecture WITHOUT loading weights, then prime the plugin
        # so its monkey-patched Worker.load_model installs capture hooks pre-compile.
        hf_config_preview = AutoConfig.from_pretrained(model_name, token=hf_token)
        architecture = hf_config_preview.architectures[0]
        overlay = get_overlay(architecture)
        resolved_dtype = dtype if dtype is not None else _dtype_from_hf_config(hf_config_preview)
        capture_specs = overlay.capture_specs(hf_config_preview)
        plugin.configure(
            capture_specs=capture_specs,
            max_num_batched_tokens=max_num_batched_tokens,
            dtype=resolved_dtype,
            enable_batching=False,  # eager batched path is a later increment
        )
        plugin.register()
        # Single-process workers are required — otherwise the plugin's _config singleton
        # isn't visible to worker subprocesses and hooks silently fail to install.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        try:
            self._llm = LLM(
                model=model_name,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                # Worker extension's tl_* methods are reachable via collective_rpc.
                worker_extension_cls=_WORKER_EXTENSION_CLS,
                # Full-vocab logprobs so _generate_capture can synthesize logits at the
                # generated position (vLLM caps logprobs to this value; default 20 is too
                # small for mech interp).
                max_logprobs=int(hf_config_preview.vocab_size),
                dtype=str(resolved_dtype).replace("torch.", "") if dtype is not None else "auto",
                **_LOCKED_VLLM_KWARGS,
                **vllm_kwargs,
            )
        finally:
            # Always clear, even on a failed boot — stale specs would make the next
            # in-process vllm.LLM(...) walk our dot-paths on a foreign model.
            plugin.clear_config()
        # Hook installation skips modules absent on a rank; a spec that landed on
        # NO rank is a broken dot-path and must fail here, not read zeros.
        verify_hook_coverage(self._llm)
        hf_config = extract_hf_config(self._llm)

        # Capture-relevant constants used by _generate_capture.
        self._d_vocab = int(hf_config.vocab_size)
        self._max_logprobs = int(hf_config_preview.vocab_size)
        self._max_num_batched_tokens = max_num_batched_tokens

        # Boundary kinds served by the vLLM overlay (decoder-only: resid_post / attn_out /
        # mlp_out). vLLM's fused execution doesn't expose block input, so resid_pre and
        # the derived resid_mid are gated — the InspectDriver consults this via the profile.
        self._kinds = _kinds_from_specs(capture_specs)
        self._capability_note = (
            "tl_bridge_vllm: vLLM's fused execution gates resid_pre (no block-input hook) "
            "and the derived resid_mid. Use boot_inspect(provider='tl_bridge') for those."
        )
        self._eval_capture = self._parse_eval_capture(model_args)

    def _generate_capture(self, input: Any, extra_args: Mapping[str, Any], config: GenerateConfig):
        """TL-driven single-token capture: push interventions to the worker, run a
        single-token generate (vLLM's prefill populates the capture buffers), read them
        back via ``collective_rpc``, and return the wire-format ``metadata["activations"]``
        + synthesized ``tl_logits`` the InspectDriver expects."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        input_ids = extra_args.get("input_ids")
        if input_ids is None:
            input_ids = self._messages_to_ids(input)[0].tolist()
        n_tokens = len(input_ids)
        if n_tokens > self._max_num_batched_tokens:
            raise ValueError(
                f"Prompt length {n_tokens} exceeds max_num_batched_tokens="
                f"{self._max_num_batched_tokens}; raise via the model_args kwarg "
                "or shorten the prompt."
            )

        # Validate capture kinds against the structural self-check (same protection the
        # HF provider gives — driver-path AND eval/extra_args entry points).
        capture_keys = list(extra_args.get("capture", []))
        for key in capture_keys:
            _, _, kind = key.partition(":")
            _require_served(kind, self._kinds, self._capability_note, f"capture {key!r}")

        # Translate wire keys ↔ TL hook names. The worker extension is keyed by hook name
        # (e.g. "blocks.0.hook_out"); the wire format uses "<layer>:<kind>".
        name_by_wire = {wk: hooks.name_from_wire_key(wk) for wk in capture_keys}
        if any(name is None for name in name_by_wire.values()):
            unknown = sorted(wk for wk, name in name_by_wire.items() if name is None)
            raise ValueError(f"unrecognised wire keys: {unknown}")
        capture_names = list(name_by_wire.values())

        interventions: Mapping[str, Any] = extra_args.get("interventions", {})
        intervention_specs: dict[str, Any] = {}
        for wk, spec in interventions.items():
            # extra_args is a documented surface: gated/capture-only kinds must fail
            # here, not deep in the worker.
            _, _, kind = wk.partition(":")
            _require_interveneable(kind, self._kinds, self._capability_note, f"intervention {wk!r}")
            if isinstance(spec, Mapping) and spec.get("pos") is not None:
                raise ValueError(
                    f"intervention {wk!r}: per-position 'pos' is not supported on the "
                    "tl_bridge_vllm provider (its worker runs without position "
                    "interventions). Use boot_inspect(provider='tl_bridge') for "
                    "position-targeted patching."
                )
            name = hooks.name_from_wire_key(wk)
            if name is None:
                raise ValueError(f"intervention wire key {wk!r} is not a fireable hook.")
            intervention_specs[name] = spec

        want_logits = bool(extra_args.get("return_logits", True))

        # Push intervention state (possibly empty — also resets stale interventions from
        # a prior call), open the per-hook capture gates (so the prefill below writes,
        # and any later forward — should this driver be reused — would self-copy until
        # the next explicit reset). Then run a single-token prefill; vLLM's prefill
        # populates the capture buffers we registered via plugin.configure.
        self._llm.collective_rpc("tl_set_interventions", args=(intervention_specs,))
        self._llm.collective_rpc("tl_reset_capture_flags")
        outputs = self._llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=list(input_ids))],
            sampling_params=SamplingParams(
                max_tokens=1,
                temperature=0.0,
                logprobs=self._max_logprobs if want_logits else None,
            ),
        )
        # collective_rpc returns one result per worker; single-rank ⇒ [0].
        worker_captures = self._llm.collective_rpc(
            "tl_read_captures", args=([n_tokens], capture_names)
        )[0]

        # Convert TL-name-keyed (n_tokens, width) tensors → wire-key-keyed numpy arrays,
        # then encode in the same envelope wire.decode_activations consumes on the driver.
        captured_wire: dict[str, np.ndarray] = {}
        for wk, name in name_by_wire.items():
            tensor = worker_captures.get(name)
            if tensor is not None:
                captured_wire[wk] = tensor.detach().float().cpu().numpy()
        metadata: dict[str, Any] = {"activations": wire.encode_activations(captured_wire)}

        if want_logits:
            logits = _synthesize_logits(outputs[0], n_tokens, self._d_vocab)
            metadata["tl_logits"] = wire.encode_array(logits[0].cpu().numpy())

        # The completion is the single generated token (matches the HF provider's shape).
        next_id = int(outputs[0].outputs[0].token_ids[0])
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
        """vLLM generation: chat input → ``llm.generate`` → completion + Logprobs + usage.
        If ``model_args['capture']`` was set, opens per-hook capture gates before the eval
        generate so prefill captures the prompt activations and decode steps self-copy
        (first-write-wins on the worker; no separate forward — see plugin._gated_capture)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        _warn_unsupported_config(config, PROVIDER_NAME)

        # Worker-side intervention buffers are persistent: any prior capture-path call that
        # pushed specs (e.g. bridge.forward(intervene=...)) would still be applied here
        # without a reset. The HF provider installs hooks per-call so it's leak-immune.
        self._llm.collective_rpc("tl_set_interventions", args=({},))
        ids = self._messages_to_ids(input, tools)[0].tolist()
        prompt_len = len(ids)
        if self._eval_capture:
            if prompt_len > self._max_num_batched_tokens:
                raise ValueError(
                    f"Prompt length {prompt_len} exceeds max_num_batched_tokens="
                    f"{self._max_num_batched_tokens}; per-turn capture cannot snapshot it."
                )
            self._llm.collective_rpc("tl_reset_capture_flags")
        max_new = int(config.max_tokens) if config.max_tokens else 16
        temperature = float(config.temperature) if config.temperature is not None else 0.0

        sp_kwargs: dict[str, Any] = {"max_tokens": max_new, "temperature": temperature}
        if config.stop_seqs:
            sp_kwargs["stop"] = list(config.stop_seqs)
        if temperature > 0:
            if config.top_p is not None:
                sp_kwargs["top_p"] = float(config.top_p)
            if config.top_k is not None:
                sp_kwargs["top_k"] = int(config.top_k)
        if config.seed is not None:
            sp_kwargs["seed"] = int(config.seed)
        if config.logprobs:
            # vLLM returns this many top logprobs per generated token (incl. the chosen).
            sp_kwargs["logprobs"] = int(config.top_logprobs) if config.top_logprobs else 1

        outputs = self._llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=ids)],
            sampling_params=SamplingParams(**sp_kwargs),
        )
        request_output = outputs[0]
        output = request_output.outputs[0]  # one prompt, one sample
        new_ids = list(output.token_ids)
        n_new = len(new_ids)
        completion = str(self._tokenizer.decode(new_ids, skip_special_tokens=True))

        logprobs = None
        if config.logprobs and output.logprobs:
            logprobs = Logprobs(
                content=[
                    self._logprob_from_dict(int(tid), step, config.top_logprobs)
                    for tid, step in zip(new_ids, output.logprobs)
                ]
            )

        tool_calls = _parse_tool_calls(completion) if len(tools) else None
        finish = (output.finish_reason or "").lower()
        stop_reason: StopReason
        if tool_calls:
            stop_reason = "tool_calls"
        elif finish == "length":
            stop_reason = "max_tokens"
        elif finish == "stop":
            stop_reason = "stop"
        else:
            stop_reason = "unknown"

        # Per-turn capture lands in metadata. First-write-wins gating made prefill the
        # only forward that wrote to the capture buffer, so we read it now (decode steps
        # left rows 1..prompt_len-1 untouched and row 0 self-copied).
        eval_metadata: dict[str, Any] = {}
        if self._eval_capture:
            capture_names = list(self._eval_capture.values())
            worker_captures = self._llm.collective_rpc(
                "tl_read_captures", args=([prompt_len], capture_names)
            )[0]
            captured_wire: dict[str, np.ndarray] = {}
            for wk, name in self._eval_capture.items():
                tensor = worker_captures.get(name)
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

    def _logprob_from_dict(self, token_id: int, step_logprobs: Any, top_n: Any) -> Logprob:
        """vLLM per-step ``{token_id: Logprob(logprob, rank, decoded_token)}`` →
        :class:`inspect_ai.model.Logprob` with optional top-k alternatives."""
        chosen = step_logprobs.get(token_id)
        chosen_lp = float(chosen.logprob) if chosen is not None else float("-inf")
        top: list[TopLogprob] = []
        if top_n:
            ranked = sorted(step_logprobs.items(), key=lambda kv: -float(kv[1].logprob))
            for tid, lp in ranked[: int(top_n)]:
                token = lp.decoded_token or self._tokenizer.decode([int(tid)])
                top.append(TopLogprob(token=str(token), logprob=float(lp.logprob), bytes=None))
        return Logprob(
            token=str(self._tokenizer.decode([int(token_id)])),
            logprob=chosen_lp,
            bytes=None,
            top_logprobs=top,
        )

    def close(self) -> None:
        """Best-effort vLLM teardown — vLLM 0.20.2 has no ``LLM.shutdown()``, so weights
        + KV cache stay resident until process exit unless we destroy the distributed env."""
        self._llm = None
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def _dtype_from_hf_config(hf_config: Any) -> torch.dtype:
    """Best-effort dtype from an HF config — vLLM prefers fp16 on GPU."""
    raw = getattr(hf_config, "torch_dtype", None)
    if isinstance(raw, torch.dtype):
        return raw
    if isinstance(raw, str):
        return getattr(torch, raw, torch.float16)
    return torch.float16


def _synthesize_logits(request_output: Any, n_tokens: int, d_vocab: int) -> torch.Tensor:
    """Build a ``(1, n_tokens, d_vocab)`` logits-like tensor from vLLM's sampler output —
    log-probs (not raw logits), with earlier positions ``-inf`` (lm_head is bypassed so
    only the generated position is populated). Matches the HF provider's ``tl_logits``
    shape so the InspectDriver consumes both the same way."""
    logits = torch.full((1, n_tokens, d_vocab), float("-inf"), dtype=torch.float32)
    gen = request_output.outputs[0] if request_output.outputs else None
    if gen is None:
        return logits
    if gen.logprobs:
        for token_id, lp in gen.logprobs[0].items():
            logits[0, -1, int(token_id)] = float(lp.logprob)
    elif gen.token_ids:
        logits[0, -1, int(gen.token_ids[0])] = 0.0
    return logits


__all__ = ["TransformerLensVLLMModelAPI"]
