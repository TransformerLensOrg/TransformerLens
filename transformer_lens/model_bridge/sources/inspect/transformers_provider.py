"""Our HF-transformers ``inspect_ai`` model provider, registered as ``tl_bridge``.

The model-runner side of the Inspect driver (this file uses torch; the consuming
``InspectDriver`` does not). On ``generate`` it reads the request from
``config.extra_body["extra_args"]`` (token ids, which ``<layer>:<kind>`` boundaries
to capture, and intervention specs), runs an HF causal LM with forward hooks that
capture residual/attn/mlp boundaries and apply affine interventions, and returns a
``ModelOutput`` whose ``metadata`` carries the activations (encoded by ``wire``)
plus the full-sequence logits.
"""
from __future__ import annotations

import contextvars
import uuid
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    GenerateConfig,
    Logprobs,
    ModelOutput,
    ModelUsage,
    StopReason,
    modelapi,
)

from . import wire
from ._provider_base import _InspectModelAPIBase, _parse_tool_calls, _require_served

# NOT "transformer_lens" — inspect_ai ships a built-in provider by that name (the
# reverse direction: serving a HookedTransformer as an Inspect model for generation).
PROVIDER_NAME = "tl_bridge"

# Per-call hook isolation. Capture/intervene hooks consult this contextvar and only fire
# for their own call's id — so concurrent inspect_eval samples (each running with their
# own contextvars copy via asyncio.to_thread) don't cross-pollute each other's activations.
_current_call_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "tl_inspect_call_id", default=""
)

# Decoder ModuleList by architecture family; each block's output is resid_post.
_LAYER_PATHS = ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers")
# Attn/MLP submodule names within a block, by family.
_ATTN_ATTRS = ("self_attn", "attn", "attention", "self_attention")  # self_attention: Falcon
_MLP_ATTRS = ("mlp", "feed_forward")


@modelapi(name=PROVIDER_NAME)
def transformer_lens_provider():
    """Lazy registration hook — returns the provider class on first use."""
    return TransformerLensTransformersModelAPI


class TransformerLensTransformersModelAPI(_InspectModelAPIBase):
    """HF-backed provider: residual/attn/mlp capture + interventions + full logits.

    Inherits generate() dispatch, _messages_to_ids, _logprob_entry, and the per-turn
    capture-config validation from :class:`_InspectModelAPIBase`; the HF-specific bits
    here are the model load, the forward-hook capture machinery, and the structural probe.
    """

    # Real per-position logits via direct HF forward — loss/both via RemoteBridge work.
    provides_sequence_logits = True

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(model_name, base_url, api_key, [], config)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = model_args.pop("device", "cpu")
        hf_kwargs = model_args.pop("model_kwargs", {})
        self._hf = (
            AutoModelForCausalLM.from_pretrained(model_name, **hf_kwargs).to(self._device).eval()
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._layers = _locate_layers(self._hf)
        self._kinds, self._capability_note = _detect_capabilities(self._hf, self._layers)
        # Per-turn capture during plain generation (agent rollouts): every _generate_eval
        # stashes these hooks in ModelOutput.metadata. Gated by the structural self-check.
        self._eval_capture = self._parse_eval_capture(model_args)

    def _generate_capture(self, input: Any, extra_args: Mapping[str, Any], config: GenerateConfig):
        """TL-driven single forward: capture residual/attn/mlp boundaries + full logits."""
        input_ids = extra_args.get("input_ids")
        if input_ids is None:
            input_ids = self._messages_to_ids(input)[0].tolist()
        # capture/interventions are keyed by "<layer>:<kind>" (hooks.wire_key).
        capture_keys = list(extra_args.get("capture", []))
        for key in capture_keys:
            _, _, kind = key.partition(":")
            _require_served(kind, self._kinds, self._capability_note, f"capture {key!r}")
        interventions: Mapping[str, Any] = extra_args.get("interventions", {})
        want_logits = bool(extra_args.get("return_logits", True))

        capture, intervene = _plan(capture_keys, interventions)
        raw: dict[tuple[int, str], np.ndarray] = {}
        call_id = uuid.uuid4().hex
        token = _current_call_id.set(call_id)
        handles = self._install_hooks(capture, intervene, raw, call_id)
        try:
            with torch.no_grad():
                ids = torch.tensor([list(input_ids)], device=self._device)
                logits = self._hf(ids).logits  # (1, seq, vocab)
        finally:
            for handle in handles:
                handle.remove()
            _current_call_id.reset(token)

        captured = _assemble(raw, capture_keys)
        metadata: dict[str, Any] = {"activations": wire.encode_activations(captured)}
        if want_logits:
            metadata["tl_logits"] = wire.encode_array(logits[0].float().cpu().numpy())

        next_id = int(logits[0, -1].argmax())
        logprobs = (
            Logprobs(content=[self._logprob_entry(next_id, logits[0, -1], config.top_logprobs)])
            if config.logprobs
            else None
        )
        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=str(self._tokenizer.decode([next_id]))),
                    stop_reason="stop",
                    logprobs=logprobs,
                )
            ],
            metadata=metadata,
        )

    def _generate_eval(self, input: Any, config: GenerateConfig, tools: Any):
        """Plain Inspect generation: HF generate from the chat input (rendering ``tools``
        into the template), honoring max_tokens/sampling, with optional per-token logprobs,
        token usage, parsed tool calls, and per-turn activation capture (agent rollouts)."""
        ids = self._messages_to_ids(input, tools)
        prompt_len = int(ids.shape[1])
        max_new = int(config.max_tokens) if config.max_tokens else 16
        temperature = config.temperature
        do_sample = temperature is not None and temperature > 0
        gen: dict[str, Any] = {
            "max_new_tokens": max_new,
            "do_sample": do_sample,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        }
        if temperature is not None and temperature > 0:
            gen["temperature"] = float(temperature)
            if config.top_p is not None:
                gen["top_p"] = float(config.top_p)
            if config.top_k is not None:
                gen["top_k"] = int(config.top_k)

        # Save BOTH CPU and CUDA RNG state — get_rng_state() is CPU-only, so seeding on a
        # CUDA model would otherwise leak its CUDA seed past this generate.
        rng_state = None
        cuda_rng_state = None
        on_cuda = "cuda" in str(self._device)
        if do_sample and config.seed is not None:
            rng_state = torch.get_rng_state()
            if on_cuda:
                cuda_rng_state = torch.cuda.get_rng_state_all()
            torch.manual_seed(int(config.seed))
        # Install per-turn capture hooks AROUND generate (not pre-): first-write-wins lets
        # the prompt forward populate them and decode forwards skip — no extra forward.
        with self._eval_capture_scope() as metadata:
            try:
                with torch.no_grad():
                    out = self._hf.generate(ids, **gen)
            finally:
                if rng_state is not None:
                    torch.set_rng_state(rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_state)

        new_ids = out.sequences[0, prompt_len:]
        completion = str(self._tokenizer.decode(new_ids, skip_special_tokens=True))
        logprobs = None
        if config.logprobs:
            content = [
                self._logprob_entry(int(tok), step[0], config.top_logprobs)
                for tok, step in zip(new_ids.tolist(), out.scores)
            ]
            logprobs = Logprobs(content=content)
        n_new = int(new_ids.shape[0])
        tool_calls = _parse_tool_calls(completion) if len(tools) else None
        eos = self._tokenizer.eos_token_id
        stop_reason: StopReason
        if tool_calls:
            stop_reason = "tool_calls"
        elif n_new and eos is not None and int(new_ids[-1]) == eos:
            stop_reason = "stop"
        else:
            stop_reason = "max_tokens"
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
            metadata=metadata or None,
        )

    @contextmanager
    def _eval_capture_scope(self) -> Iterator[dict[str, Any]]:
        """Install per-turn ``capture=[...]`` hooks for the duration of a generate. First-
        write-wins lets the prompt forward populate the raw dict (decode forwards find it
        populated and skip), so this adds no extra forward. Yields a metadata dict (empty
        when capture isn't configured) that the caller folds into ``ModelOutput.metadata``.
        Contextvar-isolated so concurrent inspect_eval samples don't cross-pollute."""
        if not self._eval_capture:
            yield {}
            return
        wire_keys = list(self._eval_capture)
        capture, _ = _plan(wire_keys, {})
        raw: dict[tuple[int, str], np.ndarray] = {}
        call_id = uuid.uuid4().hex
        token = _current_call_id.set(call_id)
        handles = self._install_hooks(capture, {}, raw, call_id)
        metadata: dict[str, Any] = {}
        try:
            yield metadata
        finally:
            for handle in handles:
                handle.remove()
            _current_call_id.reset(token)
            metadata["activations"] = wire.encode_activations(_assemble(raw, wire_keys))

    def _install_hooks(self, capture, intervene, raw, call_id: str) -> list:
        """Hook each block's pre/attn/mlp/post boundaries that need capture or intervention.
        Hooks consult ``_current_call_id`` and only fire for ``call_id`` (concurrent calls)."""
        handles = []
        for layer, block in enumerate(self._layers):
            cap_kinds = capture.get(layer, set())
            iv_kinds = intervene.get(layer, {})
            if not cap_kinds and not iv_kinds:
                continue
            attn = _first_attr(block, _ATTN_ATTRS)
            mlp = _first_attr(block, _MLP_ATTRS)
            if "resid_pre" in cap_kinds or "resid_pre" in iv_kinds:
                handles.append(
                    block.register_forward_pre_hook(
                        _pre_hook(
                            layer, "resid_pre" in cap_kinds, iv_kinds.get("resid_pre"), raw, call_id
                        ),
                        with_kwargs=True,
                    )
                )
            if attn is not None and ("attn_out" in cap_kinds or "attn_out" in iv_kinds):
                handles.append(
                    attn.register_forward_hook(
                        _out_hook(
                            layer,
                            "attn_out",
                            "attn_out" in cap_kinds,
                            iv_kinds.get("attn_out"),
                            raw,
                            call_id,
                        )
                    )
                )
            if mlp is not None and ("mlp_out" in cap_kinds or "mlp_out" in iv_kinds):
                handles.append(
                    mlp.register_forward_hook(
                        _out_hook(
                            layer,
                            "mlp_out",
                            "mlp_out" in cap_kinds,
                            iv_kinds.get("mlp_out"),
                            raw,
                            call_id,
                        )
                    )
                )
            if "resid_post" in cap_kinds or "resid_post" in iv_kinds:
                handles.append(
                    block.register_forward_hook(
                        _out_hook(
                            layer,
                            "resid_post",
                            "resid_post" in cap_kinds,
                            iv_kinds.get("resid_post"),
                            raw,
                            call_id,
                        )
                    )
                )
        return handles


def _plan(capture_keys, interventions):
    """Resolve wire keys → per-layer kinds to capture (resid_mid needs pre+attn) and intervene."""
    capture: dict[int, set[str]] = defaultdict(set)
    for key in capture_keys:
        layer, _, kind = key.partition(":")
        layer = int(layer)
        if kind == "resid_mid":
            capture[layer] |= {"resid_pre", "attn_out"}  # derived = resid_pre + attn_out
        else:
            capture[layer].add(kind)
    intervene: dict[int, dict[str, Any]] = defaultdict(dict)
    for key, spec in interventions.items():
        layer, _, kind = key.partition(":")
        intervene[int(layer)][kind] = spec
    return capture, intervene


def _assemble(raw, capture_keys) -> dict[str, np.ndarray]:
    """Build the emitted ``{wire_key: (seq, d)}`` map, deriving resid_mid as needed."""
    out: dict[str, np.ndarray] = {}
    for key in capture_keys:
        layer, _, kind = key.partition(":")
        layer = int(layer)
        if kind == "resid_mid":
            pre, attn = raw.get((layer, "resid_pre")), raw.get((layer, "attn_out"))
            if pre is not None and attn is not None:
                out[key] = pre + attn
        elif (layer, kind) in raw:
            out[key] = raw[(layer, kind)]
    return out


def _pre_hook(layer, want_capture, spec, raw, call_id):
    # with_kwargs=True: hidden_states is args[0] for most decoders, but some pass it as
    # the hidden_states kwarg — handle both so the right tensor is read/modified.
    # First-write-wins on raw so install-around-generate captures the prompt forward
    # (subsequent decode forwards find raw populated and skip — no extra prompt forward).
    def hook(_module, args, kwargs):
        if _current_call_id.get() != call_id:
            return None  # different concurrent call's hook
        kw_key = None if args else "hidden_states"
        hidden = args[0] if args else kwargs["hidden_states"]
        if spec is not None:
            hidden = _apply_affine(hidden, spec)
        if want_capture and (layer, "resid_pre") not in raw:
            raw[(layer, "resid_pre")] = hidden[0].detach().float().cpu().numpy()
        if spec is None:
            return None
        if kw_key is None:
            return (hidden, *args[1:]), kwargs
        return args, {**kwargs, kw_key: hidden}

    return hook


def _out_hook(layer, kind, want_capture, spec, raw, call_id):
    def hook(_module, _inputs, output):
        if _current_call_id.get() != call_id:
            return None  # different concurrent call's hook
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        if spec is not None:
            hidden = _apply_affine(hidden, spec)
        if want_capture and (layer, kind) not in raw:
            raw[(layer, kind)] = hidden[0].detach().float().cpu().numpy()
        if spec is None:
            return None
        return (hidden, *output[1:]) if is_tuple else hidden

    return hook


def _affine_op(sub: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    """One affine op: suppress→0, scale→·factor, add→+value, set→value. ``value`` broadcasts
    (scalar, width-shaped, or per-position ``(n_pos, width)``)."""
    op = spec["op"]
    if op == "suppress":
        return torch.zeros_like(sub)
    if op == "scale":
        return sub * float(spec["factor"])
    value = torch.as_tensor(spec["value"], dtype=sub.dtype, device=sub.device)
    if op == "add":
        return sub + value
    return torch.zeros_like(sub) + value  # set


def _apply_affine(t: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    """Affine intervention on a captured tensor ``(..., seq, width)``.

    Without ``pos`` the op spans every position (the original width-broadcast form). With
    ``pos`` (an int or list of sequence indices) it touches only those positions — the
    activation-patching primitive — and ``value`` may be per-position ``(len(pos), width)``
    to transplant a captured activation (path/causal tracing) rather than a single vector.
    """
    pos = spec.get("pos")
    if pos is None:
        return _affine_op(t, spec)
    idx = [pos] if isinstance(pos, int) else list(pos)
    out = t.clone()
    out[..., idx, :] = _affine_op(t[..., idx, :], spec)
    return out


def _detect_capabilities(model: Any, layers: Any) -> tuple[frozenset, str]:
    """Structural self-check: which boundary kinds this model can serve faithfully.

    resid_pre/resid_post are the block in/out (always); attn_out/mlp_out need their
    submodules locatable; resid_mid is gated unless its derivation holds (see
    :func:`_resid_mid_derivable`). Returns (kinds, note); note explains any gating, '' if none.
    """
    block = layers[0]
    attn = _first_attr(block, _ATTN_ATTRS)
    mlp = _first_attr(block, _MLP_ATTRS)
    kinds = {"resid_pre", "resid_post"}
    gated = []
    if attn is not None:
        kinds.add("attn_out")
    else:
        gated.append("attn_out (no attention submodule found)")
    if mlp is not None:
        kinds.add("mlp_out")
    else:
        gated.append("mlp_out (no MLP submodule found)")
    if attn is not None and mlp is not None and _resid_mid_derivable(model, block, attn, mlp):
        kinds.add("resid_mid")
    else:
        gated.append(
            "resid_mid (resid_pre + attn_out doesn't hold — parallel or norm-variant block)"
        )
    note = (
        ""
        if not gated
        else "InspectDriver: this architecture's block layout gates "
        + ", ".join(gated)
        + ". Remaining boundaries are served; use boot_transformers() for the gated ones."
    )
    return frozenset(kinds), note


def _resid_mid_derivable(model: Any, block: Any, attn: Any, mlp: Any) -> bool:
    """True iff ``resid_mid = resid_pre + attn_out`` holds, via two tiny probe forwards.
    Requires both the linear identity ``resid_post = resid_pre + attn_out + mlp_out`` (broken
    by post-norm/multiplier blocks — Gemma2/OLMo2/Granite) and attn feeding mlp (broken by
    parallel blocks — GPTNeoX/GPT-J, where mlp reads resid_pre directly)."""
    cap: dict[str, Any] = {}

    def grab(key: str):  # type: ignore[no-untyped-def]
        def hook(_m: Any, _i: Any, out: Any) -> None:
            t = out[0] if isinstance(out, tuple) else out
            cap[key] = t.detach().float()

        return hook

    def grab_in(key: str):  # type: ignore[no-untyped-def]
        def hook(_m: Any, args: Any, kwargs: Any) -> None:
            t = args[0] if args else kwargs.get("hidden_states")
            cap[key] = None if t is None else t.detach().float()

        return hook

    def perturb_attn(_m: Any, _i: Any, out: Any):  # type: ignore[no-untyped-def]
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        # Local generator: the probe must not reset the caller's global RNG. Non-uniform
        # noise so layernorm's mean-subtraction can't cancel it (a constant would).
        gen = torch.Generator(device=h.device).manual_seed(0)
        h = h + torch.empty_like(h).normal_(generator=gen)
        return (h, *out[1:]) if is_tuple else h

    ids = torch.tensor([[0, 1, 2]], device=next(model.parameters()).device)
    try:
        with torch.no_grad():
            handles = [
                block.register_forward_pre_hook(grab_in("resid_pre"), with_kwargs=True),
                attn.register_forward_hook(grab("attn_out")),
                mlp.register_forward_hook(grab("mlp_out")),
                mlp.register_forward_pre_hook(grab_in("mlp_in"), with_kwargs=True),
                block.register_forward_hook(grab("resid_post")),
            ]
            model(ids)
            for h in handles:
                h.remove()
            mlp_in_clean = cap.pop("mlp_in", None)
            handles = [
                mlp.register_forward_pre_hook(grab_in("mlp_in"), with_kwargs=True),
                attn.register_forward_hook(perturb_attn),
            ]
            model(ids)
            for h in handles:
                h.remove()
            mlp_in_perturbed = cap.get("mlp_in")
    except Exception:
        return False  # can't probe (exotic signature) → conservatively gate resid_mid

    rp = cap.get("resid_pre")
    ao = cap.get("attn_out")
    mo = cap.get("mlp_out")
    rpost = cap.get("resid_post")
    if rp is None or ao is None or mo is None or rpost is None:
        return False
    if mlp_in_clean is None or mlp_in_perturbed is None:
        return False
    if not (rp.shape == ao.shape == mo.shape == rpost.shape == mlp_in_clean.shape):
        return False
    # (1) sub-block outputs add to the residual without intervening norm/scale.
    identity = (rpost - rp - ao - mo).abs().max().item()
    identity_ok = identity <= 1e-3 * (rpost.abs().max().item() + 1e-6)
    # (2) perturbing attn moves mlp's input (sequential, not parallel).
    causal_ok = (mlp_in_clean - mlp_in_perturbed).abs().max().item() > 1e-6
    return bool(identity_ok and causal_ok)


def _locate_layers(model: Any) -> Any:
    for path in _LAYER_PATHS:
        target: Any = model
        for seg in path.split("."):
            target = getattr(target, seg, None)
            if target is None:
                break
        if target is not None:
            return target
    raise RuntimeError(
        f"Could not locate decoder layers on {type(model).__name__}; tried {_LAYER_PATHS}."
    )


def _first_attr(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        found = getattr(obj, name, None)
        if found is not None:
            return found
    return None
