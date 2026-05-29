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

from collections import defaultdict
from typing import Any, Mapping

import numpy as np
import torch
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    modelapi,
)

from . import hooks, wire

# NOT "transformer_lens" — inspect_ai ships a built-in provider by that name (the
# reverse direction: serving a HookedTransformer as an Inspect model for generation).
PROVIDER_NAME = "tl_bridge"

# Decoder ModuleList by architecture family; each block's output is resid_post.
_LAYER_PATHS = ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers")
# Attn/MLP submodule names within a block, by family.
_ATTN_ATTRS = ("self_attn", "attn", "attention")
_MLP_ATTRS = ("mlp", "feed_forward")


@modelapi(name=PROVIDER_NAME)
def transformer_lens_provider():
    """Lazy registration hook — returns the provider class on first use."""
    return TransformerLensModelAPI


class TransformerLensModelAPI(ModelAPI):
    """HF-backed provider: residual/attn/mlp capture + interventions + full logits."""

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

    def supported_kinds(self) -> frozenset:
        """Boundary kinds this model is structurally able to serve (resid_pre/resid_post
        always; attn_out/mlp_out if locatable; resid_mid only if attn feeds mlp)."""
        return self._kinds

    def capability_note(self) -> str:
        """Human-readable reason for any gated boundary, or '' if all are served."""
        return self._capability_note

    async def generate(self, input, tools, tool_choice, config):  # type: ignore[override]
        extra_args: Mapping[str, Any] = (config.extra_body or {}).get("extra_args", {})
        input_ids = extra_args.get("input_ids")
        if input_ids is None:
            input_ids = self._tokenizer(_last_user_text(input)).input_ids
        # capture/interventions are keyed by "<layer>:<kind>" (hooks.wire_key).
        capture_keys = list(extra_args.get("capture", []))
        interventions: Mapping[str, Any] = extra_args.get("interventions", {})
        want_logits = bool(extra_args.get("return_logits", True))

        capture, intervene = _plan(capture_keys, interventions)
        raw: dict[tuple[int, str], np.ndarray] = {}
        handles = self._install_hooks(capture, intervene, raw)

        with torch.no_grad():
            try:
                ids = torch.tensor([list(input_ids)], device=self._device)
                logits = self._hf(ids).logits  # (1, seq, vocab)
            finally:
                for handle in handles:
                    handle.remove()

        captured = _assemble(raw, capture_keys)
        metadata: dict[str, Any] = {"activations": wire.encode_activations(captured)}
        if want_logits:
            metadata["tl_logits"] = wire.encode_array(logits[0].float().cpu().numpy())

        next_id = int(logits[0, -1].argmax())
        completion = str(self._tokenizer.decode([next_id]))
        return ModelOutput(
            model=self.model_name,
            choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=completion))],
            metadata=metadata,
        )

    def _install_hooks(self, capture, intervene, raw) -> list:
        """Hook each block's pre/attn/mlp/post boundaries that need capture or intervention."""
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
                        _pre_hook(layer, "resid_pre" in cap_kinds, iv_kinds.get("resid_pre"), raw),
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
                        )
                    )
                )
            if mlp is not None and ("mlp_out" in cap_kinds or "mlp_out" in iv_kinds):
                handles.append(
                    mlp.register_forward_hook(
                        _out_hook(
                            layer, "mlp_out", "mlp_out" in cap_kinds, iv_kinds.get("mlp_out"), raw
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


def _pre_hook(layer, want_capture, spec, raw):
    # with_kwargs=True: hidden_states is args[0] for most decoders, but some pass it
    # as the hidden_states kwarg — handle both so the right tensor is read/modified.
    def hook(_module, args, kwargs):
        kw_key = None if args else "hidden_states"
        hidden = args[0] if args else kwargs["hidden_states"]
        if spec is not None:
            hidden = _apply_affine(hidden, spec)
        if want_capture:
            raw[(layer, "resid_pre")] = hidden[0].detach().float().cpu().numpy()
        if spec is None:
            return None
        if kw_key is None:
            return (hidden, *args[1:]), kwargs
        return args, {**kwargs, kw_key: hidden}

    return hook


def _out_hook(layer, kind, want_capture, spec, raw):
    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        if spec is not None:
            hidden = _apply_affine(hidden, spec)
        if want_capture:
            raw[(layer, kind)] = hidden[0].detach().float().cpu().numpy()
        if spec is None:
            return None
        return (hidden, *output[1:]) if is_tuple else hidden

    return hook


def _apply_affine(t: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    """suppress→0, scale→·factor, add→+value, set→value (value scalar or width-shaped)."""
    op = spec["op"]
    if op == "suppress":
        return torch.zeros_like(t)
    if op == "scale":
        return t * float(spec["factor"])
    value = torch.as_tensor(spec["value"], dtype=t.dtype, device=t.device)
    if op == "add":
        return t + value
    return torch.zeros_like(t) + value  # set


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


def _last_user_text(input: Any) -> str:
    """Best-effort prompt text from Inspect messages (only used when input_ids absent)."""
    if isinstance(input, str):
        return input
    for message in reversed(list(input)):
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    return ""
