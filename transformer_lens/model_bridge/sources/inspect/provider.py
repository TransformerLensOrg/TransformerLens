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
                        _pre_hook(layer, "resid_pre" in cap_kinds, iv_kinds.get("resid_pre"), raw)
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
    def hook(_module, inputs):
        hidden = inputs[0]
        if spec is not None:
            hidden = _apply_affine(hidden, spec)
        if want_capture:
            raw[(layer, "resid_pre")] = hidden[0].detach().float().cpu().numpy()
        if spec is None:
            return None
        return (hidden, *inputs[1:])

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
