"""Our HF-transformers ``inspect_ai`` model provider, registered as ``transformer_lens``.

The model-runner side of the Inspect driver (this file uses torch; the consuming
``InspectDriver`` does not). On ``generate`` it reads the request from
``config.extra_body["extra_args"]`` (token ids, which layers' residual stream to
capture, and intervention specs), runs an HF causal LM with forward hooks that
capture the post-block residual stream and apply affine interventions, and returns
a ``ModelOutput`` whose ``metadata`` carries the activations (wire-aligned with
vllm-lens) plus the exact last-position logits.
"""
from __future__ import annotations

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

from . import wire

# NOT "transformer_lens" — inspect_ai ships a built-in provider by that name (the
# reverse direction: serving a HookedTransformer as an Inspect model for generation).
PROVIDER_NAME = "tl_bridge"

# Attribute paths to the decoder ModuleList, by architecture family. The block's
# forward output (residual stream after the block) is hook_resid_post.
_LAYER_PATHS = (
    "model.layers",  # Llama, Mistral, Qwen, Gemma, Phi3
    "transformer.h",  # GPT2, GPT-J, GPT-Neo
    "gpt_neox.layers",  # GPT-NeoX
    "model.decoder.layers",  # OPT
)


@modelapi(name=PROVIDER_NAME)
def transformer_lens_provider():
    """Lazy registration hook — returns the provider class on first use."""
    return TransformerLensModelAPI


class TransformerLensModelAPI(ModelAPI):
    """HF-backed provider that returns residual-stream activations + last logits."""

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
        capture_layers = set(extra_args.get("output_residual_stream", range(len(self._layers))))
        interventions: Mapping[str, Any] = extra_args.get("interventions", {})

        captured: dict[int, np.ndarray] = {}
        handles = []
        for i, block in enumerate(self._layers):
            spec = interventions.get(str(i))
            if i in capture_layers or spec is not None:
                handles.append(
                    block.register_forward_hook(_make_hook(i, i in capture_layers, spec, captured))
                )

        with torch.no_grad():
            try:
                ids = torch.tensor([list(input_ids)], device=self._device)
                logits = self._hf(ids).logits  # (1, seq, vocab)
            finally:
                for handle in handles:
                    handle.remove()

        last_logits = logits[0, -1].float().cpu().numpy()
        next_id = int(last_logits.argmax())
        completion = str(self._tokenizer.decode([next_id]))

        return ModelOutput(
            model=self.model_name,
            choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=completion))],
            metadata={
                "activations": wire.encode_activations(captured),
                "tl_last_logits": wire.encode_array(last_logits),
            },
        )


def _make_hook(layer_idx: int, want_capture: bool, spec: Mapping[str, Any] | None, captured: dict):
    """Forward hook that optionally applies an affine intervention then captures resid_post."""

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        if spec is not None:
            hidden = _apply_affine(hidden, spec)
        if want_capture:
            captured[layer_idx] = hidden[0].detach().float().cpu().numpy()  # (seq, d_model)
        if spec is None:
            return None  # capture-only: leave the forward unchanged
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
    """Return the decoder-block ModuleList for common HF architectures."""
    for path in _LAYER_PATHS:
        target: Any = model
        for seg in path.split("."):
            target = getattr(target, seg, None)
            if target is None:
                break
        if target is not None:
            return target
    raise RuntimeError(
        f"Could not locate decoder layers on {type(model).__name__}; "
        f"tried {_LAYER_PATHS}. Add this architecture's path to _LAYER_PATHS."
    )


def _last_user_text(input: Any) -> str:
    """Best-effort prompt text from Inspect messages (only used when input_ids absent)."""
    if isinstance(input, str):
        return input
    for message in reversed(list(input)):
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    return ""
