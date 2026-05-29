"""Shared base for ``tl_bridge``-style Inspect providers (HF, vLLM).

Subclasses load their model + tokenizer, run their structural self-check, and implement
backend-specific ``_generate_capture`` / ``_generate_eval`` — this base owns the
``generate()`` dispatch, message-to-ids rendering (chat template + tools), logprob
construction, and validation of the per-turn ``capture=[...]`` config.

Subclass contract:
- Set ``self._tokenizer`` (HF-style ``AutoTokenizer``), ``self._device`` (str),
  ``self._kinds`` (frozenset of served boundary kinds), ``self._capability_note`` (str
  explaining any gating), and ``self._eval_capture`` (dict ``{wire_key: hook_name}``,
  built via :meth:`_parse_eval_capture`) before any ``generate()`` call.
- Implement ``_generate_capture(input, extra_args, config)`` (TL-driven single forward) and
  ``_generate_eval(input, config, tools)`` (multi-token chat generation).
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Mapping

import torch
from inspect_ai.model import GenerateConfig, Logprob, ModelAPI, TopLogprob
from inspect_ai.tool import ToolCall

from . import hooks


def _message_text(message: Any) -> str:
    """Text of an Inspect chat message — its ``.text`` (handles multimodal content),
    falling back to string content, else ''."""
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""


def _tool_schema(tool: Any) -> dict[str, Any]:
    """Inspect ``ToolInfo`` → OpenAI-style function schema for ``apply_chat_template``."""
    params = tool.parameters
    params = params.model_dump() if hasattr(params, "model_dump") else dict(params)
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": params},
    }


# Tool-call blocks emitted by common instruct templates (Qwen/Hermes-style); the bare-JSON
# fallback covers models that emit a single {"name", "arguments"} object.
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _parse_tool_calls(text: str) -> list[ToolCall] | None:
    """Best-effort parse of tool calls from a completion. Model-specific formats vary; this
    handles ``<tool_call>{json}</tool_call>`` blocks and a single bare ``{name, arguments}``."""
    blocks = _TOOL_CALL_BLOCK.findall(text)
    if not blocks:
        match = re.search(r"\{.*\}", text, re.S)
        blocks = [match.group(0)] if match else []
    calls = []
    for block in blocks:
        try:
            obj = json.loads(block)
        except (ValueError, TypeError):
            continue
        name = obj.get("name") if isinstance(obj, dict) else None
        if not isinstance(name, str):
            continue
        args = obj.get("arguments") or obj.get("parameters") or {}
        calls.append(
            ToolCall(
                id=uuid.uuid4().hex[:8],
                function=name,
                arguments=args if isinstance(args, dict) else {},
            )
        )
    return calls or None


def _require_served(kind: str, served: frozenset[str], note: str, context: str) -> None:
    """Raise if ``kind`` was gated by the structural self-check — without this the eval path
    would silently return a derivation (e.g. ``resid_mid``) the driver path excludes."""
    if kind not in served:
        raise ValueError(
            f"{context} requests kind {kind!r} which this model gated. {note} "
            f"Served kinds: {sorted(served)}."
        )


class _InspectModelAPIBase(ModelAPI):
    """Shared Inspect ModelAPI scaffolding for ``tl_bridge``-style providers.

    Subclasses populate ``self._tokenizer``/``_device``/``_kinds``/``_capability_note``/
    ``_eval_capture`` in ``__init__`` and implement the backend-specific generate paths.
    """

    # Attributes set by subclass __init__ before any generate() call (declared here so the
    # shared helpers' type-checking sees them; not initialized to avoid masking bugs).
    _tokenizer: Any
    _device: Any
    _kinds: frozenset
    _capability_note: str
    _eval_capture: dict[str, str]

    # --- subclass contract -------------------------------------------------------------

    def _generate_capture(
        self, input: Any, extra_args: Mapping[str, Any], config: GenerateConfig
    ) -> Any:
        """TL-driven single-forward capture (residual/attn/mlp boundaries + full logits)."""
        raise NotImplementedError

    def _generate_eval(self, input: Any, config: GenerateConfig, tools: Any) -> Any:
        """Plain Inspect generation: chat input → multi-token completion + Logprobs +
        ModelUsage (+ parsed tool calls when ``tools`` is non-empty + per-turn capture)."""
        raise NotImplementedError

    # --- shared API --------------------------------------------------------------------

    def supported_kinds(self) -> frozenset:
        """Boundary kinds this model is structurally able to serve."""
        return self._kinds

    def capability_note(self) -> str:
        """Human-readable reason for any gated boundary, or '' if all are served."""
        return self._capability_note

    async def generate(self, input, tools, tool_choice, config):  # type: ignore[override]
        # Two callers: the TL driver (extra_args carries input_ids/capture/interventions —
        # single-forward activation capture) and a plain Inspect eval (chat messages, real
        # multi-token generation). Branch on whether a TL request is present.
        extra_args: Mapping[str, Any] = (config.extra_body or {}).get("extra_args", {})
        if extra_args.get("input_ids") is not None or extra_args.get("capture"):
            return self._generate_capture(input, extra_args, config)
        return self._generate_eval(input, config, tools or [])

    # --- shared helpers ----------------------------------------------------------------

    def _parse_eval_capture(self, model_args: dict[str, Any]) -> dict[str, str]:
        """Validate ``model_args["capture"]`` against the structural self-check (same
        protection as the driver path) and key by wire key. Returns ``{wire_key: name}``."""
        eval_capture: dict[str, str] = {}
        for name in model_args.pop("capture", None) or []:
            resolved = hooks.resolve(name)
            if resolved is None:
                raise ValueError(f"capture={name!r} is not a fireable hook name.")
            _require_served(resolved[1], self._kinds, self._capability_note, f"capture={name!r}")
            eval_capture[hooks.wire_key(*resolved)] = name
        return eval_capture

    def _messages_to_ids(self, input: Any, tools: Any = ()) -> Any:
        """Render Inspect chat messages (+ any ``tools``) to input ids — chat template
        when the tokenizer has one, else newline-joined message text (e.g. gpt2)."""
        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
        else:
            messages = [
                {"role": getattr(m, "role", "user"), "content": _message_text(m)} for m in input
            ]
        template = getattr(self._tokenizer, "chat_template", None)
        if len(tools) and not template:
            raise NotImplementedError(
                f"tl_bridge: tool use needs a tool-aware chat template; {self.model_name} has "
                "none. Serve a tool-capable instruct model for agentic evals."
            )
        if template:
            kwargs: dict[str, Any] = {"add_generation_prompt": True}
            if len(tools):
                kwargs["tools"] = [_tool_schema(t) for t in tools]
            token_ids = self._tokenizer.apply_chat_template(messages, **kwargs)
        else:
            token_ids = self._tokenizer("\n".join(m["content"] for m in messages)).input_ids
        return torch.tensor([list(token_ids)], device=self._device)

    def _logprob_entry(self, token_id: int, step_logits: Any, top_n: Any) -> Logprob:
        """One token's log-prob (+ top-k alternatives) from a position's logits."""
        lp = torch.log_softmax(step_logits.float(), dim=-1)
        top = []
        if top_n:
            vals, idx = lp.topk(int(top_n))
            top = [
                TopLogprob(
                    token=str(self._tokenizer.decode([int(i)])), logprob=float(v), bytes=None
                )
                for v, i in zip(vals.tolist(), idx.tolist())
            ]
        return Logprob(
            token=str(self._tokenizer.decode([int(token_id)])),
            logprob=float(lp[int(token_id)]),
            bytes=None,
            top_logprobs=top,
        )
