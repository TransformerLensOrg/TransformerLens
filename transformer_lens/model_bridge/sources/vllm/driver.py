"""vLLM Driver: forward dispatches via ``llm.generate``; captures via ``collective_rpc``."""
from __future__ import annotations

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


class VLLMDriver(DriverBase):
    """Driver wrapping a vLLM ``LLM``; captures via ``collective_rpc``."""

    # vLLM owns the model in a worker — no torch surface (parameters/state_dict/grads).
    _supported_features = frozenset()

    def __init__(
        self,
        llm: Any,
        adapter: Any,
        tokenizer: Any,
        overlay: Any,
        hf_config: Any,
        max_num_batched_tokens: int,
    ) -> None:
        super().__init__(adapter.cfg, tokenizer)
        self._llm = llm
        self._max_num_batched_tokens = max_num_batched_tokens

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
            raise ValueError("VLLMDriver requires input_ids")
        if int(max_new_tokens) != 1:
            raise NotImplementedError(
                "VLLMDriver supports max_new_tokens=1 only — decode-step writes "
                "overwrite the prefill buffer; multi-step capture is multi-buffer work."
            )
        intervene_specs = self._validate_interventions(intervene or {})

        ids_list = self._normalize_input_ids(input_ids)
        if len(ids_list) > self._max_num_batched_tokens:
            # Worker buffers silently clamp on overflow — fail loud here instead.
            raise ValueError(
                f"Prompt length {len(ids_list)} exceeds max_num_batched_tokens="
                f"{self._max_num_batched_tokens}; raise the boot_vllm kwarg or "
                "shorten the prompt."
            )

        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        # Push intervention state (possibly empty) before generate — this also
        # resets stale interventions from prior forwards.
        self._llm.collective_rpc("tl_set_interventions", args=(intervene_specs,))
        # Request full-vocab logprobs so the driver can populate position -1 of
        # the synthesized logits with the real next-token distribution. vLLM's
        # ``max_logprobs`` was set to d_vocab at boot to make this legal.
        outputs = self._llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=ids_list)],
            sampling_params=SamplingParams(
                max_tokens=int(max_new_tokens),
                temperature=0.0,
                logprobs=self.bridge_config.d_vocab if return_logits else None,
            ),
        )

        n_tokens = len(ids_list)
        # collective_rpc returns one result per worker; single-rank, so [0].
        worker_captures = self._llm.collective_rpc("tl_read_captures", args=([n_tokens],))[0]
        # Add batch dim: vLLM hands back (n_tokens, width); bridge expects (1, n_tokens, width).
        captured = {name: t.unsqueeze(0) for name, t in worker_captures.items()}

        logits: torch.Tensor | None = None
        if return_logits:
            logits = self._synthesize_logits(outputs[0], n_tokens, self.bridge_config.d_vocab)

        return ForwardResult(logits=logits, captured=captured, raw_output=outputs[0])

    def close(self) -> None:
        # Detach hooks before dropping the LLM so they don't stay registered on
        # worker modules for the life of the process (long-running notebooks).
        if self._llm is not None:
            try:
                self._llm.collective_rpc("tl_remove_hooks")
            except Exception as e:
                # Best-effort: engine may already be torn down or the RPC surface
                # gone. Log so hook-leak debugging has a thread to pull.
                logging.getLogger("transformer_lens.vllm").debug(
                    "tl_remove_hooks failed during close(): %s", e
                )
        self._llm = None

    @staticmethod
    def _synthesize_logits(request_output: Any, n_tokens: int, d_vocab: int) -> torch.Tensor:
        """Build a (1, n_tokens, d_vocab) logits-like tensor from vLLM's sampler output.

        vLLM's lm_head bypass means our hook never fires; the sampler returns
        full-vocab logprobs (we set ``max_logprobs=d_vocab`` at boot to allow this).
        Position -1 — the input's last token, = next-token prediction — is populated
        from those logprobs. Earlier positions stay at ``-inf`` so any argmax there
        is loud rather than silently misleading; populating them would need
        ``prompt_logprobs`` requested per call (much more expensive).
        """
        logits = torch.full((1, n_tokens, d_vocab), float("-inf"), dtype=torch.float16)
        gen = request_output.outputs[0] if request_output.outputs else None
        if gen is None:
            return logits
        # Prefer real logprobs; fall back to the generated token id (one-hot-ish)
        # if logprobs weren't requested (e.g. return_logits=False elsewhere).
        if gen.logprobs:
            for token_id, lp_obj in gen.logprobs[0].items():
                logits[0, -1, int(token_id)] = float(lp_obj.logprob)
        elif gen.token_ids:
            logits[0, -1, int(gen.token_ids[0])] = 0.0
        return logits

    def _validate_interventions(self, intervene: Mapping[str, Any]) -> dict:
        """Reject callables, validate spec format and hook names; return a plain dict."""
        out: dict = {}
        for hook_name, spec in intervene.items():
            if callable(spec):
                raise NotImplementedError(
                    "VLLMDriver requires intervention specs (dict), not callables. "
                    "Supported ops: suppress, scale (factor: float), add (value: scalar or width-shaped), "
                    "set (value: scalar or width-shaped)."
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
                raise ValueError(f"Intervention {hook_name!r}: op='scale' requires 'factor' (float).")
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
        """Coerce input_ids to a flat list[int] for ``TokensPrompt``; batch_size=1 only."""
        if isinstance(input_ids, torch.Tensor):
            ids_list = input_ids.tolist()
        else:
            ids_list = list(input_ids)
        if ids_list and isinstance(ids_list[0], list):
            if len(ids_list) != 1:
                raise NotImplementedError("VLLMDriver supports batch_size=1 only.")
            ids_list = ids_list[0]
        return ids_list
