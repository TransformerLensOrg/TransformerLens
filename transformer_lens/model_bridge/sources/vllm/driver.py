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
                "VLLMDriver v1 supports max_new_tokens=1 only — decode-step writes "
                "overwrite the prefill buffer; multi-step capture is multi-buffer work."
            )
        if intervene:
            raise NotImplementedError(
                "VLLMDriver intervention support not yet implemented (Phase B chunk 3+)."
            )

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
        self._llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=ids_list)],
            sampling_params=SamplingParams(max_tokens=int(max_new_tokens), temperature=0.0),
        )

        n_tokens = len(ids_list)
        # collective_rpc returns one result per worker; v1 is single-rank so [0].
        worker_captures = self._llm.collective_rpc("tl_read_captures", args=([n_tokens],))[0]
        # Add batch dim: vLLM hands back (n_tokens, width); bridge expects (1, n_tokens, width).
        captured = {name: t.unsqueeze(0) for name, t in worker_captures.items()}

        logits: torch.Tensor | None = None
        if return_logits:
            logits = captured.get("unembed.hook_out")

        return ForwardResult(logits=logits, captured=captured, raw_output=None)

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
    def _normalize_input_ids(input_ids: Any) -> list:
        """Coerce input_ids to a flat list[int] for ``TokensPrompt``. v1 is batch_size=1."""
        if isinstance(input_ids, torch.Tensor):
            ids_list = input_ids.tolist()
        else:
            ids_list = list(input_ids)
        if ids_list and isinstance(ids_list[0], list):
            if len(ids_list) != 1:
                raise NotImplementedError("VLLMDriver v1 supports batch_size=1 only.")
            ids_list = ids_list[0]
        return ids_list
