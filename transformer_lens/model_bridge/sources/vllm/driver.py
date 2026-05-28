"""vLLM Driver: forward dispatches via ``llm.generate``; captures via ``collective_rpc``."""
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


class VLLMDriver(DriverBase):
    """Driver wrapping a vLLM ``LLM``; captures via ``collective_rpc``."""

    # vLLM owns the model in a worker — no torch surface (parameters/state_dict/grads).
    _supported_features = frozenset()
    # Logits synthesized for the final position only (sampler bypass).
    provides_sequence_logits = False

    def __init__(
        self,
        llm: Any,
        adapter: Any,
        tokenizer: Any,
        overlay: Any,
        hf_config: Any,
        max_num_batched_tokens: int,
        enable_batching: bool = False,
    ) -> None:
        super().__init__(adapter.cfg, tokenizer)
        self._llm = llm
        self._max_num_batched_tokens = max_num_batched_tokens
        self._enable_batching = enable_batching
        # Logprobs per forward = real vocab (boot's max_logprobs). d_vocab can be
        # padded larger, which vLLM would reject; the logits tensor stays d_vocab.
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
            raise ValueError("VLLMDriver requires input_ids")
        if int(max_new_tokens) != 1:
            raise NotImplementedError(
                "VLLMDriver supports max_new_tokens=1 only — decode-step writes "
                "overwrite the prefill buffer; multi-step capture is multi-buffer work."
            )
        intervene_specs = self._validate_interventions(intervene or {})

        # Restrict the GPU→CPU read to these hooks (None = all). run_with_cache
        # doesn't derive this from names_filter yet — only explicit forward(capture=).
        names = list(capture) or None

        if self._enable_batching:
            return self._forward_batched(input_ids, intervene_specs, return_logits, names)

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
        # Full-vocab logprobs → position -1 of the synthesized logits (see _n_logprobs).
        outputs = self._llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=ids_list)],
            sampling_params=SamplingParams(
                max_tokens=int(max_new_tokens),
                temperature=0.0,
                logprobs=self._n_logprobs if return_logits else None,
            ),
        )

        n_tokens = len(ids_list)
        # collective_rpc returns one result per worker; single-rank, so [0].
        worker_captures = self._llm.collective_rpc("tl_read_captures", args=([n_tokens], names))[0]
        # Add batch dim: vLLM hands back (n_tokens, width); bridge expects (1, n_tokens, width).
        captured = {name: t.unsqueeze(0) for name, t in worker_captures.items()}

        logits: torch.Tensor | None = None
        if return_logits:
            logits = self._synthesize_logits(outputs[0], n_tokens, self.bridge_config.d_vocab)

        return ForwardResult(logits=logits, captured=captured, raw_output=outputs[0])

    def _forward_batched(
        self,
        input_ids: TensorLike,
        intervene_specs: dict,
        return_logits: bool,
        names: list[str] | None = None,
    ) -> ForwardResult:
        """Eager batched path: per-request capture, right-padded to (B, S, W).

        No per-prompt length gate — chunked prefill accumulates long prompts
        across forwards. Interventions are global across the batch. ``names``
        restricts the returned hooks (``None`` = all).
        """
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        prompts_ids = self._normalize_input_ids_batched(input_ids)
        prompt_lens = [len(ids) for ids in prompts_ids]

        # Reset accumulators so prior-forward chunks don't leak into the cat.
        self._llm.collective_rpc("tl_reset_accumulators")
        self._llm.collective_rpc("tl_reset_counter")
        self._llm.collective_rpc("tl_set_batched_interventions", args=(intervene_specs,))

        d_vocab = self.bridge_config.d_vocab
        outputs = self._llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=ids) for ids in prompts_ids],
            sampling_params=SamplingParams(
                max_tokens=1,
                temperature=0.0,
                logprobs=self._n_logprobs if return_logits else None,
            ),
        )

        # Keyed by req_id (no guaranteed order) — _assemble_padded joins to slot
        # k via outputs[k].request_id, not by position.
        worker_captures = self._llm.collective_rpc("tl_read_batched_captures", args=(names,))[0]
        captured = self._assemble_padded(outputs, worker_captures, prompt_lens)

        logits: torch.Tensor | None = None
        if return_logits:
            logits = self._synthesize_logits_batched(outputs, prompt_lens, d_vocab)

        return ForwardResult(logits=logits, captured=captured, raw_output=outputs)

    @staticmethod
    def _assemble_padded(
        outputs: list,
        worker_captures: Mapping[str, Mapping[str, torch.Tensor]],
        prompt_lens: list[int],
    ) -> dict[str, torch.Tensor]:
        """Stack per-request captures into right-padded ``(batch, max_seq, width)``.

        Pad is a cache-assembly artifact only: vLLM computes each request
        independently, so real-token activations don't depend on the padding.
        """
        batch = len(outputs)
        max_seq = max(prompt_lens) if prompt_lens else 0
        # Worker keys are engine-internal req_ids ("10-83c3532c"); RequestOutput
        # carries only the public id ("10"). Join exact-or-prefix; the "-" keeps
        # "1" from matching "10-...".
        worker_keys = list(worker_captures.keys())

        def _captures_for(public_rid: str) -> Mapping[str, torch.Tensor]:
            if public_rid in worker_captures:
                return worker_captures[public_rid]
            matches = [k for k in worker_keys if k.startswith(f"{public_rid}-")]
            # Raise, never silently zero-fill the row: a missing or ambiguous join
            # is indistinguishable from a genuine zero activation, which is silent
            # data loss on the collection path.
            if len(matches) != 1:
                raise RuntimeError(
                    f"Cannot join request {public_rid!r} to worker captures: found "
                    f"{len(matches)} key(s) in {worker_keys}. Expected exactly one "
                    f"(exact or '{public_rid}-<hash>')."
                )
            return worker_captures[matches[0]]

        per_slot = [_captures_for(o.request_id) for o in outputs]

        hook_names: set[str] = set()
        for caps in per_slot:
            hook_names |= set(caps.keys())

        assembled: dict[str, torch.Tensor] = {}
        for name in hook_names:
            sample = next(caps[name] for caps in per_slot if name in caps)
            buf = torch.zeros(batch, max_seq, sample.shape[-1], dtype=sample.dtype)
            for k, caps in enumerate(per_slot):
                t = caps.get(name)
                if t is not None:
                    buf[k, : t.shape[0]] = t
            assembled[name] = buf
        return assembled

    @staticmethod
    def _synthesize_logits_batched(
        outputs: list, prompt_lens: list[int], d_vocab: int
    ) -> torch.Tensor:
        """Build ``(batch, max_seq, d_vocab)`` logits; next-token dist at each row's
        ``prompt_lens[k] - 1``, never ``-1`` (a pad position for shorter prompts)."""
        batch = len(outputs)
        max_seq = max(prompt_lens) if prompt_lens else 0
        logits = torch.full((batch, max_seq, d_vocab), float("-inf"), dtype=torch.float16)
        for k, request_output in enumerate(outputs):
            gen = request_output.outputs[0] if request_output.outputs else None
            if gen is None:
                continue
            pos = prompt_lens[k] - 1
            if gen.logprobs:
                for token_id, lp_obj in gen.logprobs[0].items():
                    logits[k, pos, int(token_id)] = float(lp_obj.logprob)
            elif gen.token_ids:
                logits[k, pos, int(gen.token_ids[0])] = 0.0
        return logits

    def get_param(self, dotted_name: str) -> torch.Tensor | None:
        """Fetch a named worker tensor (e.g. ``model.norm.weight``) for conversions
        the bridge can't otherwise do (ln_final post→pre-weight; see the overlay).
        None if closed or the path is missing."""
        if self._llm is None:
            return None
        return self._llm.collective_rpc("tl_get_param", args=(dotted_name,))[0]

    def close(self) -> None:
        # Detach hooks before dropping the LLM so they don't stay registered on
        # worker modules for the life of the process (long-running notebooks).
        log = logging.getLogger("transformer_lens.vllm")
        if self._llm is not None:
            try:
                self._llm.collective_rpc("tl_remove_hooks")
            except Exception as e:
                # Best-effort: engine may already be torn down or the RPC surface
                # gone. Log so hook-leak debugging has a thread to pull.
                log.debug("tl_remove_hooks failed during close(): %s", e)
        self._llm = None
        # vLLM 0.20.2 has no LLM.shutdown() — model weights and KV cache stay
        # resident until process exit unless we explicitly tear down the
        # distributed environment vLLM set up at construction. Both calls are
        # best-effort: they're no-ops if there's no distributed state.
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception as e:
            log.debug("vLLM distributed teardown failed during close(): %s", e)
        # Free the caching allocator's blocks here so the caller doesn't have to.
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            log.debug("torch.cuda.empty_cache failed during close(): %s", e)

    @staticmethod
    def _synthesize_logits(request_output: Any, n_tokens: int, d_vocab: int) -> torch.Tensor:
        """Build a (1, n_tokens, d_vocab) logits-like tensor from vLLM's sampler output.

        Values are **log-probs**, not raw logits (vLLM returns log_softmax): fine
        for argmax/next-token, wrong for absolute scale (temperature, logit-lens).
        lm_head is bypassed so only position -1 is filled (next-token); earlier
        positions stay -inf — populating them needs prompt_logprobs per call.
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

    @staticmethod
    def _normalize_input_ids_batched(input_ids: Any) -> list[list[int]]:
        """Coerce to ``list[list[int]]`` (one per prompt); accepts 1-D/2-D tensor,
        flat list (single prompt), or ragged list-of-lists."""
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 1:
                return [input_ids.tolist()]
            return [row.tolist() for row in input_ids]
        seq = list(input_ids)
        if seq and isinstance(seq[0], (list, tuple)):
            return [list(row) for row in seq]
        return [list(seq)]
