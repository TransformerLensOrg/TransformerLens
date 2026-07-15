"""vLLM Driver: forward dispatches via ``llm.generate``; captures via ``collective_rpc``."""
from __future__ import annotations

import gc
import logging
import threading
import warnings
from typing import Any, Mapping, Optional, Sequence, Union

import torch

from transformer_lens.model_bridge.driver_protocol import (
    ForwardResult,
    Intervention,
    TensorLike,
)
from transformer_lens.model_bridge.sources._driver_base import DriverBase

from .intervention_specs import validate_spec
from .worker_extension import _TL_TENSOR_KEY, decode_tensor

# vLLM's distributed teardown operates on process-wide globals, so close() may only run
# it when no other VLLMDriver-owned engine is alive (notebook re-binding boots B before
# dropping A; A's __del__ must not destroy the process groups B is using).
_LIVE_DRIVERS = 0
_LIVE_DRIVERS_LOCK = threading.Lock()


class VLLMDriver(DriverBase):
    """Driver wrapping a vLLM ``LLM``; captures via ``collective_rpc``."""

    # vLLM owns the model in a worker — no torch module surface (parameters/state_dict/
    # grads). Named-weight reads ARE served: get_param() returns CPU clones via the
    # tl_get_param RPC (what logit reconstruction and direct-logit-attribution use).
    _supported_features = frozenset({"weight_access"})
    # Full-sequence logits reconstructed host-side (ln_final @ lm_head.weight.T); vLLM's
    # sampler only hands back the final position, so the driver rebuilds the rest.
    provides_sequence_logits = True

    # Post-weight final-norm capture that lm_head consumes — reconstruction reads it.
    _LN_FINAL = "ln_final.hook_normalized"

    def __init__(
        self,
        llm: Any,
        adapter: Any,
        tokenizer: Any,
        overlay: Any,
        hf_config: Any,
        max_num_batched_tokens: int,
        enable_batching: bool = False,
        enable_position_interventions: bool = False,
        tensor_parallel_size: int = 1,
    ) -> None:
        super().__init__(adapter.cfg, tokenizer)
        self._llm = llm
        self._max_num_batched_tokens = max_num_batched_tokens
        self._enable_batching = enable_batching
        # Every overlay hook point is post-all-reduce (replicated across TP ranks),
        # so captures read rank 0; the first capture-bearing forward cross-checks
        # all ranks (see _verify_tp_replication) and then trusts rank 0.
        self._tp_size = tensor_parallel_size
        self._tp_verified = tensor_parallel_size <= 1
        # Position-scoped 'pos' interventions need (max_n, width) affine buffers,
        # allocated at boot only when this is set (see plugin.patched_load_model).
        self._enable_position_interventions = enable_position_interventions
        # Logprobs per forward = real vocab (boot's max_logprobs). d_vocab can be
        # padded larger, which vLLM would reject; the logits tensor stays d_vocab.
        self._n_logprobs = int(getattr(hf_config, "vocab_size", self.bridge_config.d_vocab))
        # Unembedding cache: (weight_fp32, bias_fp32|None) once probe_logit_reconstruction
        # runs — a per-forward re-fetch clones a d_vocab×d_model tensor to CPU every call.
        self._unembed: tuple[torch.Tensor, Any] | None = None
        self._unembed_probed = False
        # fp32 reciprocal of the guarded final-norm weight; None after a failed probe.
        self._lnf_inv_denom: torch.Tensor | None = None
        self._lnf_probed = False

        capture_specs = overlay.capture_specs(hf_config)
        self._hook_widths = {name: width for name, (_path, width) in capture_specs.items()}
        self._capture_paths = {name: path for name, (path, _width) in capture_specs.items()}
        self.supported_hook_points = frozenset(capture_specs.keys())

        global _LIVE_DRIVERS
        with _LIVE_DRIVERS_LOCK:
            _LIVE_DRIVERS += 1

        n_layers = getattr(hf_config, "num_hidden_layers", 0)
        if not isinstance(n_layers, int) or n_layers <= 0:
            # A raw "{i}" template would land unexpanded in non_fireable_hook_points —
            # a broken config should fail at boot, not surface as a garbled hook name.
            raise ValueError(
                f"VLLMDriver: hf_config.num_hidden_layers={n_layers!r} — expected a "
                "positive int; the config is missing or malformed."
            )
        nonfiring: list[str] = []
        for tmpl in overlay.nonfiring_hooks():
            if "{i}" in tmpl:
                nonfiring.extend(tmpl.replace("{i}", str(i)) for i in range(n_layers))
            else:
                nonfiring.append(tmpl)
        self.non_fireable_hook_points = frozenset(nonfiring)

    def forward(
        self,
        # Wider than the protocol's TensorLike: the batched path documents plain
        # (ragged) list[int] / list[list[int]] prompts, which have no shape/dtype.
        input_ids: Optional[Union[TensorLike, Sequence[Any]]] = None,
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
        # Pad tokens fed to vLLM are real content to it (no mask concept) — honor a
        # caller-supplied mask by trimming rows to their true lengths, never swallow it.
        attention_mask = kwargs.pop("attention_mask", None)

        # capture is authoritative — the bridge sends exactly the hooked names, so ()
        # means "capture nothing" and a plain forward(tokens) skips the GPU→CPU copy
        # entirely. (The worker's None-means-all convention is never triggered from here;
        # an empty tuple used to collapse to None and silently copy every buffer.)
        names = list(capture)

        if self._enable_batching:
            return self._forward_batched(
                input_ids, intervene_specs, return_logits, names, attention_mask
            )

        ids_list = self._normalize_input_ids(input_ids)
        if attention_mask is not None:
            n_real = int(torch.as_tensor(attention_mask).sum())
            ids_list = ids_list[:n_real]
        if len(ids_list) > self._max_num_batched_tokens:
            # Worker buffers silently clamp on overflow — fail loud here instead.
            raise ValueError(
                f"Prompt length {len(ids_list)} exceeds max_num_batched_tokens="
                f"{self._max_num_batched_tokens}; raise the boot_vllm kwarg or "
                "shorten the prompt."
            )
        # 'pos'-scoped edits target affine-buffer rows, but the compiled hook only reads
        # rows [0, len(ids_list)); a pos past the prompt length would be a silent no-op.
        self._reject_pos_beyond_seq(intervene_specs, len(ids_list))

        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        # Push intervention state (possibly empty) before generate — this also
        # resets stale interventions from prior forwards.
        self._llm.collective_rpc("tl_set_interventions", args=(intervene_specs,))
        # Open per-hook capture gates; first-write-wins means a fresh prefill writes and
        # subsequent forwards self-copy. Without this, repeated bridge.forward calls would
        # see the gate closed from the prior call and read stale buffers.
        self._llm.collective_rpc("tl_reset_capture_flags")
        outputs = self._llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=ids_list)],
            sampling_params=SamplingParams(
                max_tokens=int(max_new_tokens),
                temperature=0.0,
                logprobs=self._sampler_logprobs(return_logits),
            ),
        )

        n_tokens = len(ids_list)
        read_names = self._read_names(names, return_logits)
        # collective_rpc returns one result per worker; captures are replicated across
        # TP ranks (post-all-reduce hook points), so rank 0 is authoritative. Nothing
        # to read (no captures, logits off) → skip the crossing altogether.
        if read_names:
            per_rank = [
                self._rpc_captures(caps)
                for caps in self._llm.collective_rpc(
                    "tl_read_captures", args=([n_tokens], read_names)
                )
            ]
            if not self._tp_verified:
                self._verify_tp_replication(per_rank)
                self._tp_verified = True
            worker_captures = per_rank[0]
        else:
            worker_captures = {}

        logits: torch.Tensor | None = None
        if return_logits:
            recon = self._reconstruct_logits(worker_captures.get(self._LN_FINAL))
            # Fall back to final-position log-probs if the unembedding isn't fetchable.
            logits = (
                recon.unsqueeze(0)
                if recon is not None
                else self._synthesize_logits(outputs[0], n_tokens, self.bridge_config.d_vocab)
            )

        captured = self._expose_captured(
            {name: t.unsqueeze(0) for name, t in worker_captures.items()}, names
        )
        return ForwardResult(logits=logits, captured=captured, raw_output=outputs[0])

    def _sampler_logprobs(self, return_logits: bool) -> int | None:
        # Sampler logprobs are the reconstruction fallback only; skipping them avoids
        # marshaling a d_vocab-entry Logprob dict host-side per request.
        return self._n_logprobs if return_logits and self._unembed is None else None

    def _read_names(self, names: list[str], return_logits: bool) -> list[str]:
        # Reconstruction needs ln_final even when uncaptured; skip once probed-unavailable.
        recon_possible = self._unembed is not None or not self._unembed_probed
        if return_logits and recon_possible and self._LN_FINAL not in names:
            return names + [self._LN_FINAL]
        return names

    def _expose_captured(
        self, captured: Mapping[str, torch.Tensor], names: list[str]
    ) -> dict[str, torch.Tensor]:
        """Filter to the caller's requested hooks (dropping any forced ln_final) and
        convert the exposed ln_final to the pre-weight convention its name promises —
        reconstruction consumed the raw post-weight value already."""
        out = {name: t for name, t in captured.items() if name in names}
        if self._LN_FINAL in out:
            out[self._LN_FINAL] = self._unfold_ln_final(out[self._LN_FINAL])
        return out

    def _forward_batched(
        self,
        input_ids: Union[TensorLike, Sequence[Any]],
        intervene_specs: dict,
        return_logits: bool,
        names: list[str],
        attention_mask: Any = None,
    ) -> ForwardResult:
        """Eager batched path: per-request capture, right-padded to (B, S, W).

        No per-prompt length gate — chunked prefill accumulates long prompts
        across forwards. Interventions are global across the batch. ``names`` is
        authoritative: exactly the hooks to return (empty = none).
        """
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        prompts_ids = self._normalize_input_ids_batched(input_ids)
        if attention_mask is not None:
            # Right-padded tensor batches carry pad ids vLLM would treat as content;
            # trim each row to its masked length so per-row final positions are real.
            mask = torch.as_tensor(attention_mask)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.shape[0] != len(prompts_ids):
                raise ValueError(
                    f"attention_mask batch dim {mask.shape[0]} != number of prompts "
                    f"{len(prompts_ids)}."
                )
            prompts_ids = [ids[: int(row.sum())] for ids, row in zip(prompts_ids, mask)]
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
                logprobs=self._sampler_logprobs(return_logits),
            ),
        )

        read_names = self._read_names(names, return_logits)
        # Keyed by req_id (no guaranteed order) — _assemble_padded joins to slot
        # k via outputs[k].request_id, not by position. Empty read → skip both the
        # crossing AND the join (_assemble_padded requires one worker key per request,
        # so it can't run on an empty dict — mirror the single path's empty captured).
        if read_names:
            worker_captures = self._llm.collective_rpc(
                "tl_read_batched_captures", args=(read_names,)
            )[0]
            # Batched runs single-rank today, but coerce anyway — multiproc RPC
            # serializes tensors to lists (see _rpc_tensor).
            worker_captures = {
                req_id: self._rpc_captures(caps) for req_id, caps in worker_captures.items()
            }
            captured = self._assemble_padded(outputs, worker_captures, prompt_lens)
        else:
            captured = {}

        logits: torch.Tensor | None = None
        if return_logits:
            recon = self._reconstruct_logits(
                captured.get(self._LN_FINAL)
            )  # (batch, max_seq, d_vocab)
            if recon is not None:
                # Pad rows reconstruct from zero-filled ln_final into finite garbage
                # (0 @ W = plausible uniform logits) — mask them to the -inf convention
                # the fallback and per-row consumers rely on.
                for k, n in enumerate(prompt_lens):
                    recon[k, n:] = float("-inf")
            logits = (
                recon
                if recon is not None
                else self._synthesize_logits_batched(outputs, prompt_lens, d_vocab)
            )
        captured = self._expose_captured(captured, names)

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
        Rank-0 read — correct for replicated params only; vocab-sharded weights go
        through ``_gather_param``. None if closed or the path is missing."""
        if self._llm is None:
            return None
        value = self._llm.collective_rpc("tl_get_param", args=(dotted_name,))[0]
        return self._rpc_tensor(value) if value is not None else None

    @staticmethod
    def _rpc_tensor(value: Any) -> torch.Tensor:
        """Decode a non-None collective_rpc payload back to a tensor.

        Worker methods return the explicit wire format (see
        ``worker_extension.encode_tensor``) because vLLM's multiproc RPC can't
        round-trip raw tensors. Raw tensors still pass through for mocks and any
        legacy in-process payloads; anything else is best-effort coerced."""
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, Mapping) and value.get(_TL_TENSOR_KEY):
            return decode_tensor(dict(value))
        return torch.as_tensor(value)

    @classmethod
    def _rpc_captures(cls, captures: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        return {name: cls._rpc_tensor(value) for name, value in captures.items()}

    def _gather_param(self, dotted_name: str, dim: int = 0) -> torch.Tensor | None:
        """Fetch a param across all TP ranks: replicated → rank 0; sharded → concat.

        Vocab-parallel weights (``lm_head.weight``, ``embed_tokens.weight``) hold
        contiguous per-rank slices in rank order, which is also collective_rpc's
        result order — concatenation along ``dim`` reassembles the full tensor
        (vocab padding lands in the tail rows, sliced off by reconstruction).
        """
        if self._llm is None:
            return None
        shards = [
            self._rpc_tensor(s)
            for s in self._llm.collective_rpc("tl_get_param", args=(dotted_name,))
            if s is not None
        ]
        if not shards:
            return None
        if len(shards) == 1:
            return shards[0]
        if all(s.shape == shards[0].shape and torch.equal(s, shards[0]) for s in shards[1:]):
            return shards[0]  # replicated (norm weights, biases on some archs)
        return torch.cat(shards, dim=dim)

    def _verify_tp_replication(self, per_rank_captures: list) -> None:
        """One-time cross-rank check that captured values really are replicated.

        Every hook point the overlay declares is post-all-reduce today; if a future
        vLLM moves one pre-all-reduce, rank-0 reads would silently return sharded
        garbage — compare all ranks on the first capture-bearing forward instead.
        Also compares per-rank fire counters (a rank whose hooks half-installed
        would diverge there first).
        """
        rank0 = per_rank_captures[0]
        for rank, captures in enumerate(per_rank_captures[1:], start=1):
            for name, t0 in rank0.items():
                t_r = captures.get(name)
                if (
                    t_r is None
                    or t0.shape != t_r.shape
                    or not torch.allclose(t0.float(), t_r.float(), atol=1e-5, rtol=1e-5)
                ):
                    diff = (
                        (t0.float() - t_r.float()).abs().max().item()
                        if t_r is not None and t0.shape == t_r.shape
                        else float("inf")
                    )
                    raise RuntimeError(
                        f"TP replication check failed for {name!r}: rank 0 vs rank {rank} "
                        f"max abs diff {diff:.3e}. This hook point is no longer replicated "
                        "across ranks (vLLM may have moved it pre-all-reduce) — rank-0 "
                        "capture reads would be silently wrong. Use tensor_parallel_size=1 "
                        "and report this."
                    )
        counters = self._llm.collective_rpc("tl_read_counter")
        if len(set(int(c) for c in counters)) > 1:
            raise RuntimeError(
                f"TP fire-counter mismatch across ranks: {counters}. Some rank's capture "
                "hooks fired a different number of times — per-rank installation is "
                "inconsistent. Use tensor_parallel_size=1 and report this."
            )

    def probe_logit_reconstruction(self) -> bool:
        """One-time unembedding fetch + cache; downgrades ``provides_sequence_logits``
        honestly when no unembedding is reachable (the fallback path is final-position
        log-probs, which cannot back a loss). Idempotent — later calls return the
        cached availability."""
        if self._unembed_probed:
            return self._unembed is not None
        self._unembed_probed = True
        # Gathered reads: under TP the unembedding is vocab-sharded per rank.
        weight = self._gather_param("lm_head.weight")
        if weight is None:  # tied embeddings expose no separate lm_head
            weight = self._gather_param("model.embed_tokens.weight")
        if weight is None:
            self.provides_sequence_logits = False
            return False
        bias = self._gather_param("lm_head.bias")
        d_vocab = int(self.bridge_config.d_vocab)
        # Slice vLLM's vocab-pad rows at cache time; fp32 residency (~2× checkpoint
        # dtype on CPU) trades memory for skipping a full-matrix upcast per forward.
        self._unembed = (
            weight.to(torch.float32)[:d_vocab],
            bias.to(torch.float32)[:d_vocab] if bias is not None else None,
        )
        self.provides_sequence_logits = True
        return True

    def _reconstruct_logits(self, ln_final: Any) -> torch.Tensor | None:
        """Rebuild real logits from the captured post-weight ln_final:
        ``ln_final @ lm_head.weight.T`` (+ bias, + Gemma-family tanh soft-cap).

        vLLM's ``ln_final.hook_normalized`` is the POST-weight RMSNorm value lm_head
        consumes (verified empirically: it equals HF's pre-weight value times the norm
        weight), so no un-fold is needed here — the raw worker value feeds this directly.
        Accepts any ``(..., d_model)`` tensor and returns ``(..., d_vocab)`` on CPU.
        ``None`` if ln_final wasn't captured or no unembedding weight is fetchable —
        the caller then falls back to the sampler's log-probs.
        """
        if ln_final is None or not self.probe_logit_reconstruction():
            return None
        assert self._unembed is not None  # probe returned True
        weight32, bias32 = self._unembed
        lf = ln_final.to(device=weight32.device, dtype=torch.float32)
        logits = lf @ weight32.T
        if bias32 is not None:
            logits = logits + bias32.to(device=logits.device)
        d_vocab = int(self.bridge_config.d_vocab)
        if logits.shape[-1] > d_vocab:
            # vLLM pads vocab embeddings to a multiple of 64; its own sampler slices to
            # org_vocab_size before sampling — mirror that, or pad columns (zero-filled
            # at load) become phantom argmax candidates and bias softmax denominators.
            logits = logits[..., :d_vocab]
        cap = getattr(self.bridge_config, "output_logits_soft_cap", None)
        if cap is not None and cap > 0:  # Gemma-family cap; -1.0 is the "disabled" sentinel
            logits = float(cap) * torch.tanh(logits / float(cap))
        if logits.shape[-1] < d_vocab:  # pad the padded-vocab tail (never predicted)
            pad = logits.new_full((*logits.shape[:-1], d_vocab - logits.shape[-1]), float("-inf"))
            logits = torch.cat([logits, pad], dim=-1)
        return logits.cpu()

    def _unfold_ln_final(self, t: torch.Tensor) -> torch.Tensor:
        """Convert vLLM's post-weight RMSNorm capture to the pre-weight value the
        canonical hook name promises (÷ weight; Gemma folds ``1 + weight``). Warns
        once and serves the raw value when the norm weight is unreachable — loud
        beats silent cross-backend mismatch."""
        if not self._lnf_probed:
            self._lnf_probed = True
            # The overlay's capture spec owns the module path; derive the weight from it.
            path = self._capture_paths.get(self._LN_FINAL, "model.norm")
            weight = self.get_param(f"{path}.weight")
            if weight is None:
                warnings.warn(
                    "ln_final.hook_normalized: norm weight unreachable — the captured "
                    "value stays POST-weight and will not match boot_transformers.",
                    UserWarning,
                    stacklevel=3,
                )
            else:
                w = weight.detach().to(torch.float32)
                denom = (1.0 + w) if "gemma" in self.architecture.lower() else w
                # Near-zero weight entries would blow up the division; identity beats inf.
                denom = torch.where(denom.abs() < 1e-6, torch.ones_like(denom), denom)
                self._lnf_inv_denom = denom.reciprocal()
        if self._lnf_inv_denom is None:
            return t
        inv = self._lnf_inv_denom.to(device=t.device)
        return (t.to(torch.float32) * inv).to(t.dtype)

    def close(self) -> None:
        if self._llm is None:  # already closed — keep the refcount single-shot
            return
        # Detach hooks before dropping the LLM so they don't stay registered on
        # worker modules for the life of the process (long-running notebooks).
        log = logging.getLogger("transformer_lens.vllm")
        try:
            self._llm.collective_rpc("tl_remove_hooks")
        except Exception as e:
            # Best-effort: engine may already be torn down or the RPC surface
            # gone. Log so hook-leak debugging has a thread to pull.
            log.debug("tl_remove_hooks failed during close(): %s", e)
        self._llm = None
        self._unembed = None
        self._lnf_inv_denom = None

        global _LIVE_DRIVERS
        with _LIVE_DRIVERS_LOCK:
            _LIVE_DRIVERS -= 1
            last_driver = _LIVE_DRIVERS <= 0
        # vLLM 0.20.2 has no LLM.shutdown(); the distributed teardown below hits
        # process-global state, so it runs only for the last live driver (see
        # _LIVE_DRIVERS). Both calls are best-effort no-ops otherwise.
        if last_driver:
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
        """Shared spec validation plus driver-side gating (hook membership, pos support)."""
        out: dict = {}
        for hook_name, spec in intervene.items():
            validated = validate_spec(hook_name, spec, width=self._hook_widths.get(hook_name))
            if hook_name not in self.supported_hook_points:
                raise ValueError(
                    f"Cannot intervene on {hook_name!r}: not in supported_hook_points."
                )
            if validated.get("pos") is not None:
                if self._enable_batching:
                    # The batched/eager path applies ops to the raw tensor, not the
                    # (max_n, width) affine buffers, so it has no position surface.
                    raise NotImplementedError(
                        f"Intervention {hook_name!r}: per-position 'pos' is not supported on the "
                        "batched/eager path. Boot the compiled path (enable_batching=False) with "
                        "enable_position_interventions=True."
                    )
                if not self._enable_position_interventions:
                    # Default affine buffers are (width,) and broadcast across every position;
                    # honoring 'pos' needs the (max_n, width) buffers allocated at boot.
                    raise NotImplementedError(
                        f"Intervention {hook_name!r}: per-position 'pos' requires "
                        "boot_vllm(enable_position_interventions=True) (its default affine "
                        "buffers broadcast across all positions). Use the Inspect/HF backend for "
                        "position-scoped patching, or drop 'pos' for a whole-sequence edit."
                    )
            out[hook_name] = validated
        return out

    @staticmethod
    def _reject_pos_beyond_seq(specs: Mapping[str, Any], seq_len: int) -> None:
        """Fail loud if a spec's 'pos' targets a row past the actual prompt length.

        The compiled hook applies the affine over rows ``[0, seq_len)`` only, so a ``pos``
        in ``[seq_len, max_num_batched_tokens)`` clears the driver's non-negativity check
        and the worker's buffer-capacity check yet is never read — a silent no-op. Bound it
        against the real sequence length (known once ``ids_list`` exists) and raise instead.
        """
        for hook_name, spec in specs.items():
            pos = spec.get("pos")
            if pos is None:
                continue
            idx = [pos] if isinstance(pos, int) else list(pos)
            bad = [p for p in idx if p >= seq_len]
            if bad:
                raise ValueError(
                    f"Intervention {hook_name!r}: 'pos' {bad} is beyond the prompt length "
                    f"{seq_len} (positions are 0-indexed); the edit would be silently ignored."
                )

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
