"""vLLM Driver: forward dispatches via ``llm.generate``; captures via ``collective_rpc``."""
from __future__ import annotations

import gc
import logging
import threading
import warnings
from typing import Any, Mapping

import torch

from transformer_lens.model_bridge.driver_protocol import (
    ForwardResult,
    Intervention,
    TensorLike,
)
from transformer_lens.model_bridge.sources._driver_base import DriverBase

from .intervention_specs import SUPPORTED_OPS

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
    ) -> None:
        super().__init__(adapter.cfg, tokenizer)
        self._llm = llm
        self._max_num_batched_tokens = max_num_batched_tokens
        self._enable_batching = enable_batching
        # Position-scoped 'pos' interventions need (max_n, width) affine buffers,
        # allocated at boot only when this is set (see plugin.patched_load_model).
        self._enable_position_interventions = enable_position_interventions
        # Logprobs per forward = real vocab (boot's max_logprobs). d_vocab can be
        # padded larger, which vLLM would reject; the logits tensor stays d_vocab.
        self._n_logprobs = int(getattr(hf_config, "vocab_size", self.bridge_config.d_vocab))
        # Unembedding cache: (weight_fp32, bias_fp32|None) once probe_logit_reconstruction
        # runs — a per-forward re-fetch clones a d_vocab×d_model tensor to CPU every call.
        self._unembed: tuple[torch.Tensor, Any] | None = None
        # None = unprobed (fetch per call, legacy path); False = probed and unavailable.
        self._recon_available: bool | None = None
        self._lnf_weight: torch.Tensor | None = None
        self._lnf_unfold_warned = False
        self._closed = False

        capture_specs = overlay.capture_specs(hf_config)
        self._hook_widths = {name: width for name, (_path, width) in capture_specs.items()}
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
        # Full-vocab logprobs are the reconstruction FALLBACK only — when the unembedding
        # is cached, skip them (vLLM would otherwise marshal a d_vocab-entry Python dict
        # of Logprob objects host-side per request).
        want_logprobs = return_logits and not self._recon_available
        outputs = self._llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=ids_list)],
            sampling_params=SamplingParams(
                max_tokens=int(max_new_tokens),
                temperature=0.0,
                logprobs=self._n_logprobs if want_logprobs else None,
            ),
        )

        n_tokens = len(ids_list)
        # Reconstructing logits needs ln_final; force it into the read even if the caller
        # didn't request it (dropped from `captured` below so the surface stays as asked).
        # Skip the forcing when the probe already established reconstruction can't run.
        read_names = names
        if return_logits and self._recon_available is not False and self._LN_FINAL not in names:
            read_names = names + [self._LN_FINAL]
        # collective_rpc returns one result per worker; single-rank, so [0]. Nothing to
        # read (no captures, logits off) → skip the crossing altogether.
        worker_captures = (
            self._llm.collective_rpc("tl_read_captures", args=([n_tokens], read_names))[0]
            if read_names
            else {}
        )

        logits: torch.Tensor | None = None
        if return_logits:
            recon = self._reconstruct_logits(worker_captures.get(self._LN_FINAL))
            # Fall back to final-position log-probs if the unembedding isn't fetchable.
            logits = (
                recon.unsqueeze(0)
                if recon is not None
                else self._synthesize_logits(outputs[0], n_tokens, self.bridge_config.d_vocab)
            )

        # Add batch dim; expose only the caller's requested hooks (drop the forced ln_final).
        captured = {name: t.unsqueeze(0) for name, t in worker_captures.items() if name in names}
        # Exposed ln_final must honor the hook name's pre-weight convention; reconstruction
        # above consumed the raw post-weight value, so convert only the user-facing copy.
        if self._LN_FINAL in captured:
            captured[self._LN_FINAL] = self._unfold_ln_final(captured[self._LN_FINAL])
        return ForwardResult(logits=logits, captured=captured, raw_output=outputs[0])

    def _forward_batched(
        self,
        input_ids: TensorLike,
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
        want_logprobs = return_logits and not self._recon_available
        outputs = self._llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=ids) for ids in prompts_ids],
            sampling_params=SamplingParams(
                max_tokens=1,
                temperature=0.0,
                logprobs=self._n_logprobs if want_logprobs else None,
            ),
        )

        # Force ln_final into the read so logits can be reconstructed (dropped below if
        # the caller didn't ask for it). Skip when the probe ruled reconstruction out.
        read_names = names
        if return_logits and self._recon_available is not False and self._LN_FINAL not in names:
            read_names = names + [self._LN_FINAL]
        # Keyed by req_id (no guaranteed order) — _assemble_padded joins to slot
        # k via outputs[k].request_id, not by position. Empty read → skip both the
        # crossing AND the join (_assemble_padded requires one worker key per request,
        # so it can't run on an empty dict — mirror the single path's empty captured).
        if read_names:
            worker_captures = self._llm.collective_rpc(
                "tl_read_batched_captures", args=(read_names,)
            )[0]
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
        if self._LN_FINAL not in names:
            captured.pop(self._LN_FINAL, None)
        elif self._LN_FINAL in captured:
            captured[self._LN_FINAL] = self._unfold_ln_final(captured[self._LN_FINAL])

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

    def probe_logit_reconstruction(self) -> bool:
        """One-time unembedding fetch + cache; downgrades ``provides_sequence_logits``
        honestly when no unembedding is reachable (the fallback path is final-position
        log-probs, which cannot back a loss)."""
        weight = self.get_param("lm_head.weight")
        if weight is None:  # tied embeddings expose no separate lm_head
            weight = self.get_param("model.embed_tokens.weight")
        if weight is None:
            self._unembed = None
            self._recon_available = False
            self.provides_sequence_logits = False
            return False
        bias = self.get_param("lm_head.bias")
        self._unembed = (
            weight.to(torch.float32),
            bias.to(torch.float32) if bias is not None else None,
        )
        self._recon_available = True
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
        if ln_final is None or self._recon_available is False:
            return None
        if self._unembed is not None:
            weight32, bias32 = self._unembed
        else:
            # Unprobed (driver constructed directly): fetch per call as before.
            weight = self.get_param("lm_head.weight")
            if weight is None:  # tied embeddings expose no separate lm_head
                weight = self.get_param("model.embed_tokens.weight")
            if weight is None:
                return None
            weight32 = weight.to(torch.float32)
            bias = self.get_param("lm_head.bias")
            bias32 = bias.to(torch.float32) if bias is not None else None
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
        canonical hook name promises (÷ weight; Gemma folds ``1 + weight``). Warns and
        returns the raw value when the norm weight is unreachable — loud beats silent
        cross-backend mismatch."""
        weight = self._lnf_weight
        if weight is None:
            weight = self.get_param("model.norm.weight")
            if weight is None:
                if not self._lnf_unfold_warned:
                    warnings.warn(
                        "ln_final.hook_normalized: norm weight unreachable — the captured "
                        "value stays POST-weight and will not match boot_transformers.",
                        UserWarning,
                        stacklevel=3,
                    )
                    self._lnf_unfold_warned = True
                return t
            self._lnf_weight = weight
        w = weight.detach().to(device=t.device, dtype=torch.float32)
        denom = (1.0 + w) if "gemma" in self.architecture.lower() else w
        # Near-zero weight entries would blow up the division; identity beats inf.
        denom = torch.where(denom.abs() < 1e-6, torch.ones_like(denom), denom)
        return (t.to(torch.float32) / denom).to(t.dtype)

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
        self._unembed = None
        self._lnf_weight = None

        global _LIVE_DRIVERS
        last_driver = False
        if not self._closed:
            self._closed = True
            with _LIVE_DRIVERS_LOCK:
                _LIVE_DRIVERS -= 1
                last_driver = _LIVE_DRIVERS <= 0
        # vLLM 0.20.2 has no LLM.shutdown() — model weights and KV cache stay
        # resident until process exit unless we explicitly tear down the
        # distributed environment vLLM set up at construction. That teardown hits
        # PROCESS-GLOBAL state (destroys the TP/world groups every live engine
        # shares), so it only runs when this is the last live driver — the
        # notebook pattern `bridge = boot_vllm(B)` re-bound over A must not let
        # A's close() break B. Both calls are best-effort no-ops otherwise.
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
            # Unknown keys must fail loud: a typo'd "position"/"positions" would
            # otherwise silently become a whole-sequence edit.
            allowed = {"op", "pos"}
            if op == "scale":
                allowed.add("factor")
            if op in ("add", "set"):
                allowed.add("value")
            extra = set(spec) - allowed
            if extra:
                raise ValueError(
                    f"Intervention {hook_name!r}: unknown spec key(s) {sorted(extra)}; "
                    f"allowed for op={op!r}: {sorted(allowed)}."
                )
            if hook_name not in self.supported_hook_points:
                raise ValueError(
                    f"Cannot intervene on {hook_name!r}: not in supported_hook_points."
                )
            value = spec.get("value")
            if value is not None and not isinstance(value, (int, float)):
                width = self._hook_widths.get(hook_name)
                try:
                    n_elements = int(torch.as_tensor(value).numel())
                except (TypeError, ValueError, RuntimeError) as exc:
                    raise ValueError(
                        f"Intervention {hook_name!r}: 'value' must be a scalar or a "
                        f"width-shaped tensor/list; got {type(value).__name__}."
                    ) from exc
                if width is not None and n_elements != width:
                    # A mis-shaped value would otherwise surface as a broadcast error
                    # mid-forward — or broadcast along the wrong axis in square cases.
                    raise ValueError(
                        f"Intervention {hook_name!r}: 'value' has {n_elements} elements "
                        f"but the hook width is {width}."
                    )
            pos = spec.get("pos")
            if pos is not None:
                if isinstance(pos, bool):
                    raise ValueError(
                        f"Intervention {hook_name!r}: 'pos' must be an int or list of ints "
                        f"(sequence positions to patch); got {pos!r}."
                    )
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
                if not (
                    isinstance(pos, int)
                    or (
                        isinstance(pos, (list, tuple))
                        and all(isinstance(p, int) and not isinstance(p, bool) for p in pos)
                    )
                ):
                    raise ValueError(
                        f"Intervention {hook_name!r}: 'pos' must be an int or list of ints "
                        f"(sequence positions to patch); got {pos!r}."
                    )
                bad = [p for p in ([pos] if isinstance(pos, int) else pos) if p < 0]
                if bad:
                    raise ValueError(
                        f"Intervention {hook_name!r}: 'pos' must be non-negative; got {bad}."
                    )
            out[hook_name] = dict(spec)
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
