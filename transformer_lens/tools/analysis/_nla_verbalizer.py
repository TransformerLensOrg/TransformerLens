"""Private generation-aware activation verbalizer execution."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis._nla_artifacts import (
    NLAMetadata,
    NLAPairConfig,
    acquire_snapshot,
    load_local_metadata,
    load_snapshot_tokenizer,
)
from transformer_lens.tools.analysis._nla_contracts import (
    ActivationProvenance,
    NLAActivation,
    NLAExplanation,
    NLASamplingConfig,
    SourceProvenance,
    detached_vector,
    parse_explanation,
)
from transformer_lens.utilities.hf_utils import call_hf_with_retry


def embedding_overwrite(
    vector: torch.Tensor, scale: float, position: int, prompt_length: int, *, cached: bool
) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    """Construct per-generation state, including one-token prefill handling."""
    source = detached_vector(vector).to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(torch.tensor(scale, dtype=torch.float64)).item() or scale <= 0:
        raise ValueError("injection scale must be finite and positive")
    if (
        type(position) is not int
        or type(prompt_length) is not int
        or not 0 <= position < prompt_length
    ):
        raise ValueError("injection position must be within the original prompt")
    if type(cached) is not bool:
        raise ValueError("cached must be bool")
    unit = source / source.abs().max()
    injected = unit / unit.norm() * scale
    first_call = True

    def overwrite(value: torch.Tensor, hook: Any) -> torch.Tensor:
        nonlocal first_call
        if cached and not first_call:
            return value
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != source.numel():
            raise ValueError("embedding hook must expose one [batch, position, d_model] stream")
        if value.shape[1] < prompt_length or (first_call and value.shape[1] != prompt_length):
            raise ValueError("embedding prefill length does not match the canonical prompt")
        replacement = injected.to(device=value.device, dtype=value.dtype)
        if not torch.isfinite(replacement).all().item():
            raise ValueError("injection scale is not representable in embedding dtype")
        output = value.clone()
        output[0, position] = replacement
        first_call = False
        return output

    return overwrite


def generate_explanation(
    model: TransformerBridge,
    metadata: NLAMetadata,
    activation: NLAActivation,
    sampling: NLASamplingConfig,
) -> NLAExplanation:
    """Execute already validated canonical tokens with scoped embedding injection."""
    if metadata.role != "av" or metadata.actor_token_ids is None:
        raise ValueError("verbalization requires validated actor metadata")
    if metadata.injection_position is None or metadata.injection_scale is None:
        raise ValueError("actor metadata is missing its injection contract")
    if activation.vector.numel() != metadata.d_model or model.cfg.d_model != metadata.d_model:
        raise ValueError("activation or model dimension does not match actor metadata")
    if model.compatibility_mode or getattr(model, "_weights_processed", False):
        raise ValueError("actor must use raw unprocessed Bridge weights")
    if model.training or model.original_model.training:
        raise ValueError("actor must be in eval mode")
    sampling.validate_prompt(len(metadata.actor_token_ids), model.cfg.n_ctx)
    config: Any = model.original_model.config
    if getattr(config, "_attn_implementation", None) != "eager":
        raise ValueError("actor requires eager attention")
    if getattr(model.original_model, "is_quantized", False):
        raise ValueError("quantized actors are not supported")
    if any(
        p.is_floating_point() and p.dtype != torch.float32
        for p in model.original_model.parameters()
    ):
        raise ValueError("actor currently requires float32 weights")
    if sampling.top_k > config.vocab_size:
        raise ValueError("top_k exceeds actor vocabulary size")
    if model.tokenizer is None:
        raise ValueError("actor tokenizer is unavailable")
    tokens = torch.tensor([metadata.actor_token_ids], dtype=torch.int64, device=model.cfg.device)
    hook = embedding_overwrite(
        activation.vector,
        metadata.injection_scale,
        metadata.injection_position,
        tokens.shape[1],
        cached=sampling.use_past_kv_cache,
    )
    with torch.inference_mode(), model.hooks(fwd_hooks=[("hook_embed", hook)]):
        output = model.generate(
            tokens,
            return_type="tokens",
            max_new_tokens=sampling.max_new_tokens,
            temperature=float(sampling.temperature),
            do_sample=sampling.do_sample,
            top_p=float(sampling.top_p) if sampling.do_sample else None,
            top_k=sampling.top_k or None,
            use_past_kv_cache=sampling.use_past_kv_cache,
            verbose=False,
        )
    if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1:
        raise ValueError("actor generation must return one token sequence")
    if output.shape[1] < tokens.shape[1] or not torch.equal(output[:, : tokens.shape[1]], tokens):
        raise ValueError("actor generation did not preserve the original prompt tokens")
    ids = tuple(output[0, tokens.shape[1] :].detach().cpu().tolist())
    raw = model.tokenizer.decode(list(ids), skip_special_tokens=True)
    return parse_explanation(raw, ids, activation.provenance)


class VerbalizerRuntime:
    """Lazy fp32 actor loading with operation-local hooks and non-reentrant ownership."""

    def __init__(
        self,
        pair: NLAPairConfig,
        artifact_name: str,
        *,
        device: str = "cpu",
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.pair = pair
        self.artifact_name = artifact_name
        self.device = device
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._model: TransformerBridge | None = None
        self._lock = threading.Lock()

    def _metadata(self) -> tuple[Path, NLAMetadata, Any]:
        directory = acquire_snapshot(
            self.pair.av,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        tokenizer = load_snapshot_tokenizer(directory)
        metadata = load_local_metadata(directory, self.pair, "av", tokenizer)
        return directory, metadata, tokenizer

    def _load(self, directory: Path, tokenizer: Any) -> TransformerBridge:
        snapshot = acquire_snapshot(
            self.pair.av,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            include_weights=True,
        )
        if snapshot != directory:
            raise ValueError("actor weights must use the metadata snapshot")
        original = call_hf_with_retry(
            AutoModelForCausalLM.from_pretrained,
            str(directory),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        model = TransformerBridge.boot_transformers(
            self.pair.av.repo_id,
            hf_model=original,
            tokenizer=tokenizer,
            device=self.device,
            dtype=torch.float32,
            trust_remote_code=False,
        )
        model.eval()
        return model

    def verbalize(
        self, activation: NLAActivation | torch.Tensor, sampling: NLASamplingConfig | None = None
    ) -> NLAExplanation:
        """Validate small assets and budget before materializing full actor weights."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent verbalizer operations are not supported")
        try:
            sampling = sampling if sampling is not None else NLASamplingConfig()
            if isinstance(activation, torch.Tensor):
                activation = NLAActivation(
                    activation,
                    ActivationProvenance(self.artifact_name, SourceProvenance.UNKNOWN_VECTOR),
                )
            if activation.provenance.artifact_name != self.artifact_name:
                raise ValueError("activation provenance belongs to a different artifact")
            directory, metadata, tokenizer = self._metadata()
            detached_vector(activation.vector)
            if activation.vector.numel() != metadata.d_model:
                raise ValueError("activation dimension does not match the artifact")
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            sampling.validate_prompt(
                len(metadata.actor_token_ids or ()), config["max_position_embeddings"]
            )
            if sampling.top_k > config["vocab_size"]:
                raise ValueError("top_k exceeds actor vocabulary size")
            if self._model is None:
                self._model = self._load(directory, tokenizer)
            return generate_explanation(self._model, metadata, activation, sampling)
        finally:
            self._lock.release()

    def unload(self) -> None:
        """Release only actor references owned by this runtime, idempotently."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot unload during a verbalizer operation")
        try:
            self._model = None
        finally:
            self._lock.release()
