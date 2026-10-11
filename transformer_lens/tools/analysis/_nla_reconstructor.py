"""Private truncated Qwen reconstructor execution with an independent value head."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis._nla_artifacts import (
    NLAMetadata,
    NLAPairConfig,
    acquire_snapshot,
    load_local_metadata,
    load_snapshot_tokenizer,
    validate_value_head,
    validate_value_head_file,
)
from transformer_lens.tools.analysis._nla_contracts import (
    NLAReconstruction,
    validate_context_length,
    validate_reconstruction_text,
)
from transformer_lens.utilities.hf_utils import call_hf_with_retry

_EXPECTED_MISSING = frozenset({"model.norm.weight", "lm_head.weight"})


def validate_loading_info(info: dict[str, Any]) -> None:
    """Accept only the released Qwen AR's omitted final output weights."""
    if set(info.get("missing_keys", ())) != _EXPECTED_MISSING:
        raise ValueError("AR missing keys must be exactly final norm and LM head weights")
    for key in ("unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise ValueError(f"AR loading reported {key}")


def prepare_critic_tokens(
    tokenizer: Any, metadata: NLAMetadata, text: str, context: int
) -> torch.Tensor:
    """Render one raw prompt with special tokens, checking BOS and suffix."""
    content = validate_reconstruction_text(text)
    if metadata.role != "ar" or not metadata.critic_suffix_ids:
        raise ValueError("reconstruction requires validated AR metadata")
    prompt = metadata.critic_template.format(explanation=content)
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    validate_context_length(len(ids), context)
    if any(type(item) is not int or item < 0 for item in ids):
        raise ValueError("critic token ids must be nonnegative integers")
    suffix = metadata.critic_suffix_ids
    if tuple(ids[-len(suffix) :]) != suffix:
        raise ValueError("critic suffix token drift")
    bos = tokenizer.bos_token_id
    if bos is not None and ids[0] != bos:
        raise ValueError("critic special-token policy must preserve BOS")
    return torch.tensor([ids], dtype=torch.int64)


def load_critic_model(
    directory: Path, pair: NLAPairConfig, tokenizer: Any, device: str
) -> TransformerBridge:
    """Load an owned Qwen AR for raw backbone execution without its LM head."""
    if pair.model_type != "qwen2":
        raise ValueError("AR loading currently supports qwen2 only")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("tie_word_embeddings") is not False:
        raise ValueError("AR loading requires an independent omitted LM head")
    loaded: Any = call_hf_with_retry(
        AutoModelForCausalLM.from_pretrained,
        str(directory),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
    )
    original, info = loaded
    validate_loading_info(info)
    if (
        original.config.num_hidden_layers != pair.layer + 1
        or original.config.hidden_size != pair.d_model
    ):
        raise ValueError("AR backbone dimension/depth mismatch")
    model = TransformerBridge.boot_transformers(
        pair.ar.repo_id,
        hf_model=original,
        tokenizer=tokenizer,
        device=device,
        dtype=torch.float32,
        trust_remote_code=False,
    )
    # Bridge setup inspects output weights; bypass normalization only after wrapping.
    original.model.norm = torch.nn.Identity()
    model.eval()
    return model


def reconstruct_vector(
    model: TransformerBridge,
    head: torch.nn.Linear,
    tokens: torch.Tensor,
    metadata: NLAMetadata,
    pair: NLAPairConfig,
    *,
    output_device: str = "cpu",
) -> NLAReconstruction:
    """Read the raw last-block final token, never a normalized output vector."""
    if metadata.role != "ar" or metadata.layer != pair.layer or metadata.d_model != pair.d_model:
        raise ValueError("AR metadata does not match the artifact")
    if model.compatibility_mode or getattr(model, "_weights_processed", False):
        raise ValueError("AR requires raw unprocessed Bridge weights")
    if model.training or model.original_model.training or head.training:
        raise ValueError("AR and value head must be in eval mode")
    if model.cfg.n_layers != pair.layer + 1 or model.cfg.d_model != pair.d_model:
        raise ValueError("AR model dimensions/depth do not match the artifact")
    if head.bias is not None:
        raise ValueError("AR value head must be bias-free")
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.dtype != torch.int64:
        raise ValueError("AR requires one int64 token sequence")
    validate_context_length(tokens.shape[1], model.cfg.n_ctx)
    validate_value_head(head.state_dict(), pair.d_model)
    hook = f"blocks.{pair.layer}.hook_out"
    captured: list[torch.Tensor] = []

    def capture(value: torch.Tensor, hook: Any) -> torch.Tensor:
        captured.append(value)
        return value

    original: Any = model.original_model
    backbone = original.base_model
    with torch.inference_mode(), model.hooks(fwd_hooks=[(hook, capture)]):
        backbone(input_ids=tokens.to(model.cfg.device), use_cache=False)
        if len(captured) != 1:
            raise ValueError("AR block output must fire exactly once")
        residual = captured[0]
        if residual.shape != (1, tokens.shape[1], pair.d_model):
            raise ValueError("AR block output must be a single residual stream")
        head.to(device=residual.device, dtype=residual.dtype)
        vector = head(residual[0, -1]).detach().to(output_device).clone()
    del captured, residual
    return NLAReconstruction(vector, pair.ar)


class ReconstructorRuntime:
    """Lazy owned AR/head lifecycle; concurrent operations are rejected."""

    def __init__(
        self,
        pair: NLAPairConfig,
        *,
        device: str = "cpu",
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.pair = pair
        self.device = device
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._model: TransformerBridge | None = None
        self._head: torch.nn.Linear | None = None
        self._lock = threading.Lock()

    def reconstruct(self, text: str, *, output_device: str = "cpu") -> NLAReconstruction:
        """Validate prompt and small assets before acquiring full backbone weights."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent reconstructor operations are not supported")
        try:
            validate_reconstruction_text(text)
            directory = acquire_snapshot(
                self.pair.ar, cache_dir=self.cache_dir, local_files_only=self.local_files_only
            )
            tokenizer = load_snapshot_tokenizer(directory)
            metadata = load_local_metadata(directory, self.pair, "ar", tokenizer)
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            tokens = prepare_critic_tokens(
                tokenizer, metadata, text, config["max_position_embeddings"]
            )
            if self._model is None:
                snapshot = acquire_snapshot(
                    self.pair.ar,
                    cache_dir=self.cache_dir,
                    local_files_only=self.local_files_only,
                    include_weights=True,
                )
                if snapshot != directory:
                    raise ValueError("AR weights must use the metadata snapshot")
                path = directory / "value_head.safetensors"
                validate_value_head_file(path, self.pair.d_model)
                head = torch.nn.Linear(self.pair.d_model, self.pair.d_model, bias=False)
                head.load_state_dict(load_file(str(path), device="cpu"), strict=True)
                head.eval()
                model = load_critic_model(directory, self.pair, tokenizer, self.device)
                self._model, self._head = model, head
            if self._head is None:
                raise RuntimeError("AR value head is unavailable")
            return reconstruct_vector(
                self._model, self._head, tokens, metadata, self.pair, output_device=output_device
            )
        finally:
            self._lock.release()

    def unload(self) -> None:
        """Drop owned backbone/head references without freeing caller tensors."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot unload during a reconstructor operation")
        try:
            self._model = None
            self._head = None
        finally:
            self._lock.release()
