"""Private orchestration of local NLA inference and owned role lifecycles."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator, Literal

import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis._nla_artifacts import (
    NLAPairConfig,
    acquire_snapshot,
    load_local_metadata,
    load_registry,
    load_snapshot_tokenizer,
)
from transformer_lens.tools.analysis._nla_contracts import (
    NLAActivation,
    NLAExplanation,
    NLAReconstruction,
    NLAReconstructionScore,
    NLASamplingConfig,
    score_reconstruction,
)
from transformer_lens.tools.analysis._nla_reconstructor import ReconstructorRuntime
from transformer_lens.tools.analysis._nla_target import extract_activation, load_target
from transformer_lens.tools.analysis._nla_verbalizer import VerbalizerRuntime


class NaturalLanguageAutoencoder:
    """Compose raw-target extraction, AV text, and AR reconstruction.

    Role models are owned here; returned targets and detached results are owned
    by the caller. Loading provenance does not establish live-weight integrity.
    Closing releases role models and permanently closes this instance. Unloading
    one role permits a later lazy reload. Operations on one instance are serial.
    """

    def __init__(
        self,
        pair: NLAPairConfig,
        artifact_name: str,
        *,
        device: str = "cpu",
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        if not isinstance(artifact_name, str) or not artifact_name.strip():
            raise ValueError("artifact_name must be nonempty text")
        self.pair = pair
        self.artifact_name = artifact_name
        self.device = device
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._av = VerbalizerRuntime(
            pair,
            artifact_name,
            device=device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self._ar = ReconstructorRuntime(
            pair, device=device, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        *,
        device: str = "cpu",
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> NaturalLanguageAutoencoder:
        """Resolve a supported pinned pair, validating metadata without model weights."""
        registry = load_registry()
        if name not in registry:
            raise ValueError(f"No verified NLA artifact is registered for {name!r}")
        pair = registry[name]
        for role in ("av", "ar"):
            ref = pair.av if role == "av" else pair.ar
            directory = acquire_snapshot(
                ref, cache_dir=cache_dir, local_files_only=local_files_only
            )
            tokenizer = load_snapshot_tokenizer(directory)
            load_local_metadata(directory, pair, role, tokenizer)
        return cls(
            pair, name, device=device, cache_dir=cache_dir, local_files_only=local_files_only
        )

    @contextmanager
    def _operation(self) -> Iterator[None]:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent NLA operations are not supported")
        try:
            if self._closed:
                raise RuntimeError("NLA instance is closed")
            yield
        finally:
            self._lock.release()

    def load_target(self) -> TransformerBridge:
        """Return a pinned raw target without keeping a strong reference to it."""
        with self._operation():
            return load_target(
                self.pair,
                device=self.device,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )

    def extract(
        self,
        target: TransformerBridge,
        prompt: str | torch.Tensor,
        *,
        position: int = -1,
        output_device: str = "cpu",
        prepend_bos: bool | None = None,
        allow_unverified_target: bool = False,
    ) -> NLAActivation:
        """Extract one detached raw activation at the artifact-fixed block."""
        with self._operation():
            return extract_activation(
                target,
                self.pair,
                self.artifact_name,
                prompt,
                position=position,
                output_device=output_device,
                prepend_bos=prepend_bos,
                allow_unverified_target=allow_unverified_target,
            )

    def verbalize(
        self,
        activation: NLAActivation | torch.Tensor,
        *,
        max_new_tokens: int = 512,
        temperature: float | int = 1.0,
        do_sample: bool = True,
        use_past_kv_cache: bool = True,
        top_p: float | int = 1.0,
        top_k: int = 0,
    ) -> NLAExplanation:
        """Generate canonical inner explanation content and raw continuation data."""
        sampling = NLASamplingConfig(
            max_new_tokens, temperature, do_sample, use_past_kv_cache, top_p, top_k
        )
        with self._operation():
            return self._av.verbalize(activation, sampling)

    def explain(
        self,
        target: TransformerBridge,
        prompt: str | torch.Tensor,
        *,
        position: int = -1,
        prepend_bos: bool | None = None,
        allow_unverified_target: bool = False,
        sampling: NLASamplingConfig | None = None,
    ) -> NLAExplanation:
        """Extract then verbalize, leaving the caller-owned target resident."""
        with self._operation():
            activation = extract_activation(
                target,
                self.pair,
                self.artifact_name,
                prompt,
                position=position,
                prepend_bos=prepend_bos,
                allow_unverified_target=allow_unverified_target,
            )
            return self._av.verbalize(activation, sampling)

    def reconstruct(self, text: str, *, output_device: str = "cpu") -> NLAReconstruction:
        """Run raw AR inference on nonempty untagged explanation content."""
        with self._operation():
            return self._ar.reconstruct(text, output_device=output_device)

    @staticmethod
    def score(
        original: torch.Tensor | NLAActivation, reconstructed: torch.Tensor | NLAReconstruction
    ) -> NLAReconstructionScore:
        """Directional similarity only, not dataset FVE or explanation faithfulness."""
        return score_reconstruction(original, reconstructed)

    def unload(self, role: Literal["av", "ar"]) -> None:
        """Release one owned role, permitting lazy reload on subsequent use."""
        with self._operation():
            if role == "av":
                self._av.unload()
            elif role == "ar":
                self._ar.unload()
            else:
                raise ValueError("role must be av or ar")

    def close(self) -> None:
        """Release both owned roles; repeated closing is harmless."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot close during an NLA operation")
        try:
            if not self._closed:
                self._av.unload()
                self._ar.unload()
                self._closed = True
        finally:
            self._lock.release()

    def __enter__(self) -> NaturalLanguageAutoencoder:
        with self._operation():
            return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
