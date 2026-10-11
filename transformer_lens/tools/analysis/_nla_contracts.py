"""Private result, text, and numerical contracts for NLA inference."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

import torch

from transformer_lens.tools.analysis._nla_artifacts import ArtifactRef, _integer


class SourceProvenance(str, Enum):
    """Loading evidence, not continuous verification of live model weights."""

    PINNED_LOAD = "pinned_load"
    CALLER_ATTESTED = "caller_attested"
    UNKNOWN_VECTOR = "unknown_vector"


@dataclass(frozen=True)
class ActivationProvenance:
    """Source identity and extraction conventions without owning a model."""

    artifact_name: str
    source: SourceProvenance
    target: ArtifactRef | None = None
    layer: int | None = None
    position: int | None = None
    tokenization_policy: str | None = None
    basis: str = "raw_hf"

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_name, str) or not self.artifact_name.strip():
            raise ValueError("artifact_name must be nonempty text")
        if not isinstance(self.source, SourceProvenance):
            raise ValueError("source must be a SourceProvenance value")
        if self.basis != "raw_hf":
            raise ValueError("only the raw_hf residual basis is supported")
        for name, value in (("layer", self.layer), ("position", self.position)):
            if value is not None:
                _integer(value, name)
        if self.tokenization_policy is not None and (
            not isinstance(self.tokenization_policy, str) or not self.tokenization_policy.strip()
        ):
            raise ValueError("tokenization_policy must be nonempty text")
        if self.source is SourceProvenance.PINNED_LOAD and (
            self.target is None
            or self.layer is None
            or self.position is None
            or self.tokenization_policy is None
        ):
            raise ValueError(
                "pinned loading provenance requires identity and extraction conventions"
            )

    @property
    def source_verified(self) -> bool:
        """Whether immutable loading provenance was recorded, not weight integrity."""
        return self.source is SourceProvenance.PINNED_LOAD


def detached_vector(vector: torch.Tensor) -> torch.Tensor:
    """Validate and copy one real activation without retaining its graph or storage."""
    if vector.ndim != 1 or vector.numel() == 0 or not vector.is_floating_point():
        raise ValueError("activation must be a nonempty one-dimensional floating-point tensor")
    if not torch.isfinite(vector).all().item():
        raise ValueError("activation must contain only finite values")
    if not torch.any(vector != 0).item():
        raise ValueError("activation must be nonzero")
    return vector.detach().clone()


@dataclass(frozen=True)
class NLAActivation:
    """Owned detached activation vector and source loading evidence."""

    vector: torch.Tensor
    provenance: ActivationProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", detached_vector(self.vector))


@dataclass(frozen=True)
class NLAReconstruction:
    """Owned detached AR vector and immutable reconstructor identity."""

    vector: torch.Tensor
    artifact: ArtifactRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", detached_vector(self.vector))


class ExplanationStatus(str, Enum):
    """Observable outcomes of canonical explanation-region parsing."""

    SUCCESS = "success"
    MISSING = "missing"
    MULTIPLE = "multiple"
    NESTED = "nested"
    INCOMPLETE = "incomplete"
    EMPTY = "empty"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class NLAExplanation:
    """Parsed inner text and untouched continuation data, including failures."""

    text: str | None
    raw_text: str
    token_ids: tuple[int, ...]
    parse_status: ExplanationStatus
    provenance: ActivationProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.raw_text, str):
            raise ValueError("raw_text must be text")
        if not isinstance(self.parse_status, ExplanationStatus):
            raise ValueError("parse_status must be an ExplanationStatus value")
        ids = tuple(_integer(item, "continuation token id") for item in self.token_ids)
        object.__setattr__(self, "token_ids", ids)
        if self.parse_status is ExplanationStatus.SUCCESS:
            text = validate_reconstruction_text(self.text)
            if text != text.strip():
                raise ValueError("parsed explanation text must have stripped outer whitespace")
        elif self.text is not None:
            raise ValueError("failed explanations must have text=None")


# Recognize broken wrappers too, rather than falling back to the full response.
_WRAPPER = re.compile(r"<\s*/?\s*explanation\b", re.IGNORECASE)
_TAG = re.compile(r"</?explanation>")


def validate_reconstruction_text(text: object) -> str:
    """Accept nonempty untagged content without guessing its semantic origin."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("reconstruction text must be nonempty explanation content")
    if _WRAPPER.search(text):
        raise ValueError("reconstruction text must not contain explanation wrappers")
    return text


def parse_explanation(
    raw_text: str, token_ids: tuple[int, ...], provenance: ActivationProvenance
) -> NLAExplanation:
    """Parse one exact region, preserving raw continuation on every outcome."""
    tags = list(_TAG.finditer(raw_text))
    wrappers = list(_WRAPPER.finditer(raw_text))
    text = None
    if not wrappers:
        status = ExplanationStatus.MISSING
    elif len(wrappers) != len(tags):
        status = ExplanationStatus.MALFORMED
    else:
        depth = 0
        regions = 0
        malformed_order = False
        nested = False
        for tag in tags:
            if tag.group() == "<explanation>":
                nested |= depth > 0
                depth += 1
            elif depth == 0:
                malformed_order = True
            else:
                depth -= 1
                if depth == 0:
                    regions += 1
        if nested:
            status = ExplanationStatus.NESTED
        elif malformed_order or depth:
            status = ExplanationStatus.INCOMPLETE
        elif regions != 1:
            status = ExplanationStatus.MULTIPLE
        else:
            text = raw_text[tags[0].end() : tags[1].start()].strip()
            status = ExplanationStatus.SUCCESS if text else ExplanationStatus.EMPTY
            if not text:
                text = None
    return NLAExplanation(text, raw_text, token_ids, status, provenance)


@dataclass(frozen=True)
class NLAReconstructionScore:
    """Directional diagnostics, not raw MSE, dataset FVE, or text faithfulness."""

    cosine: float
    normalized_mse: float


def score_reconstruction(
    original: torch.Tensor | NLAActivation, reconstructed: torch.Tensor | NLAReconstruction
) -> NLAReconstructionScore:
    """Compute 2*(1-cosine) using detached, scale-safe CPU float64 accumulation."""
    a = detached_vector(original.vector if isinstance(original, NLAActivation) else original)
    b = detached_vector(
        reconstructed.vector if isinstance(reconstructed, NLAReconstruction) else reconstructed
    )
    if a.shape != b.shape:
        raise ValueError("original and reconstructed dimensions must match")
    a = a.to(device="cpu", dtype=torch.float64)
    b = b.to(device="cpu", dtype=torch.float64)
    # Rescale before taking norms to avoid overflow and underflow at finite extremes.
    a = a / a.abs().max()
    b = b / b.abs().max()
    cosine = float(torch.dot(a / a.norm(), b / b.norm()).item())
    # Clamp only roundoff-sized overshoot; larger violations indicate a defect.
    if not math.isfinite(cosine) or abs(cosine) > 1 + 1e-12:
        raise ValueError("cosine outside its numerical range")
    cosine = max(-1.0, min(1.0, cosine))
    return NLAReconstructionScore(cosine, 2 * (1 - cosine))


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class NLASamplingConfig:
    """Bounded AV generation policy with explicit sampling versus greedy mode."""

    max_new_tokens: int = 512
    temperature: float | int = 1.0
    do_sample: bool = True
    use_past_kv_cache: bool = True
    top_p: float | int = 1.0
    top_k: int = 0

    def __post_init__(self) -> None:
        _integer(self.max_new_tokens, "max_new_tokens", minimum=1)
        _integer(self.top_k, "top_k")
        if type(self.do_sample) is not bool or type(self.use_past_kv_cache) is not bool:
            raise ValueError("sampling and caching flags must be bool")
        temperature = _number(self.temperature, "temperature")
        top_p = _number(self.top_p, "top_p")
        if temperature < 0 or (self.do_sample and temperature == 0):
            raise ValueError("temperature must be positive for sampling, nonnegative for greedy")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not self.do_sample and (top_p != 1 or self.top_k != 0):
            raise ValueError("top_p and top_k filtering require sampling")

    def validate_prompt(self, prompt_length: int, context_length: int) -> None:
        """Reject prompt plus generation budgets exceeding the model context."""
        validate_context_length(prompt_length, context_length, self.max_new_tokens)


def validate_context_length(
    token_count: int, context_length: int, generation_budget: int = 0
) -> None:
    """Check exact tokenized lengths without truncating prompts or budgets."""
    _integer(token_count, "token_count", minimum=1)
    _integer(context_length, "context_length", minimum=1)
    _integer(generation_budget, "generation_budget")
    if token_count + generation_budget > context_length:
        raise ValueError("tokenized prompt plus generation budget exceeds model context length")
