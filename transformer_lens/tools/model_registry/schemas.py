"""Data schemas for the model registry.

This module defines the dataclasses used throughout the model registry for
representing supported models, architecture gaps, and related metadata.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


@dataclass
class ModelMetadata:
    """Metadata for a model from HuggingFace.

    Attributes:
        downloads: Total download count for the model
        likes: Number of likes/stars on HuggingFace
        last_modified: When the model was last updated
        tags: List of tags associated with the model
        parameter_count: Estimated number of parameters (if available)
    """

    downloads: int = 0
    likes: int = 0
    last_modified: Optional[datetime] = None
    tags: list[str] = field(default_factory=list)
    parameter_count: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "downloads": self.downloads,
            "likes": self.likes,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "tags": self.tags,
            "parameter_count": self.parameter_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        """Create from a dictionary."""
        last_modified = None
        if data.get("last_modified"):
            last_modified = datetime.fromisoformat(data["last_modified"])
        return cls(
            downloads=data.get("downloads", 0),
            likes=data.get("likes", 0),
            last_modified=last_modified,
            tags=data.get("tags", []),
            parameter_count=data.get("parameter_count"),
        )


@dataclass
class ModelEntry:
    """A single model entry in the supported models list.

    Attributes:
        architecture_id: The architecture type (e.g., "GPT2LMHeadModel")
        model_id: The HuggingFace model ID (e.g., "gpt2", "openai-community/gpt2")
        status: Verification status (0=unverified, 1=verified, 2=skipped, 3=failed)
        verified_date: Date when verification was performed
        metadata: Optional metadata from HuggingFace
        note: Optional note (skip/fail reason, e.g. "Estimated 48 GB exceeds 16 GB limit")
        phase1_score: Benchmark Phase 1 score (HF vs Bridge), 0-100 or None
        phase2_score: Benchmark Phase 2 score (Bridge vs HT unprocessed), 0-100 or None
        phase3_score: Benchmark Phase 3 score (Bridge vs HT processed), 0-100 or None
    """

    architecture_id: str
    model_id: str
    status: int = 0
    verified_date: Optional[date] = None
    metadata: Optional[ModelMetadata] = None
    note: Optional[str] = None
    phase1_score: Optional[float] = None
    phase2_score: Optional[float] = None
    phase3_score: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "architecture_id": self.architecture_id,
            "model_id": self.model_id,
            "status": self.status,
            "verified_date": self.verified_date.isoformat() if self.verified_date else None,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "note": self.note,
            "phase1_score": self.phase1_score,
            "phase2_score": self.phase2_score,
            "phase3_score": self.phase3_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelEntry":
        """Create from a dictionary."""
        verified_date = None
        if data.get("verified_date"):
            verified_date = date.fromisoformat(data["verified_date"])
        metadata = None
        if data.get("metadata"):
            metadata = ModelMetadata.from_dict(data["metadata"])
        # Backwards compat: convert old "verified" bool to new "status" int
        if "status" in data:
            status = data["status"]
        elif data.get("verified", False):
            status = 1
        else:
            status = 0
        return cls(
            architecture_id=data["architecture_id"],
            model_id=data["model_id"],
            status=status,
            verified_date=verified_date,
            metadata=metadata,
            note=data.get("note"),
            phase1_score=data.get("phase1_score"),
            phase2_score=data.get("phase2_score"),
            phase3_score=data.get("phase3_score"),
        )


@dataclass
class ArchitectureGap:
    """An unsupported architecture with model count and relevancy metrics.

    Attributes:
        architecture_id: The architecture type not supported by TransformerLens
        total_models: Number of models on HuggingFace using this architecture
        sample_models: Top models by downloads for this architecture (up to 10)
        total_downloads: Aggregate download count across all models of this architecture
        min_param_count: Parameter count of the smallest model (None if unknown)
        relevancy_score: Composite relevancy score (0-100), or None if not computed
    """

    architecture_id: str
    total_models: int
    sample_models: list[str] = field(default_factory=list)
    total_downloads: int = 0
    min_param_count: Optional[int] = None
    relevancy_score: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        d: dict = {
            "architecture_id": self.architecture_id,
            "total_models": self.total_models,
            "total_downloads": self.total_downloads,
            "min_param_count": self.min_param_count,
            "relevancy_score": self.relevancy_score,
            "sample_models": self.sample_models,
        }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ArchitectureGap":
        """Create from a dictionary."""
        return cls(
            architecture_id=data["architecture_id"],
            total_models=data["total_models"],
            sample_models=data.get("sample_models", []),
            total_downloads=data.get("total_downloads", 0),
            min_param_count=data.get("min_param_count"),
            relevancy_score=data.get("relevancy_score"),
        )


@dataclass
class ScanInfo:
    """Metadata about a scraping run.

    Attributes:
        total_scanned: Total number of models scanned in this run
        task_filter: HuggingFace task filter used (e.g., "text-generation")
        scan_duration_seconds: How long the scan took in seconds (if available)
    """

    total_scanned: int
    task_filter: str
    scan_duration_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        d: dict = {
            "total_scanned": self.total_scanned,
            "task_filter": self.task_filter,
        }
        if self.scan_duration_seconds is not None:
            d["scan_duration_seconds"] = self.scan_duration_seconds
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ScanInfo":
        """Create from a dictionary."""
        return cls(
            total_scanned=data["total_scanned"],
            task_filter=data["task_filter"],
            scan_duration_seconds=data.get("scan_duration_seconds"),
        )


@dataclass
class SupportedModelsReport:
    """Report containing all supported models.

    Attributes:
        generated_at: Date when this report was generated
        scan_info: Metadata about the scraping run
        total_architectures: Number of unique supported architectures
        total_models: Total number of supported models
        total_verified: Number of models that have been verified
        models: List of all model entries
    """

    generated_at: date
    total_models: int
    models: list[ModelEntry]
    scan_info: Optional[ScanInfo] = None
    total_architectures: int = 0
    total_verified: int = 0

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        d: dict = {
            "generated_at": self.generated_at.isoformat(),
            "scan_info": self.scan_info.to_dict() if self.scan_info else None,
            "total_architectures": self.total_architectures,
            "total_models": self.total_models,
            "total_verified": self.total_verified,
            "models": [m.to_dict() for m in self.models],
        }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "SupportedModelsReport":
        """Create from a dictionary."""
        scan_info = None
        if data.get("scan_info"):
            scan_info = ScanInfo.from_dict(data["scan_info"])
        return cls(
            generated_at=date.fromisoformat(data["generated_at"]),
            scan_info=scan_info,
            total_architectures=data.get("total_architectures", 0),
            total_models=data.get("total_models", len(data.get("models", []))),
            total_verified=data.get("total_verified", 0),
            models=[ModelEntry.from_dict(m) for m in data["models"]],
        )


@dataclass
class ArchitectureGapsReport:
    """Report containing unsupported architectures.

    Attributes:
        generated_at: Date when this report was generated
        scan_info: Metadata about the scraping run
        total_unsupported_architectures: Number of unsupported architectures
        total_unsupported_models: Total models across all unsupported architectures
        gaps: List of architecture gaps sorted by model count
    """

    generated_at: date
    gaps: list[ArchitectureGap]
    scan_info: Optional[ScanInfo] = None
    total_unsupported_architectures: int = 0
    total_unsupported_models: int = 0

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "scan_info": self.scan_info.to_dict() if self.scan_info else None,
            "total_unsupported_architectures": self.total_unsupported_architectures,
            "total_unsupported_models": self.total_unsupported_models,
            "gaps": [g.to_dict() for g in self.gaps],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArchitectureGapsReport":
        """Create from a dictionary."""
        scan_info = None
        if data.get("scan_info"):
            scan_info = ScanInfo.from_dict(data["scan_info"])
        gaps = [ArchitectureGap.from_dict(g) for g in data["gaps"]]
        return cls(
            generated_at=date.fromisoformat(data["generated_at"]),
            scan_info=scan_info,
            total_unsupported_architectures=data.get(
                "total_unsupported_architectures",
                data.get("total_unsupported", len(gaps)),
            ),
            total_unsupported_models=data.get(
                "total_unsupported_models",
                sum(g.total_models for g in gaps),
            ),
            gaps=gaps,
        )


@dataclass
class ArchitectureStats:
    """Statistics about an architecture including supported and gap info.

    Attributes:
        architecture_id: The architecture identifier
        is_supported: Whether TransformerLens supports this architecture
        model_count: Number of models using this architecture
        verified_count: Number of verified models (if supported)
        example_models: Sample model IDs for this architecture
    """

    architecture_id: str
    is_supported: bool
    model_count: int
    verified_count: int = 0
    example_models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "architecture_id": self.architecture_id,
            "is_supported": self.is_supported,
            "model_count": self.model_count,
            "verified_count": self.verified_count,
            "example_models": self.example_models,
        }


@dataclass
class ArchitectureAnalysis:
    """Analysis result for prioritizing architecture support.

    Attributes:
        architecture_id: The architecture identifier
        total_models: Total models using this architecture
        total_downloads: Sum of downloads across all models
        priority_score: Computed priority score for implementation
        top_models: Most popular models for this architecture
    """

    architecture_id: str
    total_models: int
    total_downloads: int
    priority_score: float
    top_models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "architecture_id": self.architecture_id,
            "total_models": self.total_models,
            "total_downloads": self.total_downloads,
            "priority_score": self.priority_score,
            "top_models": self.top_models,
        }
