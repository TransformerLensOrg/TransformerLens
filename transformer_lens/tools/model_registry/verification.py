"""Verification tracking for model compatibility.

This module provides dataclasses and utilities for tracking which models
have been verified to work with TransformerLens.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


@dataclass
class VerificationRecord:
    """A record of a model verification.

    Attributes:
        model_id: The HuggingFace model ID that was verified
        architecture_id: The architecture type of the model
        verified_date: Date when verification was performed
        verified_by: Who performed the verification (user, CI, etc.)
        transformerlens_version: Version of TransformerLens used
        notes: Optional notes about the verification
        invalidated: Whether this verification has been invalidated
        invalidation_reason: Reason for invalidation if applicable
    """

    model_id: str
    verified_date: date
    architecture_id: str = "Unknown"
    verified_by: Optional[str] = None
    transformerlens_version: Optional[str] = None
    notes: Optional[str] = None
    invalidated: bool = False
    invalidation_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "model_id": self.model_id,
            "architecture_id": self.architecture_id,
            "verified_date": self.verified_date.isoformat(),
            "verified_by": self.verified_by,
            "transformerlens_version": self.transformerlens_version,
            "notes": self.notes,
            "invalidated": self.invalidated,
            "invalidation_reason": self.invalidation_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VerificationRecord":
        """Create from a dictionary."""
        return cls(
            model_id=data["model_id"],
            architecture_id=data.get("architecture_id", "Unknown"),
            verified_date=date.fromisoformat(data["verified_date"]),
            verified_by=data.get("verified_by"),
            transformerlens_version=data.get("transformerlens_version"),
            notes=data.get("notes"),
            invalidated=data.get("invalidated", False),
            invalidation_reason=data.get("invalidation_reason"),
        )


@dataclass
class VerificationHistory:
    """History of all model verifications.

    Attributes:
        records: List of all verification records
        last_updated: When this history was last updated
    """

    records: list[VerificationRecord] = field(default_factory=list)
    last_updated: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "records": [r.to_dict() for r in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VerificationHistory":
        """Create from a dictionary."""
        last_updated = None
        if data.get("last_updated"):
            last_updated = datetime.fromisoformat(data["last_updated"])
        return cls(
            records=[VerificationRecord.from_dict(r) for r in data.get("records", [])],
            last_updated=last_updated,
        )

    def get_record(self, model_id: str) -> Optional[VerificationRecord]:
        """Get the most recent valid verification record for a model.

        Args:
            model_id: The model ID to look up

        Returns:
            The verification record, or None if not found or invalidated
        """
        for record in reversed(self.records):
            if record.model_id == model_id and not record.invalidated:
                return record
        return None

    def is_verified(self, model_id: str) -> bool:
        """Check if a model has a valid verification.

        Args:
            model_id: The model ID to check

        Returns:
            True if the model has a valid (non-invalidated) verification
        """
        return self.get_record(model_id) is not None

    def add_record(self, record: VerificationRecord) -> None:
        """Add a new verification record.

        Args:
            record: The verification record to add
        """
        self.records.append(record)
        self.last_updated = datetime.now()

    def invalidate(self, model_id: str, reason: str) -> bool:
        """Invalidate the most recent verification for a model.

        Args:
            model_id: The model ID to invalidate
            reason: Reason for invalidation

        Returns:
            True if a record was invalidated, False if not found
        """
        record = self.get_record(model_id)
        if record:
            record.invalidated = True
            record.invalidation_reason = reason
            self.last_updated = datetime.now()
            return True
        return False
