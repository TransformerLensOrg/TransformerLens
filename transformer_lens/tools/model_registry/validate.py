"""JSON schema validation for the model registry output files.

This module provides functions to validate that the JSON output files in the data/
directory conform to the expected schemas defined by the dataclasses in schemas.py
and verification.py.
"""

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a validation error in a JSON file.

    Attributes:
        path: JSON path where the error occurred (e.g., "models[0].architecture_id")
        message: Description of the validation error
        value: The actual value that caused the error (if applicable)
    """

    path: str
    message: str
    value: Any = None

    def __str__(self) -> str:
        """Return a human-readable error message."""
        if self.value is not None:
            return f"{self.path}: {self.message} (got: {self.value!r})"
        return f"{self.path}: {self.message}"


@dataclass
class ValidationResult:
    """Result of validating a JSON file against its schema.

    Attributes:
        valid: Whether the file passed validation
        errors: List of validation errors (empty if valid)
        schema_type: The schema type that was validated against
    """

    valid: bool
    errors: list[ValidationError]
    schema_type: str

    @property
    def error_count(self) -> int:
        """Return the number of validation errors."""
        return len(self.errors)


def _validate_string(
    value: Any, path: str, required: bool = True, min_length: int = 0
) -> list[ValidationError]:
    """Validate that a value is a string.

    Args:
        value: The value to validate
        path: JSON path for error reporting
        required: Whether the field is required (None not allowed)
        min_length: Minimum string length (only checked if value is not None)

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    if value is None:
        if required:
            errors.append(ValidationError(path, "required field is missing or null"))
    elif not isinstance(value, str):
        errors.append(ValidationError(path, f"expected string, got {type(value).__name__}", value))
    elif min_length > 0 and len(value) < min_length:
        errors.append(
            ValidationError(path, f"string must be at least {min_length} characters", value)
        )
    return errors


def _validate_int(
    value: Any, path: str, required: bool = True, min_value: int | None = None
) -> list[ValidationError]:
    """Validate that a value is an integer.

    Args:
        value: The value to validate
        path: JSON path for error reporting
        required: Whether the field is required (None not allowed)
        min_value: Minimum allowed value (only checked if value is not None)

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    if value is None:
        if required:
            errors.append(ValidationError(path, "required field is missing or null"))
    elif not isinstance(value, int) or isinstance(value, bool):
        errors.append(ValidationError(path, f"expected integer, got {type(value).__name__}", value))
    elif min_value is not None and value < min_value:
        errors.append(ValidationError(path, f"value must be >= {min_value}", value))
    return errors


def _validate_bool(value: Any, path: str, required: bool = True) -> list[ValidationError]:
    """Validate that a value is a boolean.

    Args:
        value: The value to validate
        path: JSON path for error reporting
        required: Whether the field is required (None not allowed)

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    if value is None:
        if required:
            errors.append(ValidationError(path, "required field is missing or null"))
    elif not isinstance(value, bool):
        errors.append(ValidationError(path, f"expected boolean, got {type(value).__name__}", value))
    return errors


def _validate_date_string(value: Any, path: str, required: bool = True) -> list[ValidationError]:
    """Validate that a value is a valid ISO date string.

    Args:
        value: The value to validate
        path: JSON path for error reporting
        required: Whether the field is required (None not allowed)

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    if value is None:
        if required:
            errors.append(ValidationError(path, "required field is missing or null"))
    elif not isinstance(value, str):
        errors.append(
            ValidationError(path, f"expected date string, got {type(value).__name__}", value)
        )
    else:
        try:
            date.fromisoformat(value)
        except ValueError:
            errors.append(ValidationError(path, "invalid ISO date format", value))
    return errors


def _validate_datetime_string(
    value: Any, path: str, required: bool = True
) -> list[ValidationError]:
    """Validate that a value is a valid ISO datetime string.

    Args:
        value: The value to validate
        path: JSON path for error reporting
        required: Whether the field is required (None not allowed)

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    if value is None:
        if required:
            errors.append(ValidationError(path, "required field is missing or null"))
    elif not isinstance(value, str):
        errors.append(
            ValidationError(path, f"expected datetime string, got {type(value).__name__}", value)
        )
    else:
        try:
            datetime.fromisoformat(value)
        except ValueError:
            errors.append(ValidationError(path, "invalid ISO datetime format", value))
    return errors


def _validate_list(value: Any, path: str, required: bool = True) -> list[ValidationError]:
    """Validate that a value is a list.

    Args:
        value: The value to validate
        path: JSON path for error reporting
        required: Whether the field is required (None not allowed)

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    if value is None:
        if required:
            errors.append(ValidationError(path, "required field is missing or null"))
    elif not isinstance(value, list):
        errors.append(ValidationError(path, f"expected list, got {type(value).__name__}", value))
    return errors


def _validate_model_metadata(data: dict, path: str) -> list[ValidationError]:
    """Validate a ModelMetadata object.

    Args:
        data: Dictionary to validate
        path: JSON path prefix for error reporting

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # downloads (optional, defaults to 0)
    if "downloads" in data:
        errors.extend(
            _validate_int(data["downloads"], f"{path}.downloads", required=False, min_value=0)
        )

    # likes (optional, defaults to 0)
    if "likes" in data:
        errors.extend(_validate_int(data["likes"], f"{path}.likes", required=False, min_value=0))

    # last_modified (optional datetime)
    if "last_modified" in data and data["last_modified"] is not None:
        errors.extend(
            _validate_datetime_string(
                data["last_modified"], f"{path}.last_modified", required=False
            )
        )

    # tags (optional list of strings)
    if "tags" in data:
        tags = data["tags"]
        if tags is not None:
            errors.extend(_validate_list(tags, f"{path}.tags", required=False))
            if isinstance(tags, list):
                for i, tag in enumerate(tags):
                    errors.extend(_validate_string(tag, f"{path}.tags[{i}]"))

    # parameter_count (optional int)
    if "parameter_count" in data and data["parameter_count"] is not None:
        errors.extend(
            _validate_int(
                data["parameter_count"], f"{path}.parameter_count", required=False, min_value=0
            )
        )

    return errors


def _validate_model_entry(data: dict, path: str) -> list[ValidationError]:
    """Validate a ModelEntry object.

    Args:
        data: Dictionary to validate
        path: JSON path prefix for error reporting

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if not isinstance(data, dict):
        return [ValidationError(path, f"expected object, got {type(data).__name__}", data)]

    # architecture_id (required string)
    errors.extend(
        _validate_string(data.get("architecture_id"), f"{path}.architecture_id", min_length=1)
    )

    # model_id (required string)
    errors.extend(_validate_string(data.get("model_id"), f"{path}.model_id", min_length=1))

    # status (optional int 0-3, defaults to 0)
    if "status" in data:
        errors.extend(_validate_int(data["status"], f"{path}.status", required=False, min_value=0))
        if isinstance(data["status"], int) and not isinstance(data["status"], bool):
            if data["status"] > 3:
                errors.append(
                    ValidationError(f"{path}.status", "value must be 0-3", data["status"])
                )

    # note (optional string)
    if "note" in data and data["note"] is not None:
        errors.extend(_validate_string(data["note"], f"{path}.note", min_length=1))

    # verified_date (optional date string)
    if "verified_date" in data and data["verified_date"] is not None:
        errors.extend(
            _validate_date_string(data["verified_date"], f"{path}.verified_date", required=False)
        )

    # metadata (optional ModelMetadata)
    if "metadata" in data and data["metadata"] is not None:
        if not isinstance(data["metadata"], dict):
            errors.append(
                ValidationError(
                    f"{path}.metadata",
                    f"expected object, got {type(data['metadata']).__name__}",
                    data["metadata"],
                )
            )
        else:
            errors.extend(_validate_model_metadata(data["metadata"], f"{path}.metadata"))

    # phase scores (optional floats, 0-100 or None)
    for phase_field in ("phase1_score", "phase2_score", "phase3_score"):
        if phase_field in data and data[phase_field] is not None:
            val = data[phase_field]
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                errors.append(
                    ValidationError(
                        f"{path}.{phase_field}",
                        f"expected number, got {type(val).__name__}",
                        val,
                    )
                )

    return errors


def _validate_architecture_gap(data: dict, path: str) -> list[ValidationError]:
    """Validate an ArchitectureGap object.

    Args:
        data: Dictionary to validate
        path: JSON path prefix for error reporting

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if not isinstance(data, dict):
        return [ValidationError(path, f"expected object, got {type(data).__name__}", data)]

    # architecture_id (required string)
    errors.extend(
        _validate_string(data.get("architecture_id"), f"{path}.architecture_id", min_length=1)
    )

    # total_models (required int >= 0)
    errors.extend(_validate_int(data.get("total_models"), f"{path}.total_models", min_value=0))

    return errors


def _validate_verification_record(data: dict, path: str) -> list[ValidationError]:
    """Validate a VerificationRecord object.

    Args:
        data: Dictionary to validate
        path: JSON path prefix for error reporting

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if not isinstance(data, dict):
        return [ValidationError(path, f"expected object, got {type(data).__name__}", data)]

    # model_id (required string)
    errors.extend(_validate_string(data.get("model_id"), f"{path}.model_id", min_length=1))

    # architecture_id (optional string, defaults to "Unknown")
    if "architecture_id" in data and data["architecture_id"] is not None:
        errors.extend(
            _validate_string(data["architecture_id"], f"{path}.architecture_id", required=False)
        )

    # verified_date (required date string)
    errors.extend(_validate_date_string(data.get("verified_date"), f"{path}.verified_date"))

    # verified_by (optional string)
    if "verified_by" in data and data["verified_by"] is not None:
        errors.extend(_validate_string(data["verified_by"], f"{path}.verified_by", required=False))

    # transformerlens_version (optional string)
    if "transformerlens_version" in data and data["transformerlens_version"] is not None:
        errors.extend(
            _validate_string(
                data["transformerlens_version"], f"{path}.transformerlens_version", required=False
            )
        )

    # notes (optional string)
    if "notes" in data and data["notes"] is not None:
        errors.extend(_validate_string(data["notes"], f"{path}.notes", required=False))

    # invalidated (optional boolean, defaults to False)
    if "invalidated" in data:
        errors.extend(_validate_bool(data["invalidated"], f"{path}.invalidated", required=False))

    # invalidation_reason (optional string)
    if "invalidation_reason" in data and data["invalidation_reason"] is not None:
        errors.extend(
            _validate_string(
                data["invalidation_reason"], f"{path}.invalidation_reason", required=False
            )
        )

    return errors


def validate_supported_models_report(data: dict) -> ValidationResult:
    """Validate a SupportedModelsReport JSON object.

    Args:
        data: Dictionary loaded from JSON to validate

    Returns:
        ValidationResult with validation status and any errors
    """
    errors = []

    if not isinstance(data, dict):
        return ValidationResult(
            valid=False,
            errors=[
                ValidationError("", f"expected object at root, got {type(data).__name__}", data)
            ],
            schema_type="SupportedModelsReport",
        )

    # generated_at (required date string)
    errors.extend(_validate_date_string(data.get("generated_at"), "generated_at"))

    # total_architectures (required int >= 0)
    errors.extend(
        _validate_int(data.get("total_architectures"), "total_architectures", min_value=0)
    )

    # total_models (required int >= 0)
    errors.extend(_validate_int(data.get("total_models"), "total_models", min_value=0))

    # total_verified (required int >= 0)
    errors.extend(_validate_int(data.get("total_verified"), "total_verified", min_value=0))

    # models (required list of ModelEntry)
    models = data.get("models")
    errors.extend(_validate_list(models, "models"))
    if isinstance(models, list):
        for i, model in enumerate(models):
            errors.extend(_validate_model_entry(model, f"models[{i}]"))

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        schema_type="SupportedModelsReport",
    )


def validate_architecture_gaps_report(data: dict) -> ValidationResult:
    """Validate an ArchitectureGapsReport JSON object.

    Args:
        data: Dictionary loaded from JSON to validate

    Returns:
        ValidationResult with validation status and any errors
    """
    errors = []

    if not isinstance(data, dict):
        return ValidationResult(
            valid=False,
            errors=[
                ValidationError("", f"expected object at root, got {type(data).__name__}", data)
            ],
            schema_type="ArchitectureGapsReport",
        )

    # generated_at (required date string)
    errors.extend(_validate_date_string(data.get("generated_at"), "generated_at"))

    # total_unsupported_architectures (required int >= 0)
    errors.extend(
        _validate_int(
            data.get("total_unsupported_architectures"),
            "total_unsupported_architectures",
            min_value=0,
        )
    )

    # total_unsupported_models (required int >= 0)
    errors.extend(
        _validate_int(
            data.get("total_unsupported_models"),
            "total_unsupported_models",
            min_value=0,
        )
    )

    # gaps (required list of ArchitectureGap)
    gaps = data.get("gaps")
    errors.extend(_validate_list(gaps, "gaps"))
    if isinstance(gaps, list):
        for i, gap in enumerate(gaps):
            errors.extend(_validate_architecture_gap(gap, f"gaps[{i}]"))

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        schema_type="ArchitectureGapsReport",
    )


def validate_verification_history(data: dict) -> ValidationResult:
    """Validate a VerificationHistory JSON object.

    Args:
        data: Dictionary loaded from JSON to validate

    Returns:
        ValidationResult with validation status and any errors
    """
    errors = []

    if not isinstance(data, dict):
        return ValidationResult(
            valid=False,
            errors=[
                ValidationError("", f"expected object at root, got {type(data).__name__}", data)
            ],
            schema_type="VerificationHistory",
        )

    # last_updated (optional datetime string)
    if "last_updated" in data and data["last_updated"] is not None:
        errors.extend(
            _validate_datetime_string(data["last_updated"], "last_updated", required=False)
        )

    # records (required list of VerificationRecord)
    records = data.get("records")
    errors.extend(_validate_list(records, "records"))
    if isinstance(records, list):
        for i, record in enumerate(records):
            errors.extend(_validate_verification_record(record, f"records[{i}]"))

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        schema_type="VerificationHistory",
    )


def validate_json_schema(file_path: Path | str, schema_type: str | None = None) -> ValidationResult:
    """Validate a JSON file against its expected schema.

    This function reads a JSON file and validates it against one of the model registry
    schemas. The schema type can be automatically inferred from the filename or
    explicitly specified.

    Args:
        file_path: Path to the JSON file to validate
        schema_type: Schema type to validate against. If None, inferred from filename.
            Supported values: "supported_models", "architecture_gaps", "verification_history"

    Returns:
        ValidationResult with validation status and any errors

    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file is not valid JSON
        ValueError: If schema_type cannot be determined
    """
    file_path = Path(file_path)

    # Infer schema type from filename if not provided
    if schema_type is None:
        filename = file_path.stem.lower()
        if "supported_models" in filename or filename == "supported_models":
            schema_type = "supported_models"
        elif "architecture_gaps" in filename or filename == "architecture_gaps":
            schema_type = "architecture_gaps"
        elif "verification" in filename or filename == "verification_history":
            schema_type = "verification_history"
        else:
            raise ValueError(
                f"Cannot infer schema type from filename '{file_path.name}'. "
                "Please specify schema_type explicitly. "
                "Supported values: 'supported_models', 'architecture_gaps', 'verification_history'"
            )

    # Read and parse the JSON file
    with open(file_path) as f:
        data = json.load(f)

    # Validate based on schema type
    if schema_type == "supported_models":
        return validate_supported_models_report(data)
    elif schema_type == "architecture_gaps":
        return validate_architecture_gaps_report(data)
    elif schema_type == "verification_history":
        return validate_verification_history(data)
    else:
        raise ValueError(
            f"Unknown schema_type: {schema_type}. "
            "Supported values: 'supported_models', 'architecture_gaps', 'verification_history'"
        )


def validate_data_directory(data_dir: Path | str | None = None) -> dict[str, ValidationResult]:
    """Validate all JSON files in the data directory.

    Validates supported_models.json, verification_history.json, and
    architecture_gaps.json.

    Args:
        data_dir: Path to the data directory. If None, uses the default data directory.

    Returns:
        Dictionary mapping filenames to their ValidationResults
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    else:
        data_dir = Path(data_dir)

    results = {}

    # Validate supported_models.json
    supported_path = data_dir / "supported_models.json"
    if supported_path.exists():
        try:
            results["supported_models.json"] = validate_json_schema(
                supported_path, "supported_models"
            )
        except json.JSONDecodeError as e:
            results["supported_models.json"] = ValidationResult(
                valid=False,
                errors=[ValidationError("", f"Invalid JSON: {e}")],
                schema_type="supported_models",
            )

    # Validate architecture_gaps.json
    gaps_path = data_dir / "architecture_gaps.json"
    if gaps_path.exists():
        try:
            results["architecture_gaps.json"] = validate_json_schema(gaps_path, "architecture_gaps")
        except json.JSONDecodeError as e:
            results["architecture_gaps.json"] = ValidationResult(
                valid=False,
                errors=[ValidationError("", f"Invalid JSON: {e}")],
                schema_type="architecture_gaps",
            )

    # Validate verification_history.json
    verification_path = data_dir / "verification_history.json"
    if verification_path.exists():
        try:
            results["verification_history.json"] = validate_json_schema(
                verification_path, "verification_history"
            )
        except json.JSONDecodeError as e:
            results["verification_history.json"] = ValidationResult(
                valid=False,
                errors=[ValidationError("", f"Invalid JSON: {e}")],
                schema_type="verification_history",
            )

    return results
