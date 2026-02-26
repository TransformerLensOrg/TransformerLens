"""Custom exceptions for the model registry API.

This module defines specific exceptions that can be raised by the model registry
to provide clear error messages for common failure scenarios.
"""


class ModelRegistryError(Exception):
    """Base exception for all model registry errors."""

    pass


class ModelNotFoundError(ModelRegistryError):
    """Raised when a requested model ID is not found in the registry.

    Attributes:
        model_id: The model ID that was not found
        suggestion: Optional suggested alternative model
    """

    def __init__(self, model_id: str, suggestion: str | None = None):
        self.model_id = model_id
        self.suggestion = suggestion
        msg = f"Model '{model_id}' not found in the registry"
        if suggestion:
            msg += f". Did you mean '{suggestion}'?"
        super().__init__(msg)


class ArchitectureNotSupportedError(ModelRegistryError):
    """Raised when an architecture is not supported by TransformerLens.

    Attributes:
        architecture_id: The architecture that is not supported
        model_count: Number of models using this architecture (if known)
    """

    def __init__(self, architecture_id: str, model_count: int | None = None):
        self.architecture_id = architecture_id
        self.model_count = model_count
        msg = f"Architecture '{architecture_id}' is not supported by TransformerLens"
        if model_count is not None:
            msg += f" ({model_count} models on HuggingFace use this architecture)"
        super().__init__(msg)


class DataNotLoadedError(ModelRegistryError):
    """Raised when registry data has not been loaded or is unavailable.

    Attributes:
        data_type: Type of data that was not loaded (e.g., "supported_models")
        path: Optional path where data was expected
    """

    def __init__(self, data_type: str, path: str | None = None):
        self.data_type = data_type
        self.path = path
        msg = f"Registry data '{data_type}' has not been loaded"
        if path:
            msg += f" (expected at: {path})"
        super().__init__(msg)


class DataValidationError(ModelRegistryError):
    """Raised when registry data fails validation.

    Attributes:
        file_path: Path to the file that failed validation
        errors: List of validation error messages
    """

    def __init__(self, file_path: str, errors: list[str]):
        self.file_path = file_path
        self.errors = errors
        msg = f"Data validation failed for '{file_path}': {len(errors)} errors"
        if errors:
            msg += f"\n  - " + "\n  - ".join(errors[:5])
            if len(errors) > 5:
                msg += f"\n  ... and {len(errors) - 5} more"
        super().__init__(msg)
