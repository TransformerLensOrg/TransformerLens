"""HF-loader-specific helpers: model-class selection, checkpoint revision resolution, registry discovery."""
from __future__ import annotations

from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
)

from transformer_lens.tools.model_registry.checkpoints import get_checkpoint_labels


def get_hf_model_class_for_architecture(architecture: str):
    """Pick the correct HuggingFace ``AutoModel*`` class for the architecture."""
    from transformer_lens.utilities.architectures import (
        AUDIO_ARCHITECTURES,
        MASKED_LM_ARCHITECTURES,
        MULTIMODAL_ARCHITECTURES,
        SEQ2SEQ_ARCHITECTURES,
    )

    if architecture in SEQ2SEQ_ARCHITECTURES:
        return AutoModelForSeq2SeqLM
    elif architecture in MASKED_LM_ARCHITECTURES:
        return AutoModelForMaskedLM
    elif architecture in MULTIMODAL_ARCHITECTURES:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    elif architecture in AUDIO_ARCHITECTURES:
        if "ForCTC" in architecture:
            from transformers import AutoModelForCTC

            return AutoModelForCTC
        from transformers import AutoModel

        return AutoModel
    else:
        return AutoModelForCausalLM


# Known training-checkpoint revision conventions on HF Hub.
_CHECKPOINT_REVISION_FORMATS: dict[str, str] = {
    "EleutherAI/pythia": "step{value}",
    "stanford-crfm": "checkpoint-{value}",
}


def _resolve_checkpoint_to_revision(
    model_name: str,
    checkpoint_index: int | None,
    checkpoint_value: int | None,
) -> str:
    """Convert a checkpoint index/value into an HF revision string, validated against ``get_checkpoint_labels``."""
    if checkpoint_index is None and checkpoint_value is None:
        raise ValueError("Must specify either checkpoint_index or checkpoint_value.")

    format_str: str | None = None
    for prefix, fmt in _CHECKPOINT_REVISION_FORMATS.items():
        if model_name.startswith(prefix):
            format_str = fmt
            break
    if format_str is None:
        raise ValueError(
            f"Model {model_name!r} does not have a known checkpoint revision convention. "
            f"Pass revision= directly if your model uses HF revisions. Known checkpoint "
            f"families: {list(_CHECKPOINT_REVISION_FORMATS.keys())}."
        )

    labels, _ = get_checkpoint_labels(model_name)
    if checkpoint_value is not None:
        if checkpoint_value not in labels:
            raise ValueError(
                f"checkpoint_value={checkpoint_value} not in available checkpoints for "
                f"{model_name!r}. {len(labels)} labels available, "
                f"first/last: {labels[0]}..{labels[-1]}."
            )
    else:
        assert checkpoint_index is not None  # narrowed by initial guard
        if not 0 <= checkpoint_index < len(labels):
            raise ValueError(
                f"checkpoint_index={checkpoint_index} out of range [0, {len(labels)}) "
                f"for {model_name!r}."
            )
        checkpoint_value = labels[checkpoint_index]
    return format_str.format(value=checkpoint_value)


def list_supported_models(
    architecture: str | None = None,
    verified_only: bool = False,
) -> list[str]:
    """List all models supported by TransformerLens.

    Args:
        architecture: Filter by architecture ID (e.g., "GPT2LMHeadModel").
        verified_only: If True, only return verified-to-work models.

    Returns:
        List of model IDs.
    """
    try:
        from transformer_lens.tools.model_registry import api

        models = api.get_supported_models(architecture=architecture, verified_only=verified_only)
        return [m.model_id for m in models]
    except ImportError:
        return []
    except Exception:
        return []


def check_model_support(model_id: str) -> dict:
    """Detailed support info for a model: ``is_supported``, ``architecture_id``, ``verified``, ``suggestion``."""
    try:
        from transformer_lens.tools.model_registry import api

        is_supported = api.is_model_supported(model_id)

        if is_supported:
            model_info = api.get_model_info(model_id)
            return {
                "is_supported": True,
                "architecture_id": model_info.architecture_id,
                "status": model_info.status,
                "verified_date": (
                    model_info.verified_date.isoformat() if model_info.verified_date else None
                ),
                "suggestion": None,
            }
        else:
            suggestion = api.suggest_similar_model(model_id)
            return {
                "is_supported": False,
                "architecture_id": None,
                "verified": False,
                "verified_date": None,
                "suggestion": suggestion,
            }
    except ImportError:
        return {
            "is_supported": None,
            "architecture_id": None,
            "verified": False,
            "verified_date": None,
            "suggestion": None,
            "error": "Model registry not available",
        }
    except Exception as e:
        return {
            "is_supported": None,
            "architecture_id": None,
            "verified": False,
            "verified_date": None,
            "suggestion": None,
            "error": str(e),
        }
