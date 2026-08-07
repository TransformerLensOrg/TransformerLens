"""HF-loader-specific helpers: model-class selection, modality processor loading, checkpoint revision resolution, registry discovery."""
from __future__ import annotations

from typing import Any

import transformers
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
        AUDIO_CLASSIFICATION_ARCHITECTURES,
        AUDIO_TEXT_ARCHITECTURES,
        BASE_AUTOMODEL_ARCHITECTURES,
        MASKED_LM_ARCHITECTURES,
        MULTIMODAL_ARCHITECTURES,
        SEQ2SEQ_ARCHITECTURES,
        VISION_ARCHITECTURES,
        VISION_CLASSIFICATION_ARCHITECTURES,
    )

    if architecture in SEQ2SEQ_ARCHITECTURES or architecture in AUDIO_TEXT_ARCHITECTURES:
        return AutoModelForSeq2SeqLM
    elif architecture in MASKED_LM_ARCHITECTURES:
        return AutoModelForMaskedLM
    elif architecture in MULTIMODAL_ARCHITECTURES:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    elif architecture in BASE_AUTOMODEL_ARCHITECTURES:
        from transformers import AutoModel

        return AutoModel
    elif architecture in AUDIO_CLASSIFICATION_ARCHITECTURES:
        from transformers import AutoModelForAudioClassification

        return AutoModelForAudioClassification
    elif architecture in AUDIO_ARCHITECTURES:
        if "ForCTC" in architecture:
            from transformers import AutoModelForCTC

            return AutoModelForCTC
        from transformers import AutoModel

        return AutoModel
    elif architecture in VISION_ARCHITECTURES:
        if architecture in VISION_CLASSIFICATION_ARCHITECTURES:
            from transformers import AutoModelForImageClassification

            return AutoModelForImageClassification
        from transformers import AutoModel

        return AutoModel
    else:
        return AutoModelForCausalLM


# Modality flag → HF auto-loader class, applied in order. Last write wins on
# bridge.processor: multimodal → audio → vision. The flags are disjoint today
# (vit.py is the only is_visual_model adapter and it doesn't set is_multimodal),
# but the order must be preserved for future dual-flag adapters.
_MODALITY_PROCESSOR_LOADERS: list[tuple[str, str]] = [
    ("is_multimodal", "AutoProcessor"),
    ("is_audio_model", "AutoFeatureExtractor"),
    ("is_visual_model", "AutoImageProcessor"),
]


def _ensure_torchvision() -> bool:
    """Import torchvision, installing it on the fly if missing; True when importable."""
    try:
        import torchvision  # noqa: F401

        return True
    except Exception:
        pass
    import importlib
    import shutil
    import subprocess
    import sys

    try:
        if shutil.which("uv"):
            subprocess.check_call(["uv", "pip", "install", "torchvision", "-q"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision", "-q"])
        importlib.invalidate_caches()
        return True
    except Exception:
        return False


def load_modality_processor(
    bridge: Any,
    cfg: Any,
    model_name: str,
    trust_remote_code: bool,
    token: str | None,
) -> None:
    """Attach the modality preprocessor to ``bridge.processor`` per the cfg's modality flags.

    Best-effort: each loader failure is swallowed so a missing/broken processor
    never blocks the boot itself.
    """
    for cfg_flag, loader_name in _MODALITY_PROCESSOR_LOADERS:
        if not getattr(cfg, cfg_flag, False):
            continue
        try:
            loader = getattr(transformers, loader_name)
            bridge.processor = loader.from_pretrained(
                model_name,
                token=token,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            # Some AutoProcessors need torchvision (e.g. LlavaOnevision); install and retry.
            if loader_name != "AutoProcessor" or not _ensure_torchvision():
                continue
            try:
                bridge.processor = transformers.AutoProcessor.from_pretrained(
                    model_name,
                    token=token,
                    trust_remote_code=trust_remote_code,
                )
            except Exception:
                pass


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
