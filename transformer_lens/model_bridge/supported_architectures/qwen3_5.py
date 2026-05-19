"""Qwen3.5 architecture adapter.

Hybrid linear-attention (GatedDeltaNet) + full-attention with dense gated MLP.
3 linear-attn layers per 1 full-attn layer. Extends Qwen3 base with
optional attention mapping and fold_ln disabled.
"""

from typing import Any

import torch
from packaging.version import InvalidVersion, Version

from transformer_lens.model_bridge.supported_architectures.qwen3 import (
    Qwen3ArchitectureAdapter,
)


class Qwen3_5ArchitectureAdapter(Qwen3ArchitectureAdapter):
    """Hybrid linear-attention + full-attention with dense gated MLP.

    Inherits Qwen3 config/attention/MLP structure. Differences:
    - Attention + linear_attn are optional (per-layer type)
    - Gated q_proj (2x wide) sliced by preprocess_weights for weight analysis
    """

    _MIN_TRANSFORMERS_VERSION = Version("5.2.0")

    def __init__(self, cfg: Any) -> None:
        self._validate_transformers_support()
        setattr(cfg, "gated_q_proj", True)
        super().__init__(cfg, hybrid=True)

    @classmethod
    def _validate_transformers_support(cls) -> None:
        """Fail clearly when the optional Qwen3.5 Transformers support is unavailable."""
        import transformers

        try:
            installed_version = Version(transformers.__version__)
        except InvalidVersion:
            installed_version = Version("0")

        if installed_version < cls._MIN_TRANSFORMERS_VERSION:
            raise ImportError(
                f"Qwen3.5 requires transformers >= {cls._MIN_TRANSFORMERS_VERSION} "
                f"(installed: {transformers.__version__}). "
                f"Upgrade with: pip install 'transformers>={cls._MIN_TRANSFORMERS_VERSION}'"
            )

        if not cls._has_qwen3_5_causal_lm(transformers):
            raise ImportError(
                "Qwen3.5 requires a Transformers build that exposes "
                "Qwen3_5ForCausalLM. Install the Qwen3.5 optional dependency "
                f"with: pip install 'transformers>={cls._MIN_TRANSFORMERS_VERSION}'"
            )

    @staticmethod
    def _has_qwen3_5_causal_lm(transformers_module: Any) -> bool:
        return hasattr(transformers_module, "Qwen3_5ForCausalLM")

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Swap multimodal Qwen3_5Config for text-only Qwen3_5TextConfig.

        Published checkpoints carry architectures=['Qwen3_5ForConditionalGeneration'].
        We replace config with text_config so AutoModelForCausalLM loads the
        text-only Qwen3_5ForCausalLM.
        """
        config = model_kwargs.get("config")
        if config is not None and hasattr(config, "text_config"):
            model_kwargs["config"] = config.text_config

    def prepare_model(self, hf_model: Any) -> None:
        """Reject full multimodal Qwen3.5 models on this text-only adapter."""
        config = getattr(hf_model, "config", None)
        architectures = getattr(config, "architectures", []) or []
        class_name = type(hf_model).__name__

        is_conditional_generation = (
            class_name == "Qwen3_5ForConditionalGeneration"
            or "Qwen3_5ForConditionalGeneration" in architectures
        )
        still_has_top_level_multimodal_config = hasattr(config, "text_config")
        if is_conditional_generation or still_has_top_level_multimodal_config:
            raise ValueError(
                "Qwen3.5 support in TransformerLens is text-only. Pass a "
                "Qwen3_5ForCausalLM / Qwen3_5TextConfig model, or load by model id "
                "with TransformerBridge.boot_transformers(...) so the text_config is "
                "selected automatically. Qwen3_5ForConditionalGeneration, image/video "
                "inputs, and Qwen3.5 MoE are not supported by this adapter."
            )

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Slice query half from gated q_proj.weight for weight-space analysis.

        In processed mode, W_Q is the pure query projection (for composition
        scores, logit lens). Gate signal available in unprocessed mode on
        full-attention layers via blocks.N.attn.hook_q_gate.
        """
        return self._preprocess_gated_q_proj(state_dict, self.cfg.n_heads, self.cfg.d_head)
