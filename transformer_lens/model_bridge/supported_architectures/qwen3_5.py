"""Qwen3.5 architecture adapter.

Hybrid linear-attention (GatedDeltaNet) + full-attention with dense gated MLP.
3 linear-attn layers per 1 full-attn layer. Extends Qwen3 base with
optional attention mapping and fold_ln disabled.
"""

from typing import Any

from transformer_lens.model_bridge.supported_architectures.qwen3 import (
    Qwen3ArchitectureAdapter,
)


class Qwen3_5ArchitectureAdapter(Qwen3ArchitectureAdapter):
    """Hybrid linear-attention + full-attention with dense gated MLP.

    Inherits Qwen3 config/attention/MLP structure. Differences:
    - Attention + linear_attn are optional (per-layer type)
    - Gated q_proj: [query|gate] is split at forward time, never in weight space
    """

    def __init__(self, cfg: Any) -> None:
        # q_proj stays 2x-wide through weight processing: HF and the attention bridge both
        # split [query|gate] per head at forward time; slicing the gate out changes outputs.
        setattr(cfg, "gated_q_proj", True)
        super().__init__(cfg, hybrid=True)

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
