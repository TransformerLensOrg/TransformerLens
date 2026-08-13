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
    - Gated q_proj (2x wide); AttentionBridge exposes a query-only W_Q view
    """

    # Multimodal wrapper architecture this text-only adapter rejects; the MoE
    # subclass swaps in its own name.
    _multimodal_arch_name: str = "Qwen3_5ForConditionalGeneration"

    def __init__(self, cfg: Any) -> None:
        setattr(cfg, "gated_q_proj", True)
        super().__init__(cfg, hybrid=True)

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Swap the multimodal config for its text_config so AutoModelForCausalLM
        loads the text-only model (published checkpoints carry the
        ForConditionalGeneration architecture)."""
        config = model_kwargs.get("config")
        if config is not None and hasattr(config, "text_config"):
            model_kwargs["config"] = config.text_config

    def prepare_model(self, hf_model: Any) -> None:
        """Reject full multimodal checkpoints on this text-only adapter."""
        config = getattr(hf_model, "config", None)
        architectures = getattr(config, "architectures", []) or []
        class_name = type(hf_model).__name__
        multimodal_arch = self._multimodal_arch_name

        is_conditional_generation = (
            class_name == multimodal_arch or multimodal_arch in architectures
        )
        still_has_top_level_multimodal_config = hasattr(config, "text_config")
        if is_conditional_generation or still_has_top_level_multimodal_config:
            causal_arch = multimodal_arch.replace("ForConditionalGeneration", "ForCausalLM")
            text_config_name = multimodal_arch.replace("ForConditionalGeneration", "TextConfig")
            raise ValueError(
                f"This adapter is text-only. Pass a {causal_arch} / "
                f"{text_config_name} model, or load by model id with "
                f"TransformerBridge.boot_transformers(...) so {multimodal_arch} "
                f"checkpoints route to the multimodal adapter automatically."
            )
