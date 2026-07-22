"""Qwen3.5-MoE architecture adapter.

Hybrid linear-attention (GatedDeltaNet) + full-attention with sparse MoE MLP
(256 experts, top-8 routing, shared expert in public checkpoints). Same hybrid
design as Qwen3.5 dense and the same MoE block family as Qwen3-Next.

Two adapters: text-only ``Qwen3_5MoeForCausalLM`` and the vision-language
``Qwen3_5MoeForConditionalGeneration`` (text backbone nested under
``model.language_model`` plus the Qwen3.5 vision tower).
"""

from typing import Any

import torch

from transformer_lens.model_bridge.generalized_components import (
    LinearBridge,
    MoEBridge,
    MoERouterBridge,
)
from transformer_lens.model_bridge.supported_architectures.qwen3 import (
    Qwen3ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.qwen3_5_multimodal import (
    Qwen3_5MultimodalArchitectureAdapter,
)


def _sparse_moe_mlp(adapter):
    """Qwen3.5 sparse MoE block: 3-tuple router, batched experts, shared expert."""
    return MoEBridge(
        name="mlp",
        config=adapter.cfg,
        submodules={
            "gate": MoERouterBridge(name="gate"),
            "experts": MoEBridge(name="experts", config=adapter.cfg),
            "shared_expert": adapter._gated_mlp(name="shared_expert"),
            "shared_expert_gate": LinearBridge(name="shared_expert_gate"),
        },
    )


class Qwen3_5MoeArchitectureAdapter(Qwen3ArchitectureAdapter):
    """Text-only Qwen3.5-MoE: hybrid GatedDeltaNet + full attention, sparse MoE MLP."""

    def __init__(self, cfg: Any) -> None:
        setattr(cfg, "gated_q_proj", True)
        super().__init__(cfg, hybrid=True)

    def _build_mlp_bridge(self):
        """Sparse MoE MLP (router + batched experts + shared expert)."""
        return _sparse_moe_mlp(self)

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Swap multimodal Qwen3_5MoeConfig for text-only Qwen3_5MoeTextConfig.

        Published checkpoints carry architectures=['Qwen3_5MoeForConditionalGeneration'].
        We replace config with text_config so AutoModelForCausalLM loads the
        text-only Qwen3_5MoeForCausalLM.
        """
        config = model_kwargs.get("config")
        if config is not None and hasattr(config, "text_config"):
            model_kwargs["config"] = config.text_config

    def prepare_model(self, hf_model: Any) -> None:
        """Reject full multimodal Qwen3.5-MoE models on this text-only adapter."""
        config = getattr(hf_model, "config", None)
        architectures = getattr(config, "architectures", []) or []
        class_name = type(hf_model).__name__

        is_conditional_generation = (
            class_name == "Qwen3_5MoeForConditionalGeneration"
            or "Qwen3_5MoeForConditionalGeneration" in architectures
        )
        still_has_top_level_multimodal_config = hasattr(config, "text_config")
        if is_conditional_generation or still_has_top_level_multimodal_config:
            raise ValueError(
                "This adapter is text-only. Pass a Qwen3_5MoeForCausalLM / "
                "Qwen3_5MoeTextConfig model, or load by model id with "
                "TransformerBridge.boot_transformers(...) so Qwen3_5MoeForConditionalGeneration "
                "checkpoints route to the multimodal adapter automatically."
            )

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Slice query half from gated q_proj.weight for weight-space analysis."""
        return self._preprocess_gated_q_proj(state_dict, self.cfg.n_heads, self.cfg.d_head)


class Qwen3_5MoeMultimodalArchitectureAdapter(Qwen3_5MultimodalArchitectureAdapter):
    """Vision-language adapter for Qwen3_5MoeForConditionalGeneration.

    Reuses the Qwen3.5 multimodal wiring (language model under
    ``model.language_model`` + vision tower) with the MLP swapped for sparse MoE.
    """

    def _build_mlp_bridge(self):
        """Sparse MoE MLP (router + batched experts + shared expert)."""
        return _sparse_moe_mlp(self)
