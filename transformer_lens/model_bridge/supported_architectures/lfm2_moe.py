"""LiquidAI LFM2 MoE architecture adapter."""

from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    DepthwiseConv1DBridge,
    EmbeddingBridge,
    Lfm2ShortConvBridge,
    LinearBridge,
    MoEBridge,
    MoERouterBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class Lfm2MoeGateBridge(MoERouterBridge):
    def get_random_inputs(
        self,
        batch_size: int = 2,
        seq_len: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Random inputs for router component testing.

        The router runs on the reshaped [N, d_model] hidden states and takes a
        second `expert_bias` arg (use_expert_bias=True); its top-k gather is
        hardcoded to dim=1, so the input must be 2D or the gather indexes the
        sequence axis out of bounds.

        Args:
            batch_size: Batch size for generated inputs
            seq_len: Sequence length for generated inputs
            device: Device to place tensors on
            dtype: Dtype for generated tensors (defaults to float32)

        Returns:
            Dictionary of input tensors matching the component's expected input signature
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        d_model = self.config.d_model if self.config and hasattr(self.config, "d_model") else 768
        num_experts = (
            self.config.num_experts if self.config and hasattr(self.config, "num_experts") else 0
        )
        hidden_states = torch.randn(batch_size * seq_len, d_model, device=device, dtype=dtype)
        expert_bias = torch.zeros(num_experts, device=device)
        return {"args": (hidden_states, expert_bias)}


class Lfm2MoeArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for LiquidAI Lfm2 MoE models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Lfm2 MoE architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()

        self.cfg.act_fn = "silu"
        self.cfg.attn_implementation = "eager"

        rope_parameters = getattr(cfg, "rope_parameters", None) or {}
        rope_theta = rope_parameters.get("rope_theta") or getattr(cfg, "rope_theta", None)
        if rope_theta is not None:
            self.cfg.rotary_base = rope_theta

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.pos_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                config=self.cfg,
                submodules={
                    "ln1": RMSNormalizationBridge(
                        name="operator_norm",
                        config=self.cfg,
                    ),
                    "ln2": RMSNormalizationBridge(
                        name="ffn_norm",
                        config=self.cfg,
                    ),
                    "attn": PositionEmbeddingsAttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        optional=True,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="out_proj"),
                            "q_norm": RMSNormalizationBridge(name="q_layernorm", config=self.cfg),
                            "k_norm": RMSNormalizationBridge(name="k_layernorm", config=self.cfg),
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    "conv": Lfm2ShortConvBridge(
                        name="conv",
                        config=self.cfg,
                        optional=True,
                        submodules={
                            "in": LinearBridge(name="in_proj"),
                            "conv": DepthwiseConv1DBridge(name="conv"),
                            "out": LinearBridge(name="out_proj"),
                        },
                    ),
                    "mlp": MoEBridge(
                        name="feed_forward",
                        config=self.cfg,
                        sparse_required=("gate",),
                        submodules={
                            "gate": Lfm2MoeGateBridge(name="gate", config=self.cfg, optional=True),
                            "dense_gate": LinearBridge(name="w1", optional=True),
                            "dense_in": LinearBridge(name="w3", optional=True),
                            "dense_out": LinearBridge(name="w2", optional=True),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.embedding_norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up model-specific references for component testing."""
        rotary_emb = hf_model.model.pos_emb

        # Set attention implementation on HF model to eager (vs sdpa default)
        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
            for layer in hf_model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"

        # Set rotary_emb on actual bridge instances
        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        # Set on template for get_generalized_component() calls
        # Find the first attention layer (LFM2 layer 0 is conv, not attn)
        layer_types = getattr(self.cfg, "layer_types", None)
        if layer_types is not None and "full_attention" in layer_types:
            first_attn_idx = layer_types.index("full_attention")
            attn_bridge = self.get_generalized_component(f"blocks.{first_attn_idx}.attn")
            attn_bridge.set_rotary_emb(rotary_emb)
