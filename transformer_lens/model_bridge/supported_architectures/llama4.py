"""Llama 4 (text) architecture adapter.

Meta's Llama 4 text decoder (``Llama4ForCausalLM``): llama-style RMS-norm
blocks whose attention adds complex-valued interleaved RoPE, NoPE layers
with temperature tuning, post-RoPE weightless L2 QK-norm, and chunked
attention masks — so attention stays delegated to HF. The feed-forward is
either a sparse MoE (batched 3D experts + top-k sigmoid router + shared
expert) or a dense gated MLP on non-MoE layers.
"""

from typing import Any

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    MoEBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    CloneOutputUnderGradMixin,
)


class _Llama4SharedExpertBridge(CloneOutputUnderGradMixin, GatedMLPBridge):
    """Llama4TextMoe accumulates routed output into the shared-expert result
    with an in-place ``add_``; clone under grad (see mixin)."""


class _Llama4MoEBridge(MoEBridge):
    """MoEBridge that fires hook_out in [batch, seq, d_model].

    Llama4TextMoe flattens to [batch * seq, d_model] internally and the
    decoder layer views the result back, so hooks are fired on an
    input-shaped view and the HF-native flat shape is returned.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        if len(args) > 0:
            hidden = self.hook_in(args[0])
            args = (hidden,) + args[1:]
        else:
            hidden = self.hook_in(kwargs["hidden_states"])
            kwargs = {**kwargs, "hidden_states": hidden}
        output = self.original_component(*args, **kwargs)
        if isinstance(output, tuple):
            flat = output[0]
            if len(output) > 1:
                self.hook_router_scores(output[1])
            hooked = self.hook_out(flat.view(hidden.shape))
            return (hooked.view(flat.shape),) + output[1:]
        assert isinstance(output, torch.Tensor)
        return self.hook_out(output.view(hidden.shape)).view(output.shape)


class Llama4ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Llama4ForCausalLM models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Llama 4 architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()
        self.cfg.attn_implementation = "eager"

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    # Complex-tensor RoPE, NoPE temperature tuning, L2 QK-norm,
                    # and chunked masks live in HF's forward; keep it native.
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                        maintain_native_attention=True,
                        requires_attention_mask=True,
                    ),
                    # The router returns a (scores, logits) tuple, so it stays
                    # unwrapped; MoEBridge.hook_router_scores captures logits.
                    # Non-MoE layers hold a dense gated MLP under the same name.
                    "mlp": _Llama4MoEBridge(
                        name="feed_forward",
                        config=self.cfg,
                        submodules={
                            "shared_expert": _Llama4SharedExpertBridge(
                                name="shared_expert",
                                config=self.cfg,
                                optional=True,
                                submodules={
                                    "gate": LinearBridge(name="gate_proj"),
                                    "in": LinearBridge(name="up_proj"),
                                    "out": LinearBridge(name="down_proj"),
                                },
                            ),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }
