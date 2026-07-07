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


class _Llama4SharedExpertBridge(GatedMLPBridge):
    """GatedMLPBridge whose output survives HF's in-place ``out.add_``.

    Llama4TextMoe accumulates routed-expert output into the shared-expert
    result in place, which autograd forbids on backward-hook views; clone
    under grad so the hooked tensor is never the in-place target.
    """

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = super().forward(*args, **kwargs)
        if torch.is_grad_enabled():
            out = out.clone()
        return out


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

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True
        self.cfg.attn_implementation = "eager"

        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

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

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Force eager attention so chunked-mask handling stays hookable."""
        config = model_kwargs.get("config")
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"

    def prepare_model(self, hf_model: Any) -> None:
        """Force eager attention on the loaded HF model."""
        if hasattr(hf_model, "config"):
            hf_model.config._attn_implementation = "eager"
