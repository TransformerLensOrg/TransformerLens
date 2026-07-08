"""LED (Longformer Encoder-Decoder) architecture adapter.

AllenAI's LED (``LEDForConditionalGeneration``: led-base/large-16384): a
BART-layout post-LN encoder-decoder whose encoder self-attention is
Longformer's sliding-window + global attention (separate query/key/value
and *_global projections behind an ``output`` projection). The encoder
attention stays delegated to HF; the decoder is plain BART attention. The
whole stack lives under the ``led.`` prefix instead of ``model.``.
"""

from typing import Any

from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    LinearBridge,
    NormalizationBridge,
    SymbolicBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    CloneOutputUnderGradMixin,
)
from transformer_lens.model_bridge.supported_architectures.bart import (
    BartArchitectureAdapter,
)


class _LEDEncoderQueryBridge(CloneOutputUnderGradMixin, LinearBridge):
    """LEDEncoderSelfAttention scales the query projection with an in-place
    ``/=``; clone under grad (see mixin)."""


class LEDArchitectureAdapter(BartArchitectureAdapter):
    """Architecture adapter for LEDForConditionalGeneration models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the LED architecture adapter."""
        super().__init__(cfg)
        assert self.component_mapping is not None

        for component in self.component_mapping.values():
            if component.name and component.name.startswith("model."):
                component.name = "led." + component.name[len("model.") :]

        self.component_mapping["encoder_blocks"] = BlockBridge(
            name="led.encoder.layers",
            hook_alias_overrides={
                "hook_mlp_in": "mlp.in.hook_in",
                "hook_mlp_out": "mlp.out.hook_out",
            },
            submodules={
                # Sliding-window + global attention; window chunking and the
                # global projections have no generic reconstruction.
                "attn": AttentionBridge(
                    name="self_attn",
                    config=self.cfg,
                    submodules={
                        "q": _LEDEncoderQueryBridge(name="longformer_self_attn.query"),
                        "k": LinearBridge(name="longformer_self_attn.key"),
                        "v": LinearBridge(name="longformer_self_attn.value"),
                        "o": LinearBridge(name="output"),
                    },
                    maintain_native_attention=True,
                ),
                "ln1": NormalizationBridge(
                    name="self_attn_layer_norm",
                    config=self.cfg,
                    use_native_layernorm_autograd=True,
                ),
                "ln2": NormalizationBridge(
                    name="final_layer_norm",
                    config=self.cfg,
                    use_native_layernorm_autograd=True,
                ),
                "mlp": SymbolicBridge(
                    submodules={
                        "in": LinearBridge(name="fc1"),
                        "out": LinearBridge(name="fc2"),
                    },
                ),
            },
        )
