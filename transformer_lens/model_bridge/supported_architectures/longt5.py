"""LongT5 architecture adapter.

Google's LongT5 (``LongT5ForConditionalGeneration``): a T5 stack whose
encoder self-attention is replaced by local windowed attention or
transient-global attention (``encoder_attention_type``). The decoder is
identical to T5. Encoder attention stays delegated to HF — its block-wise
position bias has a [1, 1, heads, block, 3*block] shape the generic
reconstruction cannot supply.
"""

from typing import Any, Union

from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    GatedMLPBridge,
    LinearBridge,
    MLPBridge,
    PosEmbedBridge,
    RMSNormalizationBridge,
    T5BlockBridge,
)
from transformer_lens.model_bridge.supported_architectures.t5 import (
    T5ArchitectureAdapter,
)

_ENCODER_ATTN_ATTR = {
    "local": "LocalSelfAttention",
    "transient-global": "TransientGlobalSelfAttention",
}


class LongT5ArchitectureAdapter(T5ArchitectureAdapter):
    """Architecture adapter for LongT5ForConditionalGeneration models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the LongT5 architecture adapter."""
        super().__init__(cfg)
        assert self.component_mapping is not None

        attn_attr = _ENCODER_ATTN_ATTR[getattr(cfg, "encoder_attention_type", "local")]

        encoder_mlp: Union[GatedMLPBridge, MLPBridge]
        if self.cfg.gated_mlp:
            encoder_mlp = self._gated_mlp(
                name="layer.1.DenseReluDense", gate="wi_0", up="wi_1", down="wo"
            )
        else:
            encoder_mlp = MLPBridge(
                name="layer.1.DenseReluDense",
                submodules={
                    "in": LinearBridge(name="wi"),
                    "out": LinearBridge(name="wo"),
                },
            )

        # Rebuild the encoder stack around the local/tglobal attention module;
        # the decoder mapping inherited from T5 is unchanged.
        self.component_mapping["pos_embed"] = PosEmbedBridge(
            name=f"encoder.block.0.layer.0.{attn_attr}.relative_attention_bias"
        )
        self.component_mapping["encoder_blocks"] = T5BlockBridge(
            name="encoder.block",
            config=self.cfg,
            is_decoder=False,
            submodules={
                "ln1": RMSNormalizationBridge(name="layer.0.layer_norm", config=self.cfg),
                "attn": AttentionBridge(
                    name=f"layer.0.{attn_attr}",
                    config=self.cfg,
                    submodules={
                        "q": LinearBridge(name="q"),
                        "k": LinearBridge(name="k"),
                        "v": LinearBridge(name="v"),
                        "o": LinearBridge(name="o"),
                    },
                    maintain_native_attention=True,
                ),
                "ln2": RMSNormalizationBridge(name="layer.1.layer_norm", config=self.cfg),
                "mlp": encoder_mlp,
            },
        )
