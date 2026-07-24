"""BART adapter and the shared BART-family encoder-decoder base (BART, Marian,
MBart, Pegasus, Blenderbot, M2M100/NLLB); per-member differences are declarative."""

from typing import Any, Dict

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    NormalizationBridge,
    PosEmbedBridge,
    SymbolicBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class BartFamilyArchitectureAdapter(ArchitectureAdapter):
    """Shared base for the BART-family encoder-decoder adapters."""

    # Blenderbot ships asymmetric stacks (e.g. 2 encoder / 24 decoder layers) and
    # follows the decoder; every other member requires symmetric stacks.
    require_symmetric_layers: bool = True
    n_layers_from: str = "encoder"
    # BART checkpoints don't scale embeddings; the rest default scale_embedding on.
    force_scale_embedding: bool = True
    # layernorm_embedding after the token+position embeds (BART, MBart).
    has_layernorm_embedding: bool = False
    # Trailing per-stack layer_norm — the pre-LN members (MBart, Pegasus,
    # Blenderbot, M2M100).
    has_final_stack_norm: bool = False

    def __init__(self, cfg: Any) -> None:
        """Validate the config, set family flags, and build the mapping."""
        super().__init__(cfg)

        name = type(self).__name__
        encoder_layers = getattr(self.cfg, "encoder_layers", self.cfg.n_layers)
        decoder_layers = getattr(self.cfg, "decoder_layers", self.cfg.n_layers)
        if self.require_symmetric_layers and encoder_layers != decoder_layers:
            raise ValueError(
                f"{name} only supports symmetric configs for now: "
                f"encoder_layers={encoder_layers}, decoder_layers={decoder_layers}."
            )

        encoder_heads = getattr(self.cfg, "encoder_attention_heads", self.cfg.n_heads)
        decoder_heads = getattr(self.cfg, "decoder_attention_heads", self.cfg.n_heads)
        if encoder_heads != decoder_heads:
            raise ValueError(
                f"{name} only supports symmetric attention heads for now: "
                f"encoder_attention_heads={encoder_heads}, decoder_attention_heads={decoder_heads}."
            )

        encoder_ffn_dim = getattr(self.cfg, "encoder_ffn_dim", self.cfg.d_mlp)
        decoder_ffn_dim = getattr(self.cfg, "decoder_ffn_dim", self.cfg.d_mlp)
        if encoder_ffn_dim != decoder_ffn_dim:
            raise ValueError(
                f"{name} only supports symmetric FFN dims for now: "
                f"encoder_ffn_dim={encoder_ffn_dim}, decoder_ffn_dim={decoder_ffn_dim}."
            )

        self.cfg.n_layers = decoder_layers if self.n_layers_from == "decoder" else encoder_layers
        self.cfg.n_heads = encoder_heads
        self.cfg.d_head = self.cfg.d_model // encoder_heads
        self.cfg.d_mlp = encoder_ffn_dim
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "standard"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        if self.force_scale_embedding and self.cfg.scale_embedding is None:
            self.cfg.scale_embedding = True

        # Post-LN members break fold-LN's pre-LN assumption; pre-LN members keep
        # the family-wide conservative default (per-stack final norms + embed
        # scaling sit outside the folding machinery).
        self.supports_fold_ln = False
        self.supports_center_writing_weights = False
        self.weight_processing_conversions = {}

        self.component_mapping = self._build_component_mapping()

    def _norm(self, name: str) -> NormalizationBridge:
        return NormalizationBridge(name=name, config=self.cfg, use_native_layernorm_autograd=True)

    def _attention(self, name: str, *, is_cross_attention: bool = False) -> AttentionBridge:
        return AttentionBridge(
            name=name,
            config=self.cfg,
            submodules={
                "q": LinearBridge(name="q_proj"),
                "k": LinearBridge(name="k_proj"),
                "v": LinearBridge(name="v_proj"),
                "o": LinearBridge(name="out_proj"),
            },
            is_cross_attention=is_cross_attention,
        )

    def _encoder_attention(self) -> AttentionBridge:
        """Encoder self-attention seam; LED swaps in its Longformer variant."""
        return self._attention("self_attn")

    def _mlp(self) -> SymbolicBridge:
        return SymbolicBridge(
            submodules={
                "in": LinearBridge(name="fc1"),
                "out": LinearBridge(name="fc2"),
            },
        )

    def _encoder_block(self) -> BlockBridge:
        return BlockBridge(
            name="model.encoder.layers",
            hook_alias_overrides={
                "hook_mlp_in": "mlp.in.hook_in",
                "hook_mlp_out": "mlp.out.hook_out",
            },
            submodules={
                "attn": self._encoder_attention(),
                "ln1": self._norm("self_attn_layer_norm"),
                "ln2": self._norm("final_layer_norm"),
                "mlp": self._mlp(),
            },
        )

    def _decoder_block(self) -> BlockBridge:
        return BlockBridge(
            name="model.decoder.layers",
            hook_alias_overrides={
                "hook_attn_in": "self_attn.hook_attn_in",
                "hook_attn_out": "self_attn.hook_out",
                "hook_q_input": "self_attn.hook_q_input",
                "hook_k_input": "self_attn.hook_k_input",
                "hook_v_input": "self_attn.hook_v_input",
                "hook_mlp_in": "mlp.in.hook_in",
                "hook_mlp_out": "mlp.out.hook_out",
            },
            submodules={
                "self_attn": self._attention("self_attn"),
                "ln1": self._norm("self_attn_layer_norm"),
                "cross_attn": self._attention("encoder_attn", is_cross_attention=True),
                "ln2": self._norm("encoder_attn_layer_norm"),
                "ln3": self._norm("final_layer_norm"),
                "mlp": self._mlp(),
            },
        )

    def _build_component_mapping(self) -> Dict[str, GeneralizedComponent]:
        mapping: Dict[str, GeneralizedComponent] = {
            "embed": EmbeddingBridge(name="model.encoder.embed_tokens"),
            "pos_embed": PosEmbedBridge(name="model.encoder.embed_positions"),
        }
        if self.has_layernorm_embedding:
            mapping["embed_ln"] = self._norm("model.encoder.layernorm_embedding")
        mapping["encoder_blocks"] = self._encoder_block()
        if self.has_final_stack_norm:
            mapping["encoder_ln_final"] = self._norm("model.encoder.layer_norm")
        mapping["decoder_embed"] = EmbeddingBridge(name="model.decoder.embed_tokens")
        mapping["decoder_pos_embed"] = PosEmbedBridge(name="model.decoder.embed_positions")
        if self.has_layernorm_embedding:
            mapping["decoder_embed_ln"] = self._norm("model.decoder.layernorm_embedding")
        mapping["decoder_blocks"] = self._decoder_block()
        if self.has_final_stack_norm:
            mapping["decoder_ln_final"] = self._norm("model.decoder.layer_norm")
        mapping["unembed"] = UnembeddingBridge(name="lm_head")
        return mapping

    def setup_hook_compatibility(self, bridge: Any) -> None:
        """Fold the trained final_logits_bias into the unembed bias.

        HF adds the buffer after lm_head, so b_U would read fabricated zeros
        and unembed.hook_out would fire pre-bias (Marian opus-mt trains it).
        Moving it into the bias UnembeddingBridge injects is numerically
        identity; zeroing the buffer keeps the (re-run) fold idempotent.
        """
        import torch

        model = getattr(bridge, "original_model", None)
        buf = getattr(model, "final_logits_bias", None)
        lm_head = getattr(model, "lm_head", None)
        if buf is None or lm_head is None or getattr(lm_head, "bias", None) is None:
            return
        with torch.no_grad():
            lm_head.bias.add_(buf.reshape(-1).to(lm_head.bias.dtype))
            buf.zero_()


class BartArchitectureAdapter(BartFamilyArchitectureAdapter):
    """Architecture adapter for BartForConditionalGeneration models.

    Post-LN with layernorm_embedding; checkpoints ship scale_embedding=False,
    so the family default-on is disabled.
    """

    force_scale_embedding = False
    has_layernorm_embedding = True
