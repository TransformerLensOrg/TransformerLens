"""Pegasus architecture adapter.

Google's PEGASUS summarization family (``PegasusForConditionalGeneration``):
pre-norm encoder-decoder with per-stack final LayerNorms (M2M100-style
layout), Marian-style deterministic sinusoidal position embeddings, and the
sqrt(d_model) embedding scale applied in the stack forward — so, as with
Marian and unlike M2M100/MBart, hooks on ``embed`` observe unscaled output.
"""

from typing import Any

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


class PegasusArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for PegasusForConditionalGeneration models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Pegasus architecture adapter."""
        super().__init__(cfg)

        encoder_layers = getattr(self.cfg, "encoder_layers", self.cfg.n_layers)
        decoder_layers = getattr(self.cfg, "decoder_layers", self.cfg.n_layers)
        if encoder_layers != decoder_layers:
            raise ValueError(
                "PegasusArchitectureAdapter only supports symmetric configs for now: "
                f"encoder_layers={encoder_layers}, decoder_layers={decoder_layers}."
            )

        encoder_heads = getattr(self.cfg, "encoder_attention_heads", self.cfg.n_heads)
        decoder_heads = getattr(self.cfg, "decoder_attention_heads", self.cfg.n_heads)
        if encoder_heads != decoder_heads:
            raise ValueError(
                "PegasusArchitectureAdapter only supports symmetric attention heads for now: "
                f"encoder_attention_heads={encoder_heads}, decoder_attention_heads={decoder_heads}."
            )

        encoder_ffn_dim = getattr(self.cfg, "encoder_ffn_dim", self.cfg.d_mlp)
        decoder_ffn_dim = getattr(self.cfg, "decoder_ffn_dim", self.cfg.d_mlp)
        if encoder_ffn_dim != decoder_ffn_dim:
            raise ValueError(
                "PegasusArchitectureAdapter only supports symmetric FFN dims for now: "
                f"encoder_ffn_dim={encoder_ffn_dim}, decoder_ffn_dim={decoder_ffn_dim}."
            )

        self.cfg.n_layers = encoder_layers
        self.cfg.n_heads = encoder_heads
        self.cfg.d_head = self.cfg.d_model // encoder_heads
        self.cfg.d_mlp = encoder_ffn_dim
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "standard"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        # Applied in the stack forward; embed hooks observe unscaled output.
        if self.cfg.scale_embedding is None:
            self.cfg.scale_embedding = True

        # Encoder-decoder LN folding is unsupported; keep raw weights.
        self.supports_fold_ln = False
        self.supports_center_writing_weights = False
        self.weight_processing_conversions = {}

        def _attention(name: str, is_cross: bool = False) -> AttentionBridge:
            return AttentionBridge(
                name=name,
                config=self.cfg,
                submodules={
                    "q": LinearBridge(name="q_proj"),
                    "k": LinearBridge(name="k_proj"),
                    "v": LinearBridge(name="v_proj"),
                    "o": LinearBridge(name="out_proj"),
                },
                is_cross_attention=is_cross,
            )

        def _norm(name: str) -> NormalizationBridge:
            return NormalizationBridge(
                name=name, config=self.cfg, use_native_layernorm_autograd=True
            )

        def _mlp() -> SymbolicBridge:
            return SymbolicBridge(
                submodules={
                    "in": LinearBridge(name="fc1"),
                    "out": LinearBridge(name="fc2"),
                },
            )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.encoder.embed_tokens"),
            "pos_embed": PosEmbedBridge(name="model.encoder.embed_positions"),
            "encoder_blocks": BlockBridge(
                name="model.encoder.layers",
                hook_alias_overrides={
                    "hook_mlp_in": "mlp.in.hook_in",
                    "hook_mlp_out": "mlp.out.hook_out",
                },
                submodules={
                    "ln1": _norm("self_attn_layer_norm"),
                    "attn": _attention("self_attn"),
                    "ln2": _norm("final_layer_norm"),
                    "mlp": _mlp(),
                },
            ),
            "encoder_ln_final": _norm("model.encoder.layer_norm"),
            "decoder_embed": EmbeddingBridge(name="model.decoder.embed_tokens"),
            "decoder_pos_embed": PosEmbedBridge(name="model.decoder.embed_positions"),
            "decoder_blocks": BlockBridge(
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
                    "ln1": _norm("self_attn_layer_norm"),
                    "self_attn": _attention("self_attn"),
                    "ln2": _norm("encoder_attn_layer_norm"),
                    "cross_attn": _attention("encoder_attn", is_cross=True),
                    "ln3": _norm("final_layer_norm"),
                    "mlp": _mlp(),
                },
            ),
            "decoder_ln_final": _norm("model.decoder.layer_norm"),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
