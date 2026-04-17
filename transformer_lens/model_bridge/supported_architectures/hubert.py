"""HuBERT architecture adapter.

Supports HubertModel (bare encoder) and HubertForCTC (with CTC head).
Encoder blocks are structurally identical to BERT (post-LN by default,
pre-LN when do_stable_layer_norm=True).
"""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.audio_feature_extractor import (
    AudioFeatureExtractorBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.conv_pos_embed import (
    ConvPosEmbedBridge,
)


class HubertArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for HuBERT audio models.

    HubertForCTC nests HubertModel under a 'hubert.' prefix;
    prepare_model() detects this and adjusts component paths.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.is_audio_model = True
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "conv"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False

        # Pre-LN (True) vs post-LN (False). Propagated from HF config in prepare_loading().
        self._do_stable_layer_norm = getattr(self.cfg, "do_stable_layer_norm", False)
        self.supports_fold_ln = self._do_stable_layer_norm

        n_heads = self.cfg.n_heads

        # Q/K/V/O rearrangement — same pattern as BERT
        self.weight_processing_conversions = {
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(h d_head) d_model -> h d_model d_head", h=n_heads
                ),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(h d_head) d_model -> h d_model d_head", h=n_heads
                ),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(h d_head) d_model -> h d_model d_head", h=n_heads
                ),
            ),
            "blocks.{i}.attn.q.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(h d_head) -> h d_head", h=n_heads),
            ),
            "blocks.{i}.attn.k.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(h d_head) -> h d_head", h=n_heads),
            ),
            "blocks.{i}.attn.v.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(h d_head) -> h d_head", h=n_heads),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (h d_head) -> h d_head d_model", h=n_heads
                ),
            ),
        }

        # Default mapping for bare HubertModel. prepare_model() rebuilds with
        # "hubert." prefix for HubertForCTC.
        self.component_mapping = self._build_component_mapping(prefix="")

    def _build_component_mapping(self, prefix: str) -> dict:
        """Build component mapping. prefix="" for HubertModel, "hubert." for HubertForCTC."""
        p = prefix
        mapping: dict[str, Any] = {
            "audio_feature_extractor": AudioFeatureExtractorBridge(
                name=f"{p}feature_extractor",
            ),
            "feat_proj": GeneralizedComponent(
                name=f"{p}feature_projection",
            ),
            "conv_pos_embed": ConvPosEmbedBridge(
                name=f"{p}encoder.pos_conv_embed",
            ),
            "embed_ln": NormalizationBridge(
                name=f"{p}encoder.layer_norm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
            "blocks": BlockBridge(
                name=f"{p}encoder.layers",
                # Redirect MLP hooks to the actual linear layer hooks (same as BERT)
                hook_alias_overrides={
                    "hook_mlp_out": "mlp.out.hook_out",
                    "hook_mlp_in": "mlp.in.hook_in",
                },
                submodules={
                    "ln1": NormalizationBridge(
                        name="layer_norm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "ln2": NormalizationBridge(
                        name="final_layer_norm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "attn": AttentionBridge(
                        name="attention",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="out_proj"),
                        },
                    ),
                    "mlp": MLPBridge(
                        name="feed_forward",
                        config=self.cfg,
                        submodules={
                            "in": LinearBridge(name="intermediate_dense"),
                            "out": LinearBridge(name="output_dense"),
                        },
                    ),
                },
            ),
        }
        return mapping

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Propagate HuBERT-specific HF config attributes to bridge config.

        Prevents silent-default bugs where adapter reads from bridge config
        but the attribute was never propagated from HF config.
        """
        hf_config = model_kwargs.get("config")
        if hf_config is None:
            return

        # Pre-LN vs post-LN — determines fold_ln safety
        do_stable = getattr(hf_config, "do_stable_layer_norm", False)
        self.cfg.do_stable_layer_norm = do_stable  # type: ignore[attr-defined]
        self._do_stable_layer_norm = do_stable
        self.supports_fold_ln = do_stable

        # hidden_act and layer_norm_eps are mapped globally in
        # map_default_transformer_lens_config()

        # Rebuild with correct LN variant
        self.component_mapping = self._build_component_mapping(prefix="")

    def prepare_model(self, hf_model: Any) -> None:
        """Detect HubertForCTC (has 'hubert.' prefix) and add CTC head."""
        if hasattr(hf_model, "hubert"):
            self.component_mapping = self._build_component_mapping(prefix="hubert.")
            self.component_mapping["unembed"] = UnembeddingBridge(name="lm_head")
