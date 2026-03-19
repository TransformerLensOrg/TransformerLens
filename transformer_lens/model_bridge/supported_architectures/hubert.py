"""HuBERT architecture adapter.

This module provides the architecture adapter for HuBERT audio models,
including HubertModel (bare encoder) and HubertForCTC (with CTC head).

HuBERT is an encoder-only audio transformer with a CNN feature extractor
front-end. Its transformer encoder blocks are structurally identical to
BERT's, using post-LN with standard multi-head self-attention and
feed-forward layers.
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

    Supports both HubertModel (bare encoder) and HubertForCTC (with CTC head).
    HubertForCTC wraps HubertModel under a 'hubert.' prefix; prepare_model()
    detects this and adjusts component paths accordingly.
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the HuBERT architecture adapter.

        Args:
            cfg: The configuration object.
        """
        super().__init__(cfg)

        self.cfg.is_audio_model = True
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "conv"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False

        # do_stable_layer_norm controls whether HuBERT uses pre-LN or post-LN.
        # - False (default, hubert-base): post-LN (like BERT) — fold_ln unsafe
        # - True (hubert-large/xlarge): pre-LN (like GPT-2) — fold_ln safe
        # This attribute is propagated from HF config in prepare_loading().
        self._do_stable_layer_norm = getattr(self.cfg, "do_stable_layer_norm", False)
        self.supports_fold_ln = self._do_stable_layer_norm

        n_heads = self.cfg.n_heads

        # Q/K/V/O weight rearrangement (same pattern as BERT)
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

        # Default component mapping for bare HubertModel (no prefix).
        # prepare_model() will prepend "hubert." for HubertForCTC.
        self.component_mapping = self._build_component_mapping(prefix="")

    def _build_component_mapping(self, prefix: str) -> dict:
        """Build the component mapping with the given module prefix.

        Args:
            prefix: Module path prefix ("" for HubertModel, "hubert." for HubertForCTC)

        Returns:
            Dictionary of component name to bridge component mappings
        """
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
                # HuBERT MLP has no single container module (intermediate_dense
                # and output_dense are inside feed_forward). Redirect hook aliases
                # to the actual linear layer hooks, same pattern as BERT.
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
        """Propagate HuBERT-specific HF config attributes to the bridge config.

        Several HuBERT config attributes affect model behavior but are not
        standard TransformerBridgeConfig fields. We propagate them here to
        avoid the silent-default bug.
        """
        hf_config = model_kwargs.get("config")
        if hf_config is None:
            return

        # do_stable_layer_norm: controls pre-LN (True) vs post-LN (False)
        do_stable = getattr(hf_config, "do_stable_layer_norm", False)
        self.cfg.do_stable_layer_norm = do_stable  # type: ignore[attr-defined]
        self._do_stable_layer_norm = do_stable
        self.supports_fold_ln = do_stable

        # hidden_act and layer_norm_eps are now mapped globally in
        # map_default_transformer_lens_config() — no per-adapter propagation needed.

        # Rebuild component mapping now that we know the LN variant
        self.component_mapping = self._build_component_mapping(prefix="")

    def prepare_model(self, hf_model: Any) -> None:
        """Adjust component mapping based on the actual HF model variant.

        HubertForCTC nests HubertModel under 'self.hubert', adding a prefix.
        HubertModel has no prefix. Also adds CTC head for HubertForCTC.
        """
        if hasattr(hf_model, "hubert"):
            # HubertForCTC — rebuild mapping with "hubert." prefix and add CTC head
            self.component_mapping = self._build_component_mapping(prefix="hubert.")
            self.component_mapping["unembed"] = UnembeddingBridge(name="lm_head")
