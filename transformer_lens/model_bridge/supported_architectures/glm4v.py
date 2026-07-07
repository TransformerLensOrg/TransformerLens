"""GLM-4V / GLM-4.1V architecture adapter.

Z.ai's GLM-4V line (``Glm4vForConditionalGeneration``, GLM-4.1V-9B
Thinking): a GLM vision tower at ``model.visual`` (RMS-normed blocks,
learned position embeddings + 2D rotary, patch merger + conv
downsample) feeding a GLM-4-0414-layout text decoder at
``model.language_model`` — sandwich norms and the joint ``gate_up_proj``
MLP. Text attention uses mRoPE (three position streams), so it stays
HF-native; the tower is delegated opaquely with the merger as the
projector.
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
    EmbeddingBridge,
    JointGateUpMLPBridge,
    LinearBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
    VisionProjectionBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.phi3 import (
    Phi3ArchitectureAdapter,
)


class Glm4vArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Glm4vForConditionalGeneration models."""

    required_libraries: list[str] = ["torchvision"]
    required_libraries_group: str = "multimodal"

    def __init__(self, cfg: Any) -> None:
        """Initialize the GLM-4V architecture adapter."""
        super().__init__(cfg)

        self.cfg.is_multimodal = True
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True
        self.cfg.attn_implementation = "eager"
        # GLM tokenizers carry no BOS token.
        self.cfg.default_prepend_bos = False

        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

        vision_cfg = getattr(cfg, "vision_config", None)
        if vision_cfg is not None:
            self.cfg.vision_hidden_size = getattr(vision_cfg, "hidden_size", None)
            self.cfg.vision_num_layers = getattr(vision_cfg, "depth", None)
            self.cfg.vision_num_heads = getattr(vision_cfg, "num_heads", None)

        # Joint gate_up_proj cannot be folded by the standard LN machinery.
        self.supports_fold_ln = False
        n_kv_heads = getattr(self.cfg, "n_key_value_heads", None) or self.cfg.n_heads
        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
            # GLM attention carries QKV biases; K/V reshape by kv-head count.
            "blocks.{i}.attn.q.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(h d_head) -> h d_head", h=self.cfg.n_heads
                ),
            ),
            "blocks.{i}.attn.k.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(h d_head) -> h d_head", h=n_kv_heads),
            ),
            "blocks.{i}.attn.v.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(h d_head) -> h d_head", h=n_kv_heads),
            ),
        }

        self.component_mapping = {
            "vision_encoder": GeneralizedComponent(name="model.visual"),
            "vision_projector": VisionProjectionBridge(name="model.visual.merger"),
            "embed": EmbeddingBridge(name="model.language_model.embed_tokens"),
            "blocks": BlockBridge(
                name="model.language_model.layers",
                submodules={
                    # GLM-4-0414 sandwich layout: post_attention_layernorm is
                    # the pre-MLP norm; the sandwich norms sit on sublayer
                    # outputs before their residual adds.
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln1_post": RMSNormalizationBridge(
                        name="post_self_attn_layernorm", config=self.cfg
                    ),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "ln2_post": RMSNormalizationBridge(name="post_mlp_layernorm", config=self.cfg),
                    # mRoPE (3-section multimodal rotary) lives in HF's forward.
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
                    "mlp": JointGateUpMLPBridge(
                        name="mlp",
                        config=self.cfg,
                        split_gate_up_matrix=Phi3ArchitectureAdapter._split_gate_up,
                        submodules={
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.language_model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Force eager attention for hookable vision/text attention."""
        config = model_kwargs.get("config")
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"

    def prepare_model(self, hf_model: Any) -> None:
        """Force eager attention on the loaded HF model."""
        if hasattr(hf_model, "config"):
            hf_model.config._attn_implementation = "eager"
