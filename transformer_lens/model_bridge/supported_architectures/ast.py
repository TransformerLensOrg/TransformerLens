"""AST (Audio Spectrogram Transformer) architecture adapter.

Supports ASTForAudioClassification
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
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class ASTArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for AST (Audio Spectrogram Transformer) audio classifiers.

    Input is a [batch, time=1024, n_mels=128] spectrogram; positions 0/1 are the
    CLS and distillation tokens. HF pools (cls+dist)/2 after ln_final and applies
    and extra LayerNorm inside the classifier head, see the unembed mapping note.
    """

    supports_generation: bool = False

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        # essential flag for audio models in V3
        self.cfg.is_audio_model = True
        self.cfg.normalization_type = "LN"

        n_heads = self.cfg.n_heads

        # Q/K/V/O rearrangement: splits hidden dims into (heads, head_dim)
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

        # default mapping for bare ASTModel (prefix="")
        self.component_mapping = self._build_component_mapping(prefix="")

    def _build_component_mapping(self, prefix: str) -> dict:
        """Build component mapping. prefix="" for ASTModel, "audio_spectrogram_transformer." for classification."""
        p = prefix

        return {
            "embed": GeneralizedComponent(name=f"{p}embeddings"),
            "ln_final": NormalizationBridge(name=f"{p}layernorm", config=self.cfg),
            "blocks": BlockBridge(
                name=f"{p}layers",
                submodules={
                    "ln1": NormalizationBridge(name="layernorm_before", config=self.cfg),
                    "ln2": NormalizationBridge(name="layernorm_after", config=self.cfg),
                    "attn": AttentionBridge(
                        name="attention",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                    ),
                    "mlp": MLPBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "in": LinearBridge(name="fc1"),
                            "out": LinearBridge(name="fc2"),
                        },
                    ),
                },
            ),
        }

    def prepare_model(self, hf_model: Any) -> None:
        """Detect classification head, rebind prefixes, guard unembed, and set n_ctx."""
        # 1. handle classification model vs bare encoder
        if hasattr(hf_model, "audio_spectrogram_transformer"):
            self.component_mapping = self._build_component_mapping(
                prefix="audio_spectrogram_transformer."
            )
            base_model = hf_model.audio_spectrogram_transformer

            # guard unembed: only map if num_labls > 0 and dense head is real
            num_labels = getattr(hf_model.config, "num_labels", 0)
            if (
                num_labels > 0
                and hasattr(hf_model, "classifier")
                and hasattr(hf_model.classifier, "dense")
            ):
                self.component_mapping["unembed"] = UnembeddingBridge(name="classifier.dense")
                self.cfg.d_vocab = num_labels
                self.cfg.d_vocab_out = num_labels
        else:
            base_model = hf_model

        # 2. dynamically grab n_ctx from whatever base model we resolved
        if hasattr(base_model, "embeddings") and hasattr(
            base_model.embeddings, "position_embeddings"
        ):
            self.cfg.n_ctx = base_model.embeddings.position_embeddings.shape[1]
