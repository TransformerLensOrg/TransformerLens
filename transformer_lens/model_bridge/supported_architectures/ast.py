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

        # V3 bridge pattern: hierarchical mapping using bridge components
        self.component_mapping = {
            "embed": GeneralizedComponent(name="audio_spectrogram_transformer.embeddings"),
            "ln_final": NormalizationBridge(
                name="audio_spectrogram_transformer.layernorm", config=self.cfg
            ),
            "unembed": UnembeddingBridge(name="classifier.dense"),
            "blocks": BlockBridge(
                name="audio_spectrogram_transformer.layers",
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
        # hook to access the live Huggingface model before boot
        # calculate n_ctx dynamically from the instantiated position embeddings
        self.cfg.n_ctx = (
            hf_model.audio_spectrogram_transformer.embeddings.position_embeddings.shape[1]
        )
