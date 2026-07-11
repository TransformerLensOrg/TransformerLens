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
    GeneralizedComponent,
)

class ASTArchitectureAdapter(ArchitectureAdapter):

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        # essential flag for audio models in V3
        self.cfg.is_audio_model = True

        n_heads = self.cfg.n_heads

        # calculate n_ctx: dynamically grab the length from the generated position embeddings
        # this resolves the n_ctx = num_patches + 2 requirement accurately regardless of config stride
        self.n_ctx = hf_model.audio_spectrogram_transformer.embeddings.position_embeddings.shape[1]

        # V3 Bridge pattern: Map transformerlens canonical names to hf AST attributes
        self.component_mapping = {
            "embed": "audio_spectrogram_transformer.embeddings",
            "blocks": "audio_spectrogram_transformer.layers",
            "ln_final": "audio_spectrogram_transformer.layernorm",
            "unembed": "classifier.dense",

            # block internals mapping
            "block.ln1": "layernorm_before",
            "block.attn.W_Q": "attention.q_proj",
            "block.attn.W_K": "attention.k_proj",
            "block.attn.W_V": "attention.v_proj",
            "block.attn.W_O": "attention.o_proj",
            "block.ln2": "layernorm_after",
            "block.mlp.W_in": "mlp.fc1",
            "block.mlp.W_out": "mlp.fc2",
        }