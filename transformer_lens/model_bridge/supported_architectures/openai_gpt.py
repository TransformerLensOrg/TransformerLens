"""OpenAI GPT (GPT-1) architecture adapter.

The original GPT (``OpenAIGPTLMHeadModel``, openai-community/openai-gpt).
GPT-2's block internals under different top-level names — combined-QKV
``c_attn`` Conv1D, ``c_fc``/``c_proj`` MLP — but POST-norm (``ln_1(x + attn)``,
``ln_2(n + mlp)``), no final LayerNorm, and embeddings at
``transformer.tokens_embed`` / ``transformer.positions_embed``.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    JointQKVAttentionBridge,
    LinearBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)


class _OpenAIGPTJointQKVAttentionBridge(JointQKVAttentionBridge):
    """GPT-1's Block concatenates ``[h] + attn_outputs[1:]`` — it requires the
    attention module to return a list, not the tuple modern blocks accept."""

    def _process_output(self, output: Any) -> Any:
        processed = super()._process_output(output)
        if isinstance(processed, tuple):
            return list(processed)
        return [processed]


class OpenAIGPTArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for OpenAIGPTLMHeadModel (GPT-1) models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the OpenAI GPT architecture adapter."""
        super().__init__(cfg)

        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "standard"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False

        # GPT-1 is post-LN; fold-LN assumes pre-LN and would fold norms into
        # the wrong sublayers.
        self.supports_fold_ln = False
        self.supports_center_writing_weights = False
        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.tokens_embed"),
            "pos_embed": PosEmbedBridge(name="transformer.positions_embed"),
            "blocks": BlockBridge(
                name="transformer.h",
                config=self.cfg,
                submodules={
                    "attn": _OpenAIGPTJointQKVAttentionBridge(
                        name="attn",
                        config=self.cfg,
                        submodules={
                            "qkv": LinearBridge(name="c_attn"),
                            "o": LinearBridge(name="c_proj"),
                        },
                    ),
                    # Post-norm: ln_1 follows attention, ln_2 follows the MLP.
                    "ln1": NormalizationBridge(name="ln_1", config=self.cfg),
                    "mlp": self._ungated_mlp(up="c_fc", down="c_proj"),
                    "ln2": NormalizationBridge(name="ln_2", config=self.cfg),
                },
                # Post-norm: ln2.hook_in is the post-MLP sum n+m; the true
                # attn->MLP mid-point n = ln_1(x+a) is mlp.hook_in.
                hook_alias_overrides={
                    "hook_resid_mid": "mlp.hook_in",
                },
            ),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
