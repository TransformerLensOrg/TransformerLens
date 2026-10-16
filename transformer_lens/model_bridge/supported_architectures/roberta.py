"""RoBERTa architecture adapter."""

from typing import Any

from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.bert import (
    BertArchitectureAdapter,
)


class RobertaArchitectureAdapter(BertArchitectureAdapter):
    """Architecture adapter for ``RobertaForMaskedLM``.

    BERT's encoder layer under ``roberta.*`` with the MLM head under ``lm_head.*``;
    flags, weight conversions and ``prepare_model`` (a no-op here: no pooler, no
    ``cls.*`` heads) are inherited. Position ids start at ``padding_idx + 1 = 2``,
    so ``pos_embed.hook_out`` is offset by two rows from BERT's;
    ``TransformerBridge.W_pos`` slices those rows off.
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the RoBERTa architecture adapter."""
        super().__init__(cfg)

        self.component_mapping = {
            "embed": EmbeddingBridge(name="roberta.embeddings.word_embeddings"),
            # Real [1, d_model] table at type_vocab_size = 1; HF adds it on every forward.
            "token_type_embed": EmbeddingBridge(name="roberta.embeddings.token_type_embeddings"),
            "pos_embed": PosEmbedBridge(name="roberta.embeddings.position_embeddings"),
            "embed_ln": NormalizationBridge(
                name="roberta.embeddings.LayerNorm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
            "blocks": BlockBridge(
                name="roberta.encoder.layer",
                # Same split intermediate.dense / output.dense layout as BertLayer, see bert.py.
                hook_alias_overrides={
                    "hook_mlp_out": "mlp.out.hook_out",
                    "hook_mlp_in": "mlp.in.hook_in",
                },
                submodules={
                    "ln1": NormalizationBridge(
                        name="attention.output.LayerNorm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "ln2": NormalizationBridge(
                        name="output.LayerNorm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "attn": AttentionBridge(
                        name="attention",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="self.query"),
                            "k": LinearBridge(name="self.key"),
                            "v": LinearBridge(name="self.value"),
                            "o": LinearBridge(name="output.dense"),
                        },
                    ),
                    "mlp": MLPBridge(
                        name=None,
                        config=self.cfg,
                        submodules={
                            "in": LinearBridge(name="intermediate.dense"),
                            "out": LinearBridge(name="output.dense"),
                        },
                    ),
                },
            ),
            "mlm_head": LinearBridge(name="lm_head.dense"),
            "unembed": UnembeddingBridge(name="lm_head.decoder"),
            "ln_final": NormalizationBridge(
                name="lm_head.layer_norm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
        }

        # max_position_embeddings counts the two reserved rows; a table of n_ctx + 2
        # rows is what makes TransformerBridge.W_pos drop them.
        self.cfg.n_ctx -= 2
