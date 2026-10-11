"""RoBERTa architecture adapter.

RoBERTa reuses BERT's encoder layer and only renames what surrounds it: the
trunk lives under ``roberta.*`` instead of ``bert.*`` and the masked-LM head
under ``lm_head.*`` instead of ``cls.predictions.*``.
"""

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
    """Architecture adapter for ``RobertaForMaskedLM`` checkpoints.

    Everything BERT sets up carries over unchanged: the post-LN config flags
    with ``supports_fold_ln = False``, the q/k/v/o rearrange conversions and
    ``supports_generation = False``. Only the HF module paths and the context
    length differ. Tokenizer policy is not set here: the boot path reads the
    ``<s>`` / ``</s>`` behaviour from each checkpoint's tokenizer.

    HF builds RoBERTa position ids inside ``RobertaEmbeddings`` starting at
    ``padding_idx + 1 = 2`` (padding keeps id 1), so ``pos_embed.hook_out``
    holds rows ``2 .. seq_len + 1`` of the table and is not row-aligned with
    BERT's. ``W_pos`` drops the two reserved rows and is aligned.

    ``BertArchitectureAdapter.prepare_model`` only reacts to ``bert.pooler``
    and ``cls.*`` task heads, which ``RobertaForMaskedLM`` lacks, so the
    inherited version is a no-op here.
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the RoBERTa architecture adapter.

        Args:
            cfg: The configuration object. ``n_ctx`` is expected to arrive as
                HF's ``max_position_embeddings`` (the full table size).
        """
        super().__init__(cfg)

        # Components are keyed by HF module path, so BERT's mapping cannot be
        # renamed in place: rebuild it with RoBERTa's paths.
        self.component_mapping = {
            "embed": EmbeddingBridge(name="roberta.embeddings.word_embeddings"),
            # A real [type_vocab_size, d_model] table that HF adds on every forward,
            # even though type_vocab_size is 1 and the row is a constant.
            "token_type_embed": EmbeddingBridge(name="roberta.embeddings.token_type_embeddings"),
            "pos_embed": PosEmbedBridge(name="roberta.embeddings.position_embeddings"),
            "embed_ln": NormalizationBridge(
                name="roberta.embeddings.LayerNorm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
            "blocks": BlockBridge(
                name="roberta.encoder.layer",
                # RobertaLayer keeps BERT's layout: intermediate.dense and output.dense
                # are siblings with no MLP module, so the MLP aliases point at the two
                # Linear hooks exactly as in bert.py.
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

        # HF's max_position_embeddings (514 for roberta-base) counts the two
        # reserved rows, so the usable context is two shorter. Subtracting rather
        # than pinning 512 keeps longer-table checkpoints right too, and W_pos
        # only slices the reserved rows off when the table has n_ctx + 2 of them.
        self.cfg.n_ctx -= 2
