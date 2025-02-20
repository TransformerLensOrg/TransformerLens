from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils import ArchitectureConversion
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)

class BertWeightConversion(ArchitectureConversion):
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        super().__init__({
            "embed.embed.W_E": "bert.embeddings.word_embeddings.weight",
            "embed.pos_embed.W_pos": "bert.embeddings.position_embeddings.weight",
            "embed.token_type_embed.W_token_type": "bert.embeddings.token_type_embeddings.weight",
            "embed.ln.w": "bert.embeddings.LayerNorm.weight",
            "embed.ln.b": "bert.embeddings.LayerNorm.bias",
            "unembed.W_U": "cls.predictions.decoder.weight.T",
            "unembed.b_U": "cls.predictions.decoder.bias",
            "mlm_head.W": "bert.cls.predictions.transform.dense.weight.T",
            "mlm_head.b": "bert.cls.predictions.transform.dense.bias",
            "mlm_head.ln.w": "bert.cls.predictions.transform.LayerNorm.weight",
            "mlm_head.ln.b": "bert.cls.predictions.transform.LayerNorm.bias",
            "mlm_head.W_U": "bert.embeddings.word_embeddings.weight.T",
            "mlm_head.b_U": "bert.cls.predictions.bias",
            "pooler.W": "bert.pooler.dense.weight.T",
            "pooler.b": "bert.pooler.dense.bias",
            "nsp_head.W": "cls.seq_relationship.weight.T",
            "nsp_head.b": "cls.seq_relationship.bias",
            "blocks": ("bert.encoder.layer", WeightConversionSet({
                "attn.W_Q": ("attention.self.query.weight", RearrangeWeightConversion("(i h) m -> i m h", i=cfg.n_heads)),
                "attn.b_Q": ("attention.self.query.bias", RearrangeWeightConversion("(i h) -> i h", i=cfg.n_heads)),
                "attn.W_K": ("attention.self.key.weight", RearrangeWeightConversion("(i h) m -> i m h", i=cfg.n_heads)),
                "attn.b_K": ("attention.self.key.bias", RearrangeWeightConversion("(i h) -> i h", i=cfg.n_heads)),
                "attn.W_V": ("attention.self.value.weight", RearrangeWeightConversion("(i h) m -> i m h", i=cfg.n_heads)),
                "attn.b_V": ("attention.self.value.bias", RearrangeWeightConversion("(i h) -> i h", i=cfg.n_heads)),
                "attn.W_O": ("attention.self.dense.weight", RearrangeWeightConversion("m (i h) -> i h m", i=cfg.n_heads)),
                "attn.b_O": "attention.output.dense.bias",
                "ln1.w": "attention.output.LayerNorm.weight",
                "ln1.b": "attention.output.LayerNorm.bias",
                "mlp.W_in": ("intermediate.dense.weight", RearrangeWeightConversion("mlp model -> model mlp")),
                "mlp.b_in": "intermediate.dense.bias",
                "mlp.W_out": ("output.dense.weight", RearrangeWeightConversion("model mlp -> mlp model")),
                "mlp.b_out": "output.dense.bias",
                "ln2.w": "output.LayerNorm.weight",
                "ln2.b": "output.LayerNorm.bias",
            }))
        })
