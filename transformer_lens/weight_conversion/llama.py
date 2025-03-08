import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils import ArchitectureConversion
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)


class LLAMAWeightConversion(ArchitectureConversion):
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        n_kv_heads = cfg.n_key_value_heads if cfg.n_key_value_heads is not None else cfg.n_heads
        using_gqa = cfg.n_key_value_heads is not None
        gqa_uscore = "_" if using_gqa else ""
        
        super().__init__(
            {
                "embed.W_E": "model.embed_tokens.weight",
                "ln_final.w": "model.norm.weight",
                "unembed.W_U": "lm_head.weight.T",
                "unembed.b_U": torch.zeros(cfg.d_vocab, dtype=cfg.dtype, device=cfg.device),
                "blocks": ("model.layers", WeightConversionSet({
                    "ln1.w": "input_layernorm.weight",
                    "attn.W_Q": ("self_attn.q_proj.weight", RearrangeWeightConversion("(n h) m->n m h", n=cfg.n_heads)),
                    "attn.{gqa_uscore}W_K": ("self_attn.k_proj.weight", RearrangeWeightConversion("(n h) m->n m h", n=n_kv_heads)),
                    "attn.{gqa_uscore}W_V": ("self_attn.v_proj.weight", RearrangeWeightConversion("(n h) m->n m h", n=n_kv_heads)),
                    "attn.b_Q": torch.zeros(
                        n_kv_heads,
                        cfg.d_head,
                        dtype=cfg.dtype,
                        device=cfg.device,
                    ),
                    "attn.{gqa_uscore}b_K": torch.zeros(
                        n_kv_heads,
                        cfg.d_head,
                        dtype=cfg.dtype,
                        device=cfg.device,
                    ),
                    "attn.{gqa_uscore}b_V": torch.zeros(
                        n_kv_heads,
                        cfg.d_head,
                        dtype=cfg.dtype,
                        device=cfg.device,
                    ),
                    "attn.W_O": ("self_attn.o_proj.weight", RearrangeWeightConversion("m (n h)->n h m", n=cfg.n_heads)),
                    "attn.b_O": torch.zeros(
                        cfg.d_model, dtype=cfg.dtype, device=cfg.device
                    ),
                    "ln2.w": "post_attention_layernorm.weight",
                    "mlp.W_in": "mlp.up_proj.weight.T",
                    "mlp.W_gate": "mlp.gate_proj.weight.T",
                    "mlp.W_out": "mlp.down_proj.weight.T",
                    "mlp.b_in": torch.zeros(
                        cfg.d_mlp, dtype=cfg.dtype, device=cfg.device
                    ),
                    "mlp.b_out": torch.zeros(
                        cfg.d_model, dtype=cfg.dtype, device=cfg.device
                    ),
                }))
            }
        )
        self.enable_quantiziation(cfg, {
            "blocks": ("model.layers", WeightConversionSet({
                "attn.W_Q": "self_attn.q_proj.weight",
                "attn.{gqa_uscore}W_K": "self_attn.k_proj.weight",
                "attn.{gqa_uscore}W_V": "self_attn.v_proj.weight",
                "attn.W_O": "self_attn.o_proj.weight",
                "mlp.W_in": "mlp.up_proj.weight",
                "mlp.W_gate": "mlp.gate_proj.weight",
                "mlp.W_out": "mlp.down_proj.weight",
            })),
        })
