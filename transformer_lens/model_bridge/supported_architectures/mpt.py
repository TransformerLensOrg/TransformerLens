"""MPT (MPTForCausalLM) adapter — ALiBi, fused Wqkv, weight-only LayerNorm, no biases."""

from typing import Any

import torch
import torch.nn as nn

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.mpt_alibi_attention import (
    MPTALiBiAttentionBridge,
)


class MPTArchitectureAdapter(ArchitectureAdapter):
    """MPT adapter: ALiBi bias; all layers bias-free (no b_Q/b_K/b_V/b_O/b_in/b_out/ln bias)."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "alibi"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        self.cfg.default_prepend_bos = False

        # Pure MHA: split_qkv yields [d_model, d_model] per head; standard rearrangements apply.
        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            "blocks": BlockBridge(
                name="transformer.blocks",
                submodules={
                    "ln1": NormalizationBridge(name="norm_1", config=self.cfg),
                    "attn": MPTALiBiAttentionBridge(
                        name="attn",
                        config=self.cfg,
                        split_qkv_matrix=self._split_mpt_qkv,
                        submodules={
                            "qkv": LinearBridge(name="Wqkv"),
                            "o": LinearBridge(name="out_proj"),
                        },
                    ),
                    "ln2": NormalizationBridge(name="norm_2", config=self.cfg),
                    "mlp": MLPBridge(
                        name="ffn",
                        submodules={
                            "in": LinearBridge(name="up_proj"),
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="transformer.norm_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def _split_mpt_qkv(self, attn_component: Any) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        """Split fused Wqkv into Q, K, V — row-wise chunk (NOT interleaved like BLOOM)."""
        w = attn_component.Wqkv.weight.detach().clone()
        w_q, w_k, w_v = torch.chunk(w, 3, dim=0)
        d_model = self.cfg.d_model

        def make_linear(weight: torch.Tensor) -> nn.Linear:
            lin = nn.Linear(d_model, d_model, bias=False, device=weight.device, dtype=weight.dtype)
            lin.weight = nn.Parameter(weight.contiguous())
            return lin

        return make_linear(w_q), make_linear(w_k), make_linear(w_v)
