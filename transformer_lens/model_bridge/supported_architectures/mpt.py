"""MPT architecture adapter.

Supports MosaicML MPT models (MPTForCausalLM) with:
- ALiBi positional encoding (no learnable positional embedding module)
- LayerNorm (bias=None — weight-only, set explicitly in MptBlock.__init__)
- Fused QKV projection (Wqkv: [3*d_model, d_model], simple row-wise concat)
- Standard 2-layer MLP (up_proj -> GELU -> down_proj), module named 'ffn'
- No biases anywhere (no_bias=True by default)
- Weight tying: lm_head.weight tied to transformer.wte.weight

Limitations
-----------
- resid_attn_dropout: MptBlock.resid_attn_dropout is an nn.Dropout child module
  with no weights. It is not in the component mapping and is skipped by weight loading.
- logit_scale: A MptConfig field that MptForCausalLM.forward() never reads or applies.
  Dead field; the adapter does not use it.
- attn_type: The adapter assumes pure MHA. MptAttention.__init__ always allocates
  Wqkv as Linear(d_model, 3*d_model) regardless of attn_type, so this is safe.
"""

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
    """Architecture adapter for MPT models (MPTForCausalLM).

    Optional Parameters (not present in state_dict):
    -------------------------------------------------
    - blocks.{i}.attn.b_Q  -- no bias on query projection
    - blocks.{i}.attn.b_K  -- no bias on key projection
    - blocks.{i}.attn.b_V  -- no bias on value projection
    - blocks.{i}.attn.b_O  -- no bias on output projection
    - blocks.{i}.mlp.b_in  -- no bias on MLP up_proj
    - blocks.{i}.mlp.b_out -- no bias on MLP down_proj
    - blocks.{i}.ln1.b     -- LayerNorm has no bias (bias=None)
    - blocks.{i}.ln2.b     -- LayerNorm has no bias (bias=None)
    - ln_final.b           -- LayerNorm has no bias (bias=None)
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the MPT architecture adapter."""
        super().__init__(cfg)

        # --- Config attributes ---
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "alibi"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        self.cfg.default_prepend_bos = False
        # eps_attr not overridden: NormalizationBridge reads LayerNorm.eps directly,
        # which is the standard PyTorch attribute name.

        # --- Weight processing conversions ---
        # After _split_mpt_qkv splits Wqkv into separate Q/K/V linears each shaped
        # [d_model, d_model] = [(n_heads * d_head), d_model], the standard rearrangements
        # apply. MPT is pure MHA so n_kv_heads == n_heads.
        #
        # MLP weight note: up_proj and down_proj use standard PyTorch [out, in] layout
        # and are copied as-is by LinearBridge. No entries are added for the MLP.
        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        # --- Component mapping ---
        # Weight tying: EmbeddingBridge("transformer.wte") and UnembeddingBridge("lm_head")
        # reference HF modules whose weights are tied via HF's tie-weights mechanism.
        # No adapter-level action is required.
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
        """Split MPT's fused Wqkv [3*d_model, d_model] into separate Q, K, V linears.

        MPT concatenates Q, K, V row-wise (NOT interleaved like BLOOM), matching
        HF's mixed_qkv.chunk(3, dim=2) decomposition in MptAttention.forward.
        """
        w = attn_component.Wqkv.weight.detach().clone()  # [3*d_model, d_model]
        # torch.chunk(w, 3, dim=0) mirrors HF's chunk(3, dim=2) on the activations
        w_q, w_k, w_v = torch.chunk(w, 3, dim=0)  # each [d_model, d_model]
        d_model = self.cfg.d_model

        def make_linear(weight: torch.Tensor) -> nn.Linear:
            lin = nn.Linear(d_model, d_model, bias=False, device=weight.device, dtype=weight.dtype)
            lin.weight = nn.Parameter(weight.contiguous())
            return lin

        return make_linear(w_q), make_linear(w_k), make_linear(w_v)
