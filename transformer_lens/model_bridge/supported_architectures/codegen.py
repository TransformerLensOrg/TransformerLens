"""CodeGen architecture adapter."""

from typing import Any

import torch.nn as nn

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    CodeGenAttentionBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    UnembeddingBridge,
)


class CodeGenArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for CodeGen models.

    CodeGen uses a parallel attention+MLP block (attn and MLP share the same
    LayerNorm input and their outputs are summed).  The attention layer uses a
    fused ``qkv_proj`` weight whose layout follows GPT-J's ``mp_num=4``
    tensor-parallel partitioning: the rows are interleaved as
    ``[Q_part, V_part, K_part]`` within each of the 4 MP partitions.

    Optional Parameters (may be absent in some CodeGen checkpoints):
    ---------------------------------------------------------------
    - No bias on qkv_proj (fused QKV has no bias)
    - No bias on out_proj
    - No bias on mlp.fc_in or mlp.fc_out
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the CodeGen architecture adapter."""
        super().__init__(cfg)

        # Config attributes
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        self.cfg.parallel_attn_mlp = True

        # After split_qkv_matrix the individual Q/K/V weights have shape
        # [n_embd, n_embd].  The conversions below rearrange them to the
        # TransformerLens format [n_heads, d_model, d_head].
        self.weight_processing_conversions = {
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
            ),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            "blocks": BlockBridge(
                name="transformer.h",
                submodules={
                    "ln1": NormalizationBridge(name="ln_1", config=self.cfg),
                    # No ln2: CodeGen uses parallel attn+MLP that both read from ln_1
                    "attn": CodeGenAttentionBridge(
                        name="attn",
                        config=self.cfg,
                        split_qkv_matrix=self.split_qkv_matrix,
                        submodules={
                            "qkv": LinearBridge(name="qkv_proj"),
                            "o": LinearBridge(name="out_proj"),
                        },
                    ),
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "in": LinearBridge(name="fc_in"),
                            "out": LinearBridge(name="fc_out"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="transformer.ln_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def split_qkv_matrix(self, attn_component: Any) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        """Split the fused QKV weight into separate Q, K, V linear modules.

        CodeGen uses GPT-J-style tensor-parallel partitioning with ``mp_num=4``
        partitions.  Within each partition the row order is
        ``[Q_part, V_part, K_part]``, i.e. **not** the conventional Q/K/V order.

        The fused weight has shape ``[3 * n_embd, n_embd]``.  We reshape to
        ``[mp_num, 3, local_dim, n_embd]``, extract the three slices, then
        flatten back to ``[n_embd, n_embd]`` for each of Q, K, V.

        Args:
            attn_component: The original ``CodeGenAttention`` module.

        Returns:
            Tuple of ``(q_linear, k_linear, v_linear)`` — three ``nn.Linear``
            modules with no bias and weight shape ``[n_embd, n_embd]``.
        """
        mp_num = 4
        n_embd = self.cfg.d_model

        weight = attn_component.qkv_proj.weight  # [3*n_embd, n_embd]

        # Partition into mp_num slices; within each: [Q_part, V_part, K_part]
        local_dim = n_embd // mp_num
        w = weight.reshape(mp_num, 3, local_dim, n_embd)

        # Index 0 = Q, 1 = V, 2 = K  (CodeGen partition ordering)
        W_Q = w[:, 0, :, :].reshape(n_embd, n_embd)
        W_V = w[:, 1, :, :].reshape(n_embd, n_embd)
        W_K = w[:, 2, :, :].reshape(n_embd, n_embd)

        q_linear = nn.Linear(n_embd, n_embd, bias=False)
        q_linear.weight = nn.Parameter(W_Q)

        k_linear = nn.Linear(n_embd, n_embd, bias=False)
        k_linear.weight = nn.Parameter(W_K)

        v_linear = nn.Linear(n_embd, n_embd, bias=False)
        v_linear.weight = nn.Parameter(W_V)

        return q_linear, k_linear, v_linear
