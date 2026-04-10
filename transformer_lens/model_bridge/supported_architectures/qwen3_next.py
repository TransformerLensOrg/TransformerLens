"""Qwen3Next architecture adapter.

Qwen3NextForCausalLM is a hybrid linear-attention + full-attention architecture
with a sparse Mixture-of-Experts MLP on every layer. Layers alternate between
GatedDeltaNet (linear attention) and standard full attention blocks, while the
MLP is always a Qwen3NextSparseMoeBlock (gate router + batched experts +
shared expert).

Since self_attn is absent on linear-attention layers, we only map submodules
that exist on ALL layers (norms, MLP). The HF native forward handles
linear/full attention dispatch internally, and MoEBridge delegates the entire
MoE forward (including router, experts, and shared expert) to the native
implementation.

Hook coverage:
- Block-level: hook_resid_pre, hook_resid_post on every layer
- Normalization: ln1 (input_layernorm), ln2 (post_attention_layernorm)
- MLP: hook_in, hook_out on the MoE block (MoEBridge)
- Attention internals are NOT individually hooked (self_attn absent on
  linear-attention layers; mapping it would crash on those layers)
- Expert-level internals are NOT individually hooked (batched expert params
  live inside Qwen3NextExperts; MoEBridge delegates to HF forward)

Optional parameters:
- n_key_value_heads: only set when using GQA (num_key_value_heads != num_attention_heads)
"""

from typing import Any

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    MoEBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class Qwen3NextArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Qwen3Next models.

    Qwen3NextForCausalLM is a hybrid linear-attention + full-attention
    architecture with sparse MoE MLPs, sharing the same design as Qwen3.5:
    - Uses RMSNorm for all normalizations
    - Uses rotary position embeddings (RoPE) with partial rotation
    - Every 4th layer is a full-attention layer (self_attn); the rest are
      GatedDeltaNet linear-attention layers (linear_attn)
    - Uses Qwen3NextSparseMoeBlock on ALL layers (decoder_sparse_step=1 and
      mlp_only_layers=[] on every real checkpoint). The MoE block contains a
      top-K router, batched Qwen3NextExperts (experts.gate_up_proj /
      experts.down_proj as 3D tensors), plus a shared_expert (gated MLP) and
      shared_expert_gate. Each expert is internally a gated MLP.
    - No biases on any linear layers
    - Full-attention layers have Q/K normalization (q_norm, k_norm)
    - Full-attention q_proj outputs n_heads * head_dim * 2 (interleaved
      query+gate layout); the preprocess_weights method slices the query half

    Since self_attn is absent on linear-attention layers, only universally
    present submodules (norms, MLP) are mapped as block submodules. The HF
    native forward handles per-layer attention dispatch internally, and
    MoEBridge delegates the MoE forward pass (including router + experts +
    shared expert) to the native Qwen3NextSparseMoeBlock implementation.

    Optional parameters:
    - n_key_value_heads: set when num_key_value_heads != num_attention_heads (GQA)
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the Qwen3Next architecture adapter."""
        super().__init__(cfg)

        # Core config attributes
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True
        self.cfg.default_prepend_bos = False

        # Disable fold_ln: ln1 is followed by self_attn on full-attention
        # layers and by linear_attn (GatedDeltaNet) on linear-attention layers,
        # but neither is mapped as a bridge submodule (see class docstring for
        # why). With no bridge-mapped target to fold into, the standard fold_ln
        # pass leaves LN weights in an inconsistent state and the processed
        # bridge output diverges from the unprocessed / HF output. Skipping
        # fold_ln keeps processed-mode forward passes numerically equivalent.
        self.supports_fold_ln = False

        # Use eager attention to support output_attentions for hook_attn_scores
        # and hook_pattern. SDPA doesn't support output_attentions.
        self.cfg.attn_implementation = "eager"

        # GQA: only set n_key_value_heads when using grouped-query attention
        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

        self.weight_processing_conversions: dict = {}
        self.component_mapping: dict = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    # Qwen3NextSparseMoeBlock has a custom Qwen3NextTopKRouter
                    # (not an nn.Linear) as `gate`, plus batched experts and a
                    # shared expert. MoEBridge wraps the whole MoE module and
                    # delegates to HF's native forward, so we don't enumerate
                    # the internal structure here.
                    "mlp": MoEBridge(name="mlp", config=self.cfg),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """No-op for hybrid models.

        Hybrid models don't map attention as a block submodule (self_attn is
        absent on linear-attention layers), so there are no rotary embedding
        references to set up.

        Note: to find which layers are full_attention at runtime, use:
            layer_types = getattr(hf_model.config, "layer_types", [])
            first_full_attn_idx = next(
                i for i, t in enumerate(layer_types) if t == "full_attention"
            )
        Do NOT use hf_model.config.full_attention_interval -- it is not stored
        on the config object (consumed during __init__ to build layer_types).
        """

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Slice query half from q_proj.weight (interleaved per-head layout).

        In Qwen3Next, q_proj.weight has shape (n_heads * head_dim * 2, hidden_size).
        Rows are organized as per-head interleaved:
          head_0_query (d_head rows), head_0_gate (d_head rows),
          head_1_query (d_head rows), head_1_gate (d_head rows), ...

        A naive first-half slice would be wrong. We must reshape by head, then
        take the first d_head rows of each head (the query half).

        Note: since self_attn is NOT currently mapped as a bridge submodule,
        these weights will not be loaded by the bridge. This method is included
        for correctness and forward-compatibility.
        """
        n_heads = self.cfg.n_heads
        d_head = self.cfg.d_head
        keys_to_update = [k for k in state_dict if k.endswith(".self_attn.q_proj.weight")]
        for key in keys_to_update:
            w = state_dict[key]  # shape: (n_heads * d_head * 2, hidden_size)
            # Reshape to expose per-head layout
            w = w.view(n_heads, d_head * 2, -1)
            # Take only the first d_head rows of each head (query half)
            state_dict[key] = w[:, :d_head, :].reshape(n_heads * d_head, -1)
        return state_dict
