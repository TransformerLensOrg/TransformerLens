"""Qwen3_5 architecture adapter.

Qwen3_5ForCausalLM is a hybrid linear-attention + full-attention architecture
with a dense gated MLP on every layer. Layers follow a repeating pattern of
3 GatedDeltaNet (linear attention) layers followed by 1 standard full-attention
layer (every 4th layer by default).

Since self_attn is absent on linear-attention layers, we only map submodules
that exist on ALL layers (norms, MLP). The HF native forward handles
linear/full attention dispatch internally, and GatedMLPBridge maps the dense
gate_proj/up_proj/down_proj structure on every layer.

Hook coverage:
- Block-level: hook_resid_pre, hook_resid_post on every layer
- Normalization: ln1 (input_layernorm), ln2 (post_attention_layernorm)
- MLP: hook_in, hook_out via GatedMLPBridge (gate_proj, up_proj, down_proj)
- Attention internals are NOT individually hooked (self_attn absent on
  linear-attention layers; mapping it would crash on those layers)

Optional parameters:
- n_key_value_heads: only set when using GQA (num_key_value_heads != num_attention_heads)
"""

from typing import Any

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class Qwen3_5ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Qwen3_5 models.

    Qwen3_5ForCausalLM is a hybrid linear-attention + full-attention
    architecture with dense gated MLPs, sharing the same hybrid design as
    Qwen3Next but replacing the sparse MoE MLP with a standard dense MLP:
    - Uses RMSNorm for all normalizations
    - Uses rotary position embeddings (RoPE) with partial rotation
    - Every 4th layer is a full-attention layer (self_attn); the rest are
      GatedDeltaNet linear-attention layers (linear_attn)
    - Uses dense gated MLP (gate_proj + up_proj -> down_proj) on ALL layers
    - No biases on any linear layers
    - Full-attention layers have Q/K normalization (q_norm, k_norm)
    - Full-attention q_proj outputs n_heads * head_dim * 2 (interleaved
      query+gate layout); the preprocess_weights method slices the query half

    Since self_attn is absent on linear-attention layers, only universally
    present submodules (norms, MLP) are mapped as block submodules. The HF
    native forward handles per-layer attention dispatch internally.

    Optional parameters:
    - n_key_value_heads: set when num_key_value_heads != num_attention_heads (GQA)
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the Qwen3_5 architecture adapter."""
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
                    # Dense gated MLP present on every layer (unlike Qwen3Next's MoE).
                    # gate_proj + up_proj feed into down_proj via SwiGLU activation.
                    "mlp": GatedMLPBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": LinearBridge(name="gate_proj"),
                            "in": LinearBridge(name="up_proj"),
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Swap the multimodal Qwen3_5Config for its text-only Qwen3_5TextConfig.

        Published Qwen3.5 checkpoints (e.g. Qwen/Qwen3.5-0.8B) carry
        model_type='qwen3_5' and architectures=['Qwen3_5ForConditionalGeneration'].
        AutoModelForCausalLM would load the full VLM (Qwen3_5ForConditionalGeneration)
        with its vision tower, wasting memory and failing the bridge.

        Instead we replace model_kwargs['config'] with the nested text_config so
        AutoModelForCausalLM loads Qwen3_5ForCausalLM (text only).
        """
        config = model_kwargs.get("config")
        if config is not None and hasattr(config, "text_config"):
            model_kwargs["config"] = config.text_config

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

        In Qwen3_5, q_proj.weight has shape (n_heads * head_dim * 2, hidden_size).
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
