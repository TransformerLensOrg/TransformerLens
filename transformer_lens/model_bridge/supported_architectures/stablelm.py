"""StableLM architecture adapter."""

from typing import Any

import torch

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    NormalizationBridge,
    PositionEmbeddingsAttentionBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class StableLmArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for StableLM models.

    StableLM uses a Llama-like architecture with separate Q/K/V projections and
    gated MLP, but differs in using standard LayerNorm (not RMSNorm) and partial
    rotary embeddings (25% of head dimensions by default).

    Supports optional features:
    - Grouped Query Attention (num_key_value_heads != num_attention_heads)
    - QKV bias (use_qkv_bias=True on some models like stable-code-3b)
    - Parallel residual connections (use_parallel_residual=True)
    - Per-head QK LayerNorm (qk_layernorm=True)

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    - blocks.{i}.attn.b_Q - Only present when use_qkv_bias=True
    - blocks.{i}.attn.b_K - Only present when use_qkv_bias=True
    - blocks.{i}.attn.b_V - Only present when use_qkv_bias=True
    - blocks.{i}.attn.b_O - No bias on output projection
    - blocks.{i}.mlp.b_in - No bias on MLP up_proj
    - blocks.{i}.mlp.b_gate - No bias on MLP gate_proj
    - blocks.{i}.mlp.b_out - No bias on MLP down_proj
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the StableLM architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = False
        # Force eager attention for numerical consistency with benchmark reference
        # PositionEmbeddingsAttentionBridge delegates to native HF attention, so
        # both bridge and reference must use the same implementation
        self.cfg.attn_implementation = "eager"

        self.default_config = {
            "d_model": cfg.d_model,
            "d_head": cfg.d_model // cfg.n_heads,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "d_vocab": cfg.d_vocab,
        }

        # GQA support
        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.default_config["n_key_value_heads"] = cfg.n_key_value_heads
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

        n_kv_heads = (
            self.cfg.n_key_value_heads
            if self.cfg.n_key_value_heads is not None
            else self.cfg.n_heads
        )

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv_heads),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv_heads),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
            ),
            # Bias conversions for models with use_qkv_bias=True
            "blocks.{i}.attn.q.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) -> n h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.k.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) -> n h", n=n_kv_heads),
            ),
            "blocks.{i}.attn.v.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) -> n h", n=n_kv_heads),
            ),
        }

        # When parallel_attn_mlp=True (HF: use_parallel_residual=True), both attn
        # and MLP read from ln1 output:
        #   x = x + attn(ln1(x)) + mlp(ln1(x))
        # When False, they are sequential with separate norms:
        #   x = x + attn(ln1(x)); x = x + mlp(ln2(x))
        # HF sets post_attention_layernorm=None when use_parallel_residual=True,
        # so we must not include ln2 in that case.
        use_parallel_residual = getattr(cfg, "parallel_attn_mlp", False)

        block_submodules: dict[str, Any] = {
            "ln1": NormalizationBridge(
                name="input_layernorm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
        }
        if not use_parallel_residual:
            block_submodules["ln2"] = NormalizationBridge(
                name="post_attention_layernorm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            )
        block_submodules["attn"] = PositionEmbeddingsAttentionBridge(
            name="self_attn",
            config=self.cfg,
            submodules={
                "q": LinearBridge(name="q_proj"),
                "k": LinearBridge(name="k_proj"),
                "v": LinearBridge(name="v_proj"),
                "o": LinearBridge(name="o_proj"),
            },
            requires_attention_mask=True,
            requires_position_embeddings=True,
        )
        block_submodules["mlp"] = GatedMLPBridge(
            name="mlp",
            config=self.cfg,
            submodules={
                "gate": LinearBridge(name="gate_proj"),
                "in": LinearBridge(name="up_proj"),
                "out": LinearBridge(name="down_proj"),
            },
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules=block_submodules,
            ),
            "ln_final": NormalizationBridge(
                name="model.norm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_hook_compatibility(self, bridge: Any) -> None:
        """Inject hook points for QK LayerNorm on models with qk_layernorm=True.

        StableLM v2 models (e.g., stablelm-2-12b) apply per-head LayerNorm to Q and K
        after projection but before rotary embedding. The native HF attention handles
        this internally, but we inject hooks so researchers can observe/intervene on
        the post-norm Q/K values.

        Adds to each attention bridge:
          - hook_q_layernorm: fires after q_layernorm(query_states)
          - hook_k_layernorm: fires after k_layernorm(key_states)

        This runs during bridge __init__ via _setup_hook_compatibility(), after
        component setup but before hook registry finalization. The hook registry
        scanner skips _original_component subtrees, so we register hooks directly
        in bridge._hook_registry with canonical TL-style names.

        Args:
            bridge: The TransformerBridge instance (fully initialized)
        """
        if not hasattr(bridge, "blocks"):
            return

        for i, block in enumerate(bridge.blocks):
            if not hasattr(block, "attn"):
                continue
            attn_bridge = block.attn
            hf_attn = getattr(attn_bridge, "original_component", None)
            if hf_attn is None:
                continue
            if not getattr(hf_attn, "qk_layernorm", False):
                continue

            # Add hook points to the attention bridge as proper submodules
            attn_bridge.add_module("hook_q_layernorm", HookPoint())
            attn_bridge.add_module("hook_k_layernorm", HookPoint())

            # Register directly in bridge's hook registry with canonical names
            # (the scanner skips _original_component subtrees so won't find these)
            q_name = f"blocks.{i}.attn.hook_q_layernorm"
            k_name = f"blocks.{i}.attn.hook_k_layernorm"
            attn_bridge.hook_q_layernorm.name = q_name
            attn_bridge.hook_k_layernorm.name = k_name
            bridge._hook_registry[q_name] = attn_bridge.hook_q_layernorm
            bridge._hook_registry[k_name] = attn_bridge.hook_k_layernorm

            # Wrap the HF q_layernorm/k_layernorm forward methods to fire hooks
            original_q_ln_forward = hf_attn.q_layernorm.forward
            original_k_ln_forward = hf_attn.k_layernorm.forward

            # Use a closure factory to capture the correct references
            def _make_hooked_forward(original_forward: Any, hook: HookPoint) -> Any:
                def hooked_forward(hidden_states: torch.Tensor) -> torch.Tensor:
                    result = original_forward(hidden_states)
                    return hook(result)

                return hooked_forward

            hf_attn.q_layernorm.forward = _make_hooked_forward(  # type: ignore[method-assign]
                original_q_ln_forward, attn_bridge.hook_q_layernorm
            )
            hf_attn.k_layernorm.forward = _make_hooked_forward(  # type: ignore[method-assign]
                original_k_ln_forward, attn_bridge.hook_k_layernorm
            )

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for StableLM component testing.

        StableLM uses RoPE (Rotary Position Embeddings) with partial rotation.
        We set the rotary_emb reference on all attention bridge instances and
        force eager attention for numerical consistency.

        Args:
            hf_model: The HuggingFace StableLM model instance
            bridge_model: The TransformerBridge model (if available)
        """
        rotary_emb = hf_model.model.rotary_emb

        # Force HF model to use "eager" attention to match bridge implementation
        # Bridge uses "eager" to support output_attentions for hook compatibility
        # SDPA and eager are mathematically equivalent but have numerical differences
        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        # Also set on all attention layers
        if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
            for layer in hf_model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"

        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
