"""Falcon architecture adapter.

Supports original Falcon models (7B, 40B, 180B) with:
- Parallel attention+MLP (both read same residual input)
- Multi-query or grouped-query attention (fused QKV)
- RoPE or ALiBi position embeddings
"""

from typing import Any

import torch

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    ALiBiJointQKVAttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    JointQKVPositionEmbeddingsAttentionBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    ParallelBlockBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


def _patch_decoder_inplace_add(layer: Any) -> None:
    """Patch FalconDecoderLayer.forward to use non-inplace addition.

    The original does `mlp_output += attention_output` which modifies
    mlp_output inplace, conflicting with backward hooks on mlp.hook_out.
    We monkey-patch the forward to use `mlp_output = mlp_output + attention_output`.
    """
    import inspect

    src = inspect.getsource(type(layer).forward)

    # Only patch if the inplace pattern exists
    if "mlp_output += attention_output" not in src:
        return

    # Get the original forward and wrap it
    orig_forward = type(layer).forward

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Call original but intercept mlp_output before inplace add.
        # Since we can't modify the source, we use a different approach:
        # register a temporary hook on self.mlp that clones output.
        clone_handle = self.mlp.register_forward_hook(
            lambda _m, _i, o: o.clone() if isinstance(o, torch.Tensor) else o
        )
        try:
            result = orig_forward(self, *args, **kwargs)
        finally:
            clone_handle.remove()
        return result

    layer.forward = patched_forward.__get__(layer, type(layer))  # type: ignore[method-assign]


class FalconArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Falcon models (FalconForCausalLM)."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self._is_alibi = getattr(cfg, "alibi", False)
        self._is_new_arch = getattr(cfg, "new_decoder_architecture", False)
        self._is_multi_query = getattr(cfg, "multi_query", False)
        is_parallel = getattr(cfg, "parallel_attn", True)

        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "alibi" if self._is_alibi else "rotary"
        self.cfg.parallel_attn_mlp = is_parallel
        self.cfg.gated_mlp = False

        if self._is_multi_query:
            self.cfg.n_key_value_heads = 1

        n_kv_heads = self.cfg.n_key_value_heads or self.cfg.n_heads
        self.weight_processing_conversions = {
            "blocks.{i}.attn.q": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.k": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv_heads),
            ),
            "blocks.{i}.attn.v": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv_heads),
            ),
            "blocks.{i}.attn.o": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
            ),
        }

        ln1_name = "ln_attn" if self._is_new_arch else "input_layernorm"

        if self._is_alibi:
            # ALiBi: reimplement attention with ALiBi bias fused into scores.
            # Splits fused QKV and fires hooks at each stage for mech interp.
            attn_bridge: Any = ALiBiJointQKVAttentionBridge(
                name="self_attention",
                config=self.cfg,
                split_qkv_matrix=self._split_falcon_qkv,
                submodules={
                    "qkv": LinearBridge(name="query_key_value"),
                    "o": LinearBridge(name="dense"),
                },
            )
        else:
            # RoPE: reimplement with position embeddings for hook access
            attn_bridge = JointQKVPositionEmbeddingsAttentionBridge(
                name="self_attention",
                config=self.cfg,
                split_qkv_matrix=self._split_falcon_qkv,
                submodules={
                    "qkv": LinearBridge(name="query_key_value"),
                    "o": LinearBridge(name="dense"),
                },
            )

        block_submodules: dict[str, Any] = {
            "ln1": NormalizationBridge(name=ln1_name, config=self.cfg),
            "attn": attn_bridge,
            "mlp": MLPBridge(
                name="mlp",
                config=self.cfg,
                submodules={
                    "in": LinearBridge(name="dense_h_to_4h"),
                    "out": LinearBridge(name="dense_4h_to_h"),
                },
            ),
        }

        if not is_parallel:
            block_submodules["ln2"] = NormalizationBridge(
                name="post_attention_layernorm", config=self.cfg
            )
        elif self._is_new_arch and getattr(cfg, "num_ln_in_parallel_attn", None) == 2:
            block_submodules["ln2"] = NormalizationBridge(name="ln_mlp", config=self.cfg)

        # Falcon has both parallel (most checkpoints) and sequential variants.
        block_cls = ParallelBlockBridge if is_parallel else BlockBridge
        self.component_mapping: dict[str, Any] = {
            "embed": EmbeddingBridge(name="transformer.word_embeddings"),
            "blocks": block_cls(name="transformer.h", submodules=block_submodules),
            "ln_final": NormalizationBridge(name="transformer.ln_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

        if not self._is_alibi:
            self.component_mapping["rotary_emb"] = RotaryEmbeddingBridge(
                name="transformer.rotary_emb", config=self.cfg
            )

    def prepare_model(self, hf_model: Any) -> None:
        """Patch Falcon modules to avoid backward hook conflicts.

        Two issues:
        1. FalconLinear does `input @ self.weight.T` where .T is a view —
           clone the transpose to break the view chain.
        2. FalconDecoderLayer does `mlp_output += attention_output` (inplace) —
           this modifies a tensor captured by mlp.hook_out's backward hook.
           Patch to use non-inplace addition.
        """

        def _make_patched_linear(mod: Any) -> Any:
            def patched_forward(input: torch.Tensor) -> torch.Tensor:
                hidden_states = input @ mod.weight.T.contiguous()
                if mod.bias is not None:
                    hidden_states = hidden_states + mod.bias
                return hidden_states

            return patched_forward

        for module in hf_model.modules():
            if type(module).__name__ == "FalconLinear":
                module.forward = _make_patched_linear(module)  # type: ignore[method-assign]

        # Patch decoder layers to avoid `mlp_output += attention_output` (inplace).
        # The patched forward registers a temporary clone hook on self.mlp
        # around each forward call, so the inplace += gets a clone, not the
        # original tensor captured by backward hooks.
        for module in hf_model.modules():
            if type(module).__name__ == "FalconDecoderLayer":
                _patch_decoder_inplace_add(module)

    def _split_falcon_qkv(
        self, original_attention_component: Any
    ) -> tuple[torch.nn.Linear, torch.nn.Linear, torch.nn.Linear]:
        """Split Falcon's fused query_key_value into separate Q, K, V projections."""
        qkv = original_attention_component.query_key_value
        weight = qkv.weight.detach().clone()
        d_model = self.cfg.d_model
        head_dim = d_model // self.cfg.n_heads
        has_bias = qkv.bias is not None

        if self._is_new_arch:
            n_kv = self.cfg.n_key_value_heads or self.cfg.n_heads
            sizes = [self.cfg.n_heads * head_dim, n_kv * head_dim, n_kv * head_dim]
            W_Q, W_K, W_V = torch.split(weight, sizes, dim=0)
            b_Q: torch.Tensor | None
            b_K: torch.Tensor | None
            b_V: torch.Tensor | None
            if has_bias:
                b_Q, b_K, b_V = torch.split(qkv.bias.detach().clone(), sizes, dim=0)
            else:
                b_Q = b_K = b_V = None
        elif self._is_multi_query:
            sizes = [d_model, head_dim, head_dim]
            W_Q, W_K, W_V = torch.split(weight, sizes, dim=0)
            if has_bias:
                b_Q, b_K, b_V = torch.split(qkv.bias.detach().clone(), sizes, dim=0)
            else:
                b_Q = b_K = b_V = None
        else:
            # Non-multi-query, non-new-arch: QKV is interleaved per head.
            # Weight layout: [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...]
            # Each chunk is head_dim rows. Deinterleave to [Q_all, K_all, V_all].
            n_heads = self.cfg.n_heads
            weight_heads = weight.view(n_heads, 3, head_dim, d_model)
            W_Q = weight_heads[:, 0, :, :].reshape(d_model, d_model)
            W_K = weight_heads[:, 1, :, :].reshape(d_model, d_model)
            W_V = weight_heads[:, 2, :, :].reshape(d_model, d_model)
            if has_bias:
                bias = qkv.bias.detach().clone()
                bias_heads = bias.view(n_heads, 3, head_dim)
                b_Q = bias_heads[:, 0, :].reshape(d_model)
                b_K = bias_heads[:, 1, :].reshape(d_model)
                b_V = bias_heads[:, 2, :].reshape(d_model)
            else:
                b_Q = b_K = b_V = None

        def build_linear(
            w: torch.Tensor, b: torch.Tensor | None, out_features: int
        ) -> torch.nn.Linear:
            linear = torch.nn.Linear(
                d_model, out_features, bias=b is not None, device=w.device, dtype=w.dtype
            )
            linear.weight = torch.nn.Parameter(w.contiguous())
            if b is not None:
                linear.bias = torch.nn.Parameter(b.contiguous())
            return linear

        return (
            build_linear(W_Q, b_Q, W_Q.shape[0]),
            build_linear(W_K, b_K, W_K.shape[0]),
            build_linear(W_V, b_V, W_V.shape[0]),
        )

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for component testing."""
        if self._is_alibi:
            return  # ALiBi handled by HF natively

        rotary_emb = hf_model.transformer.rotary_emb

        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
