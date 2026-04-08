"""Phi-3 architecture adapter."""

from typing import Any

import torch

from transformer_lens.conversion_utils.conversion_steps import (
    RearrangeTensorConversion,
    SplitTensorConversion,
)
from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
    BaseTensorConversion,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    JointGateUpMLPBridge,
    JointQKVPositionEmbeddingsAttentionBridge,
    LinearBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class _SizedSplitConversion(BaseTensorConversion):
    """Split a tensor using explicit sizes (for GQA where Q/K/V have different dimensions)."""

    def __init__(self, sizes: list[int], index: int, dim: int = 0):
        super().__init__()
        self.sizes = sizes
        self.index = index
        self.dim = dim

    def handle_conversion(self, input_value: torch.Tensor, *full_context: Any) -> torch.Tensor:
        parts = torch.split(input_value, self.sizes, dim=self.dim)
        return parts[self.index]


class Phi3ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Phi-3 models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Phi-3 architecture adapter.

        Args:
            cfg: The configuration object.
        """
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False

        self.cfg.uses_rms_norm = True

        # Standard fold_ln can't handle joint qkv/gate_up projections (shape mismatch).
        # LN folding is handled in preprocess_weights() instead.
        self.supports_fold_ln = False

        # GQA: Q has n_heads * d_head, K/V have n_kv_heads * d_head
        d_head = cfg.d_model // cfg.n_heads
        n_kv_heads = cfg.n_key_value_heads or cfg.n_heads
        q_size = cfg.n_heads * d_head
        kv_size = n_kv_heads * d_head
        qkv_sizes = [q_size, kv_size, kv_size]

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q": ParamProcessingConversion(
                tensor_conversion=_SizedSplitConversion(qkv_sizes, 0),
                source_key="model.layers.{i}.self_attn.qkv_proj.weight",
            ),
            "blocks.{i}.attn.k": ParamProcessingConversion(
                tensor_conversion=_SizedSplitConversion(qkv_sizes, 1),
                source_key="model.layers.{i}.self_attn.qkv_proj.weight",
            ),
            "blocks.{i}.attn.v": ParamProcessingConversion(
                tensor_conversion=_SizedSplitConversion(qkv_sizes, 2),
                source_key="model.layers.{i}.self_attn.qkv_proj.weight",
            ),
            "blocks.{i}.attn.o": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                source_key="model.layers.{i}.self_attn.o_proj.weight",
            ),
            "blocks.{i}.mlp.in": ParamProcessingConversion(
                tensor_conversion=SplitTensorConversion(1, 2),
                source_key="model.layers.{i}.mlp.gate_up_proj.weight",
            ),
            "blocks.{i}.mlp.gate": ParamProcessingConversion(
                tensor_conversion=SplitTensorConversion(0, 2),
                source_key="model.layers.{i}.mlp.gate_up_proj.weight",
            ),
        }

        # Set up component mapping
        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": JointQKVPositionEmbeddingsAttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        split_qkv_matrix=self._split_phi3_qkv,
                        submodules={
                            "qkv": LinearBridge(name="qkv_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                    ),
                    "mlp": JointGateUpMLPBridge(
                        name="mlp",
                        config=self.cfg,
                        split_gate_up_matrix=self._split_gate_up,
                        submodules={
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    @staticmethod
    def _split_gate_up(
        original_mlp_component: Any,
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Split Phi-3's fused gate_up_proj into separate gate and up Linear modules."""
        fused_weight = original_mlp_component.gate_up_proj.weight
        gate_w, up_w = torch.tensor_split(fused_weight, 2, dim=0)
        d_model = fused_weight.shape[1]
        d_mlp = gate_w.shape[0]

        has_bias = (
            hasattr(original_mlp_component.gate_up_proj, "bias")
            and original_mlp_component.gate_up_proj.bias is not None
        )
        gate_b: torch.Tensor | None
        up_b: torch.Tensor | None
        if has_bias:
            gate_b, up_b = torch.tensor_split(original_mlp_component.gate_up_proj.bias, 2, dim=0)
        else:
            gate_b = up_b = None

        gate_proj = torch.nn.Linear(d_model, d_mlp, bias=has_bias)
        gate_proj.weight = torch.nn.Parameter(gate_w)
        if gate_b is not None:
            gate_proj.bias = torch.nn.Parameter(gate_b)

        up_proj = torch.nn.Linear(d_model, d_mlp, bias=has_bias)
        up_proj.weight = torch.nn.Parameter(up_w)
        if up_b is not None:
            up_proj.bias = torch.nn.Parameter(up_b)

        return gate_proj, up_proj

    def _split_phi3_qkv(
        self, original_attention_component: Any
    ) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
        """Split Phi-3's fused qkv_proj into separate Q, K, V linear modules."""
        qkv_weight = original_attention_component.qkv_proj.weight
        d_model = qkv_weight.shape[1]

        # GQA: Q has n_heads * d_head, K/V have n_kv_heads * d_head each
        d_head = self.cfg.d_model // self.cfg.n_heads
        n_kv_heads = self.cfg.n_key_value_heads or self.cfg.n_heads
        q_size = self.cfg.n_heads * d_head
        kv_size = n_kv_heads * d_head
        q_weight, k_weight, v_weight = torch.split(qkv_weight, [q_size, kv_size, kv_size], dim=0)

        has_bias = (
            hasattr(original_attention_component.qkv_proj, "bias")
            and original_attention_component.qkv_proj.bias is not None
        )
        q_bias: torch.Tensor | None
        k_bias: torch.Tensor | None
        v_bias: torch.Tensor | None
        if has_bias:
            q_bias, k_bias, v_bias = torch.split(
                original_attention_component.qkv_proj.bias, [q_size, kv_size, kv_size], dim=0
            )
        else:
            q_bias = k_bias = v_bias = None

        q_linear = torch.nn.Linear(d_model, q_weight.shape[0], bias=has_bias)
        q_linear.weight = torch.nn.Parameter(q_weight)
        if q_bias is not None:
            q_linear.bias = torch.nn.Parameter(q_bias)

        k_linear = torch.nn.Linear(d_model, k_weight.shape[0], bias=has_bias)
        k_linear.weight = torch.nn.Parameter(k_weight)
        if k_bias is not None:
            k_linear.bias = torch.nn.Parameter(k_bias)

        v_linear = torch.nn.Linear(d_model, v_weight.shape[0], bias=has_bias)
        v_linear.weight = torch.nn.Parameter(v_weight)
        if v_bias is not None:
            v_linear.bias = torch.nn.Parameter(v_bias)

        return q_linear, k_linear, v_linear

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for Phi-3 component testing.

        Args:
            hf_model: The HuggingFace Phi-3 model instance
            bridge_model: The TransformerBridge model (if available)
        """
        rotary_emb = hf_model.model.rotary_emb

        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Patch cached Phi-3 remote code for transformers v5 compatibility."""
        uses_remote_code = model_kwargs.get("trust_remote_code", False)
        if not uses_remote_code:
            return

        config = model_kwargs.get("config")
        if config is not None:
            rope_scaling = getattr(config, "rope_scaling", None)
            if isinstance(rope_scaling, dict) and rope_scaling.get("rope_type") == "default":
                config.rope_scaling = None

        # Monkey-patch DynamicCache methods removed in transformers v5.
        try:
            from transformers.cache_utils import DynamicCache

            if not hasattr(DynamicCache, "from_legacy_cache"):

                @classmethod  # type: ignore[misc]
                def _from_legacy_cache(cls, past_key_values=None):
                    cache = cls()
                    if past_key_values is not None:
                        for layer_idx, layer_past in enumerate(past_key_values):
                            cache.update(layer_past[0], layer_past[1], layer_idx)
                    return cache

                DynamicCache.from_legacy_cache = _from_legacy_cache  # type: ignore[attr-defined]

            if not hasattr(DynamicCache, "get_usable_length"):

                def _get_usable_length(self, new_seq_len: int = 0, layer_idx: int = 0) -> int:
                    return self.get_seq_length(layer_idx)

                DynamicCache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]

            if not hasattr(DynamicCache, "to_legacy_cache"):

                def _to_legacy_cache(self):
                    legacy_cache = []
                    for layer in self.layers:
                        legacy_cache.append((layer.keys, layer.values))
                    return tuple(legacy_cache)

                DynamicCache.to_legacy_cache = _to_legacy_cache  # type: ignore[attr-defined]
        except Exception:
            pass

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Fold layer norms into joint QKV/gate_up projections.

        Standard fold_ln can't handle joint projections (shape mismatch on round-trip),
        so we scale the full joint weights directly.
        """
        fold_ln = getattr(self, "_fold_ln_requested", True)
        if not fold_ln:
            return state_dict

        n_layers = self.cfg.n_layers

        for i in range(n_layers):
            ln1_key = f"blocks.{i}.ln1.weight"
            ln2_key = f"blocks.{i}.ln2.weight"

            # Fold ln1 into qkv_proj
            if ln1_key in state_dict:
                ln1_w = state_dict[ln1_key].float()
                for qkv_key in [
                    f"blocks.{i}.attn.q.weight",
                    f"blocks.{i}.attn.k.weight",
                    f"blocks.{i}.attn.v.weight",
                ]:
                    if qkv_key in state_dict:
                        orig_dtype = state_dict[qkv_key].dtype
                        state_dict[qkv_key] = (state_dict[qkv_key].float() * ln1_w[None, :]).to(
                            orig_dtype
                        )
                state_dict[ln1_key] = torch.ones_like(state_dict[ln1_key])

            # Fold ln2 into gate_up_proj
            if ln2_key in state_dict:
                ln2_w = state_dict[ln2_key].float()
                for mlp_key in [
                    f"blocks.{i}.mlp.gate.weight",
                    f"blocks.{i}.mlp.in.weight",
                ]:
                    if mlp_key in state_dict:
                        orig_dtype = state_dict[mlp_key].dtype
                        state_dict[mlp_key] = (state_dict[mlp_key].float() * ln2_w[None, :]).to(
                            orig_dtype
                        )
                state_dict[ln2_key] = torch.ones_like(state_dict[ln2_key])

        # Fold ln_final into unembed
        ln_final_key = "ln_final.weight"
        unembed_key = "unembed.weight"
        if ln_final_key in state_dict and unembed_key in state_dict:
            ln_final_w = state_dict[ln_final_key].float()
            unembed_w = state_dict[unembed_key].float()
            orig_dtype = state_dict[unembed_key].dtype
            if unembed_w.shape[-1] == ln_final_w.shape[0]:
                state_dict[unembed_key] = (unembed_w * ln_final_w[None, :]).to(orig_dtype)
            elif unembed_w.shape[0] == ln_final_w.shape[0]:
                state_dict[unembed_key] = (unembed_w * ln_final_w[:, None]).to(orig_dtype)
            state_dict[ln_final_key] = torch.ones_like(state_dict[ln_final_key])

        return state_dict
