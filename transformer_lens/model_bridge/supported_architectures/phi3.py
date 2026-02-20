"""Phi-3 architecture adapter."""

from typing import Any

import torch

from transformer_lens.conversion_utils.conversion_steps import (
    RearrangeTensorConversion,
    SplitTensorConversion,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)


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

        # Phi-3 uses joint qkv_proj and gate_up_proj. Standard fold_layer_norm splits
        # Q from qkv_proj, scales it, then tries to write [d_model, d_model] back to
        # qkv_proj (shape [3*d_model, d_model]) -> shape mismatch crash.
        # Instead, disable standard fold_ln and handle it in preprocess_weights() by
        # scaling the full joint weight directly.
        self.supports_fold_ln = False

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q": ParamProcessingConversion(
                tensor_conversion=SplitTensorConversion(
                    0,
                    3,
                ),
                source_key="model.layers.{i}.self_attn.qkv_proj.weight",
            ),
            "blocks.{i}.attn.k": ParamProcessingConversion(
                tensor_conversion=SplitTensorConversion(
                    1,
                    3,
                ),
                source_key="model.layers.{i}.self_attn.qkv_proj.weight",
            ),
            "blocks.{i}.attn.v": ParamProcessingConversion(
                tensor_conversion=SplitTensorConversion(
                    2,
                    3,
                ),
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
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        requires_position_embeddings=True,
                        requires_attention_mask=True,
                        submodules={
                            # Phi-3 uses combined qkv_proj, but we still need submodules for hooks
                            "q": LinearBridge(name="qkv_proj"),
                            "k": LinearBridge(name="qkv_proj"),
                            "v": LinearBridge(name="qkv_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                    ),
                    "mlp": GatedMLPBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            # Phi-3 uses joint gate_up_proj, but we need submodules for hooks
                            "gate": LinearBridge(name="gate_up_proj"),
                            "in": LinearBridge(name="gate_up_proj"),
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Fix compatibility issues for Phi-3 models with trust_remote_code=True.

        Only applies patches when trust_remote_code is being used, since the
        built-in transformers Phi-3 implementation works correctly without them.

        Applies two fixes for cached Phi-3 model code vs transformers v5:
        1. rope_scaling format: cached code uses rope_scaling["type"] but
           newer configs use {"rope_type": "default", ...}. Set to None
           when rope_type is "default" (no special scaling).
        2. DynamicCache.from_legacy_cache: removed in transformers v5.
           Add back as a classmethod that converts legacy tuple format.
        """
        uses_remote_code = model_kwargs.get("trust_remote_code", False)
        if not uses_remote_code:
            return

        config = model_kwargs.get("config")
        if config is not None:
            rope_scaling = getattr(config, "rope_scaling", None)
            if isinstance(rope_scaling, dict) and rope_scaling.get("rope_type") == "default":
                config.rope_scaling = None

        # Monkey-patch DynamicCache methods removed in transformers v5.
        # The cached modeling_phi3.py uses several removed/renamed DynamicCache APIs.
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
                # get_usable_length(new_seq_len, layer_idx) → get_seq_length(layer_idx)
                def _get_usable_length(self, new_seq_len: int = 0, layer_idx: int = 0) -> int:
                    return self.get_seq_length(layer_idx)

                DynamicCache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]

            if not hasattr(DynamicCache, "to_legacy_cache"):
                # to_legacy_cache() → convert DynamicCache back to tuple-of-tuples format
                # In transformers v5, DynamicCache uses .layers[i].keys/.values
                def _to_legacy_cache(self):
                    legacy_cache = []
                    for layer in self.layers:
                        legacy_cache.append((layer.keys, layer.values))
                    return tuple(legacy_cache)

                DynamicCache.to_legacy_cache = _to_legacy_cache  # type: ignore[attr-defined]
        except Exception:
            # DynamicCache may not exist or may already have these methods.
            # Safe to skip - only needed for cached remote code compatibility.
            pass

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Fold layer norms directly into joint QKV/gate_up projections.

        Phi-3 uses joint qkv_proj (Q+K+V concatenated) and gate_up_proj (gate+up
        concatenated). Standard fold_layer_norm reads the Q split, scales it, then
        tries to write [d_model, d_model] back to qkv_proj (shape [3*d_model, d_model])
        causing a shape mismatch. We handle LN folding here by scaling the full
        joint weight directly, bypassing the split/revert round-trip.

        The bridge state_dict uses TL-format keys:
          - "blocks.{i}.attn.q.weight" = full qkv_proj weight [3*d_model, d_model]
          - "blocks.{i}.attn.k.weight" = same qkv_proj (separate dict entry)
          - "blocks.{i}.attn.v.weight" = same qkv_proj (separate dict entry)
          - "blocks.{i}.mlp.gate.weight" = full gate_up_proj [2*d_mlp, d_model]
          - "blocks.{i}.mlp.in.weight"  = same gate_up_proj (separate dict entry)
        """
        n_layers = self.cfg.n_layers

        for i in range(n_layers):
            ln1_key = f"blocks.{i}.ln1.weight"
            ln2_key = f"blocks.{i}.ln2.weight"

            # Fold ln1 into the joint qkv_proj (stored under q, k, v keys)
            if ln1_key in state_dict:
                ln1_w = state_dict[ln1_key].float()
                for qkv_key in [
                    f"blocks.{i}.attn.q.weight",
                    f"blocks.{i}.attn.k.weight",
                    f"blocks.{i}.attn.v.weight",
                ]:
                    if qkv_key in state_dict:
                        orig_dtype = state_dict[qkv_key].dtype
                        state_dict[qkv_key] = (
                            state_dict[qkv_key].float() * ln1_w[None, :]
                        ).to(orig_dtype)
                state_dict[ln1_key] = torch.ones_like(state_dict[ln1_key])

            # Fold ln2 into the joint gate_up_proj (stored under gate and in keys)
            if ln2_key in state_dict:
                ln2_w = state_dict[ln2_key].float()
                for mlp_key in [
                    f"blocks.{i}.mlp.gate.weight",
                    f"blocks.{i}.mlp.in.weight",
                ]:
                    if mlp_key in state_dict:
                        orig_dtype = state_dict[mlp_key].dtype
                        state_dict[mlp_key] = (
                            state_dict[mlp_key].float() * ln2_w[None, :]
                        ).to(orig_dtype)
                state_dict[ln2_key] = torch.ones_like(state_dict[ln2_key])

        # Fold ln_final into the unembedding weight
        ln_final_key = "ln_final.weight"
        unembed_key = "unembed.weight"
        if ln_final_key in state_dict and unembed_key in state_dict:
            ln_final_w = state_dict[ln_final_key].float()
            unembed_w = state_dict[unembed_key].float()
            orig_dtype = state_dict[unembed_key].dtype
            # lm_head weight is [vocab_size, d_model] — scale each row
            if unembed_w.shape[-1] == ln_final_w.shape[0]:
                state_dict[unembed_key] = (unembed_w * ln_final_w[None, :]).to(orig_dtype)
            elif unembed_w.shape[0] == ln_final_w.shape[0]:
                state_dict[unembed_key] = (unembed_w * ln_final_w[:, None]).to(orig_dtype)
            state_dict[ln_final_key] = torch.ones_like(state_dict[ln_final_key])

        return state_dict
