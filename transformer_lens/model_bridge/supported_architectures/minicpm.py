"""MiniCPM architecture adapter.

Supports MiniCPMForCausalLM checkpoints (trust_remote_code) with dense attention,
e.g. openbmb/MiniCPM4-0.5B and openbmb/MiniCPM-1B-sft-bf16. Opt-in InfLLM v2
sparse attention (a non-null ``sparse_config``) is rejected.
"""

import math
from typing import Any

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.buffer_restore import restore_rotary_inv_freq
from transformer_lens.model_bridge.compat import patch_dynamic_cache_v5
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    ScaledResidualBlockBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures._remote_code_compat import (
    backfill_is_torch_fx_available,
    create_dynamic_cache_when_missing,
    force_import_remote_class,
    hide_default_rope_scaling_from_init_rope,
    iter_remote_modeling_modules,
    retie_weights_keys_v5,
)


class _MiniCPMAttentionBridge(PositionEmbeddingsAttentionBridge):
    """Reconstructed attention speaking MiniCPM's decoder-layer contract.

    The decoder layer passes ``position_ids`` instead of ``position_embeddings``
    (each attention owns a ``rotary_emb(x, seq_len)`` returning the whole cos/sin
    table), names the cache ``past_key_value``, and unpacks a 3-tuple.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("past_key_values") is None and "past_key_value" in kwargs:
            kwargs["past_key_values"] = kwargs.pop("past_key_value")
        if kwargs.get("position_embeddings") is None:
            kwargs["position_embeddings"] = self._rotary_position_embeddings(args, kwargs)
        attn_output, attn_weights = super().forward(*args, **kwargs)
        return attn_output, attn_weights, kwargs.get("past_key_values")

    def _rotary_position_embeddings(
        self, args: tuple, kwargs: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            position_ids = torch.arange(
                hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)
        rotary_emb: Any = getattr(self.original_component, "rotary_emb")
        cos, sin = rotary_emb(hidden_states, seq_len=int(position_ids.max()) + 1)
        return cos[position_ids], sin[position_ids]


class MiniCPMArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for dense MiniCPM models (MiniCPMForCausalLM).

    Llama-shaped (RMSNorm, RoPE incl. LongRoPE, GQA, gated SiLU MLP, tied
    embeddings) plus three muP scalings, all applied by the HF forward:

    - embeddings are multiplied by ``scale_emb`` (hook_embed exposes the product);
    - each sublayer output is multiplied by ``scale_depth / sqrt(num_hidden_layers)``
      before the residual add (hook_attn_out / hook_mlp_out expose the product);
    - the final hidden state is divided by ``hidden_size / dim_model_base`` before
      lm_head (reproduced by apply_output_logits_transform).

    Nothing is folded into weights: HF applies the scales on the raw and the
    processed path alike, and weight processing is linear, so it commutes.

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    - blocks.{i}.attn.b_Q/b_K/b_V/b_O - attention_bias=False on shipped models
    - blocks.{i}.mlp.b_in/b_gate/b_out - No bias on MLP projections
    - blocks.{i}.ln1.b, blocks.{i}.ln2.b, ln_final.b - RMSNorm has no bias
    """

    _attention_bridge_cls = _MiniCPMAttentionBridge

    def __init__(self, cfg: Any) -> None:
        """Initialize the MiniCPM architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()
        # LlamaTokenizer with add_bos_token=True: HF inputs always start with <s>.
        self.cfg.default_prepend_bos = True

        # Defaults are MiniCPMConfig's, so an absent attr means what it means to HF.
        embedding_scale = float(getattr(cfg, "scale_emb", 1.0))
        residual_scale = float(getattr(cfg, "scale_depth", 1.0)) / math.sqrt(cfg.n_layers)

        self.weight_processing_conversions = {**self._qkvo_weight_conversions()}
        self.component_mapping = {
            "embed": EmbeddingBridge(
                name="model.embed_tokens", output_contribution_scale=embedding_scale
            ),
            "blocks": ScaledResidualBlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": self._qkvo_attention_bridge(),
                    "mlp": self._gated_mlp(),
                },
                residual_contribution_scale=residual_scale,
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def apply_output_logits_transform(self, logits: torch.Tensor) -> torch.Tensor:
        """Match MiniCPM's ``lm_head(h / (hidden_size / dim_model_base))`` output path.

        lm_head is bias-free, so dividing its output equals dividing its input.
        """
        divisor = self.cfg.d_model / float(getattr(self.cfg, "dim_model_base", 1.0))
        return super().apply_output_logits_transform(logits / divisor)

    def output_logits_scale(self) -> float:
        return float(getattr(self.cfg, "dim_model_base", 1.0)) / self.cfg.d_model

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Patch the remote code's transformers-v5 incompatibilities before loading."""
        super().prepare_loading(model_name, model_kwargs)

        config = model_kwargs.get("config")
        tp = getattr(config, "pretraining_tp", 1) if config is not None else 1
        if tp > 1:
            # The remote tensor-parallel lm_head branch skips the dim_model_base scaling.
            raise ValueError(
                f"MiniCPM adapter does not support pretraining_tp={tp}; "
                "only pretraining_tp=1 is supported for logit correctness."
            )
        if getattr(config, "sparse_config", None) is not None:
            # The reconstructed attention is dense; InfLLM v2 would silently diverge.
            raise ValueError(
                "MiniCPM adapter does not support sparse attention (sparse_config / "
                "InfLLM v2); load the checkpoint with sparse_config=None for dense attention."
            )

        # MiniCPM 1/2-era code (e.g. MiniCPM-1B) still calls the legacy-cache methods.
        patch_dynamic_cache_v5()
        # modeling_minicpm imports this v5-removed name at module level, so it
        # must exist before the force-import below.
        backfill_is_torch_fx_available()
        force_import_remote_class(
            model_name,
            "modeling_minicpm.MiniCPMForCausalLM",
            revision=model_kwargs.get("code_revision") or model_kwargs.get("revision"),
        )
        for module in iter_remote_modeling_modules("minicpm"):
            retie_weights_keys_v5(
                getattr(module, "MiniCPMForCausalLM", None),
                {"lm_head.weight": "model.embed_tokens.weight"},
            )
            # MiniCPM4-era MiniCPMModel.forward raises on use_cache=True without a
            # Cache object, which its config enables by default.
            model_cls = getattr(module, "MiniCPMModel", None)
            if model_cls is not None:
                create_dynamic_cache_when_missing(model_cls)
            # v5 turns a null rope_scaling into a default-type dict, which the 1/2-era
            # _init_rope indexes as a scaled RoPE ("type"). Patched on the class so the
            # verify_models HF reference, which loads its own config, is covered too.
            attention_cls = getattr(module, "MiniCPMAttention", None)
            if attention_cls is not None:
                hide_default_rope_scaling_from_init_rope(attention_cls)

    def prepare_model(self, hf_model: Any) -> None:
        """Rebuild the per-layer rotary tables meta-device loading leaves uninitialized."""
        super().prepare_model(hf_model)
        restore_rotary_inv_freq(hf_model)
