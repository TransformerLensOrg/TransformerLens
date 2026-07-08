"""PhiMoE architecture adapter."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MoEBridge,
    NormalizationBridge,
    UnembeddingBridge,
)


class PhiMoEArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Microsoft PhiMoE models.

    PhiMoE is a Phi-style decoder with LayerNorm, split Q/K/V attention, and a
    sparse MoE block. This adapter targets the native Transformers implementation
    (``trust_remote_code=False``); the archived remote implementation is not
    compatible with modern Transformers generation/cache semantics.
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the PhiMoE architecture adapter."""
        super().__init__(cfg)

        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = False
        self.cfg.attn_implementation = "eager"
        self.cfg.default_prepend_bos = False

        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.n_key_value_heads
        if hasattr(cfg, "num_experts"):
            self.cfg.num_experts = cfg.num_experts
        if hasattr(cfg, "experts_per_token"):
            self.cfg.experts_per_token = cfg.experts_per_token
        if hasattr(cfg, "router_jitter_noise"):
            setattr(self.cfg, "router_jitter_noise", cfg.router_jitter_noise)
        if hasattr(cfg, "input_jitter_noise"):
            setattr(self.cfg, "input_jitter_noise", cfg.input_jitter_noise)
        if hasattr(cfg, "attention_bias"):
            setattr(self.cfg, "attention_bias", cfg.attention_bias)
        if hasattr(cfg, "lm_head_bias"):
            setattr(self.cfg, "lm_head_bias", cfg.lm_head_bias)
        if hasattr(cfg, "eos_token_id") and cfg.eos_token_id is not None:
            # PhiMoE chat templates terminate assistant turns with <|end|>, while
            # the tokenizer's primary EOS is <|endoftext|>. Stop on either by
            # default so generate() does not continue into a new assistant turn.
            setattr(self.cfg, "eos_token_id", [cfg.eos_token_id, 32007])

        rope_parameters = getattr(cfg, "rope_parameters", None) or {}
        rope_theta = rope_parameters.get("rope_theta") or getattr(cfg, "rope_theta", None)
        if rope_theta is not None:
            self.cfg.rotary_base = rope_theta

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(
                include_biases=bool(getattr(self.cfg, "attention_bias", False))
            ),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": NormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": NormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    # Keep PhiMoE attention delegated to HF so native RoPE, GQA,
                    # and cache behavior stay aligned with Transformers.
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                        maintain_native_attention=True,
                        requires_attention_mask=True,
                    ),
                    # Native Transformers names the sparse MoE block "mlp" and
                    # its router "router"; the archived remote code used other names.
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": LinearBridge(name="router"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Force eager attention for consistent hookable generation."""
        # The archived remote PhiMoE code is incompatible with current
        # Transformers cache/generation semantics; always use the native class.
        model_kwargs["trust_remote_code"] = False
        config = model_kwargs.get("config")
        if config is not None:
            config._attn_implementation = "eager"

    def prepare_model(self, hf_model: Any) -> None:
        """Force eager attention on the loaded HF model."""
        if hasattr(hf_model, "config"):
            hf_model.config._attn_implementation = "eager"
        if hasattr(hf_model, "model") and hasattr(hf_model.model, "_attn_implementation"):
            hf_model.model._attn_implementation = "eager"
