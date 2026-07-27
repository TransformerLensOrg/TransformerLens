"""Dream diffusion LM architecture adapter.

HKU-NLP's Dream 7B (``DreamModel``, remote code; also Apple's DiffuCoder):
a discrete-diffusion text model initialized from Qwen2.5, so the module
tree is exactly Qwen2 (biased q/k/v, gated SiLU MLP, RMS norms, shared
``model.rotary_emb``, untied ``lm_head``) — but attention is fully
bidirectional (``is_causal = False``) and generation is iterative
denoising via ``diffusion_generate``, not autoregressive decoding.

Attention is therefore delegated to HF wholesale: the bridge's
reimplemented attention assumes causal masking. Q/K/V/O hooks fire on the
wrapped projections; there is no reconstructed pattern hook.

The remote code targets transformers 4.46; v5 removed the "default" key
from ``ROPE_INIT_FUNCTIONS``, so ``prepare_loading`` re-registers it with
the v4 semantics (plain inverse-frequency rope, attention factor 1.0).
"""

from typing import Any

import torch

from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    LinearBridge,
)
from transformer_lens.model_bridge.supported_architectures.qwen2 import (
    Qwen2ArchitectureAdapter,
)


def _patch_eager_attention_mask(attn_cls: Any) -> None:
    """Teach Dream's eager attention the ``"full"`` mask sentinel -- the bridge forces
    eager, whose path (unlike SDPA) raises on the non-tensor sentinel, so normalize it to None."""
    if getattr(attn_cls, "_tl_mask_sentinel_patched", False):
        return

    original_forward = attn_cls.forward

    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        mask = kwargs.get("attention_mask")
        if mask is not None and not isinstance(mask, torch.Tensor):
            kwargs["attention_mask"] = None
        elif args and len(args) > 1 and args[1] is not None:
            if not isinstance(args[1], torch.Tensor):
                args = (args[0], None) + args[2:]
        return original_forward(self, *args, **kwargs)

    setattr(attn_cls, "forward", forward)
    setattr(attn_cls, "_tl_mask_sentinel_patched", True)


def _patch_from_model_config(gen_cfg_cls: Any) -> None:
    """Rebuild DreamGenerationConfig directly -- v5's from_model_config raises on Dream's
    diffusion fields, and rebuilding preserves the null-by-default ``mask_token_id``."""
    if getattr(gen_cfg_cls, "_tl_from_model_config_patched", False):
        return

    def from_model_config(cls: Any, model_config: Any) -> Any:
        generation_config = cls()
        for key in ("bos_token_id", "eos_token_id", "pad_token_id", "mask_token_id"):
            value = getattr(model_config, key, None)
            if value is not None:
                setattr(generation_config, key, value)
        generation_config._from_model_config = True
        return generation_config

    setattr(gen_cfg_cls, "from_model_config", classmethod(from_model_config))
    setattr(gen_cfg_cls, "_tl_from_model_config_patched", True)


def _v4_default_rope_parameters(
    config: Any = None, device: Any = None, seq_len: Any = None, **rope_kwargs: Any
) -> tuple:
    """transformers 4.x ``_compute_default_rope_parameters``, removed in v5."""
    if config is not None:
        base = config.rope_theta
        partial = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial)
    else:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, 1.0


def _register_default_rope_init() -> None:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    ROPE_INIT_FUNCTIONS.setdefault("default", _v4_default_rope_parameters)


class DreamArchitectureAdapter(Qwen2ArchitectureAdapter):
    """Architecture adapter for DreamModel diffusion LMs."""

    # Sampling is diffusion, not autoregressive; P4 scores the native
    # sampler's text (benchmarks route through diffusion_generate).
    applicable_phases: list[int] = [1, 2, 3, 4]
    supports_generation: bool = False
    # Sampling is iterative denoising, not left-to-right; Dream ships the
    # schedule as a mixin method whose per-step forward goes through __call__,
    # so bridge hooks fire during sampling.
    native_sampler: str = "diffusion_generate"

    def native_sampler_kwargs(self, max_new_tokens: int, prompt_len: int) -> dict:
        """Dream denoises a fixed-length canvas; one step per token is its default ratio."""
        return {"max_new_tokens": max_new_tokens, "steps": max_new_tokens}

    def _build_attention_bridge(self):
        """Bidirectional diffusion attention; the bridge reimplementation
        assumes causal masking, so delegate to HF."""
        return AttentionBridge(
            name="self_attn",
            config=self.cfg,
            submodules={
                "q": LinearBridge(name="q_proj"),
                "k": LinearBridge(name="k_proj"),
                "v": LinearBridge(name="v_proj"),
                "o": LinearBridge(name="o_proj"),
            },
            maintain_native_attention=True,
        )

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Shim the remote code's two transformers-v4 dependencies."""
        _register_default_rope_init()
        # DreamGenerationConfig.validate is a no-op with the v4 signature
        # (is_init=False); v5 passes user_set_attributes. Replace with a
        # kwargs-tolerant no-op.
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            gen_cfg_cls = get_class_from_dynamic_module(
                "generation_utils.DreamGenerationConfig", model_name
            )
            setattr(gen_cfg_cls, "validate", lambda self, *args, **kwargs: None)
            _patch_from_model_config(gen_cfg_cls)
            _patch_eager_attention_mask(
                get_class_from_dynamic_module("modeling_dream.DreamAttention", model_name)
            )
        except Exception:
            pass
        super().prepare_loading(model_name, model_kwargs)

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Delegated attention computes rotary inside HF; nothing to wire."""
