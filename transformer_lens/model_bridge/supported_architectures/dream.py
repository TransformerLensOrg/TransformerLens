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
from transformer_lens.model_bridge.supported_architectures._remote_code_compat import (
    compute_default_rope_inv_freq,
    force_import_remote_class,
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


def _register_default_rope_init() -> None:
    """Restore the global ``ROPE_INIT_FUNCTIONS["default"]`` entry v5 removed;
    Dream's remote code (and llada2_moe's) looks it up by that key."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    ROPE_INIT_FUNCTIONS.setdefault("default", compute_default_rope_inv_freq)


class DreamArchitectureAdapter(Qwen2ArchitectureAdapter):
    """Architecture adapter for DreamModel diffusion LMs."""

    # Sampling is diffusion, not autoregressive; P4 scores the native
    # sampler's text (benchmarks route through diffusion_generate).
    applicable_phases: list[int] = [1, 2, 3, 4]
    supports_generation: bool = False
    # Bidirectional masked-denoising objective; shifted causal CE is undefined.
    supports_causal_loss: bool = False
    # Sampling is iterative denoising, not left-to-right; Dream ships the
    # schedule as a mixin method whose per-step forward goes through __call__,
    # so bridge hooks fire during sampling.
    native_sampler: str = "diffusion_generate"
    # Delegated attention computes rotary inside HF; nothing to wire.
    _testing_eager = None
    _testing_wire_rotary = False

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
        gen_cfg_cls = force_import_remote_class(
            model_name, "generation_utils.DreamGenerationConfig"
        )
        if gen_cfg_cls is not None:
            setattr(gen_cfg_cls, "validate", lambda self, *args, **kwargs: None)
            _patch_from_model_config(gen_cfg_cls)
        attn_cls = force_import_remote_class(model_name, "modeling_dream.DreamAttention")
        if attn_cls is not None:
            _patch_eager_attention_mask(attn_cls)
        super().prepare_loading(model_name, model_kwargs)
