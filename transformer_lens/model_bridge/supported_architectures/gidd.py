"""Gidd architecture adapter.

Dimitri von Rütte's GIDD (``GiddForDiffusionLM``, remote code): the only
open uniform-noise (non-masked) diffusion LM at scale, with self-correction
sampling. The decoder is bidirectional (config.is_causal=False) with
softcap attention variants, optional per-head QK norms, ScaledLinear
projections (weight-scaled at forward), per-layer scaled residual adds
(resid_scale/num_layers), an ungated up/down MLP, and rotary positions
held as a model-level buffer rather than a module.

Everything nonstandard lives inside delegated modules: attention delegates
wholesale (softcap + bidirectional), ScaledLinear wraps as plain hookable
Linears, and generation is the model's own diffusion sampler, reached via
``bridge.diffusion_generate`` — no autoregressive generation, no folding
into scaled projections.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)


def restore_frequencies(hf_model: Any) -> bool:
    """Recompute GIDD's ``frequencies`` rotary table on a loaded model.

    The buffer is non-persistent (absent from the checkpoint) and built in
    ``GiddModel.__init__``; under v5's meta-device load it materializes as
    uninitialized memory — different garbage per load, sometimes NaN — which
    silently corrupts every forward. Recomputing from config is exact.
    Shared with the benchmark so bridge and HF reference agree.
    """
    import sys

    inner = getattr(hf_model, "model", None)
    old = getattr(inner, "frequencies", None)
    if inner is None or old is None:
        return False
    module = sys.modules.get(type(inner).__module__)
    compute = getattr(module, "compute_basic_frequencies", None)
    if compute is None:
        return False
    config = hf_model.config
    freqs = compute(
        base=config.rope_theta,
        rotary_dim=config.hidden_size // config.num_attention_heads,
        max_position_embeddings=config.max_position_embeddings,
    )
    inner.frequencies = freqs.to(device=old.device, dtype=old.dtype)
    return True


class GiddArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GiddForDiffusionLM models."""

    applicable_phases: list[int] = [1, 2, 3, 4]
    supports_generation: bool = False
    # Block-wise denoising with self-correction, shipped on the model class.
    native_sampler: str = "generate"
    # ScaledLinear applies a runtime weight scale; folding norms into those
    # projections (or centering through scaled residual adds) is unsound.
    supports_fold_ln = False

    def native_sampler_kwargs(self, max_new_tokens: int, prompt_len: int) -> dict:
        """Gidd's max_length counts generated tokens: its windows start at
        prompt_length and span max_length, so adding the prompt over-generates."""
        return {
            "max_length": max_new_tokens,
            "block_length": min(128, max_new_tokens),
            "steps": max_new_tokens,
        }

    def __init__(self, cfg: Any) -> None:
        """Initialize the Gidd architecture adapter."""
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.uses_rms_norm = True
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.gated_mlp = False  # ungated up/down MLP
        self.cfg.attn_only = False
        self.cfg.final_rms = True

        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "blocks": BlockBridge(
                name="model.layers",
                config=self.cfg,
                submodules={
                    "ln1": RMSNormalizationBridge(name="attn_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="mlp_layernorm", config=self.cfg),
                    # Bidirectional softcap attention: delegate; QK norms only
                    # exist when use_qk_norm is set.
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                            "q_norm": RMSNormalizationBridge(
                                name="q_norm", config=self.cfg, optional=True
                            ),
                            "k_norm": RMSNormalizationBridge(
                                name="k_norm", config=self.cfg, optional=True
                            ),
                        },
                        maintain_native_attention=True,
                    ),
                    "mlp": self._ungated_mlp(),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Patch the remote class before from_pretrained runs.

        Like BD3LM, the remote code's attribute handling raises on v5's
        all_tied_weights_keys lookup (the checkpoint is untied anyway).
        """
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            model_class = get_class_from_dynamic_module(
                "modeling_gidd.GiddForDiffusionLM", model_name
            )
            setattr(model_class, "all_tied_weights_keys", {})
            # v5 walks _init_weights over every module post-materialization;
            # the remote _init_weights assumes module.weight exists (crashes
            # on containers) and would re-randomize loaded tensors anyway.
            # Skip modules whose params are already real (internlm2 pattern).
            pretrained_cls = model_class.__mro__[1]
            if not getattr(pretrained_cls, "_tl_patched", False):
                original_init_weights = getattr(pretrained_cls, "_init_weights")

                def safe_init_weights(self, mod, _original=original_init_weights):
                    first_param = next(mod.parameters(), None)
                    if first_param is not None and first_param.device.type != "meta":
                        return
                    _original(self, mod)

                setattr(pretrained_cls, "_init_weights", safe_init_weights)
                setattr(pretrained_cls, "_tl_patched", True)
        except Exception:
            pass
        super().prepare_loading(model_name, model_kwargs)

    def prepare_model(self, hf_model: Any) -> None:
        """Restore the rotary table lost to meta-device loading."""
        super().prepare_model(hf_model)
        restore_frequencies(hf_model)

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Delegated attention reads the rotary buffer inside HF; nothing to wire."""
