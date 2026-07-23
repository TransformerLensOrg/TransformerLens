"""RWKV architecture adapter.

BlinkDL's RWKV-4 (``RwkvForCausalLM``, native in transformers): the
canonical WKV linear-attention RNN, trained on the Pile in a
Pythia-parallel suite. Blocks pair a time-mix module (token-shift
interpolation into key/value/receptance projections, recurrent WKV
kernel, gated output) with a channel-mix module (token-shift key/
receptance, squared-relu value) under pre-LNs, plus an extra pre_ln on
layer 0 before anything else. Both mixers delegate to HF (the WKV
recurrence has no attention-shaped reconstruction); their projections
are wrapped for hooks.

HF rescales attention.output/feed_forward.value weights by 2^(layer //
rescale_every) at eval (the forward divides hidden states to compensate,
so the function is unchanged); bridge and reference both keep the default
so their weights match exactly. use_cache is forced off: the recurrent
state buffers are written in-place per layer, which breaks autograd under
backward hooks — and generation (the only state consumer) runs on a
bespoke ``state`` kwarg the bridge's loop doesn't speak, so generation
phases are excluded for now.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class _RwkvBlockBridge(BlockBridge):
    """RWKV blocks have no attention: replace the attention-flavored alias set
    with time-mix/channel-mix names (resid_mid is ln2's input, pre-mix)."""

    hook_aliases = {
        "hook_resid_pre": "hook_in",
        "hook_resid_mid": "ln2.hook_in",
        "hook_resid_post": "hook_out",
        "hook_time_mix_in": "time_mix.hook_in",
        "hook_time_mix_out": "time_mix.hook_out",
        "hook_channel_mix_in": "channel_mix.hook_in",
        "hook_channel_mix_out": "channel_mix.hook_out",
    }


class RwkvArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for RwkvForCausalLM models."""

    applicable_phases: list[int] = [1, 2, 3, 4]
    supports_generation: bool = True
    # HF threads recurrence through a bespoke `state` kwarg, not past_key_values;
    # generation recomputes the full prefix per step (exact, O(n^2)).
    supports_kv_cache: bool = False
    # RwkvModel ignores attention_mask entirely, so left-padding would silently
    # poison the recurrent state instead of being masked out.
    supports_batched_generation: bool = False
    # Pre-LNs feed the mixers, but the recurrent WKV kernel consumes raw and
    # time-shifted inputs — no fold target exists.
    supports_fold_ln = False

    def __init__(self, cfg: Any) -> None:
        """Initialize the RWKV architecture adapter."""
        super().__init__(cfg)

        self.cfg.normalization_type = "LN"
        self.cfg.uses_rms_norm = False
        self.cfg.positional_embedding_type = "none"
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        self.cfg.final_rms = False

        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="rwkv.embeddings"),
            "blocks": _RwkvBlockBridge(
                name="rwkv.blocks",
                config=self.cfg,
                submodules={
                    # Layer 0 only: an extra LN before the block body.
                    "pre_ln": NormalizationBridge(
                        name="pre_ln",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                        optional=True,
                    ),
                    "ln1": NormalizationBridge(
                        name="ln1", config=self.cfg, use_native_layernorm_autograd=True
                    ),
                    "time_mix": GeneralizedComponent(
                        name="attention",
                        submodules={
                            "key": LinearBridge(name="key"),
                            "value": LinearBridge(name="value"),
                            "receptance": LinearBridge(name="receptance"),
                            "output": LinearBridge(name="output"),
                        },
                    ),
                    "ln2": NormalizationBridge(
                        name="ln2", config=self.cfg, use_native_layernorm_autograd=True
                    ),
                    # MLPBridge so the component harness sizes inputs by the
                    # true in/out dims (key: d_model->4d, value: 4d->d_model).
                    "channel_mix": MLPBridge(
                        name="feed_forward",
                        config=self.cfg,
                        submodules={
                            "in": LinearBridge(name="key"),
                            "receptance": LinearBridge(name="receptance"),
                            "out": LinearBridge(name="value"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(
                name="rwkv.ln_out", config=self.cfg, use_native_layernorm_autograd=True
            ),
            "unembed": UnembeddingBridge(name="head"),
        }

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Force use_cache off: per-layer in-place state writes break autograd
        under backward hooks, and only recurrent generation consumes them."""
        config = model_kwargs.get("config")
        if config is not None:
            config.use_cache = False
        super().prepare_loading(model_name, model_kwargs)
