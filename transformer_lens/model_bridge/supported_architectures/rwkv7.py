"""RWKV-7 ("Goose") architecture adapter (RWKV7ForCausalLM).

Model family: ``fla-hub/rwkv7-*`` (e.g. ``fla-hub/rwkv7-0.1B-g1``), an
attention-free recurrent language model from the flash-linear-attention (``fla``)
library. Loaded via remote code (``trust_remote_code=True``); the modeling
classes live in ``fla.models.rwkv7``.

Architecture overview
---------------------
RWKV-7 is a flat stack of recurrent blocks over a shared residual width, wrapped
by standard (biased) LayerNorm rather than RMSNorm and with no positional
embeddings::

    embeddings -> [ RWKV7Block x N ] -> norm -> lm_head

Each ``RWKV7Block`` is a pre-norm pair of a time-mixing and a channel-mixing
sublayer::

    x = x + attn(attn_norm(x))     # time-mixing  (RWKV7Attention)
    x = x + ffn(ffn_norm(x))       # channel-mixing (RWKV7FeedForward)

with an extra ``pre_norm`` LayerNorm on layer 0 only (``config.norm_first``).
The time-mixing sublayer is the RWKV-7 "generalized delta rule": token-shifted
lerp coefficients (``x_r``..``x_g``), receptance / key / value / output
projections (``r_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj``), low-rank LoRAs
for the log-space decay / value blend / a-coefficient / gate, and a GroupNorm.
The channel-mixing sublayer is a token-shifted squared-ReLU MLP (``key`` up,
``value`` down).

Cross-block ``v_first`` threading: ``RWKV7Model.forward`` initialises
``v_first = torch.zeros_like(hidden_states)`` and threads it through every block
(layer 0 fills it; later layers blend their value with it). This is managed
entirely by the HF model-level forward, so the bridge — which delegates each
block's forward to HF — gets it for free.
"""

import sys
from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    LinearBridge,
    NormalizationBridge,
    OpaqueBlockBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class RWKV7ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for RWKV7ForCausalLM (RWKV-7 "Goose").

    Attention-free recurrent decoder: a flat stack of pre-norm blocks, each a
    generalized-delta-rule time-mixing sublayer plus a token-shifted squared-ReLU
    channel-mixing sublayer, wrapped by standard biased LayerNorm. The recurrence
    and the cross-block ``v_first`` threading live inside the ``fla`` remote-code
    forward, which the bridge delegates to; see the module docstring for the full
    set of adapter decisions.
    """

    # Attention-free and recurrent — off the transformer-shaped verify_models
    # path; correctness lives in the integration tests (bridge-vs-HF parity).
    applicable_phases: list[int] = []

    def __init__(self, cfg: Any) -> None:
        """Initialize the RWKV-7 architecture adapter."""
        super().__init__(cfg)

        # Standard biased LayerNorm, no positional embeddings, ungated FFN.
        self.cfg.normalization_type = "LN"
        self.cfg.uses_rms_norm = False
        self.cfg.positional_embedding_type = "none"
        self.cfg.final_rms = False
        self.cfg.attn_only = False
        self.cfg.gated_mlp = False

        # fla drives recurrent decode state via its own Cache, not the
        # cache_params convention, so the adapter does not advertise a cache.
        setattr(self.cfg, "is_stateful", False)

        # Surface the RWKV-7 shape attributes on cfg so they are present on both
        # the HF-boot path (also via _HF_PASSTHROUGH_ATTRS) and the synthetic
        # config path used by the unit tests. getattr-with-default keeps a bare
        # TransformerBridgeConfig from raising.
        # head_dim is a read-only property on TransformerBridgeConfig (aliases
        # d_head), so it is read but never assigned. num_heads defaults to
        # d_model // head_dim.
        head_dim = getattr(cfg, "head_dim", 64) or 64
        num_heads = getattr(cfg, "num_heads", None) or max(1, self.cfg.d_model // head_dim)
        value_dim = getattr(cfg, "value_dim", None) or [self.cfg.d_model] * self.cfg.n_layers
        # setattr (not direct assignment) for the RWKV-7-specific names so mypy
        # does not flag them as undeclared on TransformerBridgeConfig.
        setattr(self.cfg, "num_heads", num_heads)
        setattr(self.cfg, "value_dim", value_dim)
        setattr(self.cfg, "decay_low_rank_dim", getattr(cfg, "decay_low_rank_dim", 64))
        setattr(self.cfg, "gate_low_rank_dim", getattr(cfg, "gate_low_rank_dim", 128))
        setattr(self.cfg, "a_low_rank_dim", getattr(cfg, "a_low_rank_dim", 64))
        setattr(self.cfg, "v_low_rank_dim", getattr(cfg, "v_low_rank_dim", 16))
        setattr(self.cfg, "norm_first", getattr(cfg, "norm_first", True))
        setattr(self.cfg, "norm_bias", getattr(cfg, "norm_bias", True))
        setattr(self.cfg, "fuse_norm", getattr(cfg, "fuse_norm", True))
        setattr(self.cfg, "attn_mode", getattr(cfg, "attn_mode", "chunk"))
        setattr(self.cfg, "hidden_act", getattr(cfg, "hidden_act", "sqrelu"))
        # LayerNorm epsilon for the reimplementing NormalizationBridge path (eps
        # is a real config field, so direct assignment is fine).
        self.cfg.eps = getattr(cfg, "norm_eps", getattr(cfg, "eps", 1e-5))

        # Full delegation to the fla forward — no HT-format weight reshaping.
        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embeddings"),
            "blocks": OpaqueBlockBridge(
                name="model.layers",
                submodules={
                    # Pre-norm before time-mixing (standard single-input LayerNorm).
                    "attn_norm": NormalizationBridge(
                        name="attn_norm", config=self.cfg, uses_rms_norm=False
                    ),
                    # Time-mixing: generalized delta rule. Delegated passthrough;
                    # only the four projections are exposed (LoRAs / GroupNorm /
                    # lerp params stay inside the fla forward).
                    "attn": GeneralizedComponent(
                        name="attn",
                        config=self.cfg,
                        submodules={
                            "r_proj": LinearBridge(name="r_proj"),
                            "k_proj": LinearBridge(name="k_proj"),
                            "v_proj": LinearBridge(name="v_proj"),
                            "o_proj": LinearBridge(name="o_proj"),
                        },
                    ),
                    # Pre-norm before channel-mixing. Under config.fuse_norm the
                    # block calls this as ffn_norm(x, residual, True) -> (normed,
                    # residual), a fused signature the reimplementing
                    # NormalizationBridge can't express, so delegate any-signature
                    # to the live HF module with I/O hooks.
                    "ffn_norm": GeneralizedComponent(name="ffn_norm", config=self.cfg),
                    # Channel-mixing: token-shifted squared-ReLU MLP. Delegated
                    # passthrough exposing the up ("key") and down ("value")
                    # projections. HF confusingly names the output proj "value".
                    # NOT MLPBridge (cf. rwkv4): its hook_in pre-fire would
                    # suppress key.hook_in, the post-token-shift input — the one
                    # interesting tensor here. Aliases give hook_pre/hook_post.
                    "ffn": GeneralizedComponent(
                        name="ffn",
                        config=self.cfg,
                        hook_alias_overrides={
                            "hook_pre": "key.hook_out",
                            "hook_post": "value.hook_in",
                        },
                        submodules={
                            "key": LinearBridge(name="key"),
                            "value": LinearBridge(name="value"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(
                name="model.norm", config=self.cfg, uses_rms_norm=False
            ),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Patch fla's RWKV-7 remote code for transformers v5 compatibility.

        Two defensive patches, mirroring raven:

        1. Tied-weights format. ``RWKV7ForCausalLM._tied_weights_keys`` is a list
           (``["lm_head.weight"]``, the 4.x form). v5's ``tie_weights`` ->
           ``get_expanded_tied_weights_keys`` calls ``.keys()`` on the mapping,
           which raises ``AttributeError`` on a list. Rewrite it to the v5 dict
           form ``{"lm_head.weight": "model.embeddings.weight"}``. RWKV-7 defaults
           ``tie_word_embeddings=False`` (so the list path is usually short-
           circuited before ``.keys()``), but the rewrite is harmless when untied
           and prevents the crash on any tied checkpoint.

        2. Weight re-init. Under v5's meta-device load-then-materialise flow,
           ``PreTrainedModel._init_weights`` is invoked on modules that already
           hold checkpoint weights, re-randomising them. Guard it to skip modules
           whose parameters are already on a real (non-meta) device — the same
           defensive patch openelm.py / raven.py apply.

        Args:
            model_name: The HuggingFace model name/path.
            model_kwargs: The kwargs dict for from_pretrained().
        """
        # Force-import the fla RWKV-7 modeling module so its classes appear in
        # sys.modules to patch. fla is normally a pip package; fall back to the
        # dynamic-module route for genuinely bundled remote code.
        try:
            import fla.models.rwkv7.modeling_rwkv7  # noqa: F401
        except Exception:
            try:
                from transformers.dynamic_module_utils import (
                    get_class_from_dynamic_module,
                )

                get_class_from_dynamic_module(
                    "modeling_rwkv7.RWKV7ForCausalLM",
                    model_name,
                )
            except Exception:
                return

        # Patch every loaded RWKV-7 modeling module (each remote revision gets its
        # own module object in sys.modules).
        for key in list(sys.modules.keys()):
            if "rwkv7" not in key.lower() or "modeling" not in key.lower():
                continue
            module = sys.modules[key]

            # Patch 1: tied-weights keys list -> v5 dict form.
            causal_lm_class = getattr(module, "RWKV7ForCausalLM", None)
            if causal_lm_class is not None and isinstance(
                getattr(causal_lm_class, "_tied_weights_keys", None), list
            ):
                causal_lm_class._tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}

            # Patch 2: don't re-randomise already-loaded weights.
            pretrained_class = getattr(module, "RWKV7PreTrainedModel", None)
            if pretrained_class is None or getattr(pretrained_class, "_tl_patched", False):
                continue
            original_init_weights = pretrained_class._init_weights

            def safe_init_weights(self, mod, _original=original_init_weights):
                # Only initialise modules still on meta device (pre-loading);
                # never re-randomise weights already read from the checkpoint.
                first_param = next(mod.parameters(), None)
                if first_param is not None and first_param.device.type != "meta":
                    return
                _original(self, mod)

            pretrained_class._init_weights = safe_init_weights
            pretrained_class._tl_patched = True
