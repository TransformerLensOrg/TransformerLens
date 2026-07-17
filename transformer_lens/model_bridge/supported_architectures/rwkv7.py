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

Key adapter decisions
---------------------
1. Full delegation. The recurrence, the token shift, the LoRAs, the GroupNorm,
   and the ``v_first`` threading all live inside the ``fla`` remote-code forward
   that the bridge delegates to, so a single forward pass is numerically correct
   with no scan handling here. ``weight_processing_conversions = {}``.

2. ``SSMBlockBridge`` for the block list, for the same reason nemotron_h / raven
   use it: ``BlockBridge``'s hook aliases hardcode a standard pre-norm attention
   flow (``hook_resid_mid``, ``ln1 -> attn -> ln2 -> mlp``) that a recurrent
   RWKV-7 block does not follow. ``SSMBlockBridge`` delegates the whole block and
   exposes only ``hook_in`` / ``hook_out`` on the residual stream, which is
   correct regardless of the internal mixing. The block's inner norms / mixers
   are still declared as submodules so they wrap the live HF modules and fire.

3. ``SSM2MixerBridge`` for both the time-mixing (``attn``) and channel-mixing
   (``ffn``) sublayers. Its forward is a pure wrap-don't-reimplement passthrough
   that hooks the input (positional or ``hidden_states`` kwarg) and the primary
   output while preserving the rest of the tuple — exactly what both RWKV-7
   sublayers need: ``attn`` returns a 4-tuple ``(hidden_states, attentions,
   past_key_values, v_first)`` (called by keyword) and ``ffn`` returns a 2-tuple
   ``(hidden_states, state)`` (called positionally). The Mamba-specific analysis
   methods and ``hook_ssm_*`` points the bridge also carries are inert here
   (``eager_scan`` defaults off); only ``hook_in`` / ``hook_out`` and the wrapped
   projections participate. The ``r_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj``
   (time-mixing) and ``key`` / ``value`` (channel-mixing) projections are exposed
   as ``LinearBridge`` submodules so their hooks fire; the LoRAs, GroupNorm, and
   per-channel lerp parameters are deliberately left inside the delegated forward.

4. Norms. RWKV-7 uses standard biased LayerNorm, so ``normalization_type = "LN"``
   and the norms are ``NormalizationBridge`` (not ``RMSNormalizationBridge``).
   ``attn_norm`` and the final ``model.norm`` are ordinary single-input norms.
   ``ffn_norm`` is the exception: under ``config.fuse_norm`` (default ``True``)
   the block calls it as ``ffn_norm(hidden_states, residual, True)`` — a fused
   add-and-norm returning ``(normed, new_residual)`` — which the reimplementing
   ``NormalizationBridge`` cannot express. It is therefore wrapped with a plain
   delegating ``GeneralizedComponent`` (generic I/O-hooked passthrough) that
   forwards any signature to the live HF module and hooks the primary output.

5. ``applicable_phases = []``. RWKV-7 is attention-free and recurrent, off the
   transformer-shaped ``verify_models`` phases; correctness is covered by the
   integration tests (bridge-vs-HF logit parity). ``ln_final`` (``model.norm``)
   feeds only ``lm_head``, so folding is unnecessary and ``supports_fold_ln`` is
   left at its default.

6. ``is_stateful = False``. ``fla`` manages recurrent decode state through its
   own ``Cache`` object, not the ``cache_params`` convention the stateful-cache
   path expects, so the adapter does not advertise a stateful cache.

Config propagation
------------------
The RWKV-7 shape attributes (``head_dim``, ``num_heads``, ``value_dim``, the four
LoRA low-rank dims, ``norm_first`` / ``norm_bias`` / ``fuse_norm``, ``attn_mode``,
``hidden_act``) are surfaced on ``self.cfg`` here AND added to the
``_HF_PASSTHROUGH_ATTRS`` list so analysis tooling can read them off a booted
bridge; the synthetic-config unit tests read the same names.

Remote-code loading (transformers v5)
------------------------------------
``fla``'s modeling code predates transformers v5; ``prepare_loading`` applies the
same two defensive patches raven does: (1) rewrite ``_tied_weights_keys`` from the
4.x list form to the v5 dict form so ``get_expanded_tied_weights_keys`` does not
call ``.keys()`` on a list (only reached on tied checkpoints — RWKV-7 defaults
``tie_word_embeddings=False`` — but harmless when untied); (2) guard
``_init_weights`` so v5's meta-device materialise pass does not re-randomise
weights already read from the checkpoint.
"""

import sys
from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    LinearBridge,
    NormalizationBridge,
    SSM2MixerBridge,
    SSMBlockBridge,
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
            "blocks": SSMBlockBridge(
                name="model.layers",
                submodules={
                    # Pre-norm before time-mixing (standard single-input LayerNorm).
                    "attn_norm": NormalizationBridge(
                        name="attn_norm", config=self.cfg, uses_rms_norm=False
                    ),
                    # Time-mixing: generalized delta rule. Delegated passthrough;
                    # only the four projections are exposed (LoRAs / GroupNorm /
                    # lerp params stay inside the fla forward).
                    "attn": SSM2MixerBridge(
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
                    "ffn": SSM2MixerBridge(
                        name="ffn",
                        config=self.cfg,
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
