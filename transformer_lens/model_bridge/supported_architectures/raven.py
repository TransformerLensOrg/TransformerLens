"""Raven / Huginn architecture adapter (RavenForCausalLM).

Family ``tomg-group-umd/huginn-0125``: depth-recurrent ("latent reasoning")
decoder loaded via remote code. Three phases over one residual width —
prelude blocks, a weight-tied recurrent core applied N times (``num_steps``
is a RUNTIME forward argument, defaulting to ``config.mean_recurrence``),
then coda blocks. Core blocks are post-residual ``SandwichBlock`` modules
(residual renormalised after each add), MHA with combined ``Wqkv`` plus a
learned additive ``qk_bias``.

Adapter decisions:
- Full delegation: recurrence, prelude re-injection, emb-scale, sandwich
  norms, and RoPE all run inside the remote-code forward.
- ``OpaqueBlockBridge`` for all three block lists: BlockBridge's hook aliases
  hardcode the standard pre-norm flow the SandwichBlock does not follow.
- Core-block ``hook_in``/``hook_out`` fire once PER recurrence step (N times
  per forward); ``run_with_cache`` keeps the final step. Per-step access is
  not expressible through the static hook names — it needs the model's native
  ``iterate_one_step``/``predict_from_latents`` interface. Deliberately not
  mapped: ``transformer.adapter`` and ``HuginnDynamicCache``'s slot layout.
- ``applicable_phases = []``: ``initialize_state`` uses ``torch.randn_like``,
  so the forward is non-deterministic unless seeded — integration tests pin
  the seed around both the bridge and HF calls.
- Only bias anywhere is ``qk_bias``; weight processing must tolerate missing
  biases via ``ProcessWeights._safe_get_tensor()``.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    OpaqueBlockBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
)
from transformer_lens.model_bridge.supported_architectures._remote_code_compat import (
    force_import_remote_class,
    iter_remote_modeling_modules,
    patch_init_weights_skip_loaded,
    retie_weights_keys_v5,
)


class RavenArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for RavenForCausalLM (Huginn depth-recurrent decoder).

    Prelude / weight-tied recurrent core / coda phases over a shared residual
    width. The recurrence and prelude re-injection live inside the remote-code
    HF forward, which the bridge delegates to; see the module docstring for the
    full set of adapter decisions.
    """

    # Huginn is off the transformer-shaped verify_models path: post-residual
    # sandwich norms, runtime recurrence count, and a random initial latent
    # state make the phases non-meaningful. Correctness lives in the
    # integration tests (seed pinned before bridge and HF calls).
    applicable_phases: list[int] = []

    # HuginnDynamicCache decode freezes K/V from earlier stochastic calls (fresh
    # randn latent per forward), so cached logits can't seed-match the full HF
    # forward we verify against; full-prefix recompute per step is exact.
    supports_kv_cache: bool = False

    # The remote forward ignores attention_mask (its compile_mask call is
    # commented out), so left-padded rows silently attend to pad tokens.
    supports_batched_generation: bool = False

    def __init__(self, cfg: Any) -> None:
        """Initialize the Raven / Huginn architecture adapter."""
        super().__init__(cfg)

        # Standard weight-processing / norm flags.
        self.cfg.normalization_type = "RMS"
        self.cfg.uses_rms_norm = True
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False

        # ln_f (transformer.ln_f) is applied after the recurrence (feeding the
        # coda) AND at the very end, so it is not a final-only norm. Folding it
        # into W_U would corrupt the coda's input.
        self.supports_fold_ln = False

        # Surface the recurrence-shape attributes on cfg so they are present on
        # both the HF-boot path (also via _HF_PASSTHROUGH_ATTRS) and the
        # synthetic-config path used by the unit tests.
        setattr(self.cfg, "mean_recurrence", getattr(cfg, "mean_recurrence", 32))
        setattr(self.cfg, "mean_backprop_depth", getattr(cfg, "mean_backprop_depth", 8))
        setattr(self.cfg, "n_layers_in_prelude", getattr(cfg, "n_layers_in_prelude", 2))
        setattr(
            self.cfg, "n_layers_in_recurrent_block", getattr(cfg, "n_layers_in_recurrent_block", 4)
        )
        setattr(self.cfg, "n_layers_in_coda", getattr(cfg, "n_layers_in_coda", 2))
        setattr(self.cfg, "injection_type", getattr(cfg, "injection_type", "linear"))
        setattr(self.cfg, "qk_bias", getattr(cfg, "qk_bias", True))

        # Full delegation to the HF forward — no HT-format weight reshaping.
        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            # Three separate physical block lists. Each uses OpaqueBlockBridge so
            # the delegated SandwichBlock forward keeps its post-residual norm
            # placement while hook_in / hook_out wrap the residual stream. Fresh
            # submodule instances per list (they bind to distinct HF modules).
            "prelude": OpaqueBlockBridge(
                name="transformer.prelude",
                submodules=self._sandwich_submodules(),
            ),
            "core_block": OpaqueBlockBridge(
                name="transformer.core_block",
                submodules=self._sandwich_submodules(),
            ),
            "coda": OpaqueBlockBridge(
                name="transformer.coda",
                submodules=self._sandwich_submodules(),
            ),
            "ln_final": RMSNormalizationBridge(name="transformer.ln_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def _sandwich_submodules(self) -> dict[str, Any]:
        """Build a fresh set of SandwichBlock submodule bridges.

        Returns new instances on every call so each of the three block lists
        wraps its own live HF modules rather than sharing bridge objects.

        Submodule keys mirror the HF attribute names (``norm_1``..``norm_4``,
        ``attn``, ``mlp``) so weight-key translation is identity. Attention is
        native (combined ``Wqkv`` + ``qk_bias``, RoPE, custom SDPA path), so it
        is delegated via ``maintain_native_attention``; only the combined
        ``qkv`` and output ``o`` projections are exposed. The gated MLP is
        likewise delegated with its combined gate+up ``fc`` and output ``proj``.
        """
        attn = AttentionBridge(
            name="attn",
            config=self.cfg,
            submodules={
                "qkv": LinearBridge(name="Wqkv"),
                "o": LinearBridge(name="proj"),
            },
            maintain_native_attention=True,
            requires_attention_mask=True,
        )
        # Raven uses a combined Wqkv projection; no separate q/k/v submodules exist.
        # Strip the default hook_q/hook_k/hook_v aliases so they don't appear as
        # dead (unresolvable) aliases in the hook-alias resolution audit.
        attn.hook_aliases = {
            k: v
            for k, v in AttentionBridge.hook_aliases.items()
            if k not in {"hook_q", "hook_k", "hook_v"}
        }
        return {
            "norm_1": RMSNormalizationBridge(name="norm_1", config=self.cfg),
            "attn": attn,
            "norm_2": RMSNormalizationBridge(name="norm_2", config=self.cfg),
            "norm_3": RMSNormalizationBridge(name="norm_3", config=self.cfg),
            "mlp": MLPBridge(
                name="mlp",
                config=self.cfg,
                submodules={
                    "in": LinearBridge(name="fc"),
                    "out": LinearBridge(name="proj"),
                },
            ),
            "norm_4": RMSNormalizationBridge(name="norm_4", config=self.cfg),
        }

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Patch Huginn's remote code for transformers v5 compatibility.

        Huginn's modeling code targets transformers 4.44; two things break under
        v5 (5.8.1), so two patches:

        1. Tied-weights format. ``RavenForCausalLM._tied_weights_keys`` is a list
           (``["lm_head.weight"]``, the 4.x format), but v5's ``tie_weights`` ->
           ``get_expanded_tied_weights_keys`` calls ``.keys()`` on it and raises
           ``AttributeError``. The model does not even construct. Rewrite it to
           the v5 dict form ``{"lm_head.weight": "transformer.wte.weight"}``
           (Huginn ties ``lm_head`` to ``transformer.wte``).

        2. Weight re-init. Under v5's meta-device load-then-materialise flow,
           ``PreTrainedModel._init_weights`` is invoked on modules that already
           hold checkpoint weights, re-randomising them. Guard it to skip modules
           whose parameters are already on a real (non-meta) device — the same
           defensive patch openelm.py applies.

        Args:
            model_name: The HuggingFace model name/path.
            model_kwargs: The kwargs dict for from_pretrained().
        """
        # Force-import the modeling module so it appears in sys.modules to patch.
        if force_import_remote_class(model_name, "raven_modeling_minimal.RavenForCausalLM") is None:
            return

        # Each checkpoint revision gets its own module in sys.modules; patch all.
        for module in iter_remote_modeling_modules("raven"):
            # Patch 1: tied-weights keys list -> v5 dict form (Huginn ties
            # lm_head to transformer.wte).
            retie_weights_keys_v5(
                getattr(module, "RavenForCausalLM", None),
                {"lm_head.weight": "transformer.wte.weight"},
            )
            # Patch 2: don't re-randomise already-loaded weights.
            pretrained_class = getattr(module, "RavenPreTrainedModel", None)
            if pretrained_class is not None:
                patch_init_weights_skip_loaded(pretrained_class)
