"""Raven / Huginn architecture adapter (RavenForCausalLM).

Model family: tomg-group-umd/huginn-0125 ("Huginn"), a depth-recurrent
("latent reasoning") decoder from Geiping et al. Loaded via remote code
(``auto_map`` → ``raven_modeling_minimal.RavenForCausalLM``), so
``trust_remote_code=True`` is required.

Architecture overview
---------------------
Huginn is NOT a flat stack of transformer layers. Its forward has three
phases, all operating on the same residual width ``n_embd`` (5280):

    wte → prelude (P physical blocks)
        → [ recurrent core: R physical blocks, applied N times ]
        → coda (C physical blocks) → ln_f → lm_head

with P = ``n_layers_in_prelude`` (2), R = ``n_layers_in_recurrent_block``
(4), C = ``n_layers_in_coda`` (2). ``num_hidden_layers`` (8) counts the
*physical* blocks (2 + 4 + 2), stored as three separate ``ModuleList``s under
``model.transformer`` (``prelude`` / ``core_block`` / ``coda``), NOT one
``model.layers``.

The recurrence is the defining feature. ``RavenForCausalLM.forward`` calls
``iterate_forward`` → ``core_block_forward``, which runs the SAME four
``core_block`` modules N times. Each step re-injects the prelude output:

    x = adapter(cat([latent_state, prelude_output], dim=-1))   # injection
    for block in core_block: x = block(x)                      # 4 blocks

``N`` (``num_steps``) is a RUNTIME argument to ``forward``, not a fixed
config value. At eval it defaults to ``config.mean_recurrence`` (32) via
``randomized_iteration_sampler``; a caller may pass any ``num_steps``.

Each block is a ``SandwichBlock``: four RMSNorms with post-residual
normalisation (``x = norm_2(attn(norm_1(x)) + x)``; ``x = norm_4(mlp(norm_3
(x)) + x)``) — the residual stream itself is renormalised after each add,
unlike a standard pre-norm transformer. Attention is MHA (55 heads == 55 kv
heads) with a COMBINED ``Wqkv`` projection plus a learned additive ``qk_bias``
parameter and RoPE (base 50000). The MLP is a gated SiLU MLP with a combined
gate+up ``fc`` projection. Embeddings are scaled by ``√n_embd`` (≈72.66) in
the HF forward, and ``lm_head`` is tied to ``wte``.

Key adapter decisions
---------------------
1. Full delegation, like Ouro. The recurrence, the prelude re-injection, the
   emb-scale, the sandwich norms and RoPE all live inside the remote-code
   ``forward`` that the bridge delegates to, so a single forward pass is
   numerically correct with no loop handling here.

2. ``SSMBlockBridge`` for all three block lists (``prelude`` / ``core_block``
   / ``coda``), for the same reason nemotron_h uses it: ``BlockBridge``'s
   hook aliases hardcode a standard pre-norm flow (hook_resid_mid, ln1→attn→
   ln2→mlp) that the post-residual ``SandwichBlock`` does not follow.
   ``SSMBlockBridge`` delegates the whole block and exposes only ``hook_in`` /
   ``hook_out`` on the residual stream, which is correct regardless of the
   internal norm placement. Each block's inner attn / mlp / norms are still
   declared as submodules so they wrap the live HF modules and their hooks
   fire.

3. Recurrent core hooks. Because the core loop lives inside the HF forward,
   the four ``core_block`` blocks' ``hook_in`` / ``hook_out`` fire once PER
   recurrence step (N times per forward), and ``run_with_cache`` keeps the
   final step. This is the load-bearing behavioural difference from a flat
   decoder and mirrors Ouro's looped-depth semantics. Separately addressing
   an individual recurrence step (e.g. logit-lens across steps) is NOT
   expressible through the static ``core_block.{i}.hook_out`` names — it
   requires the model's native ``iterate_one_step`` / ``predict_from_latents``
   interface. Deliberately not mapped: ``transformer.adapter`` (the injection
   Linear) and ``HuginnDynamicCache``'s block-indexed slot layout.

4. ``applicable_phases = []``. Huginn diverges too far from the transformer-
   shaped ``verify_models`` phases to score meaningfully: post-residual
   sandwich norms, a runtime recurrence count, combined QKV + ``qk_bias``,
   and — decisively — a RANDOM initial latent state (``initialize_state``
   uses ``torch.randn_like``), which makes the forward non-deterministic
   across calls unless the RNG is seeded or ``input_states`` is supplied.
   Correctness is covered in the integration tests, where the seed is pinned
   before both the bridge and HF calls (same decision spirit as nemotron_h /
   mamba). ``ln_f`` is applied mid-network (after the recurrence, feeding the
   coda) as well as at the end, so ``supports_fold_ln = False`` — folding it
   into ``W_U`` would corrupt the coda input.

Config propagation
------------------
The recurrence-shape attributes (``mean_recurrence``, ``n_layers_in_prelude``
/ ``_recurrent_block`` / ``_coda``, ``mean_backprop_depth``, ``injection_type``,
``qk_bias``) are surfaced on ``self.cfg`` here AND added to the two
``_HF_PASSTHROUGH_ATTRS`` lists so analysis tooling can read them off a booted
bridge.

Remote-code loading (transformers v5)
------------------------------------
Huginn's remote code targets transformers 4.44 and breaks under v5 (5.8.1) in
two ways that ``prepare_loading`` patches: (1) ``_tied_weights_keys`` is a list,
but v5's ``tie_weights`` expects a dict, so the model does not even construct;
(2) v5's meta-device load re-invokes ``PreTrainedModel._init_weights`` on
already-materialised modules and would re-randomise the checkpoint. See
``prepare_loading`` for both. (Huginn does not use ``ROPE_INIT_FUNCTIONS``; it
precomputes its own ``freqs_cis``, so no RoPE patch is needed.)

Optional parameters (may be absent from a state_dict)
----------------------------------------------------
Huginn has ``bias=False`` everywhere except the attention ``qk_bias``
parameter; RMSNorm has no bias. Weight processing must tolerate missing
biases via ``ProcessWeights._safe_get_tensor()``.
"""

import sys
from typing import Any

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    RMSNormalizationBridge,
    SSMBlockBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
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
        self.cfg.mean_recurrence = getattr(cfg, "mean_recurrence", 32)
        self.cfg.mean_backprop_depth = getattr(cfg, "mean_backprop_depth", 8)
        self.cfg.n_layers_in_prelude = getattr(cfg, "n_layers_in_prelude", 2)
        self.cfg.n_layers_in_recurrent_block = getattr(cfg, "n_layers_in_recurrent_block", 4)
        self.cfg.n_layers_in_coda = getattr(cfg, "n_layers_in_coda", 2)
        self.cfg.injection_type = getattr(cfg, "injection_type", "linear")
        self.cfg.qk_bias = getattr(cfg, "qk_bias", True)

        # Full delegation to the HF forward — no HT-format weight reshaping.
        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            # Three separate physical block lists. Each uses SSMBlockBridge so
            # the delegated SandwichBlock forward keeps its post-residual norm
            # placement while hook_in / hook_out wrap the residual stream. Fresh
            # submodule instances per list (they bind to distinct HF modules).
            "prelude": SSMBlockBridge(
                name="transformer.prelude",
                submodules=self._sandwich_submodules(),
            ),
            "core_block": SSMBlockBridge(
                name="transformer.core_block",
                submodules=self._sandwich_submodules(),
            ),
            "coda": SSMBlockBridge(
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
        return {
            "norm_1": RMSNormalizationBridge(name="norm_1", config=self.cfg),
            "attn": AttentionBridge(
                name="attn",
                config=self.cfg,
                submodules={
                    "qkv": LinearBridge(name="Wqkv"),
                    "o": LinearBridge(name="proj"),
                },
                maintain_native_attention=True,
                requires_attention_mask=True,
            ),
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
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            get_class_from_dynamic_module(
                "raven_modeling_minimal.RavenForCausalLM",
                model_name,
            )
        except Exception:
            return

        # Each checkpoint revision gets its own module in sys.modules; patch all.
        for key in list(sys.modules.keys()):
            if "raven" not in key.lower() or "modeling" not in key.lower():
                continue
            module = sys.modules[key]

            # Patch 1: tied-weights keys list -> v5 dict form.
            causal_lm_class = getattr(module, "RavenForCausalLM", None)
            if causal_lm_class is not None and isinstance(
                getattr(causal_lm_class, "_tied_weights_keys", None), list
            ):
                causal_lm_class._tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

            # Patch 2: don't re-randomise already-loaded weights.
            pretrained_class = getattr(module, "RavenPreTrainedModel", None)
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
