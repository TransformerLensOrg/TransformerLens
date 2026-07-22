"""Zamba2 hybrid Mamba2-Transformer architecture adapter.

Supports Zamba2ForCausalLM (e.g. Zyphra/Zamba2-1.2B, Zamba2-7B, Zamba2-7B-Instruct).

Architecture overview:
- Heterogeneous layers defined by ``config.layers_block_type`` — surfaced with
  canonical names: ``"linear_attention"`` (pure Mamba-2 SSM, HF ``"mamba"``) or
  ``"hybrid"`` (Mamba-2 + shared global-attention block).
- Most layers are ``Zamba2MambaDecoderLayer``: a single pre-norm
  (``input_layernorm``) followed by a Mamba-2 mixer (``.mamba``).
- A recurring subset are ``Zamba2HybridLayer``: each wraps a Mamba-2 decoder
  layer plus a SHARED ``Zamba2AttentionDecoderLayer``. The shared attention
  block's weights are tied across all hybrid layers, cycling through
  ``config.num_mem_blocks`` unique blocks. When
  ``config.use_shared_attention_adapter=True``, each hybrid layer carries
  an independent per-layer LoRA adapter on top of the shared attention.
- No model-level rotary embedding module is wired by the bridge — the
  attention block handles RoPE internally via ``position_ids``.
- Generation runs on the standard KV-cache path: HF threads a single unified
  ``Zamba2HybridDynamicCache`` via ``past_key_values`` (carrying both KV-cache
  entries for attention and SSM conv/recurrent states for Mamba-2), so the
  bridge does not use the Mamba-specific ``cache_params`` stateful path.

Key adapter decisions:
- ``SSMBlockBridge`` is used for all layers. Its forward delegates entirely to
  the HF layer, giving ``hook_in`` / ``hook_out`` on every layer regardless
  of type.
- For Mamba layers: ``norm`` (-> ``.input_layernorm``) and ``mixer``
  (-> ``.mamba``) are declared as submodules and expose inner hooks
  (in_proj, conv1d, inner_norm, out_proj).
- For Hybrid layers: ``norm`` and ``mixer`` are marked ``optional=True`` so
  component_setup skips them gracefully (Hybrid layers have no top-level
  ``.input_layernorm`` or ``.mamba``). Block-level ``hook_in``/``hook_out``
  still fire on every layer.
- ``applicable_phases = [1, 2, 3, 4]``: P1 is exact vs raw HF (pure
  passthrough); P2/P3 skip without a HookedTransformer; P4 exercises
  ``past_key_values`` cache threading across Mamba-2 and attention layers.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    DepthwiseConv1DBridge,
    EmbeddingBridge,
    GatedRMSNormBridge,
    LinearBridge,
    RMSNormalizationBridge,
    SSM2MixerBridge,
    SSMBlockBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


def _make_optional(component: "GeneralizedComponent") -> "GeneralizedComponent":
    """Mark a GeneralizedComponent submodule as optional.

    ``component_setup.py`` reads ``getattr(submodule, 'optional', False)`` at
    setup time, so setting the attribute directly is safe regardless of whether
    the bridge class's ``__init__`` accepts an ``optional`` keyword argument.
    """
    component.optional = True
    return component


class Zamba2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Zamba2ForCausalLM.

    Hybrid Mamba-2 + shared global-attention model. Most layers are pure
    Mamba-2 SSM (``"mamba"``); a recurring subset are hybrid layers
    (``"hybrid"``) that route through a shared attention block before the
    Mamba-2 step.
    """

    # P1: exact passthrough vs raw HF; P2/P3: skip without HookedTransformer;
    # P4: generation with past_key_values cache threading.
    applicable_phases: list[int] = [1, 2, 3, 4]

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.uses_rms_norm = True
        # RoPE is handled internally by the attention block (position_ids are
        # threaded through HF layers); no model-level rotary bridge needed.
        self.cfg.positional_embedding_type = "none"
        # MLP inside the shared attention block uses GELU, not SwiGLU.
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        self.cfg.final_rms = True
        # NOTE: is_stateful stays False even though Mamba-2 layers carry SSM
        # state. In this bridge, ``is_stateful=True`` selects the *Mamba* cache
        # path, which threads the cache as ``cache_params=`` and drives a
        # conv-kernel ``cache_position``. Zamba2's HF forward instead threads a
        # single unified ``Zamba2HybridDynamicCache`` via ``past_key_values=``
        # (both KV entries and SSM conv/recurrent states live in that one
        # object). The standard KV-cache generation path already threads
        # ``past_key_values`` correctly and matches HF ``generate`` bit-for-bit,
        # so we use it rather than the Mamba path (whose ``cache_params`` kwarg
        # collides with Zamba2's own ``cache_params=past_key_values`` call).
        self.cfg.is_stateful = False

        # Expose the per-layer type list so analysis tools can identify which
        # layers are Mamba-only vs hybrid: "linear_attention" | "hybrid".
        setattr(self.cfg, "layers_block_type", self._canonical_layer_types(cfg))

        # Number of unique shared attention weight blocks (hybrid layers cycle
        # through num_mem_blocks independent attention weight sets).
        setattr(self.cfg, "num_mem_blocks", getattr(cfg, "num_mem_blocks", 1))

        # Whether per-layer LoRA adapters are active on the shared attention.
        setattr(
            self.cfg,
            "use_shared_attention_adapter",
            getattr(cfg, "use_shared_attention_adapter", False),
        )

        # Mamba-2 dimensional config (mirrors Mamba2ArchitectureAdapter /
        # NemotronHArchitectureAdapter patterns).
        mamba_intermediate_size = int(getattr(cfg, "mamba_expand", 2) * self.cfg.d_model)
        n_groups = getattr(cfg, "mamba_ngroups", 1)
        ssm_state_size = getattr(cfg, "mamba_d_state", 64)
        conv_dim = mamba_intermediate_size + 2 * n_groups * ssm_state_size
        setattr(self.cfg, "mamba_intermediate_size", mamba_intermediate_size)
        setattr(self.cfg, "conv_dim", conv_dim)

        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "blocks": SSMBlockBridge(
                name="model.layers",
                submodules={
                    # Pre-norm: present on Zamba2MambaDecoderLayer (.input_layernorm),
                    # absent on Zamba2HybridLayer (no top-level pre-norm) -> optional.
                    "norm": _make_optional(
                        RMSNormalizationBridge(name="input_layernorm", config=self.cfg)
                    ),
                    # Mamba-2 mixer: present on Zamba2MambaDecoderLayer (.mamba),
                    # absent at the top level of Zamba2HybridLayer -> optional.
                    "mixer": _make_optional(
                        SSM2MixerBridge(
                            name="mamba",
                            config=self.cfg,
                            submodules={
                                # -- Mamba-2 inner submodules (all optional) --
                                "in_proj": LinearBridge(name="in_proj", optional=True),
                                "conv1d": DepthwiseConv1DBridge(name="conv1d", optional=True),
                                # HF names the gated RMS norm "norm" inside the
                                # mixer; TL uses "inner_norm" to avoid colliding
                                # with the block-level norm declared above.
                                "inner_norm": _make_optional(GatedRMSNormBridge(name="norm")),
                                "out_proj": LinearBridge(name="out_proj", optional=True),
                            },
                        )
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.final_layernorm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
