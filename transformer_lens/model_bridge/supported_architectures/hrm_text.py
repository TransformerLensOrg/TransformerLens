"""HRM-Text architecture adapter.

HRM-Text (Sapient Intelligence) is a hierarchical two-timescale recurrent model:
two transformer stacks (H = slow/planning, L = fast/computation) iterate in a
nested loop with additive cross-stack coupling.

Architecture notes:
    - **Two physical stacks**: ``model.L_module`` and ``model.H_module``, each with
      ``num_layers_per_stack`` layers. The stacks share identical internal structure
      but have separate weights.
    - **Recurrence**: outer H-cycle iterates ``H_cycles`` times; each iteration runs
      ``L_cycles`` inner L-cycle iterations. Total forward passes through the layer
      stacks = ``H_cycles * (L_cycles + 1)``. The config field ``num_hidden_layers``
      is rewritten by HF to ``num_layers_per_stack * H_cycles * (L_cycles + 1)`` to
      size the KV cache slots.
    - **Parameterless RMSNorm**: ``input_layernorm``, ``post_attention_layernorm``,
      and each stack's ``final_norm`` have no learnable weight tensor.
    - **Sigmoid attention gate**: each attention block has a ``gate_proj`` linear
      that produces a per-head sigmoid gate applied to the attention output before
      ``o_proj``. Delegated to HF; hookable via ``L_blocks.{i}.attn.gate.hook_out``.
    - **Embedding scale**: ``inputs_embeds *= embedding_scale`` (default ~39.19 for
      HRM-Text-1B). Applied at runtime by ``HrmTextModel.forward``; must NOT be
      folded into ``embed.weight`` — same reasoning as ``gemma1.py``.
    - **PrefixLM mask**: instruction tokens attend bidirectionally when
      ``token_type_ids`` is passed to HF forward; delegated, not modeled by bridge.

Known limitations:
    1. Hooks on ``L_blocks.{i}.*`` fire ``H_cycles * L_cycles`` times per forward;
       on ``H_blocks.{i}.*`` they fire ``H_cycles`` times. No per-iteration index is
       exposed; per-cycle disambiguation is future work.
    2. Compat-mode with PrefixLM (``token_type_ids``) inputs is untested in v1.
    3. ``supports_fold_ln = False`` — parameterless norms cannot be folded.
    4. ``supports_center_writing_weights = False`` — block naming (``L_blocks`` /
       ``H_blocks`` instead of ``blocks``) is incompatible with weight-centering
       iteration over ``range(cfg.n_layers)``.
    5. Requires ``transformers >= 5.9.0`` at runtime.
"""

from typing import Any

import torch

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class HrmTextArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for HRM-Text (Sapient Intelligence).

    Exposes ``L_blocks`` (fast/low-level stack) and ``H_blocks`` (slow/high-level
    stack) as sibling block lists. The nested recurrence loop is owned by HF's
    forward; hooks fire once per iteration through the physical layers.
    """

    supports_fold_ln = False
    supports_center_writing_weights = False
    applicable_phases = [1, 2, 3]

    def __init__(self, cfg: Any) -> None:
        """Initialize the HRM-Text architecture adapter."""
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True

        if hasattr(cfg, "num_key_value_heads") and cfg.num_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.num_key_value_heads
        elif hasattr(cfg, "num_attention_heads"):
            self.cfg.n_key_value_heads = cfg.num_attention_heads

        for attr in (
            "H_cycles",
            "L_cycles",
            "L_bp_cycles",
            "num_layers_per_stack",
            "embedding_scale",
            "prefix_lm",
        ):
            if hasattr(cfg, attr):
                setattr(self.cfg, attr, getattr(cfg, attr))

        n_kv_heads = (
            self.cfg.n_key_value_heads
            if hasattr(self.cfg, "n_key_value_heads") and self.cfg.n_key_value_heads is not None
            else self.cfg.n_heads
        )
        self.weight_processing_conversions = self._build_weight_conversions(n_kv_heads)

        def _make_block_submodules():
            return {
                "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                "attn": PositionEmbeddingsAttentionBridge(
                    name="self_attn",
                    config=self.cfg,
                    submodules={
                        "q": LinearBridge(name="q_proj"),
                        "k": LinearBridge(name="k_proj"),
                        "v": LinearBridge(name="v_proj"),
                        "o": LinearBridge(name="o_proj"),
                        "gate": LinearBridge(name="gate_proj"),
                    },
                    requires_attention_mask=True,
                    requires_position_embeddings=True,
                ),
                "mlp": GatedMLPBridge(
                    name="mlp",
                    config=self.cfg,
                    submodules={
                        "gate": LinearBridge(name="gate_proj"),
                        "in": LinearBridge(name="up_proj"),
                        "out": LinearBridge(name="down_proj"),
                    },
                ),
            }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "L_blocks": BlockBridge(
                name="model.L_module.layers",
                submodules=_make_block_submodules(),
            ),
            "H_blocks": BlockBridge(
                name="model.H_module.layers",
                submodules=_make_block_submodules(),
            ),
            "L_ln_final": RMSNormalizationBridge(name="model.L_module.final_norm", config=self.cfg),
            "H_ln_final": RMSNormalizationBridge(name="model.H_module.final_norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def _build_weight_conversions(self, n_kv_heads: int) -> dict[str, ParamProcessingConversion | str]:
        """Build weight processing conversions for both L and H block stacks.

        Each Q/K/V/O weight under ``L_blocks.{i}`` and ``H_blocks.{i}`` needs
        the same ``(n_heads * d_head, d_model) → (n_heads, d_head, d_model)``
        rearrangement as a standard decoder adapter, but with the ``L_blocks`` /
        ``H_blocks`` prefix instead of the ``blocks`` prefix.
        """
        block_prefixes = ["L_blocks", "H_blocks"]
        conversions: dict[str, ParamProcessingConversion | str] = {}
        for prefix in block_prefixes:
            conversions.update(
                {
                    f"{prefix}.{{i}}.attn.q.weight": ParamProcessingConversion(
                        tensor_conversion=RearrangeTensorConversion(
                            "(n h) m -> n m h", n=self.cfg.n_heads
                        ),
                    ),
                    f"{prefix}.{{i}}.attn.k.weight": ParamProcessingConversion(
                        tensor_conversion=RearrangeTensorConversion(
                            "(n h) m -> n m h", n=n_kv_heads
                        ),
                    ),
                    f"{prefix}.{{i}}.attn.v.weight": ParamProcessingConversion(
                        tensor_conversion=RearrangeTensorConversion(
                            "(n h) m -> n m h", n=n_kv_heads
                        ),
                    ),
                    f"{prefix}.{{i}}.attn.o.weight": ParamProcessingConversion(
                        tensor_conversion=RearrangeTensorConversion(
                            "m (n h) -> n h m", n=self.cfg.n_heads
                        ),
                    ),
                }
            )
        return conversions

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for HRM-Text component testing.

        HRM-Text uses RoPE. We set the rotary_emb reference on all attention bridge
        instances so component-level isolation tests can run.
        """
        rotary_emb = hf_model.model.rotary_emb

        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        for stack_attr in ("L_module", "H_module"):
            stack = getattr(hf_model.model, stack_attr, None)
            if stack is not None and hasattr(stack, "layers"):
                for layer in stack.layers:
                    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                        layer.self_attn.config._attn_implementation = "eager"

        if bridge_model is not None:
            for blocks_attr in ("L_blocks", "H_blocks"):
                blocks = getattr(bridge_model, blocks_attr, None)
                if blocks is not None:
                    for block in blocks:
                        if hasattr(block, "attn"):
                            block.attn.set_rotary_emb(rotary_emb)

        for blocks_path in ("L_blocks.0.attn", "H_blocks.0.attn"):
            try:
                attn_bridge = self.get_generalized_component(blocks_path)
                attn_bridge.set_rotary_emb(rotary_emb)
            except (KeyError, AttributeError):
                pass
