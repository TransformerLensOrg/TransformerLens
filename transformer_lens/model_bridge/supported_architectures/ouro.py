"""Ouro architecture adapter."""

import sys
from typing import Any, Optional

import torch

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


def _compute_default_rope_parameters(
    config: Any,
    device: Optional[torch.device] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs: Any,
) -> tuple[torch.Tensor, float]:
    """Standard (unscaled) RoPE inverse frequencies, as transformers v4 defined them.

    Transformers v5 removed the "default" entry from ROPE_INIT_FUNCTIONS (standard
    RoPE moved to a per-model static method), but Ouro's remote code still looks it
    up. Signature and return match the v4 contract the remote code calls with:
    (config, device) -> (inv_freq, attention_scaling).
    """
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", None) or 1.0
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


class OuroArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for ByteDance Ouro (LoopLM) models.

    Ouro is a looped-depth ("Universal Transformer") decoder: the remote-code
    ``OuroModel.forward`` applies the same ``num_hidden_layers``-deep stack
    ``total_ut_steps`` times (4 for the released checkpoints) within a single
    forward pass, applying ``model.norm`` after every pass. The loop lives
    entirely inside the HF forward, which the bridge delegates to, so logits
    and generation are correct with no loop handling here. ``n_layers`` counts
    the physical layers; each block's hooks fire once per loop step, and a
    cache records the final step's value. The same holds for ``ln_final``
    (``model.norm``): it runs after EVERY UT pass, so its hooks fire
    ``total_ut_steps`` times per forward and ``run_with_cache`` keeps only the
    last pass.

    The backbone is Qwen2/Llama-shaped (RoPE, no-bias q/k/v/o projections,
    SwiGLU gate/up/down MLP, untied lm_head) with one twist: sandwich
    normalization. Each decoder layer has FOUR RMSNorms; the extra two
    (``input_layernorm_2``, ``post_attention_layernorm_2``) apply to the
    sublayer outputs before the residual add, exactly like Gemma2's
    ``ln1_post``/``ln2_post`` but without Gemma's +1.0 RMSNorm offset.

    Deliberately not mapped by this adapter:

    - per-loop-step hooks (a cache holds the final UT step only)
    - ``model.early_exit_gate``, the adaptive-exit halting head
    - the ``UniversalTransformerCache`` slot layout (``step * n_layers + layer``)

    Loading requires ``trust_remote_code=True`` (``auto_map`` to
    ``modeling_ouro``).

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    Ouro models do NOT have biases on any mapped linear layers:

    - blocks.{i}.attn.b_Q / b_K / b_V / b_O - no attention biases
    - blocks.{i}.mlp.b_gate / b_in / b_out - no MLP biases
    - blocks.{i}.ln1.b / ln1_post.b / ln2.b / ln2_post.b - RMSNorm has no bias
    - ln_final.b - RMSNorm has no bias

    Weight processing must handle these missing biases gracefully using
    ProcessWeights._safe_get_tensor() or by checking for None values.
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the Ouro architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True
        # default_prepend_bos stays at the framework default: the GPT2-style BPE
        # tokenizer (bos == eos == <|endoftext|>) does not prepend BOS itself.

        # ln_final (model.norm) is applied after EVERY UT pass, feeding the next
        # pass and the early-exit gate, so it is not a final-only norm. Folding
        # it into W_U resets the live module's norm weight the loop reuses and
        # corrupts UT passes 1..N-1.
        self.supports_fold_ln = False

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }
        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                config=self.cfg,
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln1_post": RMSNormalizationBridge(name="input_layernorm_2", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "ln2_post": RMSNormalizationBridge(
                        name="post_attention_layernorm_2", config=self.cfg
                    ),
                    "attn": PositionEmbeddingsAttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
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
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Patch Ouro's remote code for compatibility with transformers v5.

        Ouro's modeling code was written against transformers 4.55, where
        standard RoPE lived in ROPE_INIT_FUNCTIONS["default"]. Transformers v5
        removed that key and instead expects each *RotaryEmbedding class to
        carry a compute_default_rope_parameters static method. Two call sites
        break, so two patches:

        1. OuroRotaryEmbedding.__init__ does ROPE_INIT_FUNCTIONS["default"]
           (KeyError). Rebind the module-level name inside the imported
           modeling_ouro module(s) to a copy with "default" restored; the
           shared transformers dict is left untouched.
        2. v5's PreTrainedModel._init_weights re-initializes RotaryEmbedding
           buffers via module.compute_default_rope_parameters(config)
           (AttributeError). Attach the same function as a static method.

        Args:
            model_name: The HuggingFace model name/path
            model_kwargs: The kwargs dict for from_pretrained()
        """
        # Force-import the modeling module so we can patch it
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            get_class_from_dynamic_module(
                "modeling_ouro.OuroForCausalLM",
                model_name,
            )
        except Exception:
            return

        # Each checkpoint revision gets its own module in sys.modules, so patch
        # every imported Ouro modeling module (same idiom as openelm.py).
        for key in list(sys.modules.keys()):
            if "ouro" in key.lower() and "modeling" in key.lower():
                module = sys.modules[key]
                rope_functions = getattr(module, "ROPE_INIT_FUNCTIONS", None)
                if rope_functions is not None and "default" not in rope_functions:
                    setattr(
                        module,
                        "ROPE_INIT_FUNCTIONS",
                        {**rope_functions, "default": _compute_default_rope_parameters},
                    )
                rope_class = getattr(module, "OuroRotaryEmbedding", None)
                if rope_class is not None and not hasattr(
                    rope_class, "compute_default_rope_parameters"
                ):
                    rope_class.compute_default_rope_parameters = staticmethod(
                        _compute_default_rope_parameters
                    )

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for Ouro component testing.

        Ouro uses RoPE (Rotary Position Embeddings) with a single shared
        ``model.rotary_emb``. We set the rotary_emb reference on all attention
        bridge instances for component testing.

        Args:
            hf_model: The HuggingFace Ouro model instance
            bridge_model: The TransformerBridge model (if available, set rotary_emb on actual instances)
        """
        # Get rotary embedding instance from the model
        rotary_emb = hf_model.model.rotary_emb

        # Set rotary_emb on actual bridge instances in bridge_model if available
        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            # Set on each layer's actual attention bridge instance
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        # Also set on the template for get_generalized_component() calls
        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
