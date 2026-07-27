"""ViT / DeiT architecture adapter.

Supports HF `ViTModel`, `ViTForImageClassification`, `DeiTModel`,
`DeiTForImageClassification` (single CLS-token classifier head). Encoder blocks
are structurally near-identical between ViT and DeiT — same field names
(layernorm_before/after, attention.{q,k,v,o}_proj, mlp.{fc1,fc2}) — differing
only in the embeddings module used (DeiT's carries an extra distillation token,
which is invisible to this adapter — see vision_embeddings.py).

NOT covered: `DeiTForImageClassificationWithTeacher` (dual cls+distillation head,
averaged). See vision_classifier_head.py's docstring for why, and prepare_model()
below raises loudly if you load one anyway rather than silently producing wrong
logits.

ViT/DeiT blocks are pre-LN (LayerNorm applied *before* attention/MLP, residual
added after — same shape as Llama/GPT2), unlike BERT's post-LN. That's why
`supports_fold_ln = True` here where BertArchitectureAdapter sets it False.
"""

from typing import Any, Dict

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
)
from transformer_lens.model_bridge.generalized_components.vision_classifier_head import (
    VisionClassifierHeadBridge,
)
from transformer_lens.model_bridge.generalized_components.vision_embeddings import (
    VisionEmbeddingsBridge,
)


class ViTArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for ViT and (non-distilled-head) DeiT vision models."""

    supports_generation: bool = False

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        # Mirrors HubertArchitectureAdapter's self.cfg.is_audio_model = True. Required —
        # bridge.py's forward() has no vision-only dispatch path without it; see
        # BRIDGE_CHANGES.md for the corresponding bridge.py edits this flag depends on.
        self.cfg.is_visual_model = True
        self.cfg.normalization_type = "LN"
        # Position embeddings here are a single learned (1, seq+1[+1], hidden) tensor
        # added inside VisionEmbeddingsBridge, not a separate lookup-by-index "pos_embed"
        # component — there's no dedicated TL positional_embedding_type value for that
        # shape, so "standard" is a placeholder. Check whether anything downstream (e.g.
        # fold_ln / HookedTransformer-compat code paths) actually branches on this string
        # for a model with no separate pos_embed component before trusting it blindly.
        self.cfg.positional_embedding_type = "standard"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        # Pre-LN blocks (see module docstring) — unlike BertArchitectureAdapter's post-LN.
        self.supports_fold_ln = True

        n_heads = self.cfg.n_heads

        # Q/K/V/O are separate nn.Linear projections (q_proj/k_proj/v_proj/o_proj),
        # same rearrangement pattern as the BERT and HuBERT adapters.
        self.weight_processing_conversions = {
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(h d_head) d_model -> h d_model d_head", h=n_heads
                ),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(h d_head) d_model -> h d_model d_head", h=n_heads
                ),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(h d_head) d_model -> h d_model d_head", h=n_heads
                ),
            ),
            "blocks.{i}.attn.q.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(h d_head) -> h d_head", h=n_heads),
            ),
            "blocks.{i}.attn.k.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(h d_head) -> h d_head", h=n_heads),
            ),
            "blocks.{i}.attn.v.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(h d_head) -> h d_head", h=n_heads),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (h d_head) -> h d_head d_model", h=n_heads
                ),
            ),
            # No q/k/v bias conversions are a no-op, not an error, when
            # config.qkv_bias=False: LinearBridge wraps whatever nn.Linear HF actually
            # built, and a bias-less Linear simply has no ".bias" key in the state dict
            # for this conversion to touch.
        }

        # Default mapping assumes a bare ViTModel (no prefix, no classifier).
        # prepare_model() rebuilds this once the actual HF model is available and we
        # can detect the "vit."/"deit." prefix and classifier presence.
        self.component_mapping = self._build_component_mapping(prefix="", with_classifier=False)

    def _build_component_mapping(self, prefix: str, with_classifier: bool) -> Dict[str, Any]:
        """Build component mapping.

        prefix="" for bare ViTModel/DeiTModel, "vit." for ViTForImageClassification,
        "deit." for DeiTForImageClassification. `classifier` itself is always a direct
        top-level attribute of the *ForImageClassification wrapper (never nested under
        the prefix) per the HF source, so it's never prefixed.
        """
        p = prefix
        mapping: Dict[str, Any] = {
            "embed": VisionEmbeddingsBridge(name=f"{p}embeddings"),
            "blocks": BlockBridge(
                name=f"{p}encoder.layer",
                # Same redirect the BERT and HuBERT adapters use — kept for consistency
                # even though ViT's MLP (unlike BERT's) is a real cohesive submodule,
                # since both existing ground-truth adapters apply it regardless.
                hook_alias_overrides={
                    "hook_mlp_out": "mlp.out.hook_out",
                    "hook_mlp_in": "mlp.in.hook_in",
                },
                submodules={
                    "ln1": NormalizationBridge(
                        name="layernorm_before",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "ln2": NormalizationBridge(
                        name="layernorm_after",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "attn": AttentionBridge(
                        name="attention",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="attention.query"),
                            "k": LinearBridge(name="attention.key"),
                            "v": LinearBridge(name="attention.value"),
                            "o": LinearBridge(name="output.dense"),
                        },
                    ),
                    "mlp": MLPBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "in": LinearBridge(name="intermediate.dense"),
                            "out": LinearBridge(name="output.dense"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(
                name=f"{p}layernorm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
        }
        if with_classifier:
            mapping["unembed"] = VisionClassifierHeadBridge(name="classifier")
        return mapping

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Propagate ViT/DeiT HF config attributes not covered by the generic mapper."""
        hf_config = model_kwargs.get("config")
        if hf_config is None:
            return

        # Some newer ViT/DeiT configs allow an explicit head_dim distinct from
        # hidden_size // num_attention_heads. Forward it if present so weight
        # reshaping doesn't silently assume the naive split (same reasoning as the
        # adapter guide's "forgetting n_key_value_heads" pitfall for GQA models).
        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is not None:
            self.cfg.d_head = head_dim  # type: ignore[attr-defined]

    def prepare_model(self, hf_model: Any) -> None:
        """Detect ViTForImageClassification vs DeiTForImageClassification vs a bare
        *Model, and add/omit the classifier head + prefix accordingly."""
        if hasattr(hf_model, "vit"):
            prefix = "vit."
        elif hasattr(hf_model, "deit"):
            prefix = "deit."
        else:
            prefix = ""  # bare ViTModel / DeiTModel, loaded directly

        if hasattr(hf_model, "cls_classifier") and hasattr(hf_model, "distillation_classifier"):
            raise NotImplementedError(
                "DeiTForImageClassificationWithTeacher (dual cls_classifier + "
                "distillation_classifier head, averaged) isn't supported by this "
                "adapter yet — see vision_classifier_head.py's docstring for why, "
                "and what to check before adding it."
            )

        with_classifier = hasattr(hf_model, "classifier")
        self.component_mapping = self._build_component_mapping(
            prefix=prefix, with_classifier=with_classifier
        )
