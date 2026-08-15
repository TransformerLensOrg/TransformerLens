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

NOTE (transformers >= the ViT/DeiT flattening refactor): as of this version of
`modeling_vit.py`, HF removed the separate `ViTEncoder` wrapper — the blocks
now live directly at `<prefix>.layers` on the model, not `<prefix>.encoder.layer`.
Attention was flattened too: `ViTAttention` now owns `q_proj`/`k_proj`/`v_proj`/
`o_proj` directly (no more nested `attention.attention.{query,key,value}` +
`output.dense`). And `ViTLayer` already exposes a flat `.mlp` submodule
(`ViTMLP` with `.fc1`/`.fc2`) instead of the old `intermediate`/`output` split.
Because of this, the old `ViTMLPWrapper` shim, the block-forward tuple-unwrapping
monkey-patch (`ViTLayer.forward` returns a plain tensor now, not a tuple), and
the `hf_model.encoder_layer = hf_model.encoder.layer` aliasing hack are all
gone — the component mapping below points straight at the real attributes.
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

    # Vision models have no tokenizer, so of the text phases only Phase 1 (HF
    # parity on pixel input) applies — Phases 2/3 need a HookedTransformer
    # counterpart and Phase 4 needs text generation. Phase 9 (vision hook/cache
    # tests) is gated by is_visual_model, not this list; _full_and_core_phases()
    # routes "vision" architectures to {1, 9}.
    applicable_phases: list[int] = [1]

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.is_visual_model = True
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "standard"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        self.supports_fold_ln = True

        n_heads = self.cfg.n_heads

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
        }

        self.component_mapping = self._build_component_mapping(prefix="", with_classifier=False)

    def _build_component_mapping(self, prefix: str, with_classifier: bool) -> Dict[str, Any]:
        p = prefix
        mapping: Dict[str, Any] = {
            "embed": VisionEmbeddingsBridge(name=f"{p}embeddings"),
            "blocks": BlockBridge(
                # HF's ViTModel/DeiTModel no longer wrap blocks in a `.encoder`
                # module — they sit directly on `<prefix>.layers`.
                name=f"{p}layers",
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
                            # Submodule paths here are resolved relative to the
                            # already-resolved "attn" bridge module itself
                            # (block.attention). ViTAttention/DeiTAttention now
                            # owns q_proj/k_proj/v_proj/o_proj directly
                            # (flattened, no nested self-attention + separate
                            # output.dense module), so no "attention." prefix.
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                    ),
                    "mlp": MLPBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            # ViTLayer.mlp is a real ViTMLP module now
                            # (fc1/fc2) — no more intermediate/output split,
                            # so no wrapper shim is needed.
                            "in": LinearBridge(name="fc1"),
                            "out": LinearBridge(name="fc2"),
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

    def prepare_model(self, hf_model: Any) -> None:
        """Detect ViTForImageClassification vs DeiTForImageClassification vs a bare
        *Model, and add/omit the classifier head + prefix accordingly.

        No structural patching of the HF model is needed any more: current
        transformers ViT/DeiT blocks already expose a flat `.mlp` (fc1/fc2) and
        return plain tensors from `forward`, and the blocks live directly on
        `<prefix>.layers` rather than behind a now-removed `.encoder` wrapper.
        This method only has to figure out the right prefix/classifier and
        build the component mapping to point at those real attributes.
        """
        if hasattr(hf_model, "cls_classifier") and hasattr(hf_model, "distillation_classifier"):
            raise NotImplementedError(
                "DeiTForImageClassificationWithTeacher (dual cls_classifier + "
                "distillation_classifier head, averaged) isn't supported by this "
                "adapter yet — see vision_classifier_head.py's docstring for why, "
                "and what to check before adding it."
            )

        # Check for the wrapper module created by HF's ForImageClassification classes.
        # Bare ViTModel / DeiTModel have components at the root, so prefix must be "".
        if hasattr(hf_model, "deit") and not hasattr(hf_model, "vit"):
            prefix = "deit."
        elif hasattr(hf_model, "vit"):
            prefix = "vit."
        else:
            prefix = ""

        with_classifier = hasattr(hf_model, "classifier")
        self.component_mapping = self._build_component_mapping(
            prefix=prefix, with_classifier=with_classifier
        )
        if not with_classifier and getattr(hf_model, "pooler", None) is not None:
            self.component_mapping["pooler"] = LinearBridge(name="pooler.dense")
