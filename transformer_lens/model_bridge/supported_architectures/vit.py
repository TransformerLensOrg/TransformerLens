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

import torch
import torch.nn as nn
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


class ViTMLPWrapper(nn.Module):
    """A transparent wrapper to group ViT's intermediate and output layers into an 'mlp' container.
    
    TransformerLens expects an 'mlp' container, but Hugging Face's ViTLayer places
    'intermediate' and 'output' directly on the block. We inject this wrapper onto
    the block as `.mlp`. We store the block reference inside a tuple to prevent PyTorch
    from registering it as a submodule, which would create a circular graph (Cycle:
    ViTLayer -> mlp -> ViTLayer) and trigger infinite recursion in PyTorch hooks.
    """
    def __init__(self, block: nn.Module):
        super().__init__()
        self._block_ref = (block,)

    @property
    def intermediate(self):
        return self._block_ref[0].intermediate

    @property
    def output(self):
        return self._block_ref[0].output


class ViTArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for ViT and (non-distilled-head) DeiT vision models."""

    supports_generation: bool = False

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
                name=f"{p}encoder.layer",
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

        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is not None:
            self.cfg.d_head = head_dim  # type: ignore[attr-defined]

        # Derive n_ctx to avoid silent fallback to 2048 in generic logic
        image_size = getattr(hf_config, "image_size", None)
        patch_size = getattr(hf_config, "patch_size", None)
        if image_size is not None and patch_size is not None:
            num_patches = (image_size // patch_size) ** 2
            # Standard ViT models add 1 cls_token to the sequence
            self.cfg.n_ctx = num_patches + 1

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

        def patch_layers(module: nn.Module):
            if type(module).__name__ == "ViTLayer":
                # 1. Inject the non-circular MLP wrapper onto every ViTLayer block.
                if not hasattr(module, "mlp"):
                    module.mlp = ViTMLPWrapper(module)
                
                # 2. Fix the tuple-chaining bug strictly for ViTLayer. 
                # TL's internal loop expects blocks to return a Tensor, but HF ViTLayer returns a tuple.
                if not getattr(module, "_tl_patched", False):
                    original_forward = module.forward
                    
                    def unwrapping_forward(*args, **kwargs):
                        out = original_forward(*args, **kwargs)
                        # If HF returned (hidden_states,), unpack it to just hidden_states
                        return out[0] if isinstance(out, tuple) else out
                        
                    module.forward = unwrapping_forward
                    module._tl_patched = True
                    
            for child in module.children():
                patch_layers(child)
                
        patch_layers(hf_model)

        with_classifier = hasattr(hf_model, "classifier")
        self.component_mapping = self._build_component_mapping(
            prefix=prefix, with_classifier=with_classifier
        )
