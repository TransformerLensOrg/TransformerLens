"""OPT architecture adapter."""

from typing import Any, Iterator

import torch

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
    BaseTensorConversion,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)


class _UnflattenTokens(BaseTensorConversion):
    """Restore [batch, seq, d] on hooks inside OPT's flattened region.

    OPTDecoderLayer reshapes hidden states to [batch*seq, d] before
    final_layer_norm/fc1/fc2 and back only after the residual add, so the MLP
    and ln2 hooks would otherwise fire 2D — silently wrong for
    position-indexed patching, crashing for `b s d` einops. The block stamps
    the live (batch, seq) at each forward entry; runs only while user hooks
    are attached.
    """

    def __init__(self) -> None:
        super().__init__()
        self.batch_seq: tuple[int, int] | None = None

    def handle_conversion(self, input_value, *full_context):
        bs = self.batch_seq
        if (
            bs is not None
            and isinstance(input_value, torch.Tensor)
            and input_value.dim() == 2
            and input_value.shape[0] == bs[0] * bs[1]
        ):
            return input_value.view(bs[0], bs[1], input_value.shape[-1])
        return input_value

    def revert(self, input_value, *full_context):
        bs = self.batch_seq
        if (
            bs is not None
            and isinstance(input_value, torch.Tensor)
            and input_value.dim() == 3
            and tuple(input_value.shape[:2]) == bs
        ):
            # reshape (not view) — hooks may return non-contiguous tensors
            return input_value.reshape(-1, input_value.shape[-1])
        return input_value


class _OptBlockBridge(BlockBridge):
    """BlockBridge that stamps (batch, seq) onto the unflatten conversions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # hook_mlp_in is the block's own HookPoint, fired from a ln2 pre-hook —
        # inside OPT's flattened region, so it needs the conversion too.
        self.hook_mlp_in.hook_conversion = _UnflattenTokens()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        hidden = args[0] if args else kwargs.get("hidden_states")
        if isinstance(hidden, torch.Tensor) and hidden.dim() == 3:
            batch_seq = (hidden.shape[0], hidden.shape[1])
            for conversion in self._unflatten_conversions():
                conversion.batch_seq = batch_seq
        return super().forward(*args, **kwargs)

    def _unflatten_conversions(self) -> Iterator[_UnflattenTokens]:
        block_conversion = getattr(self.hook_mlp_in, "hook_conversion", None)
        if isinstance(block_conversion, _UnflattenTokens):
            yield block_conversion
        for key in ("mlp", "ln2"):
            component = self.submodules.get(key)
            if component is None:
                continue
            members = [component, *getattr(component, "submodules", {}).values()]
            for member in members:
                for hook_name in ("hook_in", "hook_out"):
                    hook_point = getattr(member, hook_name, None)
                    conversion = getattr(hook_point, "hook_conversion", None)
                    if isinstance(conversion, _UnflattenTokens):
                        yield conversion


class OptArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for OPT models."""

    @staticmethod
    def _with_unflatten(component: Any) -> Any:
        """Attach _UnflattenTokens to the component's (and submodules') hooks."""
        members = [component, *getattr(component, "submodules", {}).values()]
        for member in members:
            for hook_name in ("hook_in", "hook_out"):
                hook_point = getattr(member, hook_name, None)
                if hook_point is not None and hook_point.hook_conversion is None:
                    hook_point.hook_conversion = _UnflattenTokens()
        return component

    def __init__(self, cfg: Any) -> None:
        """Initialize the OPT architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "standard"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False

        # OPT models were trained with BOS tokens (inherits default_prepend_bos = True)

        # Post-norm: disable fold_ln and center_writing_weights (pre-norm only).
        is_post_norm = not getattr(self.cfg, "do_layer_norm_before", True)
        if is_post_norm:
            self.supports_fold_ln = False
            self.supports_center_writing_weights = False

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
            ),
        }

        # OPT-350m is uniquely the only OPT size where word_embed_proj_dim (512)
        # != hidden_size (1024).  It uses project_in/project_out linear layers
        # instead of a final_layer_norm.  Detect this and conditionally include
        # ln_final only when the model actually has one.
        word_embed_proj_dim = getattr(self.cfg, "word_embed_proj_dim", self.cfg.d_model)
        has_final_layer_norm = word_embed_proj_dim == self.cfg.d_model

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.decoder.embed_tokens"),
            "pos_embed": PosEmbedBridge(name="model.decoder.embed_positions"),
            "blocks": _OptBlockBridge(
                name="model.decoder.layers",
                # fc2 IS the mlp output (no container fires hook_out). No
                # hook_mlp_in override: pre-norm, the block already provides it.
                hook_alias_overrides={"hook_mlp_out": "mlp.out.hook_out"},
                submodules={
                    "ln1": NormalizationBridge(
                        name="self_attn_layer_norm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        requires_attention_mask=True,  # OPT requires attention_mask
                        attention_mask_4d=True,  # OPT expects 4D mask [batch, 1, tgt_len, src_len]
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="out_proj"),
                        },
                    ),
                    "ln2": self._with_unflatten(
                        NormalizationBridge(
                            name="final_layer_norm",
                            config=self.cfg,
                            use_native_layernorm_autograd=True,
                        )
                    ),
                    # Containerless fc1/fc2, as BERT. ln2/fc1/fc2 run inside
                    # HF's [batch*seq, d] region — hence the unflatten wrap.
                    "mlp": self._with_unflatten(
                        MLPBridge(
                            name=None,
                            config=self.cfg,
                            submodules={
                                "in": LinearBridge(name="fc1"),
                                "out": LinearBridge(name="fc2"),
                            },
                        )
                    ),
                },
            ),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
        if has_final_layer_norm:
            self.component_mapping["ln_final"] = NormalizationBridge(
                name="model.decoder.final_layer_norm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            )
        # project_in/project_out bridge word_embed_proj_dim <-> hidden_size.
        if not has_final_layer_norm:
            self.component_mapping["project_in"] = LinearBridge(name="model.decoder.project_in")
            self.component_mapping["project_out"] = LinearBridge(name="model.decoder.project_out")
