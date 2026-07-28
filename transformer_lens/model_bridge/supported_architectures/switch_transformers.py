"""Switch Transformers (``SwitchTransformersForConditionalGeneration``) adapter:
T5 encoder-decoder with top-1-routed sparse-MoE feed-forwards. Blocks delegate
wholesale (v5's plain tensor protocol), FF is a delegated MoEBridge with an optional router."""

from typing import Any

from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    LinearBridge,
    MoEBridge,
    RMSNormalizationBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.t5 import (
    T5ArchitectureAdapter,
)


class _SwitchBlockBridge(BlockBridge):
    """v5 Switch blocks take and return bare tensors; the stack's minimal
    layer call would otherwise trip the tuple-normalizing heuristic."""

    @staticmethod
    def _is_standalone_hidden_state_call(args: tuple, kwargs: dict) -> bool:
        return False


class SwitchTransformersArchitectureAdapter(T5ArchitectureAdapter):
    """Architecture adapter for SwitchTransformersForConditionalGeneration models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Switch Transformers architecture adapter."""
        super().__init__(cfg)

        def attention(name: str, *, cross: bool = False) -> AttentionBridge:
            return AttentionBridge(
                name=name,
                config=self.cfg,
                submodules={
                    "q": LinearBridge(name="q"),
                    "k": LinearBridge(name="k"),
                    "v": LinearBridge(name="v"),
                    "o": LinearBridge(name="o"),
                },
                is_cross_attention=cross,
                maintain_native_attention=True,
            )

        self.components["encoder_blocks"] = _SwitchBlockBridge(
            name="encoder.block",
            config=self.cfg,
            submodules={
                "ln1": RMSNormalizationBridge(name="layer.0.layer_norm", config=self.cfg),
                "attn": attention("layer.0.SelfAttention"),
                "ln2": RMSNormalizationBridge(name="layer.1.layer_norm", config=self.cfg),
                "mlp": self._build_ff_bridge("layer.1"),
            },
        )
        self.components["decoder_blocks"] = _SwitchBlockBridge(
            name="decoder.block",
            config=self.cfg,
            hook_alias_overrides={
                "hook_attn_in": "self_attn.hook_attn_in",
                "hook_attn_out": "self_attn.hook_out",
                "hook_q_input": "self_attn.hook_q_input",
                "hook_k_input": "self_attn.hook_k_input",
                "hook_v_input": "self_attn.hook_v_input",
            },
            submodules={
                "ln1": RMSNormalizationBridge(name="layer.0.layer_norm", config=self.cfg),
                "self_attn": attention("layer.0.SelfAttention"),
                "ln2": RMSNormalizationBridge(name="layer.1.layer_norm", config=self.cfg),
                "cross_attn": attention("layer.1.EncDecAttention", cross=True),
                "ln3": RMSNormalizationBridge(name="layer.2.layer_norm", config=self.cfg),
                "mlp": self._build_ff_bridge("layer.2"),
            },
        )

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """The google/switch-base-* repos ship pytorch_model.bin only; skip
        v5's Hub-side safetensors auto-conversion (it needs a conversion PR)."""
        model_kwargs.setdefault("use_safetensors", False)
        super().prepare_loading(model_name, model_kwargs)

    def _build_ff_bridge(self, layer_prefix: str) -> MoEBridge:
        """Dense or sparse per layer parity; router only exists on sparse."""
        return MoEBridge(
            name=f"{layer_prefix}.mlp",
            config=self.cfg,
            submodules={
                "gate": GeneralizedComponent(name="router", optional=True),
                # Dense-layer projections (absent on sparse layers); in/out per
                # T5-family convention (nonstandard names dodge the component
                # prober's down-projection skip and get crash-probed).
                "in": LinearBridge(name="wi", optional=True),
                "out": LinearBridge(name="wo", optional=True),
            },
        )
