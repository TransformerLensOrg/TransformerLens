"""DeepSeek V4 architecture adapter.

DeepSeek V4 replaces V2/V3's MLA path with a hybrid local/compressed attention
stack and keeps ``hc_mult`` residual streams alive between blocks through
manifold-constrained Hyper-Connections (mHC). The adapter delegates those
architecture-specific calculations to Transformers while exposing the modules
that are useful for interpretability: mHC collapse/mix tensors, compressed KV
states and masks, Lightning Indexer selections, attention projections, and MoE
routing/expert outputs.
"""

from typing import Any, Dict, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    MoEBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class DeepseekV4HyperConnectionBridge(GeneralizedComponent):
    """Bridge an mHC module without discarding its three distinct outputs.

    ``hook_in`` sees the full ``[batch, pos, hc_mult, d_model]`` residual stack.
    ``hook_post`` and ``hook_comb`` expose the learned expansion and stream-mix
    weights, while ``hook_out`` exposes the collapsed conventional residual that
    enters attention or the MLP.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
    ) -> None:
        super().__init__(name, config, submodules=submodules or {})
        self.hook_post = HookPoint()
        self.hook_comb = HookPoint()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the native mHC module and hook each returned tensor separately."""
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        if args and isinstance(args[0], torch.Tensor):
            args = (self.hook_in(args[0]),) + args[1:]
        elif isinstance(kwargs.get("hidden_streams"), torch.Tensor):
            kwargs["hidden_streams"] = self.hook_in(kwargs["hidden_streams"])

        output = self.original_component(*args, **kwargs)
        if not isinstance(output, tuple) or len(output) != 3:
            raise RuntimeError(
                f"DeepSeek V4 hyper-connection {self.name} returned an unexpected output"
            )

        post, comb, collapsed = output
        return self.hook_post(post), self.hook_comb(comb), self.hook_out(collapsed)


class DeepseekV4CompressorBridge(GeneralizedComponent):
    """Bridge CSA/HCA compression and expose compressed KV plus block bias."""

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        optional: bool = False,
    ) -> None:
        super().__init__(
            name,
            config,
            submodules=submodules or {},
            optional=optional,
        )
        self.hook_block_bias = HookPoint()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the native compressor, preserving and hooking both outputs."""
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        if args and isinstance(args[0], torch.Tensor):
            args = (self.hook_in(args[0]),) + args[1:]
        elif isinstance(kwargs.get("hidden_states"), torch.Tensor):
            kwargs["hidden_states"] = self.hook_in(kwargs["hidden_states"])

        output = self.original_component(*args, **kwargs)
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(f"DeepSeek V4 compressor {self.name} returned an unexpected output")

        compressed_kv, block_bias = output
        compressed_kv = self.hook_out(compressed_kv)
        if isinstance(block_bias, torch.Tensor):
            block_bias = self.hook_block_bias(block_bias)
        return compressed_kv, block_bias


class DeepseekV4BlockBridge(BlockBridge):
    """Block bridge whose input/output hooks carry the full mHC stream stack.

    Standard residual aliases are intentionally omitted: V4's block boundary is
    four-dimensional, and presenting it as a conventional single residual stream
    would make otherwise-valid patching code silently target the wrong tensor.
    The collapsed attention/MLP inputs are available at ``attn_hc.hook_out`` and
    ``mlp_hc.hook_out`` respectively.
    """

    hook_aliases: dict[str, str | list[str]] = {}
    hook_out_is_single_residual_stream: bool = False
    maintain_native_attention: bool = True


def _compressor_bridge(cfg: Any) -> DeepseekV4CompressorBridge:
    """Build the common CSA/HCA compressor mapping, including optional indexer."""
    return DeepseekV4CompressorBridge(
        name="compressor",
        config=cfg,
        optional=True,
        submodules={
            "kv_proj": LinearBridge(name="kv_proj"),
            "gate_proj": LinearBridge(name="gate_proj"),
            "kv_norm": RMSNormalizationBridge(name="kv_norm", config=cfg),
            "rotary_emb": RotaryEmbeddingBridge(name="rotary_emb", config=cfg),
            "indexer": GeneralizedComponent(
                name="indexer",
                optional=True,
                submodules={
                    "kv_proj": LinearBridge(name="kv_proj"),
                    "gate_proj": LinearBridge(name="gate_proj"),
                    "kv_norm": RMSNormalizationBridge(name="kv_norm", config=cfg),
                    "q_b_proj": LinearBridge(name="q_b_proj"),
                    "weights_proj": LinearBridge(name="weights_proj"),
                    "rotary_emb": RotaryEmbeddingBridge(name="rotary_emb", config=cfg),
                },
            ),
        },
    )


class DeepSeekV4ArchitectureAdapter(ArchitectureAdapter):
    """Adapter for ``DeepseekV4ForCausalLM`` (Flash and Pro variants)."""

    # The isolated component harness assumes a three-dimensional residual. V4's
    # mHC stack is four-dimensional, so parity is covered by integration tests and
    # verify_models' whole-model hook/text phases instead of isolated Phase 1.
    applicable_phases: list[int] = [2, 4]

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.uses_rms_norm = True
        self.cfg.final_rms = True
        self.cfg.rmsnorm_uses_offset = False
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.gated_mlp = True
        self.cfg.attn_implementation = "eager"

        # Folding/centering assumes one additive residual stream. Applying either
        # transform to mHC's learned collapse/expand path is not basis preserving.
        self.supports_fold_ln = False
        self.supports_center_writing_weights = False
        self.weight_processing_conversions = {}

        def hyper_connection(name: str) -> DeepseekV4HyperConnectionBridge:
            return DeepseekV4HyperConnectionBridge(
                name=name,
                config=self.cfg,
                submodules={
                    "input_norm": GeneralizedComponent(name="input_norm"),
                },
            )

        attention = GeneralizedComponent(
            name="self_attn",
            submodules={
                "q_a_proj": LinearBridge(name="q_a_proj"),
                "q_a_norm": RMSNormalizationBridge(name="q_a_norm", config=self.cfg),
                "q_b_proj": LinearBridge(name="q_b_proj"),
                "q_b_norm": GeneralizedComponent(name="q_b_norm"),
                "kv_proj": LinearBridge(name="kv_proj"),
                "kv_norm": RMSNormalizationBridge(name="kv_norm", config=self.cfg),
                "compressor": _compressor_bridge(self.cfg),
                "o_a_proj": GeneralizedComponent(name="o_a_proj"),
                "o_b_proj": LinearBridge(name="o_b_proj"),
            },
        )

        mlp = MoEBridge(
            name="mlp",
            config=self.cfg,
            submodules={
                "gate": GeneralizedComponent(name="gate"),
                "experts": GeneralizedComponent(name="experts"),
                "shared_experts": GatedMLPBridge(
                    name="shared_experts",
                    config=self.cfg,
                    submodules={
                        "gate": LinearBridge(name="gate_proj"),
                        "in": LinearBridge(name="up_proj"),
                        "out": LinearBridge(name="down_proj"),
                    },
                ),
            },
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": DeepseekV4BlockBridge(
                name="model.layers",
                config=self.cfg,
                submodules={
                    "attn_hc": hyper_connection("attn_hc"),
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "attn": attention,
                    "mlp_hc": hyper_connection("ffn_hc"),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "mlp": mlp,
                },
            ),
            "hc_head": GeneralizedComponent(
                name="model.hc_head",
                submodules={
                    "input_norm": GeneralizedComponent(name="input_norm"),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Force eager attention so the delegated attention path is deterministic."""
        model_kwargs["attn_implementation"] = "eager"

    def prepare_model(self, hf_model: Any) -> None:
        """Force eager attention on a pre-loaded model before installing bridges."""
        if hasattr(hf_model, "config"):
            hf_model.config._attn_implementation = "eager"
        model = getattr(hf_model, "model", None)
        if model is not None and hasattr(model, "layers"):
            for layer in model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"
