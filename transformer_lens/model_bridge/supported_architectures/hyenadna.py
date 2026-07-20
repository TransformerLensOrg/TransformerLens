"""HyenaDNA architecture adapter.

HazyResearch/Stanford's HyenaDNA (``HyenaDNAForCausalLM``, remote code):
genomic language models built on the Hyena operator — attention-free
long-convolution mixing (in_proj -> short conv + implicit modulated
filters -> gated multiplicative recombination -> out_proj). Blocks are
otherwise transformer-shaped (pre-LN, gelu fc1/fc2 MLP), stacked under
``hyena.backbone`` with LayerNorms and single-character DNA vocab.

The mixer delegates to HF wholesale (the implicit filter has no
attention-shaped reconstruction); in_proj/out_proj are wrapped for
hooks. The HF port ships no generate() (not a GenerationMixin), so
generation phases are excluded.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    NormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.mlp import MLPBridge


class _HyenaBlockBridge(BlockBridge):
    """HyenaBlock takes and returns a bare tensor, and the backbone's minimal
    ``layer(hidden_states)`` call looks exactly like a standalone block call —
    the tuple-normalizing heuristic would feed block 1 a one-element tuple.
    The alias set drops BlockBridge's attention names (no attention here) and
    exposes the mixer instead; hook_resid_mid is ln2's input (pre-norm block)."""

    hook_aliases = {
        "hook_resid_pre": "hook_in",
        "hook_resid_mid": "ln2.hook_in",
        "hook_resid_post": "hook_out",
        "hook_mixer_in": "mixer.hook_in",
        "hook_mixer_out": "mixer.hook_out",
        "hook_mlp_in": "mlp.in.hook_in",
        "hook_mlp_out": "mlp.out.hook_out",
    }

    @staticmethod
    def _is_standalone_hidden_state_call(args: tuple, kwargs: dict) -> bool:
        return False


class HyenaDNAArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for HyenaDNAForCausalLM models."""

    applicable_phases: list[int] = [1, 2, 3]
    supports_generation: bool = False
    # Attention-free (Hyena convolutions): no q/k/v projections to fold into,
    # matching the other delegated-mixer adapters (rwkv, gidd, llada2_moe).
    supports_fold_ln: bool = False

    def __init__(self, cfg: Any) -> None:
        """Initialize the HyenaDNA architecture adapter."""
        super().__init__(cfg)

        self.cfg.normalization_type = "LN"
        self.cfg.uses_rms_norm = False
        self.cfg.positional_embedding_type = "none"
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        self.cfg.final_rms = False

        # No attention weights to rearrange.
        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="hyena.backbone.embeddings.word_embeddings"),
            "blocks": _HyenaBlockBridge(
                name="hyena.backbone.layers",
                config=self.cfg,
                submodules={
                    "ln1": NormalizationBridge(
                        name="norm1", config=self.cfg, use_native_layernorm_autograd=True
                    ),
                    # Hyena long-conv mixer: delegated (implicit filters have no
                    # attention-shaped reconstruction); projections hookable.
                    "mixer": GeneralizedComponent(
                        name="mixer",
                        submodules={
                            "in_proj": LinearBridge(name="in_proj"),
                            "out_proj": LinearBridge(name="out_proj"),
                        },
                    ),
                    "ln2": NormalizationBridge(
                        name="norm2", config=self.cfg, use_native_layernorm_autograd=True
                    ),
                    "mlp": MLPBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "in": LinearBridge(name="fc1"),
                            "out": LinearBridge(name="fc2"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(
                name="hyena.backbone.ln_f", config=self.cfg, use_native_layernorm_autograd=True
            ),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
