"""DeepSeek V2 adapter and the shared DeepSeek-MLA family base (DeepSeek V2/V3,
GLM-MoE-DSA, GLM-4.7-Flash; Youtu via V2); per-member differences are declarative.

DeepSeek V2 support covers DeepSeek-V2, DeepSeek-V2-Lite, and DeepSeek-Coder-V2
(all use DeepseekV2ForCausalLM).
"""

from typing import Any, Dict, Optional, Type

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    LinearBridge,
    MLAAttentionBridge,
    MLABlockBridge,
    MoEBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class DeepSeekMLAFamilyArchitectureAdapter(ArchitectureAdapter):
    """Shared base for DeepSeek-style Multi-head Latent Attention + MoE decoders.

    Members share LoRA-compressed attention (q_a/q_b and kv_a/kv_b projections with
    decoupled RoPE), RMSNorm, no biases, MLA weights kept in HF layout (no QKVO
    rearrangements, no LN folding), and a routed MoE whose router/shared experts are
    absent on dense layers. Divergence is declarative:

    - ``q_lora_optional``: some checkpoints (DeepSeek-V2-Lite; GigaChat3 on the V3
      class) set q_lora_rank=None — no compressed-Q pair, direct q_proj instead — so
      the whole Q path is optional; ``_build_q_a_layernorm`` picks the norm's type.
      Families whose checkpoints always compress Q keep the projections REQUIRED so
      a genuinely missing weight fails loudly; never relax them family-wide.
    - ``attention_cls``: GLM-MoE-DSA swaps in its sparse-attention bridge.
    - ``_build_mlp_bridge`` / ``_build_router`` seams: Youtu maps all-dense MLPs;
      GLM-4.7-Flash uses a MoERouterBridge router.
    """

    attention_cls: Type[MLAAttentionBridge] = MLAAttentionBridge
    q_lora_optional: bool = False
    # Sets cfg.attn_implementation="eager" (member-specific WHY at each override).
    eager_attention: bool = False
    # Sets cfg.default_prepend_bos; None leaves the config default untouched.
    prepend_bos: Optional[bool] = None

    def __init__(self, cfg: Any) -> None:
        """Apply the family config knobs and build the shared mapping."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()
        if self.eager_attention:
            self.cfg.attn_implementation = "eager"
        if self.prepend_bos is not None:
            self.cfg.default_prepend_bos = self.prepend_bos

        # MLA has no per-head q/k/v to fold into; skip LN folding.
        self.supports_fold_ln = False

        # MLA weights keep their HF layout; no QKVO rearrangements apply.
        self.weight_processing_conversions = {}

        self.component_mapping = self._build_component_mapping()

    def _build_component_mapping(self) -> dict:
        """PRE-NORM mapping: ln1 = input_layernorm (before attention),
        ln2 = post_attention_layernorm (before MLP)."""
        return {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": MLABlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": self.attention_cls(
                        name="self_attn",
                        config=self.cfg,
                        submodules=self._build_attention_submodules(),
                    ),
                    "mlp": self._build_mlp_bridge(),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def _build_attention_submodules(self) -> Dict[str, GeneralizedComponent]:
        """MLA projection submodules; the Q path follows ``q_lora_optional``."""
        optional = self.q_lora_optional
        submodules: Dict[str, GeneralizedComponent] = {
            "q_a_proj": LinearBridge(name="q_a_proj", optional=optional),
            "q_a_layernorm": self._build_q_a_layernorm(optional),
            "q_b_proj": LinearBridge(name="q_b_proj", optional=optional),
        }
        if optional:
            # Direct Q projection, mutually exclusive with the compressed pair above;
            # MLAAttentionBridge.forward picks the path from q_lora_rank.
            submodules["q_proj"] = LinearBridge(name="q_proj", optional=True)
        submodules.update(
            {
                # KV path — always compressed, present in every family member.
                "kv_a_proj_with_mqa": LinearBridge(name="kv_a_proj_with_mqa"),
                "kv_a_layernorm": RMSNormalizationBridge(name="kv_a_layernorm", config=self.cfg),
                "kv_b_proj": LinearBridge(name="kv_b_proj"),
                "o": LinearBridge(name="o_proj"),
            }
        )
        return submodules

    def _build_q_a_layernorm(self, optional: bool) -> GeneralizedComponent:
        """Compressed-Q norm. Its forward is called directly by MLAAttentionBridge, so
        on the optional (direct-Q) path a plain GeneralizedComponent suffices; V3
        overrides to keep the full norm bridge."""
        if optional:
            return GeneralizedComponent(name="q_a_layernorm", optional=True)
        return RMSNormalizationBridge(name="q_a_layernorm", config=self.cfg)

    def _build_router(self) -> GeneralizedComponent:
        """Router is a custom Module, not nn.Linear; absent on dense layers."""
        return GeneralizedComponent(name="gate", optional=True)

    def _build_mlp_bridge(self):
        """Routed MoE with optional shared experts — on dense layers (e.g. idx <
        first_k_dense_replace) router and shared_experts are absent, so setup skips
        the optional submodules; Youtu (all-dense) overrides.

        Dense-prefix layers bind as gated MLPs with neuron-basis
        hook_pre/hook_pre_linear/hook_post (#1645).
        """
        return MoEBridge(
            name="mlp",
            config=self.cfg,
            sparse_required=("gate",),
            submodules={
                "gate": self._build_router(),
                "shared_experts": self._gated_mlp(name="shared_experts", optional=True),
                # Dense-layer projections (present only on the dense layers of
                # this interleaved stack); their presence is what makes MoEBridge
                # bind gated-MLP neuron hooks there (#1645).
                "dense_gate": LinearBridge(name="gate_proj", optional=True),
                "dense_in": LinearBridge(name="up_proj", optional=True),
                "dense_out": LinearBridge(name="down_proj", optional=True),
            },
        )


class DeepSeekV2ArchitectureAdapter(DeepSeekMLAFamilyArchitectureAdapter):
    """Architecture adapter for DeepSeek V2 / V2-Lite / Coder-V2 models.

    Uses RMSNorm, MLA with compressed Q/KV projections (or direct Q projection
    when q_lora_rank is None), partial RoPE, MoE on most layers (dense MLP on
    first few), and no biases.
    """

    _testing_eager = None

    # V2-Lite sets q_lora_rank=None: no q_a_proj/q_b_proj in the state_dict,
    # direct q_proj instead — MLAAttentionBridge.forward handles both paths.
    q_lora_optional = True
