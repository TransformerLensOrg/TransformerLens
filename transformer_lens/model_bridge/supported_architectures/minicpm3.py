"""MiniCPM3 architecture adapter.

OpenBMB's MiniCPM3 (``MiniCPM3ForCausalLM``, native in transformers), e.g.
openbmb/MiniCPM3-4B: DeepSeek-V2's Multi-head Latent Attention (q LoRA +
compressed KV with decoupled rope) with every layer a dense gated MLP, plus
MiniCPM's muP scalings. Two things differ from the DeepSeek MLA family:

- RoPE is the half-split cos/sin variant. The HF rotary module returns a
  (cos, sin) tuple and the config has no ``rope_interleave``, so
  MLAAttentionBridge already takes the half-split path.
- Each sublayer output is multiplied by ``scale_depth / sqrt(num_hidden_layers)``
  before the residual add, so blocks are scaled-residual MLA blocks.
"""

import math
from typing import Any, Dict, Optional

from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    ScaledResidualBlockBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.deepseek_v2 import (
    DeepSeekMLAFamilyArchitectureAdapter,
)


class _MiniCPM3BlockBridge(ScaledResidualBlockBridge):
    """Scaled-residual block around MLA attention.

    MLA has no single input that becomes Q/K/V, so the q/k/v/attn input aliases
    are dropped the same way MLABlockBridge drops them.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        residual_contribution_scale: float = 1.0,
    ):
        super().__init__(
            name,
            config=config,
            submodules=submodules,
            residual_contribution_scale=residual_contribution_scale,
        )
        if self.hook_aliases is BlockBridge.hook_aliases:
            self.hook_aliases = dict(self.hook_aliases)
        for alias in ("hook_q_input", "hook_k_input", "hook_v_input", "hook_attn_in"):
            self.hook_aliases.pop(alias, None)


class MiniCPM3ArchitectureAdapter(DeepSeekMLAFamilyArchitectureAdapter):
    """Architecture adapter for MiniCPM3ForCausalLM models.

    The three muP scalings are all applied by the HF forward:

    - embeddings are multiplied by ``scale_emb`` inside MiniCPM3ScaledWordEmbedding,
      so hook_embed already sees the product (no output_contribution_scale);
    - each sublayer output is multiplied by ``scale_depth / sqrt(num_hidden_layers)``
      (hook_attn_out / hook_mlp_out expose the product);
    - the final hidden state is divided by ``hidden_size / dim_model_base`` before
      lm_head (reproduced by output_logits_scale).
    """

    # LlamaTokenizer with add_bos_token=True (openbmb/MiniCPM3-4B).
    prepend_bos = True

    # MiniCPM3Config's defaults, so an absent attr means what it means to HF.
    _DEFAULT_SCALE_DEPTH = 1.4
    _DEFAULT_DIM_MODEL_BASE = 256

    def _build_component_mapping(self) -> dict:
        mapping = super()._build_component_mapping()
        scale_depth = float(getattr(self.cfg, "scale_depth", self._DEFAULT_SCALE_DEPTH))
        residual_scale = scale_depth / math.sqrt(self.cfg.n_layers)
        mapping["blocks"] = _MiniCPM3BlockBridge(
            name="model.layers",
            submodules=mapping["blocks"].submodules,
            residual_contribution_scale=residual_scale,
        )
        return mapping

    def _build_mlp_bridge(self):
        """Every layer is a plain gated MLP, as in Youtu."""
        return self._gated_mlp()

    def output_logits_scale(self) -> float:
        """Match MiniCPM3's ``lm_head(h / (hidden_size / dim_model_base))``.

        lm_head is bias-free, so dividing its output equals dividing its input.
        """
        dim_model_base = getattr(self.cfg, "dim_model_base", self._DEFAULT_DIM_MODEL_BASE)
        return float(dim_model_base) / self.cfg.d_model
