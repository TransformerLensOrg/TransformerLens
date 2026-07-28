"""EXAONE architecture adapter.

Supports LG AI Research's EXAONE-3.0 / 3.5 / Deep families
(``ExaoneForCausalLM``, trust_remote_code checkpoints). Llama-style RMSNorm +
RoPE + GQA + gated MLP under GPT-2-flavored module names: ``transformer.wte``,
``transformer.h[i]``, ``transformer.ln_f``, and a double-nested attention
(``attn.attention``). EXAONE-4.0 is a separate native-transformers
architecture (Exaone4ForCausalLM) and is not covered here.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class ExaoneArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for ExaoneForCausalLM (EXAONE-3.x) models.

    The remote modeling code follows current HF conventions (Cache API,
    position_embeddings tuples), so the standard bridges delegate cleanly.
    Naming quirks: attention projections live one level deeper than usual
    (``attn.attention.q_proj``), the gated MLP uses ``c_fc_0`` (gate) /
    ``c_fc_1`` (up) / ``c_proj`` (down), and rotary sits at
    ``transformer.rotary``.
    """

    _testing_lm_attr = "transformer"
    _testing_rotary_attr = "rotary"
    _testing_eager = "config"

    def __init__(self, cfg: Any) -> None:
        """Initialize the EXAONE architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()
        # Verified against LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct: no BOS prepended.
        self.cfg.default_prepend_bos = False

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            "rotary_emb": RotaryEmbeddingBridge(name="transformer.rotary", config=self.cfg),
            "blocks": BlockBridge(
                name="transformer.h",
                submodules={
                    "ln1": RMSNormalizationBridge(name="ln_1", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="ln_2", config=self.cfg),
                    "attn": PositionEmbeddingsAttentionBridge(
                        name="attn.attention",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="out_proj"),
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    "mlp": self._gated_mlp(gate="c_fc_0", up="c_fc_1", down="c_proj"),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="transformer.ln_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def prepare_model(self, hf_model: Any) -> Any:
        """Shim the EXAONE-3.x remote module for transformers >= 5.13, which renamed
        create_causal_mask's ``input_embeds`` kwarg to ``inputs_embeds``."""
        result = super().prepare_model(hf_model)
        model = result if result is not None else hf_model

        import sys

        from transformers.masking_utils import create_causal_mask

        module = sys.modules.get(type(model).__module__)
        if module is not None and hasattr(module, "create_causal_mask"):

            def _shim(*args: Any, **kwargs: Any) -> Any:
                if "input_embeds" in kwargs:
                    kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
                kwargs.pop("cache_position", None)
                return create_causal_mask(*args, **kwargs)

            setattr(module, "create_causal_mask", _shim)
        return result
