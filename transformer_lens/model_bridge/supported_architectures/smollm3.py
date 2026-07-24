"""SmolLM3 architecture adapter.

SmolLM3 (the HuggingFaceTB SmolLM3 family, base and instruct) is a Llama-family
decoder. It pairs pre-norm RMSNorm blocks with grouped-query attention (GQA), a
SwiGLU gated MLP, rotary position embeddings (RoPE), tied input and output
embeddings, and no biases on any projection. The one feature that sets it apart
from a plain Llama or Qwen2 decoder is NoPE (No Positional Encoding): RoPE is
skipped on a periodic subset of layers. That behaviour is the only piece of this
adapter that is not a near-verbatim clone of qwen2.py, and it is handled by the
small _SmolLM3AttentionBridge subclass below.
"""

from typing import Any

import torch

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


class _SmolLM3AttentionBridge(PositionEmbeddingsAttentionBridge):
    """Attention bridge that honours SmolLM3's per-layer NoPE setting.

    SmolLM3 disables rotary position embeddings on a periodic subset of layers
    (every no_rope_layer_interval-th layer, default every 4th, controlled by
    config.no_rope_layers). The wrapped HF SmolLM3Attention module records this
    choice as an integer flag use_rope: 1 means apply RoPE, 0 means this is a
    NoPE layer. HF honours the flag inside its own forward by only calling
    apply_rotary_pos_emb when use_rope is truthy.

    The base PositionEmbeddingsAttentionBridge reimplements attention so that all
    hook points fire at the right stage, and it applies RoPE whenever a
    position_embeddings tuple is passed. It never consults use_rope. On a NoPE
    layer that would rotate Q and K while native HF does not, diverging from the
    reference and failing logit-equivalence checks on roughly a quarter of the
    layers.

    To match HF exactly we suppress position_embeddings on NoPE layers before
    delegating to the base forward. The base forward only rotates when
    position_embeddings is not None, so passing None skips the rotation while
    every non-rotary hook (hook_q, hook_k, hook_v, hook_attn_scores,
    hook_pattern, hook_z) still fires identically. RoPE layers (use_rope == 1)
    are left untouched and behave exactly like the qwen2.py attention bridge.
    """

    # Nulls position_embeddings on NoPE layers by design.
    rope_optional = True

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Drop position_embeddings on NoPE layers, then run the base forward."""
        hf_attn = self.original_component
        # use_rope is 1 on RoPE layers and 0 on NoPE layers. Default to RoPE-on
        # when the attribute is somehow absent so standard layers never break.
        if hf_attn is not None and not getattr(hf_attn, "use_rope", 1):
            kwargs["position_embeddings"] = None
            # SmolLM3DecoderLayer (inherited from LlamaDecoderLayer) passes
            # position_embeddings as a keyword, so the line above is what fires
            # in practice. The positional branch below is defensive: if a caller
            # ever passes (hidden_states, position_embeddings, ...) positionally,
            # the second slot holds the (cos, sin) tuple, not a tensor, so we
            # null it out there too.
            if len(args) >= 2 and not isinstance(args[1], torch.Tensor):
                args = (args[0], None) + args[2:]
        return super().forward(*args, **kwargs)


class SmolLM3ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for SmolLM3 models.

    SmolLM3 is a pre-norm decoder with RMSNorm, grouped-query attention (GQA),
    a SwiGLU gated MLP, rotary position embeddings (RoPE), tied input and output
    embeddings, and no biases on any projection. The block shape matches Llama
    and Qwen2 exactly, so the component mapping and weight conversions mirror
    qwen2.py.

    NoPE (No Positional Encoding): SmolLM3 disables RoPE on every
    no_rope_layer_interval-th layer (default every 4th) via config.no_rope_layers.
    That per-layer toggle lives inside HF's SmolLM3Attention.forward, but the
    bridge reimplements attention and would otherwise rotate Q and K on those
    layers. The _SmolLM3AttentionBridge subclass handles it by suppressing
    position embeddings on NoPE layers, so the reimplemented attention matches HF.

    No Q/K normalization: unlike Qwen3, SmolLM3 has no per-head Q or K RMSNorm,
    so the attention block uses the plain q/k/v/o submodules.

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    SmolLM3 models do NOT have biases on any linear layers:

    - blocks.{i}.attn.b_Q - No bias on query projection
    - blocks.{i}.attn.b_K - No bias on key projection
    - blocks.{i}.attn.b_V - No bias on value projection
    - blocks.{i}.attn.b_O - No bias on output projection
    - blocks.{i}.mlp.b_in - No bias on MLP input (up_proj)
    - blocks.{i}.mlp.b_gate - No bias on MLP gate projection
    - blocks.{i}.mlp.b_out - No bias on MLP output (down_proj)
    - blocks.{i}.ln1.b - RMSNorm has no bias
    - blocks.{i}.ln2.b - RMSNorm has no bias
    - ln_final.b - RMSNorm has no bias

    Weight processing must handle these missing biases gracefully using
    ProcessWeights._safe_get_tensor() or by checking for None values.
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the SmolLM3 architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()

        self.cfg.default_prepend_bos = False
        # The bridge reimplements attention and reads output_attentions, so the
        # HF model must run in eager mode for the scores and pattern hooks to
        # match the reference. Set it on cfg so weight processing and
        # setup_component_testing agree without relying on boot()'s default.
        self.cfg.attn_implementation = "eager"

        # Standard separate q_proj/k_proj/v_proj/o_proj layout, GQA-aware. No
        # biases anywhere (attention_bias=False, mlp_bias=False), so no bias
        # conversions are needed.
        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                config=self.cfg,
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": _SmolLM3AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    "mlp": self._gated_mlp(),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }
