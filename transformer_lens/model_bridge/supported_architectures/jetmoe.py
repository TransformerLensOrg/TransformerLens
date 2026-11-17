"""JetMoE architecture adapter.

MIT-IBM's JetMoE (``JetMoeForCausalLM``, native in transformers): the only
open at-scale Mixture-of-Attention-heads model — attention Q and output
projections are per-expert parallel 3D tensors behind a top-k router
(``experts``: JetMoeMoA), with a shared fused KV projection, alongside a
conventional parallel-experts MoE MLP. Both routers are hookable; the
mixers delegate to HF (per-expert 3D projections have no uniform
reconstruction, so no fold target exists either). ``attn.hook_z`` is the
per-head attention output; per-head results are computed from it and the
attention router's logits.
"""

from typing import TYPE_CHECKING, Any, Tuple

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MoEBridge,
    MoERouterBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)

if TYPE_CHECKING:
    from transformer_lens.ActivationCache import ActivationCache


class _JetMoeAttentionBridge(AttentionBridge):
    """Mixture-of-Attention: no separate q/k/v/o Linears to alias — Q and O
    live inside the per-expert MoA; only the shared fused KV is a Linear.

    Each token routes to ``top_k`` attention experts. Expert slot ``k`` owns heads
    ``k * n_kv_heads .. (k + 1) * n_kv_heads - 1``; their outputs go through that
    expert's ``output_linear`` and are weighted by its gate before the MoA bias is
    added. ``hook_z`` is the per-head attention output entering ``experts.reduce``.
    Per-head results depend on the routing, so there is no static ``W_O``: they
    are computed from ``hook_z`` and the router logits instead.
    """

    hook_aliases = {
        "hook_kv": "kv.hook_out",
    }
    # hook_result fires inside the wrapped experts.reduce when use_attn_result is on.
    supports_attn_result = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hook_z = HookPoint()

    def _moa(self) -> Any:
        """HF ``JetMoeMoA`` (``experts`` is itself bridged, for the router hooks)."""
        attention: Any = self.original_component
        experts = attention.experts
        original = getattr(experts, "original_component", None)
        return experts if original is None else original

    @property
    def b_O(self) -> torch.Tensor:
        """MoA output bias, added once after the expert outputs are merged."""
        return self._moa().bias

    def _routed_head_results(
        self, z: torch.Tensor, experts_idx: torch.Tensor, gates: torch.Tensor
    ) -> torch.Tensor:
        """``[..., n_heads, d_model]`` contributions from ``z`` ``[..., n_heads, d_head]``.

        ``experts_idx`` / ``gates`` are ``[..., top_k]``: the expert and gate of each slot.
        """
        moa = self._moa()
        weight = moa.output_linear.weight  # [num_experts, d_model, n_kv_heads * d_head]
        top_k = experts_idx.shape[-1]
        *lead, n_heads, d_head = z.shape
        n_kv = n_heads // top_k
        z_slots = z.reshape(*lead, top_k, n_kv, d_head)
        w_heads = weight.reshape(weight.shape[0], weight.shape[1], n_kv, d_head)
        out = z_slots.new_zeros(*lead, top_k, n_kv, weight.shape[1])
        for expert in torch.unique(experts_idx).tolist():
            mask = (experts_idx == expert).to(z.dtype)[..., None, None]
            out = out + mask * torch.einsum("...kjd,mjd->...kjm", z_slots, w_heads[expert])
        out = out * gates.to(out.dtype)[..., None, None]
        return out.reshape(*lead, n_heads, weight.shape[1])

    def head_results_from_cache(self, cache: "ActivationCache", layer: int) -> torch.Tensor:
        """Per-head results for ``layer``, routed as HF's ``JetMoeTopKGating`` does."""
        moa = self._moa()
        z = cache[f"blocks.{layer}.attn.hook_z"]
        logits = cache[f"blocks.{layer}.attn.experts.router.hook_out"]
        logits = logits.reshape(*z.shape[:-2], logits.shape[-1]).float()
        top_logits, experts_idx = logits.topk(moa.top_k, dim=-1)
        gates = torch.softmax(top_logits, dim=-1).to(z.dtype)
        return self._routed_head_results(z, experts_idx, gates)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        attention: Any = self.original_component
        moa = self._moa()
        original_reduce = moa.reduce

        def reduce(layer_input: torch.Tensor, topo_info: Tuple[Any, ...]) -> torch.Tensor:
            *lead, top_k, hidden = layer_input.shape
            d_head = attention.head_dim
            z = self.hook_z(layer_input.reshape(*lead, top_k * (hidden // d_head), d_head))
            if not getattr(self.config, "use_attn_result", False):
                return original_reduce(z.reshape(*lead, top_k, hidden), topo_info)
            experts_idx, gates = self._slot_routing(topo_info, lead, top_k, moa.num_experts)
            per_head = self.hook_result(self._routed_head_results(z, experts_idx, gates))
            return per_head.sum(dim=-2) + moa.bias

        moa.reduce = reduce
        try:
            return super().forward(*args, **kwargs)
        finally:
            del moa.reduce

    @staticmethod
    def _slot_routing(
        topo_info: Tuple[Any, ...], lead: list, top_k: int, num_experts: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-slot expert ids and gates from HF's expert-sorted routing topology."""
        index_sorted_experts, _, batch_gates, expert_size = topo_info
        sorted_experts = torch.repeat_interleave(
            torch.arange(num_experts, device=batch_gates.device),
            torch.as_tensor(expert_size, device=batch_gates.device),
        )
        experts_idx = torch.empty_like(sorted_experts)
        experts_idx[index_sorted_experts] = sorted_experts
        gates = torch.empty_like(batch_gates)
        gates[index_sorted_experts] = batch_gates
        return experts_idx.reshape(*lead, top_k), gates.reshape(*lead, top_k)


class JetMoeArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for JetMoeForCausalLM models."""

    # Per-expert 3D Q/O projections: nothing to fold a norm into.
    supports_fold_ln = False
    # TopKGating's forward sorts/scatters expert assignments and crashes on
    # the harness's isolated probes; routers stay hookable at runtime.
    component_test_skip_suffixes = ("mlp.gate", "attn.experts.router")
    # Delegated attention computes rotary inside HF; nothing to wire.
    _testing_eager = None
    _testing_wire_rotary = False

    def __init__(self, cfg: Any) -> None:
        """Initialize the JetMoE architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()

        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                config=self.cfg,
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    # MoA: delegate; the attention router and shared KV are
                    # hookable, per-expert Q/O stay inside the delegated MoA.
                    "attn": _JetMoeAttentionBridge(
                        name="self_attention",
                        config=self.cfg,
                        submodules={
                            "kv": LinearBridge(name="kv_proj"),
                            "experts": GeneralizedComponent(
                                name="experts",
                                submodules={
                                    # JetMoeTopKGating puts logits last in its 5-tuple.
                                    "router": MoERouterBridge(
                                        name="router",
                                        logits_index=-1,
                                        weights_index=None,
                                        indices_index=None,
                                    ),
                                },
                            ),
                        },
                        maintain_native_attention=True,
                    ),
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": MoERouterBridge(
                                name="router",
                                logits_index=-1,
                                weights_index=None,
                                indices_index=None,
                            ),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
