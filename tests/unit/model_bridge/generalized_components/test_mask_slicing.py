"""_apply_reconstruct_attention_mask on full-sequence masks during cached decode.

Baichuan-13B fuses ALiBi slopes into a full [n_head, seq, seq] mask and its own
layer takes the LAST query row on decode; the bridge helper took the FIRST,
silently giving every decode step position-0's bias row.
"""

from __future__ import annotations

import torch

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)


def _attn_bridge():
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(
        make_bridge_cfg("LlamaForCausalLM", d_model=8, n_heads=2, d_head=4)
    )
    import copy

    return copy.deepcopy(adapter.component_mapping["blocks"].submodules["attn"])


def test_cached_decode_takes_the_last_mask_rows() -> None:
    bridge = _attn_bridge()
    kv_len, n_heads = 5, 2
    # Position-dependent full mask: row i is filled with i (stand-in for ALiBi
    # slopes, which differ per row).
    full_mask = (
        torch.arange(kv_len, dtype=torch.float32)
        .view(1, 1, kv_len, 1)
        .expand(1, n_heads, kv_len, kv_len)
    ).clone()
    scores = torch.zeros(1, n_heads, 1, kv_len)

    out = bridge._apply_reconstruct_attention_mask(scores, full_mask, seq_len=kv_len, q_seq_len=1)
    # The single decode query is the LAST position: its row is filled with 4.
    assert torch.all(out == 4.0), out


def test_prefill_full_mask_is_untouched() -> None:
    """When q_seq_len matches the mask rows nothing is sliced."""
    bridge = _attn_bridge()
    kv_len, n_heads = 4, 2
    mask = torch.zeros(1, n_heads, kv_len, kv_len)
    scores = torch.randn(1, n_heads, kv_len, kv_len)
    out = bridge._apply_reconstruct_attention_mask(scores, mask, seq_len=kv_len)
    torch.testing.assert_close(out, scores)


class TestPerLayerHeadCountHookShapes:
    """Hook shapes must not flip convention mid-model on per-layer head counts.

    Cfg-only dims no-op on OpenELM/laguna's non-conforming layers, leaving hook_q
    per-head on some layers and flat on others.
    """

    def test_nonconforming_layer_still_gets_per_head_hooks(self) -> None:
        import copy

        from transformer_lens.model_bridge.component_setup import setup_submodules

        adapter = ArchitectureAdapterFactory.select_architecture_adapter(
            make_bridge_cfg("LlamaForCausalLM", d_model=64, n_heads=8, d_head=8)
        )
        bridge = copy.deepcopy(adapter.component_mapping["blocks"].submodules["attn"])

        class _Attn(torch.nn.Module):
            """This layer has 4 heads; the cfg majority says 8."""

            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(64, 4 * 8, bias=False)
                self.k_proj = torch.nn.Linear(64, 4 * 8, bias=False)
                self.v_proj = torch.nn.Linear(64, 4 * 8, bias=False)
                self.o_proj = torch.nn.Linear(4 * 8, 64, bias=False)

            def forward(self, hidden_states, *args, **kwargs):
                value = self.v_proj(hidden_states)
                self.q_proj(hidden_states), self.k_proj(hidden_states)
                return self.o_proj(value), None

        module = _Attn()
        bridge.set_original_component(module)
        setup_submodules(bridge, adapter, module)
        bridge.setup_hook_compatibility()

        seen: dict = {}
        bridge.q.hook_out.add_hook(lambda t, hook: seen.__setitem__("q", tuple(t.shape)))
        bridge.o.hook_in.add_hook(lambda t, hook: seen.__setitem__("z", tuple(t.shape)))
        # Drive the projections directly: the conversions under test live on
        # the LinearBridge hook points, independent of the attention math.
        with torch.no_grad():
            bridge.q(torch.randn(2, 5, 64))
            bridge.o(torch.randn(2, 5, 32))

        # THIS layer's head count (4), not the cfg majority (8) — and not flat.
        assert seen.get("q") == (2, 5, 4, 8), seen
        assert seen.get("z") == (2, 5, 4, 8), seen


class TestPerLayerHeadDimHookShapes:
    """Per-layer d_head must not be factorized with the majority scalar.

    Minority widths divide evenly by both (16 = 4x4 = 2x8), so majority-only
    division mis-factorizes silently where cfg-only dims were visibly flat.
    """

    def _hook_shapes(self, width: int, layer_idx=1, head_dim=None) -> dict:
        import copy

        from transformer_lens.model_bridge.component_setup import setup_submodules

        adapter = ArchitectureAdapterFactory.select_architecture_adapter(
            make_bridge_cfg("LlamaForCausalLM", d_model=32, n_heads=4, d_head=8)
        )
        bridge = copy.deepcopy(adapter.component_mapping["blocks"].submodules["attn"])
        bridge.config.per_layer_head_dim = [8, 4]

        class _Attn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(32, width, bias=False)
                self.k_proj = torch.nn.Linear(32, width, bias=False)
                self.v_proj = torch.nn.Linear(32, width, bias=False)
                self.o_proj = torch.nn.Linear(width, 32, bias=False)
                if layer_idx is not None:
                    self.layer_idx = layer_idx
                if head_dim is not None:
                    self.head_dim = head_dim

            def forward(self, hidden_states, *args, **kwargs):
                return self.o_proj(self.v_proj(hidden_states)), None

        module = _Attn()
        bridge.set_original_component(module)
        setup_submodules(bridge, adapter, module)
        bridge.setup_hook_compatibility()

        seen: dict = {}
        bridge.q.hook_out.add_hook(lambda t, hook: seen.__setitem__("q", tuple(t.shape)))
        bridge.o.hook_in.add_hook(lambda t, hook: seen.__setitem__("z", tuple(t.shape)))
        with torch.no_grad():
            bridge.q(torch.randn(2, 5, 32))
            bridge.o(torch.randn(2, 5, width))
        return seen

    def test_minority_layer_factorizes_with_its_own_d_head(self) -> None:
        # Layer 1 is 4 heads x d_head 4. Majority-d_head division yields
        # (2, 5, 2, 8) — and the reshape guard PASSES on it, so nothing visible
        # breaks; every "head" is two real heads concatenated.
        shapes = self._hook_shapes(width=16, layer_idx=1)
        assert shapes == {"q": (2, 5, 4, 4), "z": (2, 5, 4, 4)}, shapes

    def test_majority_layer_unchanged(self) -> None:
        shapes = self._hook_shapes(width=32, layer_idx=0)
        assert shapes == {"q": (2, 5, 4, 8), "z": (2, 5, 4, 8)}, shapes

    def test_unresolvable_per_layer_geometry_stays_visibly_flat(self) -> None:
        # No layer_idx and no module head_dim: width division is disabled
        # rather than trusted with the majority — flat beats silently wrong.
        shapes = self._hook_shapes(width=16, layer_idx=None)
        assert shapes == {"q": (2, 5, 16), "z": (2, 5, 16)}, shapes

    def test_module_head_dim_resolves_when_index_missing(self) -> None:
        shapes = self._hook_shapes(width=16, layer_idx=None, head_dim=4)
        assert shapes == {"q": (2, 5, 4, 4), "z": (2, 5, 4, 4)}, shapes
