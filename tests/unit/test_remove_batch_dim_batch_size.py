"""`remove_batch_dim=True` only makes sense for a batch of one.

Every caching path must refuse a larger batch the way ``ActivationCache.remove_batch_dim``
does, instead of keeping the batch dim or silently caching only the first example.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from transformer_lens import HookedRootModule
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge

NAME = "blocks.0.hook_out"


@pytest.fixture(scope="module")
def bridge():
    cfg = TransformerBridgeConfig(
        d_model=32,
        d_head=16,
        n_heads=2,
        n_layers=1,
        n_ctx=8,
        d_vocab=16,
        d_mlp=64,
        act_fn="gelu",
        normalization_type="LN",
        seed=0,
    )
    return TransformerBridge.boot_native(cfg)


def _tokens(batch_size: int) -> torch.Tensor:
    return torch.randint(0, 16, (batch_size, 6))


@pytest.mark.parametrize("return_cache_object", [True, False])
def test_run_with_cache_refuses_batch_gt_1(bridge, return_cache_object):
    with torch.no_grad(), pytest.raises(AssertionError, match="batch size 2"):
        bridge.run_with_cache(
            _tokens(2),
            names_filter=NAME,
            remove_batch_dim=True,
            return_cache_object=return_cache_object,
        )


def test_run_with_cache_dict_removes_batch_dim_for_batch_of_one(bridge):
    tokens = _tokens(1)
    with torch.no_grad():
        _, batched = bridge.run_with_cache(tokens, names_filter=NAME, return_cache_object=False)
        _, squeezed = bridge.run_with_cache(
            tokens, names_filter=NAME, remove_batch_dim=True, return_cache_object=False
        )
    assert isinstance(squeezed, dict)
    torch.testing.assert_close(squeezed[NAME], batched[NAME][0])


def test_get_caching_hooks_refuses_batch_gt_1(bridge):
    _, fwd_hooks, _ = bridge.get_caching_hooks(names_filter=NAME, remove_batch_dim=True)
    with torch.no_grad(), bridge.hooks(fwd_hooks=fwd_hooks):
        with pytest.raises(AssertionError, match="batch size 2"):
            bridge.forward(_tokens(2))


def test_get_caching_hooks_removes_batch_dim_for_batch_of_one(bridge):
    tokens = _tokens(1)
    with torch.no_grad():
        _, ref = bridge.run_with_cache(tokens, names_filter=NAME)
    cache, fwd_hooks, _ = bridge.get_caching_hooks(names_filter=NAME, remove_batch_dim=True)
    with torch.no_grad(), bridge.hooks(fwd_hooks=fwd_hooks):
        bridge.forward(tokens)
    torch.testing.assert_close(cache[NAME], ref[NAME][0])


class _TinyHooked(HookedRootModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.hook_out = HookPoint()
        self.setup()

    def forward(self, x):
        return self.hook_out(self.linear(x))


@pytest.mark.parametrize("method", ["get_caching_hooks", "add_caching_hooks"])
def test_hooked_root_module_caching_hooks_refuse_batch_gt_1(method):
    model = _TinyHooked()
    if method == "get_caching_hooks":
        _, fwd_hooks, _ = model.get_caching_hooks(names_filter="hook_out", remove_batch_dim=True)
        for name, hook in fwd_hooks:
            model.mod_dict[name].add_hook(hook)
    else:
        model.add_caching_hooks(names_filter="hook_out", remove_batch_dim=True)
    try:
        with torch.no_grad(), pytest.raises(AssertionError, match="batch size 2"):
            model(torch.randn(2, 3, 4))
    finally:
        model.reset_hooks()


def test_hooked_root_module_run_with_cache_removes_batch_dim_for_batch_of_one():
    model = _TinyHooked()
    x = torch.randn(1, 3, 4)
    with torch.no_grad():
        out, cache = model.run_with_cache(x, remove_batch_dim=True)
    torch.testing.assert_close(cache["hook_out"], out[0])
