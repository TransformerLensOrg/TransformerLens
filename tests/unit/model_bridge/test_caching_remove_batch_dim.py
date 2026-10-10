"""remove_batch_dim on the caching paths.

The hook paths (``get_caching_hooks`` / ``add_caching_hooks``) drop a leading dimension of
size 1 and leave every other tensor alone. ``run_with_cache`` raises for a batch larger than 1
on both models, whether the bridge returns an ``ActivationCache`` or a plain dict, and takes the
batch size from its input rather than from the cached shapes.
"""

from __future__ import annotations

import pytest
import torch

from transformer_lens import HookedRootModule
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.bridge_core import _input_batch_size
from transformer_lens.utilities import remove_batch_dim

NAMES = ["embed.hook_in", "blocks.0.hook_out", "blocks.0.attn.hook_pattern"]


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


def _tokens(bridge, batch: int) -> torch.Tensor:
    return torch.randint(0, bridge.cfg.d_vocab, (batch, 6))


def _cached_with_hooks(bridge, tokens, **kwargs) -> dict:
    cache, fwd_hooks, _ = bridge.get_caching_hooks(names_filter=NAMES, **kwargs)
    with torch.no_grad(), bridge.hooks(fwd_hooks=fwd_hooks):
        bridge.forward(tokens)
    return cache


def test_hook_path_keeps_every_example_of_a_larger_batch(bridge):
    tokens = _tokens(bridge, 3)
    with torch.no_grad():
        _, ref = bridge.run_with_cache(tokens, names_filter=NAMES)

    cache = _cached_with_hooks(bridge, tokens, remove_batch_dim=True)

    for name in NAMES:
        torch.testing.assert_close(cache[name], ref[name])


def test_hook_path_drops_a_batch_of_one(bridge):
    tokens = _tokens(bridge, 1)
    with torch.no_grad():
        _, ref = bridge.run_with_cache(tokens, names_filter=NAMES, remove_batch_dim=True)

    cache = _cached_with_hooks(bridge, tokens, remove_batch_dim=True)

    for name in NAMES:
        torch.testing.assert_close(cache[name], ref[name])


def test_hook_path_leaves_a_flattened_hook_alone_at_batch_size_one(bridge):
    # MoE router and OPT ln2 hooks see [batch * pos, ...], so their leading dim is pos even
    # at batch size 1. A check on the leading dim being 1 would reject or mangle them.
    cache, fwd_hooks, _ = bridge.get_caching_hooks(
        names_filter="blocks.0.hook_out", remove_batch_dim=True
    )
    flattened = torch.randn(6, bridge.cfg.d_model)
    hook = bridge.get_hook_point("blocks.0.hook_out")

    _, save_hook = fwd_hooks[0]
    save_hook(flattened, hook=hook)

    torch.testing.assert_close(cache["blocks.0.hook_out"], flattened)


def test_add_caching_hooks_keeps_every_example_of_a_larger_batch(bridge):
    tokens = _tokens(bridge, 2)
    cache = bridge.add_caching_hooks(names_filter="blocks.0.hook_out", remove_batch_dim=True)
    try:
        with torch.no_grad():
            bridge.forward(tokens)
    finally:
        bridge.reset_hooks()

    assert cache["blocks.0.hook_out"].shape == (2, 6, bridge.cfg.d_model)


def test_run_with_cache_raises_for_a_larger_batch_as_a_plain_dict(bridge):
    tokens = _tokens(bridge, 2)

    with pytest.raises(AssertionError, match="batch size 2"):
        bridge.run_with_cache(
            tokens, names_filter=NAMES, remove_batch_dim=True, return_cache_object=False
        )


def test_run_with_cache_drops_a_batch_of_one_as_a_plain_dict(bridge):
    tokens = _tokens(bridge, 1)
    with torch.no_grad():
        _, as_cache = bridge.run_with_cache(tokens, names_filter=NAMES, remove_batch_dim=True)
        _, as_dict = bridge.run_with_cache(
            tokens, names_filter=NAMES, remove_batch_dim=True, return_cache_object=False
        )

    assert set(as_dict) == set(as_cache.cache_dict)
    for name in as_dict:
        torch.testing.assert_close(as_dict[name], as_cache[name])


class _Toy(HookedRootModule):
    def __init__(self):
        super().__init__()
        self.hook_act = HookPoint()
        self.setup()

    def forward(self, x):
        return self.hook_act(x) + 1


def test_hooked_root_module_hook_path_keeps_every_example_of_a_larger_batch():
    model = _Toy()
    x = torch.randn(3, 4)

    cache = model.add_caching_hooks(remove_batch_dim=True)
    model(x)
    model.reset_hooks()

    torch.testing.assert_close(cache["hook_act"], x)


def test_hooked_root_module_run_with_cache_raises_for_a_larger_batch():
    with pytest.raises(AssertionError, match="batch size 3"):
        _Toy().run_with_cache(torch.randn(3, 4), remove_batch_dim=True)


def test_hooked_root_module_drops_a_batch_of_one():
    model = _Toy()
    x = torch.randn(1, 4)

    _, cache = model.run_with_cache(x, remove_batch_dim=True)

    torch.testing.assert_close(cache["hook_act"], x[0])


def test_remove_batch_dim_utility_accepts_a_scalar():
    scalar = torch.tensor(1.0)

    assert remove_batch_dim(scalar) is scalar
    assert remove_batch_dim(torch.zeros(1, 3)).shape == (3,)
    assert remove_batch_dim(torch.zeros(2, 3)).shape == (2, 3)


@pytest.mark.parametrize("return_cache_object", [True, False])
def test_run_with_cache_takes_the_batch_size_from_its_input(
    bridge, monkeypatch, return_cache_object
):
    # Caching only flattened or position-indexed hooks makes the cached shapes suggest a
    # batch size equal to the sequence length.
    monkeypatch.setattr(ActivationCache, "_batch_size", lambda self: 6)

    with torch.no_grad():
        _, cache = bridge.run_with_cache(
            _tokens(bridge, 1),
            names_filter="blocks.0.hook_out",
            remove_batch_dim=True,
            return_cache_object=return_cache_object,
        )
        with pytest.raises(AssertionError, match="batch size 2"):
            bridge.run_with_cache(
                _tokens(bridge, 2),
                names_filter="blocks.0.hook_out",
                remove_batch_dim=True,
                return_cache_object=return_cache_object,
            )

    assert cache["blocks.0.hook_out"].shape == (6, bridge.cfg.d_model)


def test_activation_cache_remove_batch_dim_uses_a_given_batch_size():
    def position_indexed_cache() -> ActivationCache:
        return ActivationCache({"rel_pos_bias": torch.randn(5, 5)}, model=None, has_batch_dim=True)

    with pytest.raises(AssertionError, match="batch size 5"):
        position_indexed_cache().remove_batch_dim()
    with pytest.raises(AssertionError, match="batch size 2"):
        position_indexed_cache().remove_batch_dim(batch_size=2)

    cache = position_indexed_cache()
    bias = cache["rel_pos_bias"].clone()
    cache.remove_batch_dim(batch_size=1)

    assert not cache.has_batch_dim
    assert torch.equal(cache["rel_pos_bias"], bias)


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (("some text",), 1),
        ((["a", "b", "c"],), 3),
        ((torch.zeros(7, dtype=torch.long),), 1),
        ((torch.zeros(4, 7, dtype=torch.long),), 4),
        ((None, torch.zeros(2, 7, dtype=torch.long)), 2),
        (([1, 2, 3],), None),
        ((None,), None),
        ((torch.tensor(1),), None),
    ],
)
def test_input_batch_size(inputs, expected):
    assert _input_batch_size(*inputs) == expected


def test_hook_path_leaves_a_tensor_that_already_lost_its_batch_dimension_alone(bridge):
    # run_with_hooks(remove_batch_dim=True) hands every hook a tensor without the batch
    # dimension, so with the flag set on both, the caching hook must not squeeze a second time.
    name = "blocks.0.hook_out"
    cache, fwd_hooks, _ = bridge.get_caching_hooks(names_filter=name, remove_batch_dim=True)
    batch_free = torch.randn(6, bridge.cfg.d_model)

    _, save_hook = fwd_hooks[0]
    save_hook(batch_free, hook=bridge.get_hook_point(name))

    torch.testing.assert_close(cache[name], batch_free)
