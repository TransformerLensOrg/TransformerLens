import pytest

from transformer_lens import HookedTransformer

MODEL = "solu-1l"

prompt = "Hello World!"
model = HookedTransformer.from_pretrained(MODEL)
embed = lambda name: name == "hook_embed"


class Counter:
    def __init__(self):
        self.count = 0

    def inc(self, *args, **kwargs):
        self.count += 1


def test_hook_attaches_normally():
    c = Counter()
    _ = model.run_with_hooks(prompt, fwd_hooks=[(embed, c.inc)])
    assert all([len(hp.fwd_hooks) == 0 for _, hp in model.hook_dict.items()])
    assert c.count == 1
    model.remove_all_hook_fns(including_permanent=True)


def test_perma_hook_attaches_normally():
    c = Counter()
    model.add_perma_hook(embed, c.inc)
    assert len(model.hook_dict["hook_embed"].fwd_hooks) == 1
    model.run_with_hooks(prompt, fwd_hooks=[])
    assert len(model.hook_dict["hook_embed"].fwd_hooks) == 1
    assert c.count == 1
    model.remove_all_hook_fns(including_permanent=True)


def test_hook_context_manager():
    c = Counter()
    with model.hooks(fwd_hooks=[(embed, c.inc)]):
        assert len(model.hook_dict["hook_embed"].fwd_hooks) == 1
        model.forward(prompt)
    assert len(model.hook_dict["hook_embed"].fwd_hooks) == 0
    assert c.count == 1
    model.remove_all_hook_fns(including_permanent=True)


def test_nested_hook_context_manager():
    c = Counter()
    with model.hooks(fwd_hooks=[(embed, c.inc)]):
        assert len(model.hook_dict["hook_embed"].fwd_hooks) == 1
        model.forward(prompt)
        assert c.count == 1
        with model.hooks(fwd_hooks=[(embed, c.inc)]):
            assert len(model.hook_dict["hook_embed"].fwd_hooks) == 2
            model.forward(prompt)
            assert c.count == 3  # 2 from outer, 1 from inner
        assert len(model.hook_dict["hook_embed"].fwd_hooks) == 1
    assert len(model.hook_dict["hook_embed"].fwd_hooks) == 0
    assert c.count == 3
    model.remove_all_hook_fns(including_permanent=True)


def test_context_manager_run_with_cache():
    c = Counter()
    with model.hooks(fwd_hooks=[(embed, c.inc)]):
        assert len(model.hook_dict["hook_embed"].fwd_hooks) == 1
        model.run_with_cache(prompt)
        assert len(model.hook_dict["hook_embed"].fwd_hooks) == 1
    assert len(model.hook_dict["hook_embed"].fwd_hooks) == 0
    assert c.count == 1
    model.remove_all_hook_fns(including_permanent=True)


def test_hook_context_manager_with_permanent_hook():
    c = Counter()
    model.add_perma_hook(embed, c.inc)
    assert len(model.hook_dict["hook_embed"].fwd_hooks) == 1
    with model.hooks(fwd_hooks=[(embed, c.inc)]):
        assert len(model.hook_dict["hook_embed"].fwd_hooks) == 2
        model.forward(prompt)
    assert len(model.hook_dict["hook_embed"].fwd_hooks) == 1
    assert c.count == 2  # 1 from permanent, 1 from context manager
    model.remove_all_hook_fns(including_permanent=True)


def test_nested_context_manager_with_failure():
    def fail_hook(z, hook):
        raise ValueError("fail")

    c = Counter()
    with model.hooks(fwd_hooks=[(embed, c.inc)]):
        with pytest.raises(ValueError):
            with model.hooks(fwd_hooks=[(embed, fail_hook)]):
                assert len(model.hook_dict["hook_embed"].fwd_hooks) == 2
                model.forward(prompt)
        assert len(model.hook_dict["hook_embed"].fwd_hooks) == 1
        assert c.count == 1
    assert len(model.hook_dict["hook_embed"].fwd_hooks) == 0
    model.remove_all_hook_fns(including_permanent=True)


def test_reset_hooks_in_context_manager():
    c = Counter()
    with model.hooks(fwd_hooks=[(embed, c.inc)]):
        assert len(model.hook_dict["hook_embed"].fwd_hooks) == 1
        model.reset_hooks()
        assert len(model.hook_dict["hook_embed"].fwd_hooks) == 0
    assert len(model.hook_dict["hook_embed"].fwd_hooks) == 0
    model.remove_all_hook_fns(including_permanent=True)


def test_remove_hook():
    c = Counter()
    model.add_perma_hook(embed, c.inc)
    assert len(model.hook_dict["hook_embed"].fwd_hooks) == 1  # 1 after adding
    model.remove_all_hook_fns()
    assert (
        len(model.hook_dict["hook_embed"].fwd_hooks) == 1
    )  # permanent not removed without flag
    model.remove_all_hook_fns(including_permanent=True)
    assert len(model.hook_dict["hook_embed"].fwd_hooks) == 0  # removed now
    model.run_with_hooks(prompt, fwd_hooks=[])
    assert c.count == 0
    model.remove_all_hook_fns(including_permanent=True)


def test_conditional_hooks():
    """Test that it's only possible to add certain hooks when certain conditions are met"""

    def identity_hook(z, hook):
        return z

    model.reset_hooks()
    model.set_use_attn_result(False)
    with pytest.raises(AssertionError):
        model.add_hook("blocks.0.attn.hook_result", identity_hook)

    model.reset_hooks()
    model.set_use_split_qkv_input(False)
    with pytest.raises(AssertionError):
        model.add_hook("blocks.0.hook_q_input", identity_hook)

    # now when we set these conditions to true, should be no errors!

    model.reset_hooks()
    model.set_use_attn_result(True)
    model.add_hook("blocks.0.attn.hook_result", identity_hook)

    model.reset_hooks()
    model.set_use_split_qkv_input(True)
    model.add_hook("blocks.0.hook_q_input", identity_hook)

    # check that things are the right shape

    cache = model.run_with_cache(
        prompt,
        names_filter=lambda x: x == "blocks.0.hook_q_input",
    )[1]

    assert len(cache) == 1, len(cache)
    assert "blocks.0.hook_q_input" in cache.keys(), cache.keys()
    assert cache["blocks.0.hook_q_input"].shape == (
        1,
        4,
        model.cfg.n_heads,
        model.cfg.d_model,
    ), cache["blocks.0.hook_q_input"].shape
