"""Tests for `set_use_split_qkv_input` / `set_use_attn_in` — independent Q/K/V residual copies."""
import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge

# Per-head einsum on 768-dim accumulates ~1e-4 float32 noise vs flat matmul.
_LOGIT_TOL = 1e-3


@pytest.fixture(scope="module")
def gpt2_bridge():
    return TransformerBridge.boot_transformers("gpt2", device="cpu")


def test_split_qkv_does_not_change_output(gpt2_bridge):
    """Toggling `use_split_qkv_input` routes through a per-head einsum; logits still match."""
    x = torch.arange(1, 9).unsqueeze(0)
    normal_output = gpt2_bridge(x)
    gpt2_bridge.set_use_split_qkv_input(True)
    assert gpt2_bridge.cfg.use_split_qkv_input is True
    try:
        split_output = gpt2_bridge(x)
    finally:
        gpt2_bridge.set_use_split_qkv_input(False)
    assert torch.allclose(normal_output, split_output, atol=_LOGIT_TOL)


def test_use_attn_in_does_not_change_output(gpt2_bridge):
    """`use_attn_in` uses one 4D residual copy for all three Q/K/V; logits still match."""
    x = torch.arange(1, 9).unsqueeze(0)
    normal_output = gpt2_bridge(x)
    gpt2_bridge.set_use_attn_in(True)
    assert gpt2_bridge.cfg.use_attn_in is True
    try:
        split_output = gpt2_bridge(x)
    finally:
        gpt2_bridge.set_use_attn_in(False)
    assert torch.allclose(normal_output, split_output, atol=_LOGIT_TOL)


def test_qkv_input_shapes(gpt2_bridge):
    """Hooks fire at `[batch, pos, n_heads (or n_kv_heads), d_model]`."""
    x = torch.arange(1, 9).unsqueeze(0)
    gpt2_bridge.set_use_split_qkv_input(True)
    captured: dict = {}

    def cap(name):
        def _hook(tensor, hook):
            captured[name] = tensor.detach().clone()
            return tensor

        return _hook

    try:
        gpt2_bridge.run_with_hooks(
            x,
            fwd_hooks=[
                ("blocks.0.attn.hook_q_input", cap("q")),
                ("blocks.0.attn.hook_k_input", cap("k")),
                ("blocks.0.attn.hook_v_input", cap("v")),
            ],
        )
    finally:
        gpt2_bridge.set_use_split_qkv_input(False)
    n_heads = gpt2_bridge.cfg.n_heads
    n_kv_heads = getattr(gpt2_bridge.cfg, "n_key_value_heads", None) or n_heads
    d_model = gpt2_bridge.cfg.d_model
    for name, expected_h in (("q", n_heads), ("k", n_kv_heads), ("v", n_kv_heads)):
        assert name in captured, f"hook_{name}_input did not fire"
        assert captured[name].shape == (1, 8, expected_h, d_model), (
            f"hook_{name}_input shape {tuple(captured[name].shape)} != "
            f"(1, 8, {expected_h}, {d_model})"
        )


def test_split_qkv_independence(gpt2_bridge):
    """Core split-qkv guarantee: patching `hook_q_input` must not affect K or V.

    Captures post-projection K and V with and without a Q-input patch;
    asserts the K and V tensors are bitwise identical across runs.
    """
    x = torch.arange(1, 9).unsqueeze(0)
    gpt2_bridge.set_use_split_qkv_input(True)
    try:
        baseline: dict = {}

        def cap_baseline(name):
            def _hook(tensor, hook):
                baseline[name] = tensor.detach().clone()
                return tensor

            return _hook

        gpt2_bridge.run_with_hooks(
            x,
            fwd_hooks=[
                ("blocks.0.attn.k.hook_out", cap_baseline("k")),
                ("blocks.0.attn.v.hook_out", cap_baseline("v")),
            ],
        )

        patched: dict = {}

        def cap_patched(name):
            def _hook(tensor, hook):
                patched[name] = tensor.detach().clone()
                return tensor

            return _hook

        def zero_q_input(tensor, hook):
            return torch.zeros_like(tensor)

        gpt2_bridge.run_with_hooks(
            x,
            fwd_hooks=[
                ("blocks.0.attn.hook_q_input", zero_q_input),
                ("blocks.0.attn.k.hook_out", cap_patched("k")),
                ("blocks.0.attn.v.hook_out", cap_patched("v")),
            ],
        )
    finally:
        gpt2_bridge.set_use_split_qkv_input(False)
    for name in ("k", "v"):
        assert torch.equal(baseline[name], patched[name]), (
            f"Patching hook_q_input leaked into {name}: max diff "
            f"{(baseline[name] - patched[name]).abs().max().item():.2e}"
        )


def test_split_qkv_mutual_exclusivity(gpt2_bridge):
    """`use_attn_in` and `use_split_qkv_input` are mutually exclusive."""
    gpt2_bridge.set_use_split_qkv_input(True)
    try:
        with pytest.raises(ValueError, match="mutually exclusive"):
            gpt2_bridge.set_use_attn_in(True)
    finally:
        gpt2_bridge.set_use_split_qkv_input(False)
    gpt2_bridge.set_use_attn_in(True)
    try:
        with pytest.raises(ValueError, match="mutually exclusive"):
            gpt2_bridge.set_use_split_qkv_input(True)
    finally:
        gpt2_bridge.set_use_attn_in(False)


def test_split_qkv_applicability_raises_on_unsupported(monkeypatch, gpt2_bridge):
    """Setter raises when the bridge has no attention class that supports the fork."""
    from torch import nn

    class _FakeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Identity()

    fake_blocks = nn.ModuleList([_FakeBlock()])
    monkeypatch.setattr(gpt2_bridge, "blocks", fake_blocks, raising=True)
    with pytest.raises(NotImplementedError, match="use_split_qkv_input"):
        gpt2_bridge.set_use_split_qkv_input(True)
    with pytest.raises(NotImplementedError, match="use_attn_in"):
        gpt2_bridge.set_use_attn_in(True)


def test_block_level_hook_alias_parity(gpt2_bridge):
    """HookedTransformer parity: `blocks[i].hook_{attn_in,q_input,k_input,v_input}`
    resolve to the four *independent* HookPoints on the attention bridge.
    """
    gpt2_bridge.enable_compatibility_mode(no_processing=True)
    block = gpt2_bridge.blocks[0]
    attn = block.attn
    assert block.hook_attn_in is attn.hook_attn_in
    assert block.hook_q_input is attn.hook_q_input
    assert block.hook_k_input is attn.hook_k_input
    assert block.hook_v_input is attn.hook_v_input
    # Independence is preserved — each alias resolves to a distinct HookPoint.
    assert (
        len(
            {
                id(x)
                for x in (
                    attn.hook_q_input,
                    attn.hook_k_input,
                    attn.hook_v_input,
                    attn.hook_attn_in,
                )
            }
        )
        == 4
    )
