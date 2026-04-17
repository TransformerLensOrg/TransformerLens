"""Tests for `set_use_attn_result` — per-head pre-sum attention output hook."""
import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge

# Float32 with per-head einsum + sum rounding differs from flat matmul by ~1e-4
# on a 768-dim model over ~8 tokens. Legacy TL tolerates the same noise.
_LOGIT_TOL = 1e-3


@pytest.fixture(scope="module")
def gpt2_bridge():
    return TransformerBridge.boot_transformers("gpt2", device="cpu")


def test_use_attn_result_does_not_change_output(gpt2_bridge):
    """Toggling `use_attn_result` is an alternate compute path; logits should still match."""
    x = torch.arange(1, 9).unsqueeze(0)
    normal_output = gpt2_bridge(x)
    gpt2_bridge.set_use_attn_result(True)
    assert gpt2_bridge.cfg.use_attn_result is True
    try:
        split_output = gpt2_bridge(x)
    finally:
        gpt2_bridge.set_use_attn_result(False)
    assert torch.allclose(normal_output, split_output, atol=_LOGIT_TOL)


def test_hook_result_shape_and_sum_equals_hook_out(gpt2_bridge):
    """`hook_result` is per-head `[batch, pos, n_heads, d_model]`; summing across the head
    dim and adding `b_O` must equal `hook_out` (the summed attention output)."""
    x = torch.arange(1, 9).unsqueeze(0)
    gpt2_bridge.set_use_attn_result(True)
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
                ("blocks.0.attn.hook_result", cap("result")),
                ("blocks.0.attn.hook_out", cap("out")),
            ],
        )
    finally:
        gpt2_bridge.set_use_attn_result(False)
    assert "result" in captured, "hook_result did not fire with use_attn_result=True"
    assert "out" in captured, "hook_out did not fire"
    result = captured["result"]
    out = captured["out"]
    assert result.ndim == 4, f"hook_result expected 4D, got shape {tuple(result.shape)}"
    assert result.shape[:2] == out.shape[:2]
    assert result.shape[2] == gpt2_bridge.cfg.n_heads
    assert result.shape[3] == gpt2_bridge.cfg.d_model
    # Sum across heads and add b_O; must equal `hook_out`.
    b_O = gpt2_bridge.blocks[0].attn.o.original_component.bias
    b_O_term = b_O if b_O is not None else 0.0
    summed = result.sum(dim=-2) + b_O_term
    assert torch.allclose(summed, out, atol=_LOGIT_TOL), (
        f"sum(hook_result, dim=heads) + b_O != hook_out; max diff "
        f"{(summed - out).abs().max().item():.2e}"
    )


def test_hook_result_does_not_fire_when_flag_off(gpt2_bridge):
    """When `use_attn_result=False` the per-head einsum path is skipped, so
    `hook_result` must NOT fire (no activation captured)."""
    x = torch.arange(1, 9).unsqueeze(0)
    assert gpt2_bridge.cfg.use_attn_result is False
    fired = {"result": False}

    def _hook(tensor, hook):
        fired["result"] = True
        return tensor

    gpt2_bridge.run_with_hooks(x, fwd_hooks=[("blocks.0.attn.hook_result", _hook)])
    assert fired["result"] is False, (
        "hook_result fired when use_attn_result was False; the flag is "
        "supposed to skip the per-head computation."
    )


def test_use_attn_result_applicability_raises_on_unsupported(monkeypatch, gpt2_bridge):
    """If no attention bridge class supports the fine-grained fork, setter raises."""
    # Clone the bridge's blocks list to a stripped-down list whose attention
    # components are plain nn.Modules (not JointQKVAttentionBridge). We monkey-
    # patch `blocks` for the duration of this test so the applicability check
    # sees no supported attention bridge.
    from torch import nn

    class _FakeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Identity()

    fake_blocks = nn.ModuleList([_FakeBlock()])
    monkeypatch.setattr(gpt2_bridge, "blocks", fake_blocks, raising=True)
    with pytest.raises(NotImplementedError, match="use_attn_result"):
        gpt2_bridge.set_use_attn_result(True)
