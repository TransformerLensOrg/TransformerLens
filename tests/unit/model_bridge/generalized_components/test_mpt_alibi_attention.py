"""Unit tests for MPTALiBiAttentionBridge.

Exercises the reimplemented MPT ALiBi attention against a live MptAttention
module — no Hub download, tiny programmatic config only.

Covers:
- Numerical match vs HF MptAttention.forward at atol=1e-5 (batch=2)
- ALiBi slicing when seq_len < max_seq_len
- Boolean causal mask enforces causal structure
- hook_q, hook_k, hook_v, hook_attn_scores, hook_pattern all fire
"""

import torch
import torch.nn as nn
from transformers import MptConfig
from transformers.models.mpt.modeling_mpt import MptAttention

from transformer_lens.model_bridge.generalized_components import LinearBridge
from transformer_lens.model_bridge.generalized_components.mpt_alibi_attention import (
    MPTALiBiAttentionBridge,
    _build_mpt_alibi_tensor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockConfig:
    """Minimal config for MPTALiBiAttentionBridge."""

    def __init__(self, n_heads: int, d_model: int) -> None:
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.n_key_value_heads = None
        self.positional_embedding_type = "alibi"


def _make_tiny_mpt_attention(d_model: int = 64, n_heads: int = 2) -> MptAttention:
    """Create a tiny MptAttention with random weights — no download."""
    cfg = MptConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=1,
        expansion_ratio=4,
        max_seq_len=32,
        vocab_size=256,
        no_bias=True,
    )
    return MptAttention(cfg)


def _make_split_fn(d_model: int) -> object:
    """Return a split_qkv_matrix function for MPT's Wqkv [3*d_model, d_model] layout.

    Splits row-wise along dim=0 (NOT the GPT-2 style dim=1 split).
    """

    def split_qkv(attn_component: object) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        w = attn_component.Wqkv.weight.detach().clone()  # [3*d_model, d_model]
        w_q, w_k, w_v = torch.chunk(w, 3, dim=0)  # each [d_model, d_model]

        def make_linear(weight: torch.Tensor) -> nn.Linear:
            lin = nn.Linear(d_model, d_model, bias=False, device=weight.device, dtype=weight.dtype)
            lin.weight = nn.Parameter(weight.contiguous())
            return lin

        return make_linear(w_q), make_linear(w_k), make_linear(w_v)

    return split_qkv


def _build_bridge(hf_attn: MptAttention) -> MPTALiBiAttentionBridge:
    """Wrap a live MptAttention in MPTALiBiAttentionBridge with the correct QKV split."""
    d_model = hf_attn.hidden_size
    n_heads = hf_attn.n_heads
    cfg = _MockConfig(n_heads=n_heads, d_model=d_model)

    bridge = MPTALiBiAttentionBridge(
        name="attn",
        config=cfg,
        split_qkv_matrix=_make_split_fn(d_model),
        submodules={
            "qkv": LinearBridge(name="Wqkv"),
            "o": LinearBridge(name="out_proj"),
        },
    )
    bridge.set_original_component(hf_attn)
    return bridge


def _make_inputs(
    d_model: int,
    n_heads: int,
    max_seq_len: int,
    seq_len: int,
    batch_size: int = 2,
) -> dict:
    """Build matching inputs for both HF MptAttention and bridge forward."""
    hidden = torch.randn(batch_size, seq_len, d_model)
    # position_bias: [n_heads, 1, max_seq_len] as MptModel.forward produces
    position_bias = _build_mpt_alibi_tensor(n_heads, max_seq_len)
    # bool causal mask: [batch, 1, seq, seq]
    causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    causal_mask = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    return {
        "hidden": hidden,
        "position_bias": position_bias,
        "causal_mask": causal_mask,
    }


# ---------------------------------------------------------------------------
# Numerical match against HF MptAttention.forward
# ---------------------------------------------------------------------------


class TestMPTALiBiMatchesHF:
    """Bridge output must numerically match HF MptAttention.forward."""

    def test_forward_matches_hf_batch2(self) -> None:
        """Primary correctness test: batch_size=2, seq_len=8.

        Uses batch_size >= 2 to catch any latent batch-broadcast bug in the
        position_bias unsqueeze(0) path.
        """
        hf_attn = _make_tiny_mpt_attention(d_model=64, n_heads=2)
        hf_attn.eval()
        bridge = _build_bridge(hf_attn)

        inputs = _make_inputs(d_model=64, n_heads=2, max_seq_len=32, seq_len=8, batch_size=2)

        with torch.no_grad():
            hf_out, *_ = hf_attn(
                inputs["hidden"],
                position_bias=inputs["position_bias"],
                attention_mask=inputs["causal_mask"],
            )
            bridge_out, *_ = bridge(
                inputs["hidden"],
                position_bias=inputs["position_bias"],
                attention_mask=inputs["causal_mask"],
            )

        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff:.2e} (threshold 1e-5)"

    def test_forward_matches_hf_batch1(self) -> None:
        """Single-batch sanity check — no batch-dim interaction."""
        hf_attn = _make_tiny_mpt_attention(d_model=64, n_heads=2)
        hf_attn.eval()
        bridge = _build_bridge(hf_attn)

        inputs = _make_inputs(d_model=64, n_heads=2, max_seq_len=32, seq_len=6, batch_size=1)

        with torch.no_grad():
            hf_out, *_ = hf_attn(
                inputs["hidden"],
                position_bias=inputs["position_bias"],
                attention_mask=inputs["causal_mask"],
            )
            bridge_out, *_ = bridge(
                inputs["hidden"],
                position_bias=inputs["position_bias"],
                attention_mask=inputs["causal_mask"],
            )

        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff:.2e}"


# ---------------------------------------------------------------------------
# ALiBi slicing: seq_len < max_seq_len
# ---------------------------------------------------------------------------


class TestALiBiSlicing:
    """position_bias covers max_seq_len; bridge must slice to current kv_len."""

    def test_short_seq_slicing_no_error(self) -> None:
        """seq_len=4 with max_seq_len=32: bridge must slice without error."""
        hf_attn = _make_tiny_mpt_attention(d_model=64, n_heads=2)
        hf_attn.eval()
        bridge = _build_bridge(hf_attn)

        # max_seq_len=32, but only use seq_len=4
        inputs = _make_inputs(d_model=64, n_heads=2, max_seq_len=32, seq_len=4, batch_size=2)

        with torch.no_grad():
            out, *_ = bridge(
                inputs["hidden"],
                position_bias=inputs["position_bias"],
                attention_mask=inputs["causal_mask"],
            )

        assert out.shape == (2, 4, 64)
        assert not torch.isnan(out).any()

    def test_slicing_matches_hf_short_seq(self) -> None:
        """Bridge output at seq_len=4 must match HF at same seq_len."""
        hf_attn = _make_tiny_mpt_attention(d_model=64, n_heads=2)
        hf_attn.eval()
        bridge = _build_bridge(hf_attn)

        inputs = _make_inputs(d_model=64, n_heads=2, max_seq_len=32, seq_len=4, batch_size=2)

        with torch.no_grad():
            hf_out, *_ = hf_attn(
                inputs["hidden"],
                position_bias=inputs["position_bias"],
                attention_mask=inputs["causal_mask"],
            )
            bridge_out, *_ = bridge(
                inputs["hidden"],
                position_bias=inputs["position_bias"],
                attention_mask=inputs["causal_mask"],
            )

        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Short-seq bridge vs HF diff = {max_diff:.2e}"


# ---------------------------------------------------------------------------
# Boolean mask enforces causal structure
# ---------------------------------------------------------------------------


class TestBoolMaskCausalStructure:
    """Boolean causal mask must zero out attention to future tokens."""

    def test_bool_mask_applies_causal_structure(self) -> None:
        """Attention pattern must be lower-triangular under a strict causal bool mask.

        Uses batch_size=2 to also exercise the bool-mask path with batched input.
        """
        d_model, n_heads, seq_len, batch_size = 64, 2, 6, 2
        hf_attn = _make_tiny_mpt_attention(d_model=d_model, n_heads=n_heads)
        hf_attn.eval()
        bridge = _build_bridge(hf_attn)

        hidden = torch.randn(batch_size, seq_len, d_model)
        position_bias = _build_mpt_alibi_tensor(n_heads, seq_len)
        causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        causal_mask = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        pattern_captured: dict[str, torch.Tensor] = {}

        def capture_pattern(tensor, hook):
            pattern_captured["pattern"] = tensor.detach().clone()
            return tensor

        bridge.hook_pattern.add_hook(capture_pattern)

        with torch.no_grad():
            bridge(hidden, position_bias=position_bias, attention_mask=causal_mask)

        assert "pattern" in pattern_captured
        pattern = pattern_captured["pattern"]  # [batch, n_heads, seq, seq]
        # Upper triangle (above diagonal) must be zero after causal masking
        upper = torch.triu(pattern[0, 0], diagonal=1)
        assert (upper.abs() < 1e-6).all(), "Future positions must have zero attention weight"


# ---------------------------------------------------------------------------
# Hook coverage
# ---------------------------------------------------------------------------


class TestHooksFire:
    """All required hooks must fire during a single forward pass."""

    def _run_forward_with_hooks(self, hook_names: list[str]) -> dict[str, torch.Tensor]:
        d_model, n_heads, seq_len = 64, 2, 8
        hf_attn = _make_tiny_mpt_attention(d_model=d_model, n_heads=n_heads)
        hf_attn.eval()
        bridge = _build_bridge(hf_attn)

        captured: dict[str, torch.Tensor] = {}

        def make_hook(name: str):
            def fn(tensor, hook):
                captured[name] = tensor.detach().clone()
                return tensor

            return fn

        for hook_name in hook_names:
            hook_obj: object = bridge
            for part in hook_name.split("."):
                hook_obj = getattr(hook_obj, part)
            hook_obj.add_hook(make_hook(hook_name))  # type: ignore[union-attr]

        inputs = _make_inputs(d_model=d_model, n_heads=n_heads, max_seq_len=32, seq_len=seq_len)
        with torch.no_grad():
            bridge(
                inputs["hidden"],
                position_bias=inputs["position_bias"],
                attention_mask=inputs["causal_mask"],
            )
        return captured

    def test_hook_q_fires(self) -> None:
        captured = self._run_forward_with_hooks(["q.hook_out"])
        assert "q.hook_out" in captured

    def test_hook_k_fires(self) -> None:
        captured = self._run_forward_with_hooks(["k.hook_out"])
        assert "k.hook_out" in captured

    def test_hook_v_fires(self) -> None:
        captured = self._run_forward_with_hooks(["v.hook_out"])
        assert "v.hook_out" in captured

    def test_hook_attn_scores_fires(self) -> None:
        captured = self._run_forward_with_hooks(["hook_attn_scores"])
        assert "hook_attn_scores" in captured

    def test_hook_pattern_fires(self) -> None:
        captured = self._run_forward_with_hooks(["hook_pattern"])
        assert "hook_pattern" in captured
