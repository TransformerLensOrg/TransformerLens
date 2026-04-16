"""Integration tests for MPT architecture adapter — Phase C.

Builds a tiny MptForCausalLM programmatically (no HF Hub download) and wraps
it in TransformerBridge via MPTArchitectureAdapter. Verifies:

- Forward output matches HF at max_diff < 1e-4
- Attention hooks fire: blocks.0.attn.hook_q/k/v, hook_attn_scores, hook_pattern
- MLP hooks fire: blocks.0.mlp.hook_in, blocks.0.mlp.hook_out
- Norm hooks fire: blocks.0.ln1.hook_out, blocks.0.ln2.hook_out
- Residual stream hooks fire: blocks.0.hook_resid_pre, blocks.0.hook_resid_post
"""

import pytest
import torch
from transformers import MptConfig
from transformers.models.mpt.modeling_mpt import MptForCausalLM

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.supported_architectures.mpt import (
    MPTArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Tiny model parameters — deterministic, no download, fits in <50 MB RAM
# ---------------------------------------------------------------------------
_D_MODEL = 64
_N_HEADS = 2
_N_LAYERS = 2
_D_MLP = 256
_D_VOCAB = 256
_MAX_SEQ_LEN = 32


def _make_hf_config() -> MptConfig:
    """Return a tiny MptConfig. max_seq_len is the MPT-specific name."""
    return MptConfig(
        d_model=_D_MODEL,
        n_heads=_N_HEADS,
        n_layers=_N_LAYERS,
        expansion_ratio=_D_MLP // _D_MODEL,
        max_seq_len=_MAX_SEQ_LEN,
        vocab_size=_D_VOCAB,
        no_bias=True,
    )


def _make_bridge() -> TransformerBridge:
    """Construct a TransformerBridge from a programmatic tiny MptForCausalLM.

    Bypasses boot_transformers (which calls AutoConfig.from_pretrained) and
    directly instantiates the adapter and bridge. Safe for CI — no download.
    """
    hf_cfg = _make_hf_config()
    hf_model = MptForCausalLM(hf_cfg)
    hf_model.eval()

    bridge_cfg = TransformerBridgeConfig(
        d_model=_D_MODEL,
        d_head=_D_MODEL // _N_HEADS,
        n_layers=_N_LAYERS,
        n_ctx=_MAX_SEQ_LEN,
        n_heads=_N_HEADS,
        d_vocab=_D_VOCAB,
        d_mlp=_D_MLP,
        default_prepend_bos=False,
        architecture="MPTForCausalLM",
        device="cpu",
    )

    adapter = MPTArchitectureAdapter(bridge_cfg)
    return TransformerBridge(hf_model, adapter, tokenizer=None)


# ---------------------------------------------------------------------------
# Module-scoped fixture — one bridge for the whole file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mpt_bridge() -> TransformerBridge:
    return _make_bridge()


# ---------------------------------------------------------------------------
# Forward pass: HF numerical equivalence
# ---------------------------------------------------------------------------


class TestMPTForwardPass:
    """Bridge forward must match HF MptForCausalLM.forward at atol=1e-4."""

    def test_forward_output_shape(self, mpt_bridge: TransformerBridge) -> None:
        tokens = torch.randint(0, _D_VOCAB, (1, 8))
        with torch.no_grad():
            out = mpt_bridge(tokens)
        assert out.shape == (1, 8, _D_VOCAB)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_matches_hf(self, mpt_bridge: TransformerBridge) -> None:
        """Logits from bridge must match HF at max_diff < 1e-4."""
        tokens = torch.randint(0, _D_VOCAB, (1, 8))
        hf_model = mpt_bridge.original_model
        with torch.no_grad():
            bridge_out = mpt_bridge(tokens)
            hf_out = hf_model(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff:.2e} (threshold 1e-4)"

    def test_forward_batch2_matches_hf(self, mpt_bridge: TransformerBridge) -> None:
        """Batch=2 check: no batch-broadcast bug in ALiBi unsqueeze(0) path."""
        tokens = torch.randint(0, _D_VOCAB, (2, 8))
        hf_model = mpt_bridge.original_model
        with torch.no_grad():
            bridge_out = mpt_bridge(tokens)
            hf_out = hf_model(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Batch=2 bridge vs HF max diff = {max_diff:.2e}"


# ---------------------------------------------------------------------------
# Hook coverage via run_with_cache
# ---------------------------------------------------------------------------


class TestMPTHookCoverage:
    """All required hooks must appear in the cache after a single forward pass."""

    @pytest.fixture(scope="class")
    def cache(self, mpt_bridge: TransformerBridge) -> dict:
        tokens = torch.randint(0, _D_VOCAB, (1, 8))
        with torch.no_grad():
            _, cache = mpt_bridge.run_with_cache(tokens)
        return cache

    # Attention hooks
    def test_hook_q_fires(self, cache: dict) -> None:
        assert "blocks.0.attn.hook_q" in cache, f"keys: {sorted(cache.keys())}"

    def test_hook_k_fires(self, cache: dict) -> None:
        assert "blocks.0.attn.hook_k" in cache

    def test_hook_v_fires(self, cache: dict) -> None:
        assert "blocks.0.attn.hook_v" in cache

    def test_hook_attn_scores_fires(self, cache: dict) -> None:
        assert "blocks.0.attn.hook_attn_scores" in cache

    def test_hook_pattern_fires(self, cache: dict) -> None:
        assert "blocks.0.attn.hook_pattern" in cache

    # MLP hooks
    def test_hook_mlp_in_fires(self, cache: dict) -> None:
        assert "blocks.0.mlp.hook_in" in cache

    def test_hook_mlp_out_fires(self, cache: dict) -> None:
        assert "blocks.0.mlp.hook_out" in cache

    # Norm hooks
    def test_hook_ln1_fires(self, cache: dict) -> None:
        assert "blocks.0.ln1.hook_out" in cache

    def test_hook_ln2_fires(self, cache: dict) -> None:
        assert "blocks.0.ln2.hook_out" in cache

    # Residual stream hooks
    def test_hook_resid_pre_fires(self, cache: dict) -> None:
        assert "blocks.0.hook_resid_pre" in cache

    def test_hook_resid_post_fires(self, cache: dict) -> None:
        assert "blocks.0.hook_resid_post" in cache

    # Shape sanity: attention pattern must be causal (lower-triangular)
    def test_attn_pattern_is_causal(self, cache: dict) -> None:
        """Attention pattern upper triangle must be zero (causal structure)."""
        pattern = cache["blocks.0.attn.hook_pattern"]  # [batch, n_heads, seq, seq]
        seq = pattern.shape[-1]
        upper = torch.triu(pattern[0, 0], diagonal=1)
        assert (
            upper.abs() < 1e-6
        ).all(), f"Attention pattern is not causal; upper-triangle max = {upper.abs().max():.2e}"
