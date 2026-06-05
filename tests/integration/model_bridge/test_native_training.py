"""Integration: train a TL-native bridge on the induction task end-to-end.

Mirrors the Realtime Telemetry demo's logic but with shorter step counts so
the test stays in CI's time budget. Thresholds are deliberately qualitative
(direction, not magnitude) so the test does not flake on
BLAS/MPS-implementation differences across CI runners.
"""
from __future__ import annotations

import numpy as np
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge


def _induction_batch(batch_size: int, seq_len: int, vocab_size: int) -> torch.Tensor:
    half = seq_len // 2
    rand = torch.randint(0, vocab_size, (batch_size, half))
    return torch.cat([rand, rand], dim=1)


def _telemetry_cfg() -> TransformerBridgeConfig:
    # Matches the demo cell-2 configuration so this test exercises the same
    # surface a user would touch.
    return TransformerBridgeConfig(
        d_model=64,
        d_head=32,
        n_heads=2,
        n_layers=2,
        n_ctx=32,
        d_vocab=64,
        d_mlp=256,
        act_fn="gelu",
        normalization_type="LN",
        seed=42,
    )


def test_native_bridge_training_decreases_loss():
    """Loss must decrease meaningfully within a small step budget."""
    torch.manual_seed(42)
    cfg = _telemetry_cfg()
    bridge = TransformerBridge.boot_native(cfg)
    bridge.train()
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-3, weight_decay=1e-4)

    initial_losses, final_losses = [], []
    for step in range(300):
        batch = _induction_batch(16, cfg.n_ctx, cfg.d_vocab)
        loss = bridge(batch, return_type="loss")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step < 5:
            initial_losses.append(loss.item())
        if step >= 295:
            final_losses.append(loss.item())

    initial_avg = sum(initial_losses) / len(initial_losses)
    final_avg = sum(final_losses) / len(final_losses)
    assert final_avg < initial_avg * 0.7, (
        f"Loss did not decrease enough: initial_avg={initial_avg:.4f}, "
        f"final_avg={final_avg:.4f} (expected < {initial_avg * 0.7:.4f})"
    )


def test_native_bridge_run_with_cache_during_training():
    """run_with_cache must populate attention-pattern hooks with [B,H,S,S] shape
    and support the demo's selective-caching pattern (call cache every K steps,
    standard forward in between)."""
    torch.manual_seed(0)
    cfg = _telemetry_cfg()
    bridge = TransformerBridge.boot_native(cfg)
    bridge.train()
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-3)

    batch = _induction_batch(8, cfg.n_ctx, cfg.d_vocab)

    # Selective caching: cache step, then plain step, then cache step.
    loss_cache_a, cache_a = bridge.run_with_cache(batch, return_type="loss")
    loss_cache_a.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_plain = bridge(batch, return_type="loss")
    loss_plain.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_cache_b, cache_b = bridge.run_with_cache(batch, return_type="loss")
    loss_cache_b.backward()
    optimizer.step()
    optimizer.zero_grad()

    for cache in (cache_a, cache_b):
        for layer in range(cfg.n_layers):
            key = f"blocks.{layer}.attn.hook_pattern"
            assert key in cache, f"Missing {key}"
            assert cache[key].shape == (8, cfg.n_heads, cfg.n_ctx, cfg.n_ctx), cache[key].shape


def test_native_bridge_induction_circuit_forms():
    """A circuit-forming proxy: at least one layer's attention coherence rises
    substantially between init and step ~500. Computes the same coherence
    metric the telemetry demo uses, but asserts a direction-only invariant."""
    torch.manual_seed(42)
    cfg = _telemetry_cfg()
    bridge = TransformerBridge.boot_native(cfg)

    def coherence_per_layer():
        batch = _induction_batch(16, cfg.n_ctx, cfg.d_vocab)
        with torch.no_grad():
            _, cache = bridge.run_with_cache(batch, return_type="loss")
        out = []
        for layer in range(cfg.n_layers):
            probs = cache[f"blocks.{layer}.attn.hook_pattern"] + 1e-9
            entropy = -torch.sum(probs * torch.log(probs), dim=-1)
            coh = 1.0 - (entropy.mean(dim=[0, 2]) / np.log(probs.shape[-1]))
            out.append(coh.mean().item())
        return out

    coherence_initial = coherence_per_layer()

    bridge.train()
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(500):
        batch = _induction_batch(16, cfg.n_ctx, cfg.d_vocab)
        loss = bridge(batch, return_type="loss")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    coherence_final = coherence_per_layer()

    rises = [(f - i) for i, f in zip(coherence_initial, coherence_final)]
    assert max(rises) > 0.2, (
        f"No layer's coherence rose meaningfully. "
        f"Initial={coherence_initial}, final={coherence_final}, deltas={rises}"
    )
