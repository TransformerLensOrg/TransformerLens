"""Integration tests for PretrainArchitectureAdapter -- numerical parity
against a real (not structural-mock) reference model.

Uses `tests/mocks/tiny_pretrain_model.py`'s `TinyPretrainModel`, which
implements genuine adjacent-pair RoPE and RMSNorm math (unlike
`_pretrain_mocks.py`'s unit-test mocks, which stub RoPE with constant
cos/sin and only need to get attribute names and call-signatures right).

Covers: source/bridge logit parity (dense, full-MoE, mixed dense/MoE,
untied embeddings); that this file's fixtures supply a `cfg` truthfully
describing the wrapped model's head/vocab dimensions (a fixture property,
not something `build_pretrain_bridge` itself enforces -- see
`TestIntegrationFixturesSupplyATruthfulConfig`); residual-stream
decomposition via `run_with_cache`; a forward-hook intervention
cross-checked against an independent way of expressing the same edit; and
float64 construction-time preservation.

Note on scope: since the adapter delegates the whole forward to the
source model rather than reimplementing attention/RoPE, exact logit
parity mainly demonstrates that wrapping and output normalization don't
alter an unhooked forward, not that the RoPE convention itself is
correct (the same implementation runs on both sides of the comparison).
`TestResidualStreamDecomposition` and `TestHookIntervention` are the
stronger evidence for correct intervention points.
"""

from __future__ import annotations

import torch

from tests.mocks.tiny_pretrain_model import TinyPretrainModel
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.supported_architectures.pretrain import (
    ARCHITECTURE_NAME,
    build_pretrain_bridge,
)


def _make_cfg(
    *,
    d_model: int,
    n_layers: int,
    n_heads: int = 2,
    d_vocab: int = 64,
) -> TransformerBridgeConfig:
    """Build a bridge config matching this file's TinyPretrainModel
    fixtures (n_heads=2, vocab_size=64 by default). See
    `TestIntegrationFixturesSupplyATruthfulConfig` for why this matching
    matters and what happens when it doesn't."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=128,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_model * 2,
        architecture=ARCHITECTURE_NAME,
    )


class TestSourceBridgeLogitParity:
    """Bridge output must equal the source model's own `"logits"` output
    exactly -- not within tolerance. The adapter never translates weights
    into a second parameter layout, so there is no floating-point drift
    to tolerate; any mismatch here is a real bug, not numerical noise."""

    def test_dense_model_logit_parity_exact(self) -> None:
        torch.manual_seed(0)
        model = TinyPretrainModel(d_model=16, n_heads=2, n_layers=2, d_ff=32, vocab_size=64)
        model.eval()
        tokens = torch.randint(0, 64, (1, 5))
        with torch.no_grad():
            source_logits = model(tokens)["logits"]

        bridge = build_pretrain_bridge(model, _make_cfg(d_model=16, n_layers=2))
        with torch.no_grad():
            bridge_logits = bridge(tokens)

        torch.testing.assert_close(bridge_logits, source_logits, atol=0, rtol=0)

    def test_full_moe_model_logit_parity_exact(self) -> None:
        """Every block's MLP is MoE (moe_layer_indices covers all layers)."""
        torch.manual_seed(0)
        model = TinyPretrainModel(
            d_model=16,
            n_heads=2,
            n_layers=2,
            d_ff=32,
            vocab_size=64,
            n_experts=4,
            top_k=2,
            moe_layer_indices=frozenset({0, 1}),
        )
        model.eval()
        tokens = torch.randint(0, 64, (1, 5))
        with torch.no_grad():
            source_logits = model(tokens)["logits"]

        bridge = build_pretrain_bridge(model, _make_cfg(d_model=16, n_layers=2))
        with torch.no_grad():
            bridge_logits = bridge(tokens)

        torch.testing.assert_close(bridge_logits, source_logits, atol=0, rtol=0)

    def test_mixed_dense_moe_model_logit_parity_exact(self) -> None:
        """One block dense, one block MoE -- the dispatcher must route each
        block's MLP correctly within a single model, not just across
        separately-constructed all-dense/all-MoE models."""
        torch.manual_seed(0)
        model = TinyPretrainModel(
            d_model=16,
            n_heads=2,
            n_layers=2,
            d_ff=32,
            vocab_size=64,
            n_experts=4,
            top_k=2,
            moe_layer_indices=frozenset({1}),
        )
        model.eval()
        tokens = torch.randint(0, 64, (1, 5))
        with torch.no_grad():
            source_logits = model(tokens)["logits"]

        bridge = build_pretrain_bridge(model, _make_cfg(d_model=16, n_layers=2))
        with torch.no_grad():
            bridge_logits = bridge(tokens)

        torch.testing.assert_close(bridge_logits, source_logits, atol=0, rtol=0)

    def test_untied_embeddings_logit_parity_exact(self) -> None:
        torch.manual_seed(0)
        model = TinyPretrainModel(
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            vocab_size=64,
            tie_embeddings=False,
        )
        model.eval()
        tokens = torch.randint(0, 64, (1, 5))
        with torch.no_grad():
            source_logits = model(tokens)["logits"]

        bridge = build_pretrain_bridge(model, _make_cfg(d_model=16, n_layers=1))
        with torch.no_grad():
            bridge_logits = bridge(tokens)

        torch.testing.assert_close(bridge_logits, source_logits, atol=0, rtol=0)


class TestIntegrationFixturesSupplyATruthfulConfig:
    """Integration fixtures must supply a config matching the source
    model. `build_pretrain_bridge` does not itself validate this -- `cfg`
    is caller-owned, per ordinary TransformerBridge convention -- so a
    mismatch shows up only in `bridge.cfg`, silently, never in the
    logits (see the second test)."""

    def test_cfg_matches_model_head_and_vocab_dimensions(self) -> None:
        torch.manual_seed(0)
        n_heads = 2
        vocab_size = 64
        model = TinyPretrainModel(
            d_model=16, n_heads=n_heads, n_layers=2, d_ff=32, vocab_size=vocab_size
        )
        model.eval()
        cfg = _make_cfg(d_model=16, n_layers=2, n_heads=n_heads, d_vocab=vocab_size)
        bridge = build_pretrain_bridge(model, cfg)

        assert bridge.cfg.n_heads == n_heads
        assert bridge.cfg.d_head == 16 // n_heads
        assert bridge.cfg.d_vocab == model.embed.num_embeddings

    def test_build_pretrain_bridge_does_not_validate_cfg_against_model(self) -> None:
        """A cfg describing a different architecture than the wrapped
        model still builds and still produces exact logit parity
        (delegation never reads cfg) -- only bridge.cfg is wrong.

        Current framework convention; not an adapter guarantee. This test
        documents present behavior, not a requirement -- if
        build_pretrain_bridge later gains cfg-vs-model validation, this
        test should be updated (or removed) rather than treated as a
        regression to preserve."""
        torch.manual_seed(0)
        model = TinyPretrainModel(d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64)
        model.eval()
        tokens = torch.randint(0, 64, (1, 5))
        with torch.no_grad():
            source_logits = model(tokens)["logits"]

        wrong_cfg = _make_cfg(d_model=16, n_layers=1, n_heads=4, d_vocab=256)
        bridge = build_pretrain_bridge(model, wrong_cfg)
        with torch.no_grad():
            bridge_logits = bridge(tokens)

        torch.testing.assert_close(bridge_logits, source_logits, atol=0, rtol=0)
        assert bridge.cfg.n_heads != model.blocks[0].attn.n_heads
        assert bridge.cfg.d_vocab != model.embed.num_embeddings


class TestResidualStreamDecomposition:
    """`run_with_cache`'s resid hooks must decompose the way the residual
    stream actually works, not just be present under the right names
    (attribute/key presence is already covered by the structural unit
    tests) -- resid_pre + attn_out == resid_mid, resid_mid + mlp_out ==
    resid_post, and one block's resid_post is the next block's resid_pre."""

    def test_resid_stream_decomposes_exactly_via_run_with_cache(self) -> None:
        torch.manual_seed(0)
        model = TinyPretrainModel(d_model=16, n_heads=2, n_layers=2, d_ff=32, vocab_size=64)
        model.eval()
        bridge = build_pretrain_bridge(model, _make_cfg(d_model=16, n_layers=2))
        tokens = torch.randint(0, 64, (1, 5))

        with torch.no_grad():
            _, cache = bridge.run_with_cache(tokens)

        torch.testing.assert_close(
            cache["blocks.0.hook_resid_pre"], cache["hook_embed"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            cache["blocks.0.hook_resid_mid"],
            cache["blocks.0.hook_resid_pre"] + cache["blocks.0.hook_attn_out"],
        )
        torch.testing.assert_close(
            cache["blocks.0.hook_resid_post"],
            cache["blocks.0.hook_resid_mid"] + cache["blocks.0.hook_mlp_out"],
        )
        torch.testing.assert_close(
            cache["blocks.1.hook_resid_pre"],
            cache["blocks.0.hook_resid_post"],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            cache["blocks.1.hook_resid_post"],
            cache["blocks.1.hook_resid_mid"] + cache["blocks.1.hook_mlp_out"],
        )


class TestHookIntervention:
    """A forward-hook edit at `blocks.{i}.mlp.hook_out` must land at the
    point the residual-stream decomposition says it should -- cross-
    checked against an independently-expressed version of the same edit
    (forcing resid_post to equal resid_mid), rather than only checking
    that *some* change occurred."""

    def test_mlp_hook_ablation_matches_independent_resid_post_override(self) -> None:
        torch.manual_seed(0)
        model = TinyPretrainModel(d_model=16, n_heads=2, n_layers=2, d_ff=32, vocab_size=64)
        model.eval()
        bridge = build_pretrain_bridge(model, _make_cfg(d_model=16, n_layers=2))
        tokens = torch.randint(0, 64, (1, 5))

        with torch.no_grad():
            baseline_logits, baseline_cache = bridge.run_with_cache(tokens)
        resid_mid0 = baseline_cache["blocks.0.hook_resid_mid"].clone()

        def zero_mlp_out(value: torch.Tensor, hook) -> torch.Tensor:
            return torch.zeros_like(value)

        def force_resid_post_to_resid_mid(value: torch.Tensor, hook) -> torch.Tensor:
            return resid_mid0

        with torch.no_grad():
            via_mlp_ablation = bridge.run_with_hooks(
                tokens, fwd_hooks=[("blocks.0.mlp.hook_out", zero_mlp_out)]
            )
            via_resid_override = bridge.run_with_hooks(
                tokens,
                fwd_hooks=[("blocks.0.hook_resid_post", force_resid_post_to_resid_mid)],
            )

        # Sanity: the intervention actually changed something.
        assert not torch.equal(via_mlp_ablation, baseline_logits)
        # The real check: two different-looking edits that are
        # mathematically the same edit must produce identical output.
        torch.testing.assert_close(via_mlp_ablation, via_resid_override, atol=0, rtol=0)


class TestDtypePreservation:
    """float64 is checked for preservation through construction only, not
    full numerical parity at that dtype -- RoPE's cos/sin computation
    involves trig functions whose float32-vs-float64 rounding can
    legitimately differ from the bridge's own dtype-casting path in ways
    unrelated to correctness, so exact-equality parity isn't a meaningful
    check at non-default dtypes the way it is at float32."""

    def test_float64_construction_preserves_dtype_and_tied_weights(self) -> None:
        torch.manual_seed(0)
        model = TinyPretrainModel(
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            vocab_size=64,
            tie_embeddings=True,
        ).to(torch.float64)
        assert model.lm_head.weight is model.embed.weight

        bridge = build_pretrain_bridge(model, _make_cfg(d_model=16, n_layers=1))

        assert next(model.parameters()).dtype == torch.float64
        assert model.lm_head.weight is model.embed.weight

        tokens = torch.randint(0, 64, (1, 4))
        with torch.no_grad():
            output = bridge(tokens)
        assert output.dtype == torch.float64
        assert torch.isfinite(output).all()
