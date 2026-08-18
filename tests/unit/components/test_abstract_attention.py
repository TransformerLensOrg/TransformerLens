import math

import torch

from transformer_lens.components import AbstractAttention, Attention
from transformer_lens.config import HookedTransformerConfig


def _init_standalone(attn: Attention) -> Attention:
    """Fill a standalone component's torch.empty weights; init_weights runs at model level."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        for parameter in attn.parameters():
            torch.nn.init.normal_(parameter, std=0.2)
    return attn


def test_attention_clip_qkv_clamps_between_projection_and_scores():
    """cfg.clip_qkv must clamp Q/K/V after projection (OLMo v1 / OLMoE semantics):
    output with clip_qkv equals a clip-free module fed pre-clamped Q/K/V."""
    cfg_kwargs = dict(n_layers=1, d_model=32, n_ctx=8, d_head=8, n_heads=4, act_fn="relu")
    # fork_rng: deterministic setup without mutating the global RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        attn_clip = Attention(HookedTransformerConfig(**cfg_kwargs, clip_qkv=0.5))
        attn_ref = Attention(HookedTransformerConfig(**cfg_kwargs))
        # Standalone components carry torch.empty weights (init_weights runs at
        # model level), so fill them explicitly before sharing the state dict.
        with torch.no_grad():
            for param in attn_clip.parameters():
                param.normal_(0, 0.02)
        attn_ref.load_state_dict(attn_clip.state_dict())

        batch, seq = 1, 4
        q0 = torch.randn(batch, seq, 4, 8) * 2
        k0 = torch.randn(batch, seq, 4, 8) * 2
        v0 = torch.randn(batch, seq, 4, 8) * 2
        x = torch.randn(batch, seq, 32)
    # clamp is active on all three of Q/K/V
    assert (q0.abs() > 0.5).any() and (k0.abs() > 0.5).any() and (v0.abs() > 0.5).any()

    attn_clip.calculate_qkv_matrices = lambda *args, **kwargs: (q0, k0, v0)
    attn_ref.calculate_qkv_matrices = lambda *args, **kwargs: (
        q0.clamp(min=-0.5, max=0.5),
        k0.clamp(min=-0.5, max=0.5),
        v0.clamp(min=-0.5, max=0.5),
    )

    with torch.no_grad():
        out_clip = attn_clip(x, x, x)
        out_ref = attn_ref(x, x, x)

    torch.testing.assert_close(out_clip, out_ref)


def test_create_alibi_slope():
    n_ctx = 100

    # Expected result computed non-vectorized way
    expected = torch.zeros((n_ctx, n_ctx))
    for row in range(n_ctx):
        for col in range(n_ctx):
            expected[row, col] = float(min(col - row, 0))

    # Check against the method's vectorized version
    result = AbstractAttention.create_alibi_slope(n_ctx)
    assert torch.allclose(expected, result)


def test_create_alibi_bias():
    n_heads = 2
    n_ctx = 4

    result = AbstractAttention.create_alibi_bias(n_heads, n_ctx, torch.device("cpu"))

    for matrix in result:
        n_row, n_col = matrix.size()
        slope = -matrix[1, 0]
        # Check if upper triangle is all zeros
        assert torch.equal(torch.triu(matrix), torch.zeros_like(matrix))

        ref_lower_triangle = torch.zeros_like(matrix)
        for i in range(1, n_row):
            for j in range(i):
                ref_lower_triangle[i, j] = -slope * (i - j)

        # Check if the lower triangle is decreasing by a constant slope (towards the bottom left corner).
        assert torch.equal(
            torch.tril(matrix, diagonal=-1), torch.tril(ref_lower_triangle, diagonal=-1)
        )


class TestLogNAttentionScaling:
    """Qwen-1's log-n scaling: queries past the training length scale by
    log_{train_len}(position), eval only. TL dropped it entirely."""

    @staticmethod
    def _attention(use_logn: bool) -> Attention:
        # n_ctx larger than the training length: the only configuration in
        # which contexts long enough to trigger log-n can run at all.
        cfg = HookedTransformerConfig(
            d_model=16,
            d_head=4,
            n_heads=4,
            n_ctx=16,
            n_layers=1,
            d_vocab=32,
            act_fn="relu",
            positional_embedding_type="rotary",
            rotary_dim=4,
            use_logn_attn=use_logn,
            train_seq_length=8,
        )
        return _init_standalone(Attention(cfg).eval())

    def test_matches_the_remote_formula_past_n_ctx(self) -> None:
        attn = self._attention(True)
        q = torch.ones(1, 12, 4, 4)
        scaled = attn._apply_logn_scaling(q, kv_cache_pos_offset=0)
        # Positions 1..8 (<= the training length) untouched; 9..12 scale by log_8(pos).
        torch.testing.assert_close(scaled[:, :8], q[:, :8])
        for position in (9, 12):
            expected = math.log(position, 8)
            torch.testing.assert_close(
                scaled[0, position - 1, 0, 0], torch.tensor(expected), atol=1e-6, rtol=0
            )

    def test_cached_decode_uses_absolute_positions(self) -> None:
        attn = self._attention(True)
        q = torch.ones(1, 1, 4, 4)  # single decode step at absolute position 11
        scaled = attn._apply_logn_scaling(q, kv_cache_pos_offset=10)
        torch.testing.assert_close(
            scaled[0, 0, 0, 0], torch.tensor(math.log(11, 8)), atol=1e-6, rtol=0
        )

    def test_flag_changes_the_forward_and_default_does_not(self) -> None:
        """End to end through the component forward, past the training length."""
        torch.manual_seed(1)
        x = torch.randn(1, 12, 16)
        on, off = self._attention(True), self._attention(False)
        off.load_state_dict(on.state_dict())
        with torch.no_grad():
            out_on = on(x, x, x)
            out_off = off(x, x, x)
        # Positions within the training length agree exactly; beyond it they differ.
        torch.testing.assert_close(out_on[:, :8], out_off[:, :8])
        assert not torch.allclose(out_on[:, 8:], out_off[:, 8:])

    def test_training_mode_is_untouched(self) -> None:
        attn = self._attention(True).train()
        x = torch.randn(1, 12, 16)
        reference = self._attention(False).train()
        reference.load_state_dict(attn.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(attn(x, x, x), reference(x, x, x))


class TestDynamicNTKRotary:
    """Qwen-1's dynamic NTK widens the rotary BASE past the training length, so
    TL dropping it left long contexts on training-length frequencies."""

    @staticmethod
    def _attention(use_ntk: bool, n_ctx: int = 64) -> Attention:
        cfg = HookedTransformerConfig(
            d_model=16,
            d_head=4,
            n_heads=4,
            n_ctx=n_ctx,
            n_layers=1,
            d_vocab=32,
            act_fn="relu",
            positional_embedding_type="rotary",
            rotary_dim=4,
            rotary_base=10000,
            use_dynamic_ntk_rope=use_ntk,
            train_seq_length=8,
        )
        return _init_standalone(Attention(cfg).eval())

    def test_alpha_matches_the_remote_step_function(self) -> None:
        """The hub's get_ntk_alpha: 2**ceil(log2(len/train) + 1) - 1, floored at 1."""
        attn = self._attention(True)
        assert attn._ntk_alpha(4, 8) == 1.0  # below the training length
        assert attn._ntk_alpha(8, 8) == 1.0  # exactly at it
        assert attn._ntk_alpha(9, 8) == 3  # just past -> first step
        assert attn._ntk_alpha(16, 8) == 3
        assert attn._ntk_alpha(17, 8) == 7  # 2x past -> next step
        assert attn._ntk_alpha(32, 8) == 7

    def test_base_widens_by_the_remote_exponent(self) -> None:
        attn = self._attention(True)
        before = attn.rotary_cos.clone()
        attn._rescale_rotary_for_ntk(32)  # alpha = 7, rotary_dim = 4
        expected_base = 10000 * 7 ** (4 / (4 - 2))
        expected_sin, expected_cos = attn.calculate_sin_cos_rotary(
            4, attn.rotary_cos.shape[0], base=expected_base, dtype=attn.cfg.dtype
        )
        torch.testing.assert_close(attn.rotary_cos, expected_cos)
        torch.testing.assert_close(attn.rotary_sin, expected_sin)
        assert not torch.allclose(attn.rotary_cos[: before.shape[0]], before)

    def test_short_context_leaves_the_table_alone(self) -> None:
        attn = self._attention(True)
        before = attn.rotary_cos.clone()
        attn._rescale_rotary_for_ntk(8)  # at the training length: alpha = 1
        torch.testing.assert_close(attn.rotary_cos, before)

    def test_forward_differs_only_past_the_training_length(self) -> None:
        torch.manual_seed(1)
        on, off = self._attention(True), self._attention(False)
        off.load_state_dict(on.state_dict())
        with torch.no_grad():
            short = torch.randn(1, 8, 16)
            torch.testing.assert_close(on(short, short, short), off(short, short, short))
            long = torch.randn(1, 32, 16)
            assert not torch.allclose(on(long, long, long), off(long, long, long))

    def test_training_mode_is_untouched(self) -> None:
        on = self._attention(True).train()
        off = self._attention(False).train()
        off.load_state_dict(on.state_dict())
        x = torch.randn(1, 32, 16)
        with torch.no_grad():
            torch.testing.assert_close(on(x, x, x), off(x, x, x))

    def test_table_reverts_when_the_context_shrinks(self) -> None:
        """alpha is recomputed per forward, so a later short sequence must not
        keep the widened base from an earlier long one."""
        attn = self._attention(True)
        baseline = attn.rotary_cos.clone()
        attn._rescale_rotary_for_ntk(32)
        assert not torch.allclose(attn.rotary_cos[: baseline.shape[0]], baseline)
        attn._rescale_rotary_for_ntk(8)
        torch.testing.assert_close(attn.rotary_cos[: baseline.shape[0]], baseline)
