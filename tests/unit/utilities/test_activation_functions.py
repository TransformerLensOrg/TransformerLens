"""Unit tests for the shared soft-cap helpers.

Covers ``softcap_enabled`` and ``apply_softcap`` in
``transformer_lens.utilities.activation_functions`` (see issue #1489).
"""

import torch

from transformer_lens.utilities.activation_functions import (
    SOFTCAP_DISABLED,
    apply_softcap,
    softcap_enabled,
)


class TestSoftcapEnabled:
    """Test cases for softcap_enabled."""

    def test_none_is_disabled(self):
        assert softcap_enabled(None) is False

    def test_zero_is_disabled(self):
        assert softcap_enabled(0.0) is False

    def test_disabled_sentinel_is_disabled(self):
        assert softcap_enabled(SOFTCAP_DISABLED) is False

    def test_negative_is_disabled(self):
        assert softcap_enabled(-5.0) is False

    def test_positive_is_enabled(self):
        assert softcap_enabled(30.0) is True


class TestApplySoftcap:
    """Test cases for apply_softcap."""

    def test_disabled_is_identity(self):
        x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
        result = apply_softcap(x, SOFTCAP_DISABLED)
        assert torch.equal(result, x)

    def test_none_is_identity(self):
        x = torch.tensor([-100.0, 0.0, 100.0])
        result = apply_softcap(x, None)
        assert torch.equal(result, x)

    def test_enabled_matches_capped_tanh_formula(self):
        x = torch.tensor([-50.0, -10.0, 0.0, 10.0, 50.0])
        cap = 30.0
        result = apply_softcap(x, cap)
        expected = cap * torch.tanh(x / cap)
        assert torch.allclose(result, expected)

    def test_enabled_output_stays_within_cap(self):
        x = torch.tensor([-1000.0, 1000.0])
        cap = 30.0
        result = apply_softcap(x, cap)
        assert torch.all(result.abs() <= cap)

    def test_disabled_sentinel_does_not_saturate_large_values(self):
        """Regression guard for the truthiness trap the issue describes:
        a naive ``if cap:`` check treats -1.0 as enabled, which computes
        ``-1.0 * tanh(x / -1.0) == tanh(x)`` and saturates every large score
        to 1.0. The disabled path must instead be a pure identity.
        """
        x = torch.tensor([50.0, 100.0])
        result = apply_softcap(x, SOFTCAP_DISABLED)
        assert torch.equal(result, x)
        assert not torch.allclose(result, torch.ones_like(x))
