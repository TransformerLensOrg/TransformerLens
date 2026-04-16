"""Unit tests for SSM bridge components.

Exercises SSMBlockBridge, SSMMixerBridge, and DepthwiseConv1DBridge with mock
modules — no HF model download required. The state-spaces/mamba-130m-hf
end-to-end test lives in tests/integration/model_bridge.
"""

import torch
import torch.nn as nn

from transformer_lens.model_bridge.generalized_components.depthwise_conv1d import (
    DepthwiseConv1DBridge,
)
from transformer_lens.model_bridge.generalized_components.ssm_block import (
    SSMBlockBridge,
)
from transformer_lens.model_bridge.generalized_components.ssm_mixer import (
    SSMMixerBridge,
)


class TestSSMBlockBridgeAliases:
    """SSMBlockBridge must NOT inherit transformer-specific hook aliases."""

    def test_has_ssm_aliases(self):
        bridge = SSMBlockBridge(name="backbone.layers")
        assert "hook_resid_pre" in bridge.hook_aliases
        assert "hook_resid_post" in bridge.hook_aliases
        assert "hook_mixer_in" in bridge.hook_aliases
        assert "hook_mixer_out" in bridge.hook_aliases

    def test_no_transformer_aliases(self):
        """Critical: SSM blocks must not expose attn/mlp/resid_mid aliases."""
        bridge = SSMBlockBridge(name="backbone.layers")
        forbidden = {
            "hook_resid_mid",
            "hook_attn_in",
            "hook_attn_out",
            "hook_mlp_in",
            "hook_mlp_out",
            "hook_q_input",
            "hook_k_input",
            "hook_v_input",
        }
        assert forbidden.isdisjoint(bridge.hook_aliases.keys())

    def test_not_subclass_of_block_bridge(self):
        """Direct GeneralizedComponent inheritance, not BlockBridge."""
        from transformer_lens.model_bridge.generalized_components.block import (
            BlockBridge,
        )

        assert not issubclass(SSMBlockBridge, BlockBridge)


class TestSSMBlockBridgeForward:
    """SSMBlockBridge forward wires hook_in/hook_out around the original component."""

    def test_hooks_fire_around_original(self):
        class MockBlock(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2.0

        bridge = SSMBlockBridge(name="backbone.layers.0")
        bridge.set_original_component(MockBlock())

        captured_in, captured_out = [], []

        def cap_in(t, hook):
            captured_in.append(t.clone())
            return t

        def cap_out(t, hook):
            captured_out.append(t.clone())
            return t

        bridge.hook_in.add_hook(cap_in)
        bridge.hook_out.add_hook(cap_out)

        x = torch.randn(1, 4, 8)
        out = bridge(x)

        assert len(captured_in) == 1
        assert len(captured_out) == 1
        assert torch.allclose(captured_in[0], x)
        assert torch.allclose(captured_out[0], x * 2.0)
        assert torch.allclose(out, x * 2.0)


class TestDepthwiseConv1DBridge:
    """DepthwiseConv1DBridge wraps nn.Conv1d and preserves causal behavior."""

    def _make_causal_conv(self, channels: int = 4, kernel: int = 3) -> nn.Conv1d:
        return nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel,
            groups=channels,  # depthwise
            padding=kernel - 1,  # causal padding (trimmed outside the bridge)
            bias=True,
        )

    def test_hooks_fire_with_channel_first_shapes(self):
        """hook_in and hook_out see the raw channel-first tensors from nn.Conv1d."""
        conv = self._make_causal_conv(channels=4, kernel=3)
        bridge = DepthwiseConv1DBridge(name="conv1d")
        bridge.set_original_component(conv)

        captured: dict = {}

        def cap_in(t, hook):
            captured["in"] = t.clone()
            return t

        def cap_out(t, hook):
            captured["out"] = t.clone()
            return t

        bridge.hook_in.add_hook(cap_in)
        bridge.hook_out.add_hook(cap_out)

        # Mamba feeds channel-first tensors: [batch, channels, seq_len]
        x = torch.randn(2, 4, 6)
        out = bridge(x)

        # hook_in captures raw [batch, channels, seq_len]
        assert captured["in"].shape == (2, 4, 6)
        # hook_out captures pre-trim output: [batch, channels, seq_len + kernel - 1]
        assert captured["out"].shape == (2, 4, 6 + 3 - 1)
        # Bridge returns the same shape as nn.Conv1d produces (no trimming)
        assert out.shape == captured["out"].shape

    def test_depthwise_equivalence(self):
        """Bridge output equals raw conv output (no arithmetic drift)."""
        conv = self._make_causal_conv(channels=4, kernel=3)
        bridge = DepthwiseConv1DBridge(name="conv1d")
        bridge.set_original_component(conv)

        x = torch.randn(1, 4, 5)
        raw = conv(x)
        bridged = bridge(x)
        assert torch.allclose(bridged, raw)

    def test_hook_mutation_affects_output(self):
        """Mutation on hook_in propagates through the wrapped conv."""
        conv = self._make_causal_conv(channels=2, kernel=2)
        bridge = DepthwiseConv1DBridge(name="conv1d")
        bridge.set_original_component(conv)

        x = torch.randn(1, 2, 4)
        baseline = bridge(x)

        def zero_input(t, hook):
            return torch.zeros_like(t)

        bridge.hook_in.add_hook(zero_input)
        muted = bridge(x)
        bridge.hook_in.remove_hooks()

        # Zeroing the input should produce a different output from baseline
        assert not torch.allclose(baseline, muted)


class TestSSMMixerBridgeAliases:
    """SSMMixerBridge exposes Mamba-1 specific submodule hook aliases."""

    def test_mamba1_alias_set(self):
        bridge = SSMMixerBridge(name="mixer")
        assert "hook_in_proj" in bridge.hook_aliases
        assert "hook_conv" in bridge.hook_aliases
        assert "hook_x_proj" in bridge.hook_aliases
        assert "hook_dt_proj" in bridge.hook_aliases
        # hook_ssm_out aliases to the mixer's own hook_out
        assert bridge.hook_aliases["hook_ssm_out"] == "hook_out"


class TestSSMMixerBridgeForward:
    """SSMMixerBridge fires hook_in/hook_out around the wrapped HF mixer."""

    def test_hooks_fire_around_wrapped_mixer(self):
        class MockMixer(nn.Module):
            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return hidden_states + 1.0

        bridge = SSMMixerBridge(name="mixer")
        bridge.set_original_component(MockMixer())

        captured_in, captured_out = [], []

        def cap_in(t, hook):
            captured_in.append(t.clone())
            return t

        def cap_out(t, hook):
            captured_out.append(t.clone())
            return t

        bridge.hook_in.add_hook(cap_in)
        bridge.hook_out.add_hook(cap_out)

        x = torch.randn(1, 3, 8)
        out = bridge(x)

        assert len(captured_in) == 1
        assert len(captured_out) == 1
        assert torch.allclose(captured_in[0], x)
        assert torch.allclose(captured_out[0], x + 1.0)
        assert torch.allclose(out, x + 1.0)
