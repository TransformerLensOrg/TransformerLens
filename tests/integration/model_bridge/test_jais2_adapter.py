"""Integration tests for the Jais2 architecture adapter.

Seeded tiny from a local Jais2Config (no hub access). Jais 2 uses plain
nn.LayerNorm in a standard pre-norm layout, so unlike its Nemotron parent
(LayerNorm1P) compatibility-mode folding is valid — the parity test pins
that the re-enabled fold path stays faithful.
"""

import pytest
import torch

from tests.integration.model_bridge.helpers import make_tiny_pair


def _tiny_jais2_pair():
    from transformers import Jais2Config

    cfg = Jais2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return make_tiny_pair(cfg, "Jais2ForCausalLM")


class TestJais2Bridge:
    def test_forward_matches_hf(self) -> None:
        bridge, ref = _tiny_jais2_pair()
        torch.manual_seed(0)
        ids = torch.randint(3, 128, (1, 12))
        with torch.no_grad():
            out = bridge(ids)
            expected = ref(input_ids=ids).logits
        max_diff = (out - expected).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"

    def test_fold_flags_reenabled(self) -> None:
        bridge, _ = _tiny_jais2_pair()
        assert bridge.adapter.supports_fold_ln is True
        assert bridge.adapter.supports_center_writing_weights is True

    def test_compatibility_mode_logit_parity(self) -> None:
        """Plain-LN folding must stay faithful to the HF reference.

        Compared on log_softmax: center_unembed shifts raw logits by a
        per-position constant by design, so raw-logit deltas are not a
        parity signal.
        """
        bridge, ref = _tiny_jais2_pair()
        torch.manual_seed(0)
        ids = torch.randint(3, 128, (1, 12))
        bridge.enable_compatibility_mode(disable_warnings=True)
        with torch.no_grad():
            out = torch.log_softmax(bridge(ids), dim=-1)
            expected = torch.log_softmax(ref(input_ids=ids).logits, dim=-1)
        max_diff = (out - expected).abs().max().item()
        assert max_diff < 1e-5, f"Compat-mode vs HF log_softmax max diff = {max_diff}"


def _perturbed_jais2_bridge():
    bridge, _ = _tiny_jais2_pair()
    # Unit LN init makes a stray (1 + w) a uniform 2x scale; non-unit weights expose it.
    with torch.no_grad():
        for module in bridge.original_model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.copy_(torch.linspace(0.7, 1.3, module.weight.numel()))
    return bridge


class TestJais2PlainLayerNormHooks:
    """Jais 2 inherits Nemotron's norm mapping but must not inherit LayerNorm1P's (1 + w)."""

    @pytest.mark.parametrize("norm_name", ["blocks.0.ln1", "blocks.0.ln2", "ln_final"])
    @pytest.mark.parametrize("hook_name", ["hook_scale", "hook_normalized"])
    @pytest.mark.parametrize("direction", ["fwd", "bwd"])
    def test_identity_hooks_preserve_logits(self, norm_name, hook_name, direction) -> None:
        bridge = _perturbed_jais2_bridge()
        ids = torch.tensor([[1, 7, 8, 9, 10]])
        with torch.no_grad():
            clean = bridge(ids)
        message = "Backward hooks" if direction == "bwd" else "reconstructed from the hooked values"
        key = f"{norm_name}.{hook_name}"
        with bridge.hooks(**{f"{direction}_hooks": [(key, lambda t, hook: t.clone())]}):
            with pytest.warns(UserWarning, match=message):
                hooked = bridge(ids)
        torch.testing.assert_close(hooked.detach(), clean, atol=1e-6, rtol=1e-5)
