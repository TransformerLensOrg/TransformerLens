"""Integration tests for the HRM-Text architecture adapter.

Builds a tiny HrmTextForCausalLM from scratch, wraps it in a TransformerBridge,
and verifies:
- Bridge creation with correct component structure
- Forward pass parity with HF
- Hook firing with expected shapes
- HRM-specific config attr propagation
"""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

_HRM_TEXT_AVAILABLE = True
try:
    from transformers import HrmTextConfig, HrmTextForCausalLM
except ImportError:
    _HRM_TEXT_AVAILABLE = False


def _make_tiny_hf_model():
    """Tiny HRM-Text model: 2 layers per stack, 1 H_cycle, 2 L_cycles."""
    cfg = HrmTextConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=16,
        intermediate_size=128,
        vocab_size=128,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        H_cycles=1,
        L_cycles=2,
        L_bp_cycles=[0, 2],
        embedding_scale=8.0,
        prefix_lm=False,
        _attn_implementation="eager",
    )
    cfg.tie_word_embeddings = False
    model = HrmTextForCausalLM(cfg)
    model.eval()
    return model


def _make_bridge(hf_model):
    from unittest.mock import MagicMock

    from transformer_lens.config.transformer_bridge_config import (
        TransformerBridgeConfig,
    )
    from transformer_lens.model_bridge.supported_architectures.hrm_text import (
        HrmTextArchitectureAdapter,
    )

    bridge_cfg = TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_heads=4,
        n_layers=2,
        n_ctx=64,
        d_vocab=128,
        d_mlp=128,
        architecture="HrmTextForCausalLM",
    )
    # HRM-Text passthrough attrs must be propagated to bridge_cfg so the adapter
    # can read them. In production this happens via _HF_PASSTHROUGH_ATTRS in
    # _bridge_builder.py / transformers.py, but in this test we construct the
    # config manually.
    for attr in ("H_cycles", "L_cycles", "L_bp_cycles", "embedding_scale", "prefix_lm"):
        setattr(bridge_cfg, attr, getattr(hf_model.config, attr))
    adapter = HrmTextArchitectureAdapter(bridge_cfg)
    return TransformerBridge(hf_model, adapter, tokenizer=MagicMock())


@pytest.mark.skipif(
    not _HRM_TEXT_AVAILABLE,
    reason="HrmTextForCausalLM not available in installed transformers",
)
class TestHrmTextBridgeCreation:
    @pytest.fixture(scope="class")
    def bridge(self):
        return _make_bridge(_make_tiny_hf_model())

    @pytest.fixture(scope="class")
    def hf_model(self):
        return _make_tiny_hf_model()

    def test_has_correct_block_structure(self, bridge):
        assert hasattr(bridge, "L_blocks")
        assert hasattr(bridge, "H_blocks")
        assert len(bridge.L_blocks) == 2
        assert len(bridge.H_blocks) == 2

    def test_has_core_components(self, bridge):
        assert hasattr(bridge, "embed")
        assert hasattr(bridge, "unembed")
        assert hasattr(bridge, "L_ln_final")
        assert hasattr(bridge, "H_ln_final")

    def test_hook_names_present(self, bridge):
        hook_keys = set(bridge.hook_dict.keys())
        assert "L_blocks.0.hook_resid_pre" in hook_keys
        assert "L_blocks.0.hook_resid_post" in hook_keys
        assert "H_blocks.0.hook_resid_pre" in hook_keys
        assert "H_blocks.0.hook_resid_post" in hook_keys

    def test_L_block_submodule_hooks(self, bridge):
        hook_keys = set(bridge.hook_dict.keys())
        assert any("L_blocks.0.ln1" in k for k in hook_keys)
        assert any("L_blocks.0.ln2" in k for k in hook_keys)
        assert any("L_blocks.0.attn" in k for k in hook_keys)
        assert any("L_blocks.0.mlp" in k for k in hook_keys)

    def test_H_block_submodule_hooks(self, bridge):
        hook_keys = set(bridge.hook_dict.keys())
        assert any("H_blocks.0.ln1" in k for k in hook_keys)
        assert any("H_blocks.0.ln2" in k for k in hook_keys)
        assert any("H_blocks.0.attn" in k for k in hook_keys)
        assert any("H_blocks.0.mlp" in k for k in hook_keys)

    def test_config_flags(self, bridge):
        assert bridge.cfg.normalization_type == "RMS"
        assert bridge.cfg.positional_embedding_type == "rotary"
        assert bridge.cfg.gated_mlp is True
        assert bridge.cfg.final_rms is True

    def test_hr_config_propagated(self, bridge, hf_model):
        assert bridge.cfg.H_cycles == hf_model.config.H_cycles
        assert bridge.cfg.L_cycles == hf_model.config.L_cycles
        assert bridge.cfg.embedding_scale == hf_model.config.embedding_scale
        assert bridge.cfg.prefix_lm == hf_model.config.prefix_lm


@pytest.mark.skipif(
    not _HRM_TEXT_AVAILABLE,
    reason="HrmTextForCausalLM not available in installed transformers",
)
class TestHrmTextForwardPass:
    @pytest.fixture(scope="class")
    def bridge(self):
        return _make_bridge(_make_tiny_hf_model())

    def test_forward_returns_logits(self, bridge):
        tokens = torch.randint(0, 128, (1, 4))
        with torch.no_grad():
            output = bridge(tokens, use_cache=False)
        assert output.shape == (1, 4, 128)
        assert not torch.isnan(output).any()

    def test_forward_matches_hf(self, bridge):
        tokens = torch.randint(0, 128, (1, 4))
        with torch.no_grad():
            hf_logits = bridge.original_model(tokens, use_cache=False).logits
            bridge_logits = bridge(tokens, use_cache=False)
        assert hf_logits.shape == bridge_logits.shape
        torch.testing.assert_close(hf_logits, bridge_logits, atol=1e-5, rtol=1e-5)

    def test_hook_activation_shapes(self, bridge):
        captured = []

        def capture_hook(tensor, hook):
            captured.append(tensor.detach().clone())
            return tensor

        tokens = torch.randint(0, 128, (1, 4))
        with torch.no_grad():
            bridge.run_with_hooks(
                tokens,
                fwd_hooks=[("L_blocks.0.mlp.hook_out", capture_hook)],
                use_cache=False,
            )
        # Each L_cycle fires the hook, so with L_cycles=2 we expect 2 firings
        assert len(captured) == 2, f"Expected 2 hook firings (one per L_cycle), got {len(captured)}"
        for output in captured:
            assert output.shape == (1, 4, 64), f"Expected (1, 4, 64), got {output.shape}"

    def test_hook_on_H_block_fires(self, bridge):
        captured = []

        def capture_hook(tensor, hook):
            captured.append(tensor.detach().clone())
            return tensor

        tokens = torch.randint(0, 128, (1, 4))
        with torch.no_grad():
            bridge.run_with_hooks(
                tokens,
                fwd_hooks=[("H_blocks.0.attn.hook_out", capture_hook)],
                use_cache=False,
            )
        # H_blocks fire once per H_cycle (H_cycles=1)
        assert len(captured) == 1, "H_blocks hook must fire"
