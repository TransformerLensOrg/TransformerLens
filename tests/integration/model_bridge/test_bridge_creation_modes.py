"""Test different bridge creation and configuration modes."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge


class TestBridgeCreationModes:
    """Test different modes of creating and configuring TransformerBridge."""

    def test_bridge_no_processing(
        self, distilgpt2_bridge_compat_no_processing, distilgpt2_goldens_unprocessed
    ):
        """Test bridge with no weight processing against the unprocessed golden loss."""
        bridge = distilgpt2_bridge_compat_no_processing
        golden = distilgpt2_goldens_unprocessed

        text = golden.scalars["ablation"]["text"]
        ref_loss = golden.scalars["long_text_ce_loss"]
        bridge_loss = bridge(text, return_type="loss")

        diff = abs(ref_loss - bridge_loss.item())
        assert diff < 0.01, f"Unprocessed bridge should match the golden loss: {diff}"
        assert 3.0 < bridge_loss < 8.0, f"Bridge loss should be reasonable: {bridge_loss}"

    def test_bridge_full_compatibility(
        self, distilgpt2_bridge_compat, distilgpt2_goldens_processed
    ):
        """Test bridge with full compatibility mode against the processed golden loss."""
        bridge = distilgpt2_bridge_compat
        golden = distilgpt2_goldens_processed

        text = golden.scalars["ablation"]["text"]
        ref_loss = golden.scalars["long_text_ce_loss"]
        bridge_loss = bridge(text, return_type="loss")

        diff = abs(ref_loss - bridge_loss.item())
        assert diff < 0.01, f"Processed bridge should match the golden loss: {diff}"
        assert 3.0 < bridge_loss < 8.0, f"Bridge loss should be reasonable: {bridge_loss}"

    def test_bridge_tokenizer_compatibility(self, distilgpt2_bridge):
        """Bridge to_tokens must reproduce the frozen legacy tokenization (BOS + GPT-2 BPE)."""
        bridge_tokens = distilgpt2_bridge.to_tokens("Hello world test")
        expected = torch.tensor([[50256, 15496, 995, 1332]])
        assert torch.equal(bridge_tokens, expected), "Tokenization drifted from the frozen ids"

    def test_bridge_configuration_persistence(self):
        # Fresh boot: tests the boot → enable_compat transition.
        bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")

        # Test configuration before compatibility mode
        assert hasattr(bridge, "cfg"), "Bridge should have configuration"

        # Enable compatibility mode and check it persists
        bridge.enable_compatibility_mode()

        # Configuration should still be accessible
        assert hasattr(bridge, "cfg"), "Configuration should persist after compatibility mode"
        assert bridge.cfg is not None, "Configuration should not be None"

    def test_bridge_device_handling(self, gpt2_bridge):
        """Test that bridge handles device specification correctly."""
        assert (
            next(gpt2_bridge.original_model.parameters()).device.type == "cpu"
        ), "Model should be on CPU device"

        # Test that bridge can process text on correct device
        test_text = "Device test"
        loss = gpt2_bridge(test_text, return_type="loss")
        assert isinstance(loss, torch.Tensor), "Should return tensor"
        assert loss.device.type == "cpu", "Loss should be on CPU"


class TestBridgeOfflineWithHfModel:
    """Bridge must reuse hf_model.config when supplied, not refetch from the Hub (#846)."""

    OFFLINE_MODEL = "trl-internal-testing/tiny-MistralForCausalLM-0.2"

    @pytest.fixture
    def hf_model(self):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(self.OFFLINE_MODEL).eval()

    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.OFFLINE_MODEL)

    def test_offline_boot_with_hf_model(self, hf_model, tokenizer):
        """Bridge boot succeeds when AutoConfig.from_pretrained would fail."""
        from unittest.mock import patch

        import transformer_lens.model_bridge.sources.transformers.source as bridge_source

        with patch.object(bridge_source, "AutoConfig") as mock_autoconfig:
            mock_autoconfig.from_pretrained.side_effect = OSError("Simulated Hub failure")

            bridge = TransformerBridge.boot_transformers(
                self.OFFLINE_MODEL, hf_model=hf_model, tokenizer=tokenizer
            )
            assert not mock_autoconfig.from_pretrained.called

            test_input = tokenizer("Hello", return_tensors="pt")["input_ids"]
            with torch.no_grad():
                logits = bridge(test_input)
            assert torch.isfinite(logits).all()

    def test_hf_model_config_not_mutated_by_bridge(self, hf_model, tokenizer):
        """hf_config_overrides / n_ctx / pad_token_id mutations must not leak into hf_model.config.

        Bridge works on a deepcopy so the user's loaded model stays clean. Catches a
        regression where the deepcopy is dropped in favor of a direct reference.
        """
        snapshot = {
            "max_position_embeddings": getattr(hf_model.config, "max_position_embeddings", None),
            "pad_token_id": getattr(hf_model.config, "pad_token_id", None),
            "output_attentions": getattr(hf_model.config, "output_attentions", None),
        }

        TransformerBridge.boot_transformers(
            self.OFFLINE_MODEL,
            hf_model=hf_model,
            tokenizer=tokenizer,
            hf_config_overrides={"max_position_embeddings": 999},
            n_ctx=128,
        )

        for attr, original_value in snapshot.items():
            assert (
                getattr(hf_model.config, attr, None) == original_value
            ), f"Bridge mutated hf_model.config.{attr}"

    def test_autoconfig_not_called_when_hf_model_provided(self, hf_model, tokenizer):
        """boot() must not call AutoConfig.from_pretrained when hf_model is supplied.

        Patches the binding in the consuming module (sources.transformers.source),
        which imports AutoConfig at module load; AutoTokenizer's internal use of
        ``transformers.AutoConfig`` is deliberately not intercepted.
        """
        from unittest.mock import patch

        import transformer_lens.model_bridge.sources.transformers.source as bridge_source

        with patch.object(bridge_source, "AutoConfig") as mock_autoconfig:
            TransformerBridge.boot_transformers(
                self.OFFLINE_MODEL, hf_model=hf_model, tokenizer=tokenizer
            )
            assert not mock_autoconfig.from_pretrained.called

    def test_attention_hooks_fire_with_preloaded_hf_model(self):
        """hook_pattern must fire on a pre-loaded model left on transformers' default sdpa.

        Under sdpa, HF attention returns attn_weights=None, so boot must force eager.
        Uses OPT because its AttentionBridge reads the pattern off the HF return tuple
        (Mistral computes its own pattern and would mask the regression).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "hf-internal-testing/tiny-random-OPTForCausalLM"
        hf_model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        bridge = TransformerBridge.boot_transformers(
            model_name, hf_model=hf_model, tokenizer=tokenizer
        )

        pattern_name = "blocks.0.attn.hook_pattern"
        _, cache = bridge.run_with_cache("Hello", names_filter=pattern_name)
        assert pattern_name in cache, "attention pattern hook never fired"
        assert torch.isfinite(cache[pattern_name]).all()
