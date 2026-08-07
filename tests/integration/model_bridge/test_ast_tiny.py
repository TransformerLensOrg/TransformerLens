"""Integration parity tests for AST architecture"""

import pytest
import torch
from transformers import ASTConfig, ASTForAudioClassification

from transformer_lens.model_bridge import TransformerBridge


@pytest.fixture(autouse=True)
def prevent_internet_processor_download(monkeypatch):
    """Prevent V3 boot from attempting to download a feature extractor."""
    monkeypatch.setattr(
        "transformers.AutoFeatureExtractor.from_pretrained", lambda *args, **kwargs: None
    )


@pytest.fixture(scope="module")
def hf_config():
    """Setup a tiny HF AST config for instantaneous local testing."""
    cfg = ASTConfig(
        hidden_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=24,
        patch_size=16,
        num_labels=2,
    )
    cfg.architectures = ["ASTForAudioClassification"]
    return cfg


class TestASTParity:
    """FP32 numerical parity assertion."""

    def test_forward_pass_parity(self, hf_config):
        hf_model = ASTForAudioClassification(hf_config)
        hf_model.eval()

        # boot the V3 transformerbridge natively
        tl_model = TransformerBridge.boot_transformers("ast", hf_model=hf_model)
        tl_model.eval()

        # 3. create dummy spectrogram input [batch, freq, time]
        dummy_spectrogram = torch.randn(1, 1024, 128)

        # 4. compare forward passes
        with torch.no_grad():
            hf_logits = hf_model(dummy_spectrogram).logits

            # bridge forward pass (extract sequence before HF's pooling)
            _, cache = tl_model.run_with_cache(dummy_spectrogram)
            resid = cache["ln_final.hook_normalized"]

            # apply AST pooling logic to the bridges residual stream
            cls_token = resid[:, 0, :]
            dist_token = resid[:, 1, :]
            pooled_out = (cls_token + dist_token) / 2.0

            # apply the final classifier (which holds 2nd layernorm + dense)
            tl_logits = hf_model.classifier(pooled_out)

        diff = (hf_logits - tl_logits).abs().max().item()

        # 5. assert parity
        assert diff < 1e-4, "Parity failed: Bridge tensors do not match HuggingFace."
