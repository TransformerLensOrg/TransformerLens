"""Session fixtures for model_bridge integration tests.

transformer_lens imports stay inside fixture bodies — jaxtyping's pytest_configure
hook must install before the package is first imported.
"""

import pytest
import torch


@pytest.fixture(scope="module")
def sample_tokens(request, bridge):
    """Deterministic tokens for the requesting module's ``bridge`` fixture.

    Length defaults to 8; modules override via a ``SAMPLE_TOKENS_LEN`` constant.
    """
    torch.manual_seed(0)
    length = getattr(request.module, "SAMPLE_TOKENS_LEN", 8)
    return torch.randint(0, bridge.cfg.d_vocab - 10, (1, length))


@pytest.fixture(scope="session")
def distilgpt2_bridge():
    """TransformerBridge wrapping distilgpt2 (no compatibility mode)."""
    from transformer_lens.model_bridge.bridge import TransformerBridge

    return TransformerBridge.boot_transformers("distilgpt2", device="cpu")


@pytest.fixture(scope="session")
def distilgpt2_bridge_compat():
    """TransformerBridge wrapping distilgpt2 with compatibility mode enabled."""
    from transformer_lens.model_bridge.bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    bridge.enable_compatibility_mode()
    return bridge


@pytest.fixture(scope="session")
def distilgpt2_hooked_processed():
    """HookedTransformer distilgpt2 with default weight processing."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("distilgpt2", device="cpu")


@pytest.fixture(scope="session")
def distilgpt2_hooked_unprocessed():
    """HookedTransformer distilgpt2 without weight processing."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained_no_processing("distilgpt2", device="cpu")


@pytest.fixture(scope="session")
def distilgpt2_bridge_compat_no_processing():
    """TransformerBridge wrapping distilgpt2 with compat mode, no weight processing."""
    from transformer_lens.model_bridge.bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    bridge.enable_compatibility_mode(no_processing=True, disable_warnings=True)
    return bridge
