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


# Golden-fixture counterparts of the two HT fixtures below (see tests/goldens.py).
# B-tier tests migrate onto these; the live-HT fixtures delete with HookedTransformer.
@pytest.fixture(scope="session")
def distilgpt2_goldens_processed():
    """Golden cell for distilgpt2 with default weight processing (full_defaults)."""
    from tests import goldens

    if not goldens.goldens_available("distilgpt2", "full_defaults"):
        pytest.skip("TL goldens dataset unavailable (set TL_GOLDENS_DIR or enable network)")
    return goldens.GoldenCell("distilgpt2", "full_defaults")


@pytest.fixture(scope="session")
def distilgpt2_goldens_unprocessed():
    """Golden cell for distilgpt2 without weight processing (no_processing)."""
    from tests import goldens

    if not goldens.goldens_available("distilgpt2", "no_processing"):
        pytest.skip("TL goldens dataset unavailable (set TL_GOLDENS_DIR or enable network)")
    return goldens.GoldenCell("distilgpt2", "no_processing")


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
