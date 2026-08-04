"""Global pytest configuration for memory management and test optimization."""

import faulthandler
import gc
import os
import random
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

# Captured in pytest_sessionfinish, used by the CI shutdown-watchdog hook below.
_SESSION_EXIT_STATUS = {"code": 0}


@pytest.fixture(autouse=True, scope="function")
def cleanup_memory():
    """Release accelerator caches after each test."""
    yield
    # gc.collect() deliberately omitted here: a full collection costs
    # ~40-200ms against the torch+transformers heap, and per-function it
    # added 10+ min to the CI coverage run. The class-scoped fixture below
    # still collects at every class boundary.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@pytest.fixture(autouse=True, scope="class")
def cleanup_class_memory():
    """Clean up memory after each test class."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


# Configure pytest to be more memory-efficient
def pytest_configure(config):
    """Configure pytest for better memory usage and reproducible randomness."""
    # Configure garbage collection to be more aggressive
    gc.set_threshold(700, 10, 10)

    # Set random seeds for consistent test parametrization across parallel workers
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture(autouse=True, scope="session")
def _enable_hf_retry_for_tests():
    """Deferred to fixture (not pytest_configure) so jaxtyping installs first."""
    from transformer_lens.utilities.hf_utils import enable_hf_retry

    enable_hf_retry()
    yield


@pytest.fixture(scope="session")
def gpt2_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


# Golden-fixture counterparts of the HookedTransformer fixtures below. They load
# frozen HT-captured reference data (see tests/goldens.py) instead of booting a
# live HT, and skip when the golden dataset is unreachable. B-tier tests migrate
# onto these; the live-HT fixtures delete with HookedTransformer.
@pytest.fixture(scope="session")
def gpt2_goldens_processed():
    """Golden cell for gpt2 with default weight processing (full_defaults)."""
    from tests import goldens

    if not goldens.goldens_available("gpt2", "full_defaults"):
        pytest.skip("TL goldens dataset unavailable (set TL_GOLDENS_DIR or enable network)")
    return goldens.GoldenCell("gpt2", "full_defaults")


@pytest.fixture(scope="session")
def gpt2_goldens_unprocessed():
    """Golden cell for gpt2 without weight processing (no_processing)."""
    from tests import goldens

    if not goldens.goldens_available("gpt2", "no_processing"):
        pytest.skip("TL goldens dataset unavailable (set TL_GOLDENS_DIR or enable network)")
    return goldens.GoldenCell("gpt2", "no_processing")


@pytest.fixture(scope="session")
def gpt2_bridge():
    """TransformerBridge wrapping gpt2 (no compatibility mode). Read-only use only."""
    from transformer_lens.model_bridge import TransformerBridge

    return TransformerBridge.boot_transformers("gpt2", device="cpu")


@pytest.fixture(scope="session")
def gpt2_bridge_compat():
    """TransformerBridge wrapping gpt2 with compatibility mode enabled. Read-only use only."""
    from transformer_lens.model_bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
    bridge.enable_compatibility_mode()
    return bridge


# Full-model fixtures for the acceptance/integration tiers — per tests/AGENTS.md, don't
# adopt them in unit-tier tests (they time out / OOM there).
@pytest.fixture(scope="session")
def gpt2_hooked_processed():
    """Read-only use only — mutations leak across the session."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("gpt2", device="cpu")


@pytest.fixture(scope="session")
def gpt2_hooked_unprocessed():
    """HookedTransformer gpt2 without weight processing. Read-only use only."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")


@pytest.fixture(scope="session")
def gpt2_bridge_compat_no_processing():
    """TransformerBridge wrapping gpt2 with compat mode, no weight processing. Read-only use only."""
    from transformer_lens.model_bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
    bridge.enable_compatibility_mode(no_processing=True)
    return bridge


def pytest_sessionfinish(session, exitstatus):
    """Clean up at the end of test session."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    _SESSION_EXIT_STATUS["code"] = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Watchdog against native-dep shutdown hangs in CI.

    Some native deps can leave threads alive that block interpreter shutdown,
    hanging CI for the whole job timeout *after* the suite has already passed
    and coverage was written. This arms a background watchdog (CI-only, opt-in
    via TL_FORCE_EXIT_AFTER_TESTS) that does nothing on a healthy run — normal
    shutdown kills the daemon thread first — but if the process is still alive a
    full minute after the session ends (i.e. a real hang), dumps every thread's
    traceback to name the culprit, then exits with the suite's real status.
    """
    if os.environ.get("TL_FORCE_EXIT_AFTER_TESTS") != "1":
        return

    def _bail_if_hung():
        time.sleep(60)  # healthy interpreter shutdown completes well within this
        sys.stderr.write(
            "\n[conftest] process still alive 60s after tests finished — shutdown is "
            "hung. Dumping all thread tracebacks, then force-exiting.\n"
        )
        sys.stderr.flush()
        faulthandler.dump_traceback()
        os._exit(_SESSION_EXIT_STATUS["code"])

    threading.Thread(target=_bail_if_hung, name="tl-shutdown-watchdog", daemon=True).start()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
