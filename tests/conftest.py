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
    """Automatically clean up memory after each test."""
    yield
    # Clear torch cache for all accelerators
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    # Force garbage collection for cleanup
    gc.collect()


@pytest.fixture(autouse=True, scope="class")
def cleanup_class_memory():
    """Clean up memory after each test class."""
    yield
    # More aggressive cleanup after test classes
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


@pytest.fixture(scope="session")
def gpt2_hooked_processed():
    """Read-only use only — mutations leak across the session."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("gpt2", device="cpu")


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
