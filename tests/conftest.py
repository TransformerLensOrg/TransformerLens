"""Global pytest configuration for memory management and test optimization."""

import gc

import pytest
import torch


@pytest.fixture(autouse=True, scope="function")
def cleanup_memory():
    """Automatically clean up memory after each test."""
    yield
    # Clear torch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Force garbage collection for cleanup
    gc.collect()


@pytest.fixture(autouse=True, scope="class")
def cleanup_class_memory():
    """Clean up memory after each test class."""
    yield
    # More aggressive cleanup after test classes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# Configure pytest to be more memory-efficient
def pytest_configure(config):
    """Configure pytest for better memory usage."""
    # Configure garbage collection to be more aggressive
    gc.set_threshold(700, 10, 10)


def pytest_sessionfinish(session, exitstatus):
    """Clean up at the end of test session."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
