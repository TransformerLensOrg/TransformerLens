import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--max-model-gb",
        action="store",
        default=1.0,
        type=float,
        help="Max model size in GB (float32) for model_accuracy tests. Default: 1.0",
    )
    parser.addoption(
        "--model",
        action="append",
        default=None,
        help="Specific model name(s) to test (can be repeated). Overrides the default list.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "model_accuracy: model accuracy tests (not run by default)")


def pytest_collection_modifyitems(config, items):
    """Skip model_accuracy tests unless explicitly selected with -m model_accuracy."""
    if "model_accuracy" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason="model_accuracy tests only run with -m model_accuracy")
    for item in items:
        if "model_accuracy" in item.keywords:
            item.add_marker(skip)
