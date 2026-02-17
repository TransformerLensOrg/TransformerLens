"""LIT (Learning Interpretability Tool) integration for TransformerLens.

This module provides integration between TransformerLens and Google's Learning
Interpretability Tool (LIT), enabling interactive visualization and analysis
of transformer models.

Quick Start:
    >>> from transformer_lens import HookedTransformer  # doctest: +SKIP
    >>> from transformer_lens.lit import HookedTransformerLIT, SimpleTextDataset, serve  # doctest: +SKIP
    >>>
    >>> # Load model and create LIT wrapper
    >>> model = HookedTransformer.from_pretrained("gpt2-small")  # doctest: +SKIP
    >>> lit_model = HookedTransformerLIT(model)  # doctest: +SKIP
    >>>
    >>> # Create a dataset
    >>> dataset = SimpleTextDataset.from_strings([  # doctest: +SKIP
    ...     "The capital of France is Paris.",
    ...     "Machine learning is a field of AI.",
    ... ])
    >>>
    >>> # Start LIT server
    >>> serve({"gpt2": lit_model}, {"examples": dataset})  # doctest: +SKIP

For Colab/Jupyter notebooks:
    >>> from transformer_lens.lit import LITWidget  # doctest: +SKIP
    >>>
    >>> widget = LITWidget({"gpt2": lit_model}, {"examples": dataset})  # doctest: +SKIP
    >>> widget.render()  # doctest: +SKIP

Features:
    - Interactive token predictions and top-k analysis
    - Attention pattern visualization across all layers and heads
    - Embedding projector for layer-wise representations
    - Token salience/gradient visualization
    - Support for IOI and Induction datasets

Requirements:
    - lit-nlp >= 1.0 (install with: pip install lit-nlp)

References:
    - LIT: https://pair-code.github.io/lit/
    - TransformerLens: https://github.com/TransformerLensOrg/TransformerLens

Note:
    This module requires the optional `lit-nlp` dependency. Install it with:
    ```
    pip install lit-nlp
    ```
    or
    ```
    pip install transformer-lens[lit]
    ```
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Union

# Check if LIT is available
from .utils import check_lit_installed

__all__ = [
    # Model wrappers
    "HookedTransformerLIT",
    "HookedTransformerLITBatched",
    "HookedTransformerLITConfig",
    # Datasets
    "SimpleTextDataset",
    "PromptCompletionDataset",
    "IOIDataset",
    "InductionDataset",
    "wrap_for_lit",
    # Server utilities
    "serve",
    "LITWidget",
    # Configuration
    "HookedTransformerLITConfig",
    # Constants
    "INPUT_FIELDS",
    "OUTPUT_FIELDS",
    # Utilities
    "check_lit_installed",
]

logger = logging.getLogger(__name__)

# Import constants (always available)
from .constants import (  # noqa: E402
    ERRORS,
    INPUT_FIELDS,
    OUTPUT_FIELDS,
    SERVER_CONFIG,
)

# Import model wrapper (handles LIT availability internally)
from .model import (  # noqa: E402
    HookedTransformerLIT,
    HookedTransformerLITConfig,
)

# Import datasets (handles LIT availability internally)
from .dataset import (  # noqa: E402
    IOIDataset,
    InductionDataset,
    PromptCompletionDataset,
    SimpleTextDataset,
    wrap_for_lit,
)

# Conditional imports that require LIT
_LIT_AVAILABLE = check_lit_installed()

if _LIT_AVAILABLE:
    from .model import HookedTransformerLITBatched  # noqa: E402
else:
    HookedTransformerLITBatched = None  # type: ignore[misc, assignment]


def serve(
    models: Union[Dict[str, Any], Any],
    datasets: Union[Dict[str, Any], Any],
    port: int = SERVER_CONFIG.DEFAULT_PORT,
    host: str = SERVER_CONFIG.DEFAULT_HOST,
    page_title: str = SERVER_CONFIG.DEFAULT_TITLE,
    **kwargs,
) -> None:
    """Start a LIT server with the given models and datasets.

    This is a convenience function to quickly start a LIT server
    for interactive model exploration.

    Args:
        models: Either a single HookedTransformer/HookedTransformerLIT, or
                a dictionary mapping model names to model wrappers.
        datasets: Either a single dataset, or a dictionary mapping
                  dataset names to datasets.
        port: Port number for the server.
        host: Host address for the server.
        page_title: Title shown in the browser tab.
        **kwargs: Additional arguments passed to LIT server.

    Example:
        >>> from transformer_lens import HookedTransformer  # doctest: +SKIP
        >>> from transformer_lens.lit import SimpleTextDataset, serve  # doctest: +SKIP
        >>>
        >>> model = HookedTransformer.from_pretrained("gpt2-small")  # doctest: +SKIP
        >>> dataset = SimpleTextDataset.from_strings(["Hello world!"])  # doctest: +SKIP
        >>>
        >>> # Simple usage with single model and dataset
        >>> serve(model, dataset)  # doctest: +SKIP
        >>>
        >>> # Or with explicit names
        >>> serve({"gpt2": model}, {"examples": dataset})  # doctest: +SKIP

    Note:
        This function will block and run the server. Press Ctrl+C to stop.
    """
    if not _LIT_AVAILABLE:
        raise ImportError(ERRORS.LIT_NOT_INSTALLED)

    from lit_nlp import dev_server

    # Handle single model vs dictionary of models
    if not isinstance(models, dict):
        # Single model passed - check if it's a HookedTransformer that needs wrapping
        model = models
        if hasattr(model, "cfg") and hasattr(model, "run_with_cache"):
            # It's a HookedTransformer, wrap it
            model = HookedTransformerLIT(model)
        models = {"model": model}

    # Handle single dataset vs dictionary of datasets
    if not isinstance(datasets, dict):
        datasets = {"dataset": datasets}

    # Wrap datasets if needed
    wrapped_datasets = {}
    for name, dataset in datasets.items():
        if hasattr(dataset, "_examples"):
            # Our custom dataset, wrap it
            wrapped_datasets[name] = wrap_for_lit(dataset)
        else:
            # Already a LIT dataset
            wrapped_datasets[name] = dataset

    # Get the LIT client root path and layout
    import os
    import lit_nlp
    from lit_nlp.api import layout as lit_layout

    client_root = os.path.join(os.path.dirname(lit_nlp.__file__), "client", "build", "default")

    # Use default layouts if not provided
    if "layouts" not in kwargs:
        kwargs["layouts"] = lit_layout.DEFAULT_LAYOUTS
    if "default_layout" not in kwargs:
        kwargs["default_layout"] = "default"

    # Create and start server
    server = dev_server.Server(
        models,
        wrapped_datasets,
        port=port,
        host=host,
        page_title=page_title,
        client_root=client_root,
        **kwargs,
    )

    logger.info(f"Starting LIT server at http://{host}:{port}")
    server.serve()


class LITWidget:
    """LIT Widget for Jupyter/Colab notebooks.

    This class provides an easy way to use LIT within notebook environments
    without needing to run a separate server.

    Example:
        >>> from transformer_lens import HookedTransformer  # doctest: +SKIP
        >>> from transformer_lens.lit import HookedTransformerLIT, SimpleTextDataset, LITWidget  # doctest: +SKIP
        >>>
        >>> model = HookedTransformer.from_pretrained("gpt2-small")  # doctest: +SKIP
        >>> lit_model = HookedTransformerLIT(model)  # doctest: +SKIP
        >>> dataset = SimpleTextDataset.from_strings(["Hello world!"])  # doctest: +SKIP
        >>>
        >>> widget = LITWidget({"gpt2": lit_model}, {"examples": dataset})  # doctest: +SKIP
        >>> widget.render()  # Displays in the notebook  # doctest: +SKIP

    Note:
        VSCode notebooks don't support iframe rendering. Use `widget.url` to
        get the URL and open it manually in your browser.
    """

    def __init__(
        self,
        models: Dict[str, Any],
        datasets: Dict[str, Any],
        height: int = 800,
        **kwargs,
    ):
        """Initialize the LIT widget.

        Args:
            models: Dictionary mapping model names to model wrappers.
            datasets: Dictionary mapping dataset names to datasets.
            height: Height of the widget in pixels.
            **kwargs: Additional arguments for the LIT widget.
        """
        if not _LIT_AVAILABLE:
            raise ImportError(ERRORS.LIT_NOT_INSTALLED)

        from lit_nlp import notebook

        # Wrap datasets if needed
        wrapped_datasets = {}
        for name, dataset in datasets.items():
            if hasattr(dataset, "_examples"):
                wrapped_datasets[name] = wrap_for_lit(dataset)
            else:
                wrapped_datasets[name] = dataset

        # LitWidget expects models and datasets as positional args
        # Remove default_layout from kwargs as it's handled internally by LitWidget
        kwargs.pop("default_layout", None)

        self._widget = notebook.LitWidget(
            models,
            wrapped_datasets,
            height=height,
            render=False,  # Don't auto-render
            **kwargs,
        )

    @property
    def url(self) -> str:
        """Get the URL of the LIT server.

        Use this to manually open LIT in a browser when notebook
        rendering doesn't work (e.g., in VSCode).

        Returns:
            The URL to access the LIT UI.
        """
        port = self._widget._server.port
        return f"http://localhost:{port}"

    def render(self, open_in_new_tab: bool = False, **kwargs):
        """Render the LIT widget.

        Args:
            open_in_new_tab: If True, opens in a new browser tab.
            **kwargs: Additional render arguments.

        Note:
            If rendering doesn't work in your environment (e.g., VSCode),
            use `print(widget.url)` and open that URL in your browser.
        """
        self._widget.render(open_in_new_tab=open_in_new_tab, **kwargs)

    def stop(self):
        """Stop the widget's server and free resources."""
        self._widget.stop()


# Version info
__version__ = "1.0.0"

