"""Unit tests for Olmo3ArchitectureAdapter.

The adapter body is `pass` — it inherits its config, component mapping, and
weight conversions unchanged from Olmo2ArchitectureAdapter (covered by
test_olmo2_adapter.py). The only contract this subclass owns is the subclass
relationship itself.
"""

from transformer_lens.model_bridge.supported_architectures.olmo2 import (
    Olmo2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.olmo3 import (
    Olmo3ArchitectureAdapter,
)


class TestOlmo3Inheritance:
    """Subclass relationship to Olmo2ArchitectureAdapter. The class body is
    `pass`, so the inherited surface is the contract worth pinning. A future
    accidental override would be caught here.
    """

    def test_subclass_of_olmo2(self) -> None:
        assert issubclass(Olmo3ArchitectureAdapter, Olmo2ArchitectureAdapter)
