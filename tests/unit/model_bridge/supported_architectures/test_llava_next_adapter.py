"""Unit tests for LlavaNextArchitectureAdapter.

The adapter body is `pass` — it inherits its config, component mapping, and
weight conversions unchanged from LlavaArchitectureAdapter (covered by
test_llava_adapter.py). The only contract this subclass owns is the subclass
relationship itself.
"""

from transformer_lens.model_bridge.supported_architectures.llava import (
    LlavaArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.llava_next import (
    LlavaNextArchitectureAdapter,
)


class TestLlavaNextInheritance:
    """Subclass relationship to LlavaArchitectureAdapter. The class body is
    `pass`; the inherited surface is the contract worth pinning so a future
    accidental override is caught.
    """

    def test_subclass_of_llava(self) -> None:
        assert issubclass(LlavaNextArchitectureAdapter, LlavaArchitectureAdapter)
