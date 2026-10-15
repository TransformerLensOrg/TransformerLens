"""Run the epsilon regression with an intentional in-memory fault.

Run from the repository root using the same environment as pytest.
Expected outcome: four assertion failures and process exit code 1.
No source file is modified. This is an analogous fault, not the June reproducer.
"""

import inspect

import pytest

import transformer_lens.model_bridge.generalized_components.normalization as module
from transformer_lens.model_bridge.generalized_components import NormalizationBridge

source = inspect.getsource(module)
old = (
    'hidden_states.pow(2).mean(-1, keepdim=True) + getattr(self.config, "eps", 1e-05)'
    '\n                ).sqrt()'
)
new = (
    'hidden_states.pow(2).mean(-1, keepdim=True)'
    '\n                ).sqrt() + getattr(self.config, "eps", 1e-05)'
)
assert source.count(old) == 1, "Normalization implementation changed; review mutation target"
namespace = dict(module.__dict__)
exec(compile(source.replace(old, new), "<epsilon-order-mutation>", "exec"), namespace)
NormalizationBridge.forward = namespace["NormalizationBridge"].forward
raise SystemExit(
    pytest.main(
        [
            "-o", "addopts=",
            "tests/integration/model_bridge/test_cohere_adapter.py",
            "-k", "folded_norm_epsilon",
            "-q", "--tb=short", "--show-capture=no",
        ]
    )
)
