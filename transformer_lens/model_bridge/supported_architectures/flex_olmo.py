"""FlexOlmo architecture adapter.

AllenAI's FlexOlmo (``FlexOlmoForCausalLM``, NeurIPS 2025): federated MoE
built by merging independently trained OLMo-2 experts, enabling
inference-time data opt-out by expert selection. Structurally it is the
exact union of the two OLMo variants already supported: OLMo-2's post-norm
blocks and full-width q/k norms, with OLMoE's batched-parameter sparse MoE
(gate_up_proj/down_proj as 3D tensors behind a top-k router) in place of
the dense MLP. The router is a raw-parameter module (not nn.Linear), so it
is wrapped for hooks as a plain delegated component.
"""

from transformer_lens.model_bridge.generalized_components import MoEBridge
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.olmo2 import (
    Olmo2ArchitectureAdapter,
)


class FlexOlmoArchitectureAdapter(Olmo2ArchitectureAdapter):
    """Architecture adapter for FlexOlmoForCausalLM models."""

    def _build_mlp_bridge(self) -> MoEBridge:
        """Batched-expert sparse MoE, delegated to HF's native forward."""
        return MoEBridge(
            name="mlp",
            config=self.cfg,
            submodules={
                "gate": GeneralizedComponent(name="gate"),
            },
        )
