"""Wav2Vec2 architecture adapter.

Wav2Vec2Model's module tree is identical to HubertModel's (zero symmetric
difference over named_modules), so the HuBERT adapter applies wholesale; only
the ForCTC nesting attribute differs (``wav2vec2.`` vs ``hubert.``).
"""

from typing import Any

from transformer_lens.model_bridge.generalized_components import UnembeddingBridge
from transformer_lens.model_bridge.supported_architectures.hubert import (
    HubertArchitectureAdapter,
)


class Wav2Vec2ArchitectureAdapter(HubertArchitectureAdapter):
    """Adapter for Wav2Vec2Model (bare encoder) and Wav2Vec2ForCTC."""

    def prepare_model(self, hf_model: Any) -> None:
        """Detect nesting under 'wav2vec2.' and add the CTC head when present.

        The registered "Wav2Vec2ForPreTraining" architecture string exists so
        checkpoints that DECLARE that class (facebook/wav2vec2-base/-large)
        boot their encoder via AutoModel -> Wav2Vec2Model. Wrapping the
        pretraining class itself is refused: its forward returns a
        Wav2Vec2ForPreTrainingOutput (projected quantizer states, no
        last_hidden_state), which no bridge output contract fits, and the
        quantizer head has no interpretability surface.
        """
        if type(hf_model).__name__ == "Wav2Vec2ForPreTraining":
            raise NotImplementedError(
                "Wav2Vec2ForPreTraining cannot be wrapped directly — boot the "
                "checkpoint without model_class to get its encoder "
                "(Wav2Vec2Model), or use Wav2Vec2ForCTC for the CTC head."
            )
        if hasattr(hf_model, "wav2vec2"):
            self.component_mapping = self._build_component_mapping(prefix="wav2vec2.")
            if hasattr(hf_model, "lm_head"):
                self.component_mapping["unembed"] = UnembeddingBridge(name="lm_head")
