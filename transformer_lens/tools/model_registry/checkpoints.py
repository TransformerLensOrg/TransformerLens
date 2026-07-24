"""Training-checkpoint label data for checkpointed model families.

Canonical home for the checkpoint schedules previously defined in
``transformer_lens/loading_from_pretrained.py``. The schedules are frozen
historical artifacts of the published training runs.
"""

import logging

from .registry_io import resolve_model_alias

# The steps for which there are checkpoints in the stanford crfm models
STANFORD_CRFM_CHECKPOINTS: list[int] = (
    list(range(0, 100, 10))
    + list(range(100, 2000, 50))
    + list(range(2000, 20000, 100))
    + list(range(20000, 400000 + 1, 1000))
)

# Linearly spaced checkpoints for Pythia models, taken every 1000 steps.
# Batch size 2,097,152 tokens, so checkpoints every 2.1B tokens
PYTHIA_CHECKPOINTS: list[int] = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(
    range(1000, 143000 + 1, 1000)
)
# Pythia V1 has log-spaced early checkpoints (see line above), but V0 doesn't
PYTHIA_V0_CHECKPOINTS: list[int] = list(range(1000, 143000 + 1, 1000))


def get_checkpoint_labels(model_name: str) -> tuple[list[int], str]:
    """Return (checkpoint labels, label type) for a checkpointed model family.

    Covers the HF-revision-checkpointed families (Pythia, stanford-crfm).
    Raises ValueError for models without published checkpoint schedules.
    """
    official_name = resolve_model_alias(model_name) or model_name
    if official_name.startswith("stanford-crfm/"):
        return STANFORD_CRFM_CHECKPOINTS, "step"
    if official_name.startswith("EleutherAI/pythia"):
        if "v0" in official_name:
            return PYTHIA_V0_CHECKPOINTS, "step"
        logging.warning(
            "Pythia models on HF were updated on 4/3/23! add '-v0' to model name to access the old models."
        )
        return PYTHIA_CHECKPOINTS, "step"
    raise ValueError(f"Model {official_name} is not checkpointed.")
