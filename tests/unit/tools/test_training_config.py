"""train() must not mutate the caller's config (device/wandb defaults)."""

import torch
from torch import nn
from torch.utils.data import Dataset

from transformer_lens.tools.training import TrainConfig, train


class _OneSample(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"tokens": torch.zeros(4, dtype=torch.long)}


class _TinyLM(nn.Module):
    """Minimal model exposing the train() contract: __call__(tokens, return_type="loss")."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8)

    def forward(self, tokens, return_type="loss"):
        logits = self.head(self.embed(tokens))
        loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 8), tokens[:, 1:].reshape(-1))
        return loss


def test_train_does_not_mutate_callers_config():
    """train() defaults device/wandb settings on an internal copy; the caller's
    object must come back exactly as it went in."""
    cfg = TrainConfig(num_epochs=1, batch_size=1, wandb=False)
    assert cfg.device is None

    train(_TinyLM(), cfg, _OneSample())

    assert cfg.device is None, "train() leaked its device default into the caller's config"
    assert cfg.wandb_project_name is None
