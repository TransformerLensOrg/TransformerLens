"""train() must not mutate the caller's config (device/wandb defaults)."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.train import HookedTransformerTrainConfig, train


def test_train_leaves_caller_config_untouched() -> None:
    model = HookedTransformer(
        HookedTransformerConfig(
            n_layers=1, d_model=16, d_head=8, n_heads=2, n_ctx=8, d_vocab=16, act_fn="gelu"
        )
    )

    class _TokensDataset(Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int) -> dict:
            return {"tokens": torch.tensor([1, 2, 3, 4])}

    dataset = _TokensDataset()
    config = HookedTransformerTrainConfig(
        num_epochs=1,
        batch_size=1,
        lr=1e-3,
        seed=0,
        device=None,
    )

    train(model, config, dataset)

    assert config.device is None, "train() wrote its resolved device onto the caller's config"
