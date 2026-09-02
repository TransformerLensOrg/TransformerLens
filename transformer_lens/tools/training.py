"""Train loop for TransformerLens models.

Utilities for training models on autoregressive language modeling tasks.
Typed against the ``model_protocol`` surface (``__call__`` with
``return_type="loss"`` plus standard ``nn.Module`` parameter access), so any
conforming model — ``TransformerBridge`` foremost — works through this loop.
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from transformer_lens import utilities as utils
from transformer_lens.model_protocol import TrainableTransformerLensModel
from transformer_lens.utilities.library_utils import is_library_available


@dataclass
class TrainConfig:
    """Configuration class to store training hyperparameters for a training run.

    Args:
        num_epochs (int): Number of epochs to train for
        batch_size (int): Size of batches to use for training
        lr (float): Learning rate to use for training
        seed (int): Random seed to use for training
        momentum (float): Momentum to use for training
        max_grad_norm (float, *optional*): Maximum gradient norm to use for
        weight_decay (float, *optional*): Weight decay to use for training
        optimizer_name (str): The name of the optimizer to use
        device (str or torch.device, *optional*): Device to use for training
        warmup_steps (int, *optional*): Number of warmup steps to use for training
        save_every (int, *optional*): After how many batches should a checkpoint be saved
        save_dir, (str, *optional*): Where to save checkpoints
        wandb (bool): Whether to use Weights and Biases for logging
        wandb_project (str, *optional*): Name of the Weights and Biases project to use
        print_every (int, *optional*): Print the loss every n steps
        max_steps (int, *optional*): Terminate the epoch after this many steps. Used for debugging.
    """

    num_epochs: int
    batch_size: int
    lr: float = 1e-3
    seed: int = 0
    momentum: float = 0.0
    max_grad_norm: Optional[float] = None
    weight_decay: Optional[float] = None
    optimizer_name: str = "Adam"
    device: Optional[Union[str, torch.device]] = None
    warmup_steps: int = 0
    save_every: Optional[int] = None
    save_dir: Optional[str] = None
    wandb: bool = False
    wandb_project_name: Optional[str] = None
    print_every: Optional[int] = 50
    max_steps: Optional[int] = None


def train(
    model: TrainableTransformerLensModel,
    config: TrainConfig,
    dataset: Dataset,
) -> TrainableTransformerLensModel:
    """Train a model on an autoregressive language modeling task.

    Args:
        model: The model to train (TrainableTransformerLensModel: callable with
            ``return_type="loss"`` and exposing torch parameters)
        config: The training configuration
        dataset: The dataset to train on - assumed set up for autoregressive language modeling.

    Returns:
        The trained model
    """

    # Work on a copy: mutating the caller's config (wandb_project_name/device
    # defaults below) was a silent side effect the caller never asked for.
    config = dataclasses.replace(config)

    torch.manual_seed(config.seed)
    model.train()

    if config.wandb:
        if not is_library_available("wandb"):
            raise ImportError("Wandb is not available")

        import wandb

        if config.wandb_project_name is None:
            config.wandb_project_name = "easy-transformer"
        wandb.init(project=config.wandb_project_name, config=vars(config))

    if config.device is None:
        config.device = utils.get_device()

    optimizer: Optimizer
    if config.optimizer_name in ["Adam", "AdamW"]:
        # Weight decay in Adam is implemented badly, so use AdamW instead (see PyTorch AdamW docs)
        if config.weight_decay is not None:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.lr,
            )
    elif config.optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=(config.weight_decay if config.weight_decay is not None else 0.0),
            momentum=config.momentum,
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer_name} not supported")

    scheduler = None
    if config.warmup_steps > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / config.warmup_steps),
        )

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model.to(config.device)

    for epoch in tqdm(range(1, config.num_epochs + 1)):
        samples = 0
        for step, batch in tqdm(enumerate(dataloader)):
            tokens = batch["tokens"].to(config.device)
            loss = model(tokens, return_type="loss")
            loss.backward()
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if config.warmup_steps > 0:
                assert scheduler is not None
                scheduler.step()
            optimizer.zero_grad()

            samples += tokens.shape[0]

            if config.wandb:
                wandb.log({"train_loss": loss.item(), "samples": samples, "epoch": epoch})

            if config.print_every is not None and step % config.print_every == 0:
                print(f"Epoch {epoch} Samples {samples} Step {step} Loss {loss.item()}")

            if (
                config.save_every is not None
                and step % config.save_every == 0
                and config.save_dir is not None
            ):
                torch.save(model.state_dict(), f"{config.save_dir}/model_{step}.pt")

            if config.max_steps is not None and step >= config.max_steps:
                break

    return model
