from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.EasyTransformerConfig import EasyTransformerConfig
from dataclasses import dataclass
from typing import Optional, Callable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import wandb
import torch


@dataclass
class EasyTransformerTrainConfig:
    """
    Configuration class to store training hyperparameters for a training run of
    an EasyTransformer model.
    Args:
        num_epochs (int): Number of epochs to train for
        batch_size (int): Size of batches to use for training
        lr (float): Learning rate to use for training
        seed (int): Random seed to use for training
        momentum (float): Momentum to use for training
        max_grad_norm (float, *optional*): Maximum gradient norm to use for
        weight_decay (float, *optional*): Weight decay to use for training
            training
        optimizer_name (str): The name of the optimizer to use
        device (str, *optional*): Device to use for training
        warmup_steps (int, *optional*): Number of warmup steps to use for training
        save_every (int, *optional*): After how many batches should a checkpoint be saved
        save_dir, (str, *optional*): Where to save checkpoints
        wandb (bool): Whether to use Weights and Biases for logging
        wandb_project (str, *optional): Name of the Weights and Biases project to use
    """

    num_epochs: int
    batch_size: int
    lr: float
    seed: int = 0
    momentum: float = 0.0
    max_grad_norm: Optional[float] = None
    weight_decay: Optional[float] = None
    optimizer_name: str = "Adam"
    device: Optional[str] = None
    warmup_steps: int = 0
    save_every: Optional[int] = None
    save_dir: Optional[str] = None
    wandb: bool = False
    wandb_project_name: Optional[str] = None


def train(
    model: EasyTransformer,
    config: EasyTransformerTrainConfig,
    dataset: Dataset,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> EasyTransformer:
    """
    Trains an EasyTransformer model.
    Args:
        model: The model to train
        config: The training configuration
        dataset: The dataset to train on
        loss_fn: The loss function to use for training
    Returns:
        The trained model
    """
    torch.manual_seed(config.seed)
    model.train()
    if config.wandb:
        if config.wandb_project_name is None:
            config.wandb_project_name = "easy-transformer"
        wandb.init(project=config.wandb_project_name, config=vars(config))

    if config.device is None:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.optimizer_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
            if config.weight_decay is not None
            else 0.0,
        )
    elif config.optimizer_name == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
            if config.weight_decay is not None
            else 1e-2,
        )
    elif config.optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
            if config.weight_decay is not None
            else 0.0,
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

    for epoch in range(config.num_epochs):
        steps = 0
        for i, (x, y) in enumerate(dataloader):
            x = x.to(config.device)
            y = y.to(config.device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if config.warmup_steps > 0:
                assert scheduler is not None
                scheduler.step()
            optimizer.zero_grad()

            steps += x.shape[0]

            if config.wandb:
                wandb.log({"train_loss": loss.item(), "steps": steps, epoch: epoch})

            if (
                config.save_every is not None
                and i % config.save_every == 0
                and config.save_dir is not None
            ):
                torch.save(model.state_dict(), f"{config.save_dir}/model_{i}.pt")

    return model
