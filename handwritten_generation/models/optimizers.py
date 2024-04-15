import torch
import torch.nn as nn

from omegaconf import DictConfig


def build_optimizer_from_config(config: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    if config.type == "adam":
        return torch.optim.Adam(model.parameters(), **config.params)
    elif config.type == "adamw":
        return torch.optim.AdamW(model.parameters(), **config.params)
    else:
        raise ValueError(f"Unsupported optimizer type. Received: {config.type}")

def build_scheduler_from_config(config: DictConfig, optimizer: torch.optim.Optimizer):
    if config.type == "none":
        return None
    elif config.type == "reduce_on_plateu":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config.params)
    elif config.type == "linear":
        return torch.optim.lr_scheduler.LinearLR(optimizer, **config.params)
    else:
        raise ValueError(f"Unsupported scheduler type. Received: {config.type}")
        

