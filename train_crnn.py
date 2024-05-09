import hydra

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.text import CharErrorRate

from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from pathlib import Path

from handwritten_generation.models.crnn import CRNN
from handwritten_generation.models.ctc_utils import ctc_decode
from handwritten_generation.models.optimizers import (
    build_optimizer_from_config,
    build_scheduler_from_config,
)
from handwritten_generation.datasets.dataset import build_dataloader_from_config
from handwritten_generation.tools.utils import set_seed, get_model_size, clip_grad


def train_step(
    model: CRNN,
    images_batch: torch.FloatTensor,
    tokens_ids_batch: torch.LongTensor,
    target_len_batch: torch.LongTensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_norm_value: float = 1.0,
):
    optimizer.zero_grad()

    logits = model(images_batch)
    log_probas = F.log_softmax(logits, dim=2)

    images_len = torch.LongTensor([logits.size(0)] * logits.size(1))

    loss = criterion(log_probas, tokens_ids_batch, images_len, target_len_batch)
    loss.backward()

    clip_grad(model, grad_norm_value)
    optimizer.step()

    return loss.detach().item(), log_probas.detach()


def train(
    model: CRNN,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metric: nn.Module,
    device: torch.device,
    num_epochs: int,
    logger: SummaryWriter,
    grad_norm_value: float = 1.0,
):
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for images_batch, tokens_ids_batch, target_len_batch in tqdm(
            dataloader, total=len(dataloader)
        ):

            images_batch = images_batch.to(device)
            tokens_ids_batch = tokens_ids_batch.to(device)
            target_len_batch = target_len_batch.to(device)

            loss, log_probas = train_step(
                model=model,
                images_batch=images_batch,
                tokens_ids_batch=tokens_ids_batch,
                target_len_batch=target_len_batch,
                criterion=criterion,
                optimizer=optimizer,
                grad_norm_value=grad_norm_value,
            )

            decoded_tokens_ids = ctc_decode(log_probas)
            decoded_tokens = dataloader.dataset.tokenizer.decode(
                decoded_tokens_ids, ignore_pad=True
            )
            decoded_target_tokens = dataloader.dataset.tokenizer.decode(
                tokens_ids_batch.detach().cpu(), ignore_pad=True
            )
            metric.update(decoded_tokens, decoded_target_tokens)
            epoch_loss += loss
            break

        epoch_loss /= len(dataloader)
        epoch_cer = metric.compute()

        print(f"Epoch:  {epoch + 1} / {num_epochs}")
        print(f"Loss:  {epoch_loss}")
        print(f"Epoch CER: {epoch_cer}")

        logger.add_scalar("Epoch loss", epoch_loss, epoch + 1)
        logger.add_scalar("Epoch CER", epoch_cer, epoch + 1)
        for param_group in optimizer.param_groups:
            logger.add_scalar("Learning rate", param_group["lr"], epoch + 1)

        if scheduler:
            scheduler.step(epoch_loss)
        return None


@torch.no_grad()
def evaluate(
    model: CRNN,
    dataloader: DataLoader,
    criterion: nn.Module,
    metric: nn.Module,
    device: torch.device,
    logger: SummaryWriter,
):
    model.eval()

    test_loss = 0.0

    for images_batch, tokens_ids_batch, target_len_batch in tqdm(
        dataloader, total=len(dataloader)
    ):
        images_batch = images_batch.to(device)
        tokens_ids_batch = tokens_ids_batch.to(device)
        target_len_batch = target_len_batch.to(device)

        logits = model(images_batch)
        log_probas = F.log_softmax(logits, dim=2)

        images_len = torch.LongTensor([logits.size(0)] * logits.size(1))

        loss = criterion(log_probas, tokens_ids_batch, images_len, target_len_batch)

        decoded_tokens_ids = ctc_decode(log_probas)
        decoded_tokens = dataloader.dataset.tokenizer.decode(
            decoded_tokens_ids, ignore_pad=True
        )
        decoded_target_tokens = dataloader.dataset.tokenizer.decode(
            tokens_ids_batch.detach().cpu(), ignore_pad=True
        )

        metric.update(decoded_tokens, decoded_target_tokens)
        test_loss += loss

    test_loss /= len(dataloader)
    test_cer = metric.compute()

    print(f"Test Loss:  {test_loss}")
    print(f"Test CER: {test_cer}")

    logger.add_scalar("Test Loss", test_loss, 1)
    logger.add_scalar("Test CER", test_cer, 1)


@hydra.main(version_base=None, config_path="configs/", config_name="train_crnn")
def main(hydra_config: DictConfig):
    OmegaConf.set_struct(hydra_config, False)

    experiment_config = hydra_config.experiment
    model_config = hydra_config.model
    dataset_config = hydra_config.dataset

    set_seed(experiment_config.seed)

    train_dataloader = build_dataloader_from_config(
        dataset_config.train_dataset,
        is_train=False,
        max_seq_len=dataset_config.max_seq_len,
    )
    test_dataloader = build_dataloader_from_config(
        dataset_config.test_dataset,
        is_train=False,
        max_seq_len=dataset_config.max_seq_len,
    )

    max_seq_len = max(len(x) for x in train_dataloader.dataset.data["tokenized_text"])
    vocab_size = len(train_dataloader.dataset.tokenizer)

    print(f"Dataset max sequence length: {max_seq_len}")
    print(f"Vocabulary size: {vocab_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device used: {device}")

    model = CRNN(config=model_config.model, num_classes=vocab_size)

    model.to(device)
    if model_config.compile:
        print("Model is compiled")
        model = torch.compile(model)

    if "checkpoint_load_path" in experiment_config:
        print(f'Checkpoint "{experiment_config.checkpoint_load_path}" is used')
        weights = torch.load(experiment_config.checkpoint_load_path)
        print(model.load_state_dict(weights["model_state_dict"]))

    optimizer = build_optimizer_from_config(experiment_config.optimizer, model)
    scheduler = build_scheduler_from_config(experiment_config.scheduler, optimizer)
    criterion = nn.CTCLoss(blank=train_dataloader.dataset.tokenizer.pad_idx)
    cer = CharErrorRate()

    num_params = get_model_size(model)
    print(f"Model parameters number: {num_params:,d}")

    with SummaryWriter() as logger:
        log_filename = Path(logger.log_dir).name

        with open(f"{logger.log_dir}/experiment_config.yaml", "x") as f:
            OmegaConf.save(config=hydra_config, f=f.name)
        train(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=cer,
            device=device,
            num_epochs=experiment_config.num_epochs,
            logger=logger,
        )

        if len(experiment_config.checkpoint_save_path) > 0:
            checkpoint_save_path = experiment_config.checkpoint_save_path
        else:
            checkpoint_save_path = "weights/{log_filename}.pth"

        torch.save({"model_state_dict": model.state_dict()}, checkpoint_save_path)

        evaluate(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            metric=cer,
            device=device,
            logger=logger,
        )


if __name__ == "__main__":
    main()
