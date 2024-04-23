import hydra
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter

from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from handwritten_generation.models.diffusion import Diffusion
from handwritten_generation.models.optimizers import build_optimizer_from_config, build_scheduler_from_config
from handwritten_generation.datasets.dataset import build_dataloader_from_config


def get_model_size(model):
    numel = 0
    for p in model.parameters():
        numel += p.numel()

    return numel

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_step(model, text_batch, image_batch, criterion, optimizer, device):
    optimizer.zero_grad()

    t = model.sample_timestapms(text_batch.size(0), device=device)
    x_t, noise = model.add_noise_to_images(image_batch, t)

    t = model.time_emb.sample(t)
    text_batch = model.text_emb(text_batch)
    
    predicted_noise = model(x_t, t, text_batch)

    loss = criterion(noise, predicted_noise)
    loss.backward()

    optimizer.step()

    return loss.detach().item()

def train(model, dataloader, criterion, optimizer, scheduler, device, num_epochs, num_log_epochs, logger, validation_text_batch=None):
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for text_batch, images_batch in tqdm(dataloader, total=len(dataloader)):
            
            text_batch, images_batch = text_batch.to(device), images_batch.to(device)
            loss = train_step(model, text_batch, images_batch, criterion, optimizer, device)
            epoch_loss += loss

        epoch_loss /= len(dataloader)
        print(f"Epoch:  {epoch + 1} / {num_epochs}")
        print(f"Loss:  {epoch_loss}")
        logger.add_scalar("Epoch loss", epoch_loss, epoch + 1)
        for param_group in optimizer.param_groups:
            logger.add_scalar("Learning rate", param_group['lr'], epoch + 1)

        if scheduler:
            scheduler.step(epoch_loss)

        if (epoch + 1) % num_log_epochs == 0:
            with torch.no_grad():
                generated_images = model.sample(validation_text_batch, device)
                for i in range(len(validation_text_batch)):
                    text = dataloader.dataset.tokenizer.decode_sequence(validation_text_batch[i])
                    logger.add_image(f"Images/{text}/Epoch_{epoch + 1}", dataloader.dataset.postprocess(generated_images[i].cpu()), epoch + 1)
                del generated_images



@hydra.main(version_base=None, config_path="configs/", config_name="train_model")
def main(hydra_config: DictConfig):
    OmegaConf.set_struct(hydra_config, False)

    experiment_config = hydra_config.experiment
    model_config = hydra_config.model
    dataset_config = hydra_config.dataset
    
    set_seed(experiment_config.seed)

    dataloader = build_dataloader_from_config(dataset_config)
    max_seq_len = max(len(x) for x in dataloader.dataset.tokenized_data)
    vocab_size = len(dataloader.dataset.tokenizer)

    print(f"Max sequence length: {max_seq_len}")
    print(f"Vocabulary size: {vocab_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device used: {device}")

    model = Diffusion(
        model_config,
        img_size=(dataset_config.preprocessor.c, dataset_config.preprocessor.h, dataset_config.preprocessor.w),
        vocab_size=vocab_size,
        max_seq_len=max_seq_len
    )

    model.to(device)
    if model_config.compile:
        print("Model is compiled")
        model = torch.compile(model)

    if experiment_config.from_checkpoint:
        print("Checkpoint is used")
        weights = torch.load(experiment_config.checkpoint_path)
        print(model.load_state_dict(weights["model_state_dict"]))
    
    optimizer = build_optimizer_from_config(experiment_config.optimizer, model)
    scheduler = build_scheduler_from_config(experiment_config.scheduler, optimizer)
    criterion = nn.MSELoss()
    
    num_params = get_model_size(model)
    print(f"Model parameters number: {num_params:,d}")

    

    validation_text_batch, _ = next(iter(dataloader))
    if validation_text_batch.size(0) < experiment_config.num_log_images:
        raise ValueError(f"Number of images to save for logs is larger than batch size. Received: {experiment_config.num_log_images}. Batch size: {validation_text_batch.size(0)}")
    validation_text_batch = validation_text_batch[:experiment_config.num_log_images].to(device)

    with SummaryWriter() as logger:
        with open(f"run_configs/{logger.log_dir}_config.yaml", "x") as f:
            OmegaConf.save(config=hydra_config, f=f.name)
        train(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=experiment_config.num_epochs,
            num_log_epochs=experiment_config.num_log_epochs,
            logger=logger,
            validation_text_batch=validation_text_batch
        )

        torch.save({"model_state_dict": model.state_dict()}, f"weights/{logger.log_dir}.pth")


if __name__ == "__main__":
    main()
