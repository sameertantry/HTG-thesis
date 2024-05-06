import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L

from omegaconf import DictConfig

from time import time
from tqdm import tqdm

from handwritten_generation.models.modules import (
    AttentionTextEmbbeding,
    PositionalEncoding,
)
from handwritten_generation.models.unet import UNet


def build_model_from_config(config: DictConfig) -> nn.Module:
    return UNet(**config)


def build_beta_schedule_from_config(config: DictConfig):
    if config.type == "lin":
        return torch.linspace(config.beta_start, config.beta_end, config.noise_steps)
    else:
        raise ValueError(f"Unsupported schedule type. Received: {config.type}.")


class Diffusion(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        img_size: tuple[int, int, int],
        vocab_size: int,
        max_seq_len: int,
    ):
        super().__init__()

        self.config = config

        self.noise_steps = self.config.diffusion.noise_steps
        self.beta_start = self.config.diffusion.beta_start
        self.beta_end = self.config.diffusion.beta_end
        beta = build_beta_schedule_from_config(config.diffusion)
        self.register_buffer("beta", beta)

        alpha = 1.0 - beta
        self.register_buffer("alpha", alpha)

        alpha_hat = torch.cumprod(alpha, dim=0)
        self.register_buffer("alpha_hat", alpha_hat)

        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

        self.model = build_model_from_config(config.model)
        self.text_emb = AttentionTextEmbbeding(
            vocab_size=vocab_size,
            emb_dim=config.model.text_emb_dim,
            max_seq_len=max_seq_len,
        )
        self.time_emb = PositionalEncoding(
            emb_dim=config.model.time_emb_dim,
            max_seq_len=self.noise_steps + 1,
        )

    def forward(self, x, time_emb, text_emb):
        return self.model(x, time_emb, text_emb)

    def add_noise_to_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_symmetric_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]

        noise = torch.randn_like(x)
        noise_images = sqrt_alpha_hat * x + sqrt_symmetric_alpha_hat * noise

        return noise_images, noise

    def sample_timestapms(self, n, device):
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=device)

    @torch.no_grad()
    def sample(self, tokenized_text: torch.Tensor, device: str):
        self.eval()

        batch_size = tokenized_text.size(0)

        text_emb = self.text_emb(tokenized_text)
        x = torch.randn((batch_size, *self.img_size), device=device)
        iterations = (
            torch.arange(1, self.noise_steps, device=device)
            .view(-1, 1)
            .repeat(1, batch_size)
        )

        for t in reversed(range(1, self.noise_steps)):
            batch_t = iterations[t - 1]

            time_emb = self.time_emb.sample(batch_t)

            predicted_noise = self(x, time_emb, text_emb)

            alpha = self.alpha[t].view(-1, 1, 1, 1)
            alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
            beta = self.beta[t].view(-1, 1, 1, 1)

            if t > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = (
                1.0
                / torch.sqrt(alpha)
                * (
                    x
                    - ((1.0 - alpha) / (torch.sqrt(1.0 - alpha_hat))) * predicted_noise
                )
                + torch.sqrt(beta) * noise
            )

        x = x.clamp(-1.0, 1.0)

        self.train()

        return x
