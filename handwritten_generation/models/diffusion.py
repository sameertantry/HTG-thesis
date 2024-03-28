import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig

from time import time
from tqdm import tqdm

from handwritten_generation.models.modules import UNet, AvgTextEmbedding, PositionalEncoding


def build_model_from_config(config: DictConfig) -> nn.Module:
    return UNet();

class Diffusion(nn.Module):
    def __init__(self, config: DictConfig, img_size: tuple[int, int], vocab_size: int, max_seq_len: int):
        super().__init__()

        self.config = config
        
        self.noise_steps = self.config.noise_steps
        self.beta_start = self.config.beta_start
        self.beta_end = self.config.beta_end
        beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        self.register_buffer("beta", beta)

        alpha = 1. - beta
        self.register_buffer("alpha", alpha)

        alpha_hat = torch.cumprod(alpha, dim=0)
        self.register_buffer("alpha_hat", alpha_hat)
        
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

        self.model = build_model_from_config(config)
        self.text_emb = AvgTextEmbedding(
            vocab_size=vocab_size,
            emb_size=config.text_emb_dim,
            max_seq_len=max_seq_len,
        )
        self.time_emb = PositionalEncoding(
            emb_size=config.time_emb_dim,
            max_seq_len=self.noise_steps+1,
        )

    def forward(self, x, time_emb, text_emb):
        return self.model(x, time_emb, text_emb)

    def add_noise_to_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1, 1)
        sqrt_symmetric_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1, 1)

        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_symmetric_alpha_hat * noise, noise
        
    def sample_timestapms(self, n, device):
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=device)

    
    def sample(self, tokenized_text: torch.Tensor, device: str = "cuda"):
        self.eval()

        text_emb = self.text_emb(tokenized_text)
        x = torch.randn((tokenized_text.size(0), 1, self.img_size[0], self.img_size[1]), device=device)
        iterations = torch.arange(1, self.noise_steps, device=device).view(-1, 1).repeat(1, tokenized_text.size(0))

        with torch.no_grad():
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
              x = 1. / torch.sqrt(alpha) * (x - ((1. - alpha) / (torch.sqrt(1. - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
  

        x = x.clamp(-1., 1.)

        self.train()
        
        return x