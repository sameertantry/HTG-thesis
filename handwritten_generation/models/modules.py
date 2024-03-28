import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConnection(nn.Module):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward(self, x):
        return x + self.module(x)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, max_seq_len: int):
        super().__init__()

        pe = torch.zeros(max_seq_len, emb_size)
        positions = torch.arange(0, max_seq_len).unsqueeze(1)
        div = torch.exp(-torch.log(torch.tensor(10000)) * torch.arange(0, emb_size, 2) / emb_size)  # sin (pos / 10000 ** (2i / emb_size))
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

    def sample(self, x):
        return self.pe[x, :]


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, max_seq_len: int):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.pe = PositionalEncoding(emb_size=emb_size, max_seq_len=max_seq_len)

    def forward(self, x):
        return self.pe(self.emb(x))

class AvgTextEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, max_seq_len: int):
        super().__init__()

        self.emb = TokenEmbedding(vocab_size=vocab_size, emb_size=emb_size, max_seq_len=max_seq_len)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return self.pooling(self.emb(x).transpose(1, 2)).transpose(1, 2).squeeze(1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, kernel_size: int = 3, residual: bool = False):
        super().__init__()
        
        mid_channels = mid_channels if mid_channels else in_channels
        padding = kernel_size // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(1, out_channels),
        )

        if residual:
            self.convs = ResConnection(self.convs)

    def forward(self, x):
        return F.gelu(self.convs(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, time_emb_dim: int, text_emb_dim: int):
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, residual=True),
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
        )

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_emb_dim,
                out_channels,
            )
        )
        self.text_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                text_emb_dim,
                out_channels,
            )
        )

    def forward(self, x, time_emb, text_emb):
        x = self.down(x)

        time_emb = self.time_proj(time_emb)
        time_emb = time_emb[:, :, None, None].repeat(1, 1, x.size(-2), x.size(-1))

        text_emb = self.text_proj(text_emb)
        text_emb = text_emb[:, :, None, None].repeat(1, 1, x.size(-2), x.size(-1))

        return F.gelu(x + time_emb + text_emb)

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, time_emb_dim: int, text_emb_dim: int):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.convs = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, residual=True),
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
        )

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_emb_dim,
                out_channels,
            )
        )
        self.text_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                text_emb_dim,
                out_channels,
            )
        )

    def forward(self, x, time_emb, text_emb, res_x):
        x = self.up(x)
        x = torch.cat([x, res_x], dim=1)
        x = self.convs(x)
        
        time_emb = self.time_proj(time_emb)[:, :, None, None].repeat(1, 1, x.size(-2), x.size(-1))
        text_emb = self.text_proj(text_emb)[:, :, None, None].repeat(1, 1, x.size(-2), x.size(-1))

        return F.gelu(x + time_emb + text_emb)


class SelfAttention(nn.Module):
    def __init__(self, num_channels: int, num_heads: int, dropout: float = 0.):
        super().__init__()

        self.mha = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.layer_norm = nn.LayerNorm(num_channels)
        self.mlp = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_channels),
            nn.GELU(),
            nn.Linear(num_channels, num_channels),
        )
        self.mlp = ResConnection(self.mlp)

    def forward(self, x):
        _, num_channels, h, w = x.size()
        
        x = x.flatten(start_dim=-2).transpose(1, 2)
        x = self.layer_norm(x)
        x = x + self.mha(x, x, x, need_weights=False)[0]
        x = self.mlp(x)

        return x.transpose(1, 2).view(-1, num_channels, h, w)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, time_emb_dim: int = 256, text_emb_dim: int = 64):
        super().__init__()

        self.time_emb_dim = time_emb_dim
        self.text_emb_dim = text_emb_dim

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=16)
        self.down1 = DownBlock(in_channels=16, out_channels=32, kernel_size=3, time_emb_dim=time_emb_dim, text_emb_dim=text_emb_dim)

        self.conv2 = ConvBlock(in_channels=32, out_channels=64)
        self.down2 = DownBlock(in_channels=64, out_channels=128, kernel_size=3, time_emb_dim=time_emb_dim, text_emb_dim=text_emb_dim)

        self.conv3 = ConvBlock(in_channels=128, out_channels=256)
        self.down3 = DownBlock(in_channels=256, out_channels=512, kernel_size=3, time_emb_dim=time_emb_dim, text_emb_dim=text_emb_dim)

        self.bn1 = ConvBlock(in_channels=512, out_channels=1024)
        self.bn2 = ConvBlock(in_channels=1024, out_channels=1024, residual=True)
        self.bn3 = ConvBlock(in_channels=1024, out_channels=512)

        self.up1 = UpBlock(in_channels=512 + 256, out_channels=256, kernel_size=3, time_emb_dim=time_emb_dim, text_emb_dim=text_emb_dim)
        self.up2 = UpBlock(in_channels=256 + 64, out_channels=128, kernel_size=3, time_emb_dim=time_emb_dim, text_emb_dim=text_emb_dim)
        self.up3 = UpBlock(in_channels=128 + 16, out_channels=64, kernel_size=3, time_emb_dim=time_emb_dim, text_emb_dim=text_emb_dim)

        self.proj = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=32),
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x, time_emb, text_emb):
        x1 = self.conv1(x)
        x2 = self.down1(x1, time_emb, text_emb)

        x2 = self.conv2(x2)
        x3 = self.down2(x2, time_emb, text_emb)

        x3 = self.conv3(x3)
        x = self.down3(x3, time_emb, text_emb)

        x = self.bn1(x)
        x = self.bn2(x)
        x = self.bn3(x)

        x = self.up1(x, time_emb, text_emb, x3)
        x = self.up2(x, time_emb, text_emb, x2)
        x = self.up3(x, time_emb, text_emb, x1)

        return self.proj(x)