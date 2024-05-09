import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


class ResConnection(nn.Module):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward(self, x):
        return x + self.module(x)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, max_seq_len: int):
        super().__init__()

        pe = torch.zeros(max_seq_len, emb_dim)
        positions = torch.arange(0, max_seq_len).unsqueeze(1)
        div = torch.exp(
            -torch.log(torch.tensor(10000)) * torch.arange(0, emb_dim, 2) / emb_dim
        )  # sin (pos / 10000 ** (2i / emb_dim))
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1), :]

    def sample(self, x):
        return self.pe[x, :]


class Attention(nn.Module):
    def __init__(
        self, hidden_dim: int, q_in: int = None, k_in: int = None, v_in: int = None
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.Q = nn.Linear(q_in if q_in else hidden_dim, hidden_dim)
        self.K = nn.Linear(k_in if k_in else hidden_dim, hidden_dim)
        self.V = nn.Linear(v_in if v_in else hidden_dim, hidden_dim)

    def forward(self, query, key, value):
        # x : B x SEQ_LEN x HIDDEN_DIM
        query = self.Q(query)
        key = self.K(key)
        value = self.V(value)

        return F.scaled_dot_product_attention(query=query, key=key, value=value)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, max_seq_len: int):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.pe = PositionalEncoding(emb_dim=emb_dim, max_seq_len=max_seq_len)

    def forward(self, x):
        return self.pe(self.emb(x))


class TextEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, max_seq_len: int):
        super().__init__()
        self.emb = TokenEmbedding(
            vocab_size=vocab_size, emb_dim=emb_dim, max_seq_len=max_seq_len
        )

    def forward(self, x):
        raise NotImplementedError("Implement 'forward' method in an inheritor")


class AvgTextEmbedding(TextEmbedding):
    def __init__(self, vocab_size: int, emb_dim: int, max_seq_len: int):
        super().__init__(
            vocab_size=vocab_size, emb_dim=emb_dim, max_seq_len=max_seq_len
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return self.pooling(self.emb(x).transpose(1, 2)).transpose(1, 2).squeeze(1)


class AttentionTextEmbbeding(TextEmbedding):
    def __init__(self, vocab_size: int, emb_dim: int, max_seq_len: int):
        super().__init__(
            vocab_size=vocab_size, emb_dim=emb_dim, max_seq_len=max_seq_len
        )
        self.attention = SelfAttention(hidden_dim=emb_dim)

    def forward(self, x):
        embeddings = self.emb(x)
        return self.attention(embeddings)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        kernel_size: int = 3,
        residual: bool = False,
    ):
        super().__init__()

        mid_channels = mid_channels if mid_channels else in_channels
        padding = kernel_size // 2

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(1, out_channels),
        )

        if residual:
            self.convs = ResConnection(self.convs)

    def forward(self, x):
        return F.gelu(self.convs(x))


# single head self-attention block
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1, dim_extend: int = 4):
        super().__init__()

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.attention = Attention(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, dim_extend * hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_extend * hidden_dim, hidden_dim),
        )

    def forward(self, x):
        x_ln = self.ln1(x)
        x = x + self.attention(query=x_ln, key=x_ln, value=x_ln)
        x = self.ln2(x)
        x = x + self.mlp(x)

        return self.ln3(x)


# single head cross-attention block
class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, context_hidden_dim: int, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)

        self.self_attention = Attention(hidden_dim=hidden_dim)
        self.cross_attention = Attention(
            hidden_dim=hidden_dim, k_in=context_hidden_dim, v_in=context_hidden_dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x, context):
        x_ln = self.ln1(x)
        x = x + self.self_attention(query=x_ln, key=x_ln, value=x_ln)
        x = self.ln2(x)
        x = x + self.cross_attention(query=x, key=context, value=context)
        x = self.ln3(x)
        x = x + self.mlp(x)

        return self.ln4(x)


class EMA:
    def __init__(self, beta, model, step_start_ema=500):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)

        self.step_start_ema = step_start_ema

    def update_model_average(self, current_model):
        for current_params, ema_params in zip(
            current_model.parameters(), self.ema_model.parameters()
        ):
            old_weight, up_weight = ema_params.data, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    @torch.no_grad()
    def step_ema(self, model):
        if self.step < self.step_start_ema:
            self.reset_parameters(model)
            self.step += 1
            return
        self.update_model_average(model)
        self.step += 1

    def reset_parameters(self, current_model):
        for current_params, ema_params in zip(
            current_model.parameters(), self.ema_model.parameters()
        ):
            old_weight, up_weight = ema_params.data, current_params.data
            ema_params.data = up_weight
