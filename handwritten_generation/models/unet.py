import torch
import torch.nn as nn
import torch.nn.functional as F

from handwritten_generation.models.modules import (
    ConvBlock,
    CrossAttention,
    SelfAttention,
)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                residual=True,
            ),
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.convs = nn.Sequential(
            ConvBlock(
                in_channels=in_channels + res_channels,
                out_channels=in_channels + res_channels,
                kernel_size=kernel_size,
                residual=True,
            ),
            ConvBlock(
                in_channels=in_channels + res_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
        )

    def forward(self, x, res_x):
        x = self.up(x)
        x = torch.cat([x, res_x], dim=1)

        return self.convs(x)


class CrossAttentionBasedCondition(nn.Module):
    def __init__(
        self, context_hidden_dim: int, num_channels: int, dropout: float = 0.1
    ):
        super().__init__()

        self.cross_attention = CrossAttention(
            hidden_dim=num_channels,
            context_hidden_dim=context_hidden_dim,
            dropout=dropout,
        )

    def forward(self, x, context):
        # cross attention with queries from image patches, keys and values from embeddings
        b, c, h, w = x.size()
        assert b == context.size(0)

        x = x.view(b, c, h * w).transpose(1, 2)
        x = self.cross_attention(x=x, context=context)

        return x.transpose(1, 2).view(b, c, h, w)


class PatchedCrossAttentionBasedCondition(CrossAttentionBasedCondition):
    def __init__(
        self,
        context_hidden_dim: int,
        num_channels: int,
        patch_size: int,
        dropout: float = 0.1,
    ):
        super().__init__(
            context_hidden_dim=context_hidden_dim,
            num_channels=num_channels,
            dropout=dropout,
        )

        self.patch_size = patch_size
        self.patchify = (
            nn.Identity()
            if patch_size == 1
            else nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=patch_size,
                stride=patch_size,
            )
        )
        self.depatchify = (
            nn.Identity()
            if patch_size == 1
            else nn.ConvTranspose2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=patch_size,
                stride=patch_size,
            )
        )
        # self.depatchify = nn.Identity() if patch_size == 1 else nn.Upsample(scale_factor=patch_size, mode="bilinear", align_corners=True)

    def forward(self, x, context):
        x = self.patchify(x)
        x = super().forward(x, context)

        return self.depatchify(x)


class TimeCondition(nn.Module):
    def __init__(self, time_emb_dim: int, num_channels: int):
        super().__init__()

        self.proj = nn.Linear(time_emb_dim, num_channels)

    def forward(self, x, time_emb):
        time_emb = self.proj(time_emb)[:, :, None, None].repeat(
            1, 1, x.size(-2), x.size(-1)
        )

        return F.gelu(x + time_emb)


class UNet(nn.Module):
    def __init__(
        self, time_emb_dim: int, text_emb_dim: int, in_channels: int, out_channels: int
    ):
        super().__init__()

        self.time_emb_dim = time_emb_dim
        self.text_emb_dim = text_emb_dim

        self.conv = ConvBlock(in_channels=in_channels, out_channels=64)

        self.down = nn.ModuleList(
            [
                DownBlock(in_channels=64, out_channels=128),
                DownBlock(in_channels=128, out_channels=256),
                DownBlock(in_channels=256, out_channels=512),
            ]
        )

        self.bottleneck = nn.Sequential(
            ConvBlock(in_channels=512, out_channels=1024),
            ConvBlock(
                in_channels=1024, out_channels=1024, mid_channels=2048, residual=True
            ),
            ConvBlock(in_channels=1024, out_channels=512),
        )

        self.up = nn.ModuleList(
            [
                UpBlock(in_channels=512, out_channels=256, res_channels=256),
                UpBlock(in_channels=256, out_channels=128, res_channels=128),
                UpBlock(in_channels=128, out_channels=64, res_channels=64),
            ]
        )

        self.time_condition = nn.ModuleList(
            [
                TimeCondition(time_emb_dim=time_emb_dim, num_channels=64),
                TimeCondition(time_emb_dim=time_emb_dim, num_channels=128),
                TimeCondition(time_emb_dim=time_emb_dim, num_channels=256),
                TimeCondition(time_emb_dim=time_emb_dim, num_channels=512),
                TimeCondition(time_emb_dim=time_emb_dim, num_channels=256),
                TimeCondition(time_emb_dim=time_emb_dim, num_channels=128),
            ]
        )

        self.text_condition = nn.ModuleList(
            [
                PatchedCrossAttentionBasedCondition(
                    context_hidden_dim=text_emb_dim, num_channels=64, patch_size=8
                ),
                PatchedCrossAttentionBasedCondition(
                    context_hidden_dim=text_emb_dim, num_channels=128, patch_size=4
                ),
                PatchedCrossAttentionBasedCondition(
                    context_hidden_dim=text_emb_dim, num_channels=256, patch_size=2
                ),
                PatchedCrossAttentionBasedCondition(
                    context_hidden_dim=text_emb_dim, num_channels=512, patch_size=2
                ),
                PatchedCrossAttentionBasedCondition(
                    context_hidden_dim=text_emb_dim, num_channels=256, patch_size=4
                ),
                PatchedCrossAttentionBasedCondition(
                    context_hidden_dim=text_emb_dim, num_channels=128, patch_size=8
                ),
            ]
        )

        self.proj = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=64, residual=True),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x, time_emb, context):
        x1 = self.conv(x)

        x1 = self.text_condition[0](x1, context)
        x1 = self.time_condition[0](x1, time_emb)
        x2 = self.down[0](x1)

        x2 = self.text_condition[1](x2, context)
        x2 = self.time_condition[1](x2, time_emb)
        x3 = self.down[1](x2)

        x3 = self.text_condition[2](x3, context)
        x3 = self.time_condition[2](x3, time_emb)
        x = self.down[2](x3)

        x = self.text_condition[3](x, context)
        x = self.time_condition[3](x, time_emb)
        x = self.up[0](x, x3)

        x = self.text_condition[4](x, context)
        x = self.time_condition[4](x, time_emb)
        x = self.up[1](x, x2)

        x = self.text_condition[5](x, context)
        x = self.time_condition[5](x, time_emb)
        x = self.up[2](x, x1)

        return self.proj(x)
