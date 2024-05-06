import torch
import torch.nn as nn
import timm

from omegaconf import DictConfig

class BidirectionalLSTM(nn.Module):
    def __init__(self, inputs_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(inputs_size, hidden_size, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(x)

        return self.linear(outputs)
    
def build_encoder_from_config(config: DictConfig) -> nn.Module:
    timm_model = timm.create_model(config.name, pretrained=False)
    encoder = nn.Sequential(
        timm_model.conv_stem,
        timm_model.bn1,
        timm_model.blocks[:config.num_layers],
        nn.Conv2d(in_channels=timm_model.blocks[config.num_layers - 1][-1].conv_pwl.out_channels, out_channels=config.out_channels, kernel_size=(4, 3), padding=(0, 1), bias=False),
        nn.BatchNorm2d(config.out_channels),
        nn.ReLU(True),
    )

    return encoder

class CRNN(nn.Module):
    def __init__(self, config: DictConfig, num_classes: int):
        super().__init__()

        self.config = config
        self.num_classes = num_classes

        self.encoder = build_encoder_from_config(config.encoder)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(config.encoder.out_channels, config.rnn_hidden_size, config.rnn_hidden_size),
            BidirectionalLSTM(config.rnn_hidden_size, config.rnn_hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)

        return self.rnn(x)
