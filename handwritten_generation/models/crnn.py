import torch
import torch.nn as nn
import timm

class BidirectionalLSTM(nn.Module):
    def __init__(self, inputs_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(inputs_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(x)

        return self.linear(outputs)


class CRNN(nn.Module):
    def __init__(self, num_classes: int, num_channels: int = 3):
        super().__init__()

        backbone = timm.create_model('mobilenetv3_small_050.lamb_in1k', pretrained=False)
        self.convs = nn.Sequential(
            backbone.conv_stem,
            backbone.bn1,
            backbone.blocks[:-1],
        )
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels=backbone.blocks[-2][0].conv.out_channels, out_channels=256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, num_classes)
        )

    def forward(self, x):
        x = self.convs(x) # batch_size x 288 x 2 x 8
        x = self.neck(x)  # bathc_size x 256 x 1 x 8
        x = x.squeeze(2)
        x = x.transpose(1, 2)

        return self.rnn(x)
