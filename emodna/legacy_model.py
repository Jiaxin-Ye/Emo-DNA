"""Exact inference modules used by the published EMODNA checkpoints.

The training-era checkpoints store four independent state dictionaries (E, G,
EMO_E, DOM_E).  Inference uses E and EMO_E only.  This small copy keeps their
module names and tensor shapes intact so the supplied checkpoints can be
verified without reconstructing the historical training harness.
"""
from __future__ import annotations

import torch
from torch import nn


class CNNPoolingBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, dropout: float, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(inputs, outputs, 3, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(outputs)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(3, stride=2)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(inputs, outputs, 3, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(outputs)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool1d(3, stride=2)
        self.dropout2 = nn.Dropout(dropout)
        self.max_net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.pool2, self.dropout1)
        self.avg_net = nn.Sequential(self.conv2, self.bn2, self.relu2, self.pool2, self.dropout2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_net(x)


class Encoder(nn.Module):
    def __init__(self, input_size: int = 40, output_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.CNN_forward_module_1 = CNNPoolingBlock(input_size, output_size, dropout)
        self.CNN_forward_module_2 = CNNPoolingBlock(output_size, output_size, dropout)
        self.CNN_forward_module_3 = CNNPoolingBlock(output_size, output_size, dropout, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.CNN_forward_module_3(
            self.CNN_forward_module_2(self.CNN_forward_module_1(x))
        )


class Chomp1d(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.chomp_size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalAttentionBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation, bias=False)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation, bias=False)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class TCN(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 5, kernel: int = 2, dropout: float = 0.1):
        super().__init__()
        self.tcn = nn.Module()
        self.tcn.conv1 = nn.Conv1d(channels, channels, 1, bias=False)
        self.tcn.TAB_forward_modules = nn.ModuleList(
            [TemporalAttentionBlock(channels, kernel, 2 ** i, dropout) for i in range(blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.tcn.TAB_forward_modules:
            x = block(x)
        return x


class EmotionEncoder(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 5, kernel: int = 2, dropout: float = 0.1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.tcn = TCN(channels, blocks, kernel, dropout)
        self.linear = nn.Linear(channels, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.tcn(x)
        return features, self.linear(self.GAP(features).squeeze(-1))


class DomainEncoder(nn.Module):
    """The original corpus encoder used by the EMOTE training branch."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.lstm1 = nn.LSTM(channels, channels, num_layers=2, batch_first=True,
                             bidirectional=True, dropout=0.5)
        self.conv1 = nn.Conv1d(channels * 2, channels, 1, bias=False)
        self.linear = nn.Sequential(nn.Linear(channels, 2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features, _ = self.lstm1(x.permute(0, 2, 1))
        features = self.conv1(features.permute(0, 2, 1))
        return features, self.linear(self.GAP(features).squeeze(-1))


class Projection(nn.Module):
    """The original 64-to-64 normalized MLP used for both contrastive losses."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.BatchNorm1d(dim * 4), nn.ReLU(inplace=True),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.head(x), dim=-1)


class LegacyInferenceModel(nn.Module):
    def __init__(self, checkpoint: dict, device: torch.device):
        super().__init__()
        self.encoder = Encoder().to(device)
        block_keys = [
            key for key in checkpoint["EMO_E"]
            if key.startswith("tcn.tcn.TAB_forward_modules.")
        ]
        blocks = max(int(key.split(".")[3]) for key in block_keys) + 1
        self.emotion = EmotionEncoder(blocks=blocks).to(device)
        self.encoder.load_state_dict(checkpoint["E"])
        self.emotion.load_state_dict(checkpoint["EMO_E"])
        self.eval()

    def predict(self, mfcc: torch.Tensor) -> torch.Tensor:
        _, logits = self.emotion(self.encoder(mfcc.transpose(1, 2)))
        return logits
