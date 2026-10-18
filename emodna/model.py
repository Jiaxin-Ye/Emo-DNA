from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .legacy_model import (
    Encoder as LegacyEncoder,
    EmotionEncoder as LegacyEmotionEncoder,
    DomainEncoder as LegacyDomainEncoder,
    Projection as LegacyProjection,
)


class ConvPoolBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            # The historical CNN block defines max_net and avg_net, but forward uses avg_net.
            nn.AvgPool1d(kernel_size=3, stride=2),
            nn.Dropout(dropout),
        )


class Backbone(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, blocks: int, dropout: float):
        layers: list[nn.Module] = []
        for index in range(blocks):
            layers.append(ConvPoolBlock(input_dim if index == 0 else hidden_dim, hidden_dim, dropout))
        super().__init__(*layers)


class Chomp1d(nn.Module):
    def __init__(self, amount: int):
        super().__init__()
        self.amount = amount

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, :, : -self.amount].contiguous() if self.amount else inputs


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        modules: list[nn.Module] = []
        for _ in range(2):
            modules.extend(
                [
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        padding=padding,
                        dilation=dilation,
                        bias=False,
                    ),
                    Chomp1d(padding),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
        self.network = nn.Sequential(*modules)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.network(inputs)


class EmotionEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        blocks: int,
        kernel_size: int,
        dropout: float,
        num_classes: int = 2,
    ):
        super().__init__()
        self.tcn = nn.Sequential(
            *[
                TemporalResidualBlock(hidden_dim, kernel_size, 2**index, dropout)
                for index in range(blocks)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.tcn(inputs)
        embedding = self.pool(sequence).squeeze(-1)
        return sequence, embedding, self.classifier(embedding)


class CorpusEncoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5 if layers > 1 else 0.0,
        )
        self.reduce = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, 2)
        nn.init.xavier_uniform_(self.reduce.weight)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence, _ = self.lstm(inputs.transpose(1, 2))
        sequence = self.reduce(sequence.transpose(1, 2))
        embedding = self.pool(sequence).squeeze(-1)
        return sequence, embedding, self.classifier(embedding)


class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(inputs), dim=-1)


class EMODNA(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden_dim = config["hidden_dim"]
        if config["input_dim"] != 40:
            raise ValueError("The paper EMOTE implementation requires input_dim=40")
        self.backbone = LegacyEncoder(output_size=hidden_dim, dropout=config["dropout"])
        self.emotion_encoder = LegacyEmotionEncoder(
            channels=hidden_dim,
            blocks=config["tcn_blocks"],
            kernel=config["kernel_size"],
            dropout=config["dropout"],
        )
        self.corpus_encoder = LegacyDomainEncoder(channels=hidden_dim)
        self.projection = LegacyProjection(hidden_dim)

    def forward(self, mfcc: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(mfcc.transpose(1, 2))
        emotion_sequence, emotion_logits = self.emotion_encoder(features)
        corpus_sequence, corpus_logits = self.corpus_encoder(features)
        emotion_embedding = self.emotion_encoder.GAP(emotion_sequence).squeeze(-1)
        corpus_embedding = self.corpus_encoder.GAP(corpus_sequence).squeeze(-1)
        return {
            "emotion_sequence": emotion_sequence,
            "emotion_embedding": emotion_embedding,
            "emotion_logits": emotion_logits,
            "corpus_sequence": corpus_sequence,
            "corpus_embedding": corpus_embedding,
            "corpus_logits": corpus_logits,
        }

    def predict(self, mfcc: torch.Tensor) -> torch.Tensor:
        return self.forward(mfcc)["emotion_logits"]
