import torch
from torch import nn


class CNNPromoterClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class LightweightTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2,
        max_len=81,
    ):
        super().__init__()

        self.max_len = max_len
        self.input_projection = nn.Linear(4, d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        seq_len = x.size(1)
        x = self.input_projection(x)
        x = x + self.positional_embedding[:, :seq_len, :]
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.output(x)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
