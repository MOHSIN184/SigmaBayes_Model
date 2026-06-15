import torch
from torch import nn


class HybridCNNKmerClassifier(nn.Module):
    def __init__(self, num_kmers, num_classes, dropout=0.3):
        super().__init__()

        self.cnn_branch = nn.Sequential(
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
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.kmer_branch = nn.Sequential(
            nn.Linear(num_kmers, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_seq, x_kmer):
        seq_features = self.cnn_branch(x_seq)
        kmer_features = self.kmer_branch(x_kmer)
        combined = torch.cat([seq_features, kmer_features], dim=1)
        return self.fusion(combined)
