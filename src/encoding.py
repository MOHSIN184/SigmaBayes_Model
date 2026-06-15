import numpy as np
import torch
from torch.utils.data import Dataset


DNA_TO_INDEX = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}


def one_hot_encode_sequence(seq, max_len=81):
    encoded = np.zeros((4, max_len), dtype=np.float32)
    sequence = str(seq).upper()

    for position, nucleotide in enumerate(sequence[:max_len]):
        channel = DNA_TO_INDEX.get(nucleotide)
        if channel is not None:
            encoded[channel, position] = 1.0

    return encoded


def encode_sequences(sequences, max_len=81):
    return np.stack(
        [one_hot_encode_sequence(sequence, max_len=max_len) for sequence in sequences],
        axis=0,
    ).astype(np.float32)


class DNADataset(Dataset):
    def __init__(self, df, max_len=81):
        self.sequences = df["sequence"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        x = one_hot_encode_sequence(self.sequences[index], max_len=self.max_len)
        y = self.labels[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def create_torch_dataset(df, max_len=81):
    return DNADataset(df, max_len=max_len)
