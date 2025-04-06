import numpy as np
import torch
from torch.utils.data import Dataset


# ECG Augmentation
class ECGAugment:
    def __call__(self, ecg):
        noise = np.random.normal(0, 0.02, ecg.shape)
        scaling_factor = np.random.uniform(0.9, 1.1)
        return ecg * scaling_factor + noise


# ECG Dataset
class ECGDataset(Dataset):
    def __init__(self, signals, labels, augment=False):
        self.signals = signals
        self.labels = labels
        self.augment = ECGAugment() if augment else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        if self.augment:
            signal = self.augment(signal)
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)