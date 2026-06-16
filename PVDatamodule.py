import torch
from torch.utils.data import RandomSampler, Sampler
import pytorch_lightning as pl

from PVDatasetWithHistory import PVDatasetWithHistory

import random

class SetShuffleSampler(Sampler[int]):
    def __init__(self, length):
        self.list = list(range(0,length))
        random.shuffle(self.list)

    def __len__(self) -> int:
        return len(self.list)

    def __iter__(self):
        yield from self.list

class PVDatamodule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.dsconfig = config.dataset
        self.train_params = config.train_params

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = PVDatasetWithHistory(split="train", **self.dsconfig)
            self.valid_dataset = PVDatasetWithHistory(split="valid", **self.dsconfig)
        if stage == "test":
            self.test_dataset = PVDatasetWithHistory(split="test", **self.dsconfig)
        # if stage == "predict":
        #     self.predict_dataset = PVDatasetWithHistory(split="test", **self.dsconfig)

    def train_dataloader(self):
        sampler = RandomSampler(self.train_dataset, replacement=True, num_samples=self.train_params.train_batch_size * self.train_params.train_batches)
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.train_params.train_batch_size,
                                           num_workers=self.train_params.num_workers,
                                           persistent_workers=True,
                                           sampler=sampler)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset,
                                           batch_size=self.train_params.valid_batch_size,
                                           num_workers=self.train_params.num_workers,
                                           persistent_workers=True,
                                           sampler=SetShuffleSampler(len(self.valid_dataset)))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.train_params.test_batch_size,
                                           num_workers=self.train_params.num_workers,
                                           persistent_workers=True,
                                           sampler=SetShuffleSampler(len(self.test_dataset)),)