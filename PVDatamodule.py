import torch
import pytorch_lightning as pl

from PVDataset import PVDataset

class PVDatamodule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.dsconfig = config.dataset
        self.train_params = config.train_params

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = PVDataset(split="train", **self.dsconfig)
            self.valid_dataset = PVDataset(split="valid", **self.dsconfig)
        if stage == "test":
            self.test_dataset = PVDataset(split="test", **self.dsconfig)
        # if stage == "predict":
        #     self.predict_dataset = PVDataset(split="test", **self.dsconfig)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_params.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.train_params.valid_batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.train_params.test_batch_size)