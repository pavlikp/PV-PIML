import torch
import pytorch_lightning as pl

from PVDataset import PVDataset

class PVDatamodule(pl.LightningDataModule):
    def __init__(self, dsconfig, train_params):
        super().__init__()
        self.dsconfig = dsconfig
        self.train_params = train_params

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = PVDataset(split="train", **self.dsconfig.SHMUDataset)
            self.valid_dataset = PVDataset(split="valid", **self.dsconfig.SHMUDataset)
        if stage == "test":
            self.test_dataset = PVDataset(split="test", **self.dsconfig.SHMUDataset)
        if stage == "predict":
            self.predict_dataset = PVDataset(split="test", **self.dsconfig.SHMUDataset)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)