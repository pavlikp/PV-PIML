import torch
import pytorch_lightning as pl

from PVDatasetWithHistory import PVDatasetWithHistory

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
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_params.train_batch_size, num_workers=self.train_params.num_workers, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.train_params.valid_batch_size, num_workers=self.train_params.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.train_params.test_batch_size, num_workers=self.train_params.num_workers, persistent_workers=True)