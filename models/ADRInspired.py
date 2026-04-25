import pytorch_lightning as pl
import torch
import torch.nn as nn

import pvlib
from pvlib import iotools, location

class ADRInspired(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.personal_device = torch.device(config.train_params.device)

        if config.model.loss.name == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f"Loss {config.model.loss.name} not implemented!")
        
        # optimization parameters
        self.lr = float(config.model.lr)
        self.lr_sch_params = config.train_params.lr_scheduler
        self.automatic_optimization = False

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        x, y, _ = batch

        y_hat = self(x)
        loss = self.criterion(y_hat, y)
            
        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()
        self.log("train_loss", loss.detach())
        return {"prediction": y_hat, "loss": loss.detach()}

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch

        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log("val_loss", loss.detach())
        return {"prediction": y_hat, "loss": loss.detach()}

    def test_step(self, batch, batch_idx):
        x, y, _ = batch

        y_hat = self(x)
        mse = nn.functional.mse_loss(y_hat, y)

        self.log("test_mse", mse.detach())
        return {"prediction": y_hat}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_sch_params.name is None:
            return optimizer
        elif self.lr_sch_params.name == "reduce_lr_on_plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, **self.lr_sch_params.kwargs
            )
            return [optimizer], [lr_scheduler]
        elif self.lr_sch_params.name == "ExponentialLR":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, **self.lr_sch_params.kwargs
            )
            return [optimizer], [lr_scheduler]
        else:
            raise NotImplementedError("Lr scheduler not defined.")
