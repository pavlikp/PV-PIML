import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.modules.UNet1D import UNet1D
from utils.normalize import normalize_inputs

class PVUNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        if config.train_params.loss == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f"Loss {config.train_params.loss} not implemented!")
        
        # optimization parameters
        self.lr = float(config.train_params.lr)
        self.lr_sch_params = config.train_params.lr_scheduler
        self.automatic_optimization = False

        self.model = UNet1D(config.model_params.kernel_size, config.model_params.out_channels, config.model_params.conv_shape)

    def forward(self, x):
        x_norm = normalize_inputs(x)
        n_channels = len(x_norm.keys())
        n_batch, n_seq = x_norm[list(x_norm.keys())[0]].shape

        input_tensor = torch.zeros(n_batch, n_channels, n_seq).to(self.device)

        for i, key in enumerate(x_norm.keys()):
            input_tensor[:, i, :] = x_norm[key]

        return self.model(input_tensor)
    
    def training_step(self, batch, batch_idx):
        x, y, meta = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss.detach())

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

    def validation_step(self, batch, batch_idx):
        x, y, meta = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss.detach())

    def test_step(self, batch, batch_idx):
        x, y, meta = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        mse = nn.functional.mse_loss(y_hat, y)
        mae_w = nn.functional.l1_loss(y_hat * meta["system_size"].unsqueeze(1), y * meta["system_size"].unsqueeze(1))

        self.log("test_mse", mse.detach())
        self.log("test_mae_watts", mae_w.detach())
        self.log("test_loss", loss.detach())
    
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
        
    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
        elif isinstance(sch, torch.optim.lr_scheduler.ExponentialLR):
            sch.step()