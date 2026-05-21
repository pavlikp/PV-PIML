import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.modules.TSMixerx import TSMixerx
from models.modules.ADRModule import ADRModule
from utils.normalize import normalize_inputs

class PVTSMixerADRResidual(pl.LightningModule):
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

        self.ADR = ADRModule()
        self.model = TSMixerx(**config.model_params)

    def forward(self, x, meta):
        adr_output = self.ADR(x, meta)
        x["ADR_output"] = adr_output

        x_norm = normalize_inputs(x)
        insample_y = x_norm.pop("production").unsqueeze(-1)

        n_channels = len(x_norm.keys())
        n_batch, n_seq = x_norm[list(x_norm.keys())[0]].shape

        futr_exog = torch.zeros(n_batch, n_channels, n_seq, 1).to(self.device)
        for i, key in enumerate(x_norm.keys()):
            futr_exog[:, i, :, 0] = x_norm[key]

        stat_exog = torch.zeros(n_batch, 2).to(self.device)
        for i, key in enumerate(["tilt", "orientation"]):
            stat_exog[:, i] = meta[key]

        insample_y = insample_y - adr_output[:, :insample_y.shape[1]].unsqueeze(2)

        batch = {
            "insample_y": insample_y,
            "hist_exog": None,
            "futr_exog": futr_exog,
            "stat_exog": None
        }

        model_output = adr_output[:, insample_y.shape[1]:].unsqueeze(2) + self.model(batch)

        model_output[x['ghi'][:,-model_output.shape[1]:] == 0] = 0
        model_output[model_output < 0] = 0

        return model_output
    
    def training_step(self, batch, batch_idx):
        x, y, meta = batch
        y_hat = self(x, meta)
        loss = self.criterion(y_hat.squeeze(), y)
        self.log("train_loss", loss)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, meta = batch
        y_hat = self(x, meta)
        loss = self.criterion(y_hat.squeeze(), y)
        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y, meta = batch
        y_hat = self(x, meta)

        loss = self.criterion(y_hat.squeeze(), y)

        mse = nn.functional.mse_loss(y_hat.squeeze(), y)
        mae_w = nn.functional.l1_loss(y_hat.squeeze() * meta["system_size"].unsqueeze(1), y * meta["system_size"].unsqueeze(1))
        mape_total = torch.abs((y.sum() - y_hat.sum()) / (y.sum() + 1e-6)) * 100

        self.log("test_mse", mse.detach())
        self.log("test_mae_watts", mae_w.detach())
        self.log("test_mape_total", mape_total.detach())
        self.log("test_loss", loss.detach())

        return loss
    
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