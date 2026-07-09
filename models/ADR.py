import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.modules.ADRModule import ADRModule
from models.modules.ADRModuleEmbedding import ADRModuleEmbedding
from models.modules.ADRModuleHybrid import ADRModuleHybrid
from utils.normalize import normalize_inputs

class ADR(pl.LightningModule):
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

        if config.train_params.mode == "global":
            self.model = ADRModule()
        elif config.train_params.mode == "embedding":
            self.model = ADRModuleEmbedding(**config.model_params)
        elif config.train_params.mode == "hybrid":
            self.model = ADRModuleHybrid(**config.model_params)

    def forward(self, x, meta):
        return self.model(x, meta)
    
    def training_step(self, batch, batch_idx):
        x, y, meta = batch
        y_hat, _ = self(x, meta)
        loss = self.criterion(y_hat.squeeze(), y)
        self.log("train_loss", loss)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, meta = batch
        y_hat, _ = self(x, meta)
        y_hat_nonneg = y_hat.clone()
        y_hat_nonneg[y_hat_nonneg < 0] = 0
        loss = self.criterion(y_hat_nonneg.squeeze(), y)
        self.log("val_loss", loss)

        if batch_idx == 0:
            self.plot_outputs(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy(), meta, "valid")

        return loss

    def test_step(self, batch, batch_idx):
        x, y, meta = batch
        y_hat, _ = self(x, meta)
        y_hat_nonneg = y_hat.clone()
        y_hat_nonneg[y_hat_nonneg < 0] = 0
        loss = self.criterion(y_hat_nonneg.squeeze(), y)

        mse = nn.functional.mse_loss(y_hat_nonneg.squeeze(), y)
        mse_daily = nn.functional.mse_loss(y_hat_nonneg.sum(axis=1).squeeze(), y.sum(axis=1))
        mae_w = nn.functional.l1_loss(y_hat_nonneg.squeeze() * meta["system_size"].unsqueeze(1), y * meta["system_size"].unsqueeze(1))
        mae_daily_kwh = nn.functional.l1_loss(y_hat_nonneg.sum(axis=1).squeeze() * meta["system_size"], y.sum(axis=1) * meta["system_size"]) / 4000
        mape_daily = (torch.abs((y.sum(axis=1) - y_hat_nonneg.sum(axis=1).squeeze()) / (y.sum(axis=1) + 1e-6)) * 100).mean()
        bias = (y_hat_nonneg.squeeze() - y).mean()
        bias_daily = (y_hat_nonneg.sum(axis=1).squeeze() - y.sum(axis=1)).mean()

        self.log("test_mse", mse.detach())
        self.log("test_mae_watts", mae_w.detach())
        self.log("test_mae_daily_kwh", mae_daily_kwh.detach())
        self.log("test_mape_daily", mape_daily.detach())
        self.log("test_mse_daily", mse_daily.detach())
        self.log("test_loss", loss.detach())
        self.log("test_bias", bias.detach())
        self.log("test_bias_daily", bias_daily.detach())

        if batch_idx == 0:
            self.plot_outputs(y.cpu().detach().numpy(), y_hat_nonneg.cpu().detach().numpy(), meta, "test")

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

    def plot_outputs(self, target, pred, metadata, stage):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 5, figsize=(25, 15), layout='constrained')
        for i in range(15):
            axes[i // 5, i % 5].plot(target[i], label="PV Output", color="black", linewidth=2)
            axes[i // 5, i % 5].plot(pred[i], label="Forecast", color="red", linewidth=2)
            axes[i // 5, i % 5].set_xlabel("Time")
            axes[i // 5, i % 5].set_ylabel("Efficiency")
            axes[i // 5, i % 5].legend(loc="upper right")
            axes[i // 5, i % 5].set_ylim(-0.2,1)
            axes[i // 5, i % 5].set_title(f"{metadata['country'][i]} {metadata['installation'][i]} {metadata['date'][i]}")

        self.logger.log_image(key=f"{stage}_outputs_plot", images=[fig])