import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.modules.TSMixerx import TSMixerx
from models.modules.ADRModule import ADRModule
from models.modules.ADRModuleEmbedding import ADRModuleEmbedding
from models.modules.ADRModuleHybrid import ADRModuleHybrid
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

        self.ghi_zero_mask = config.model_params.ghi_zero_mask
        self.residual_kind = config.model_params.residual_kind
        if config.model_params.activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif config.model_params.activation == "exp":
            self.activation = torch.exp
        elif config.model_params.activation == None or config.model_params.activation == "none":
            self.activation = torch.nn.Identity()
        else:
            raise NotImplementedError(f"Activation {config.model_params.activation} not implemented!")


        if config.model_params.ADR.mode == "global":
            self.ADR = ADRModule()
        elif config.model_params.ADR.mode == "embedding":
            self.ADR = ADRModuleEmbedding(**config.model_params.ADR.params)
        elif config.model_params.ADR.mode == "hybrid":
            self.ADR = ADRModuleHybrid(**config.model_params.ADR.params)

        if config.model_params.ADR.checkpoint is not None:
            checkpoint = torch.load(config.model_params.ADR.checkpoint, weights_only=False)
            self.ADR.load_state_dict(checkpoint["state_dict"], strict=False)
            if config.model_params.ADR.freeze:
                for param in self.ADR.parameters():
                    param.requires_grad = False

        self.model = TSMixerx(**config.model_params)

    def forward(self, x, meta):
        adr_output, miscellaneous = self.ADR(x, meta)
        x["ADR_output"] = adr_output
        x = {**x, **miscellaneous}

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


        if self.residual_kind == "additive":
            insample_y = insample_y - adr_output[:, :insample_y.shape[1]].unsqueeze(2)
        elif self.residual_kind == "multiplicative":
            insample_y = torch.log((insample_y + 1e-6) / (adr_output[:, :insample_y.shape[1]].unsqueeze(2) + 1e-6))

        batch = {
            "insample_y": insample_y,
            "hist_exog": None,
            "futr_exog": futr_exog,
            "stat_exog": None
        }

        if self.residual_kind == "additive":
            model_output = adr_output[:, insample_y.shape[1]:].unsqueeze(2) + self.activation(self.model(batch))
        elif self.residual_kind == "multiplicative":
            model_output = adr_output[:, insample_y.shape[1]:].unsqueeze(2) * (self.activation(self.model(batch)))

        if self.ghi_zero_mask or not self.training:
            mask = (x['ghi'][:, -model_output.shape[1]:] != 0).float()
            model_output = model_output * mask.unsqueeze(2)

        return model_output, adr_output[:, insample_y.shape[1]:].unsqueeze(2)
    
    def training_step(self, batch, batch_idx):
        x, y, meta = batch
        y_hat, adr_output = self(x, meta)
        loss = self.criterion(y_hat.squeeze(), y)
        self.log("train_loss", loss)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, meta = batch
        y_hat, adr_output = self(x, meta)
        y_hat_nonneg = y_hat.clone()
        y_hat_nonneg[y_hat_nonneg < 0] = 0
        loss = self.criterion(y_hat_nonneg.squeeze(), y)
        self.log("val_loss", loss)

        if batch_idx == 0:
            self.plot_outputs(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy(), adr_output.cpu().detach().numpy(), meta, "valid")

        return loss

    def test_step(self, batch, batch_idx):
        x, y, meta = batch
        y_hat, adr_output = self(x, meta)
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
            self.plot_outputs(y.cpu().detach().numpy(), y_hat_nonneg.cpu().detach().numpy(), adr_output.cpu().detach().numpy(), meta, "test")

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

    def plot_outputs(self, target, pred, adr, metadata, stage):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 5, figsize=(25, 15), layout='constrained')
        for i in range(15):
            axes[i // 5, i % 5].plot(target[i], label="PV Output", color="black", linewidth=2)
            axes[i // 5, i % 5].plot(adr[i], label="ADR", color="green", linewidth=2)
            axes[i // 5, i % 5].plot(pred[i], label="Forecast", color="red", linewidth=2)
            axes[i // 5, i % 5].set_xlabel("Time")
            axes[i // 5, i % 5].set_ylabel("Efficiency")
            axes[i // 5, i % 5].legend(loc="upper right")
            axes[i // 5, i % 5].set_ylim(-0.2,1)
            axes[i // 5, i % 5].set_title(f"{metadata['country'][i]} {metadata['installation'][i]} {metadata['date'][i]}")

        self.logger.log_image(key=f"{stage}_outputs_plot", images=[fig])