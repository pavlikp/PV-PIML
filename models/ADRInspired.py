import pytorch_lightning as pl
import torch
import torch.nn as nn

import pvlib
from pvlib import iotools, location

import pandas as pd

class ADRInspired(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.personal_device = torch.device(config.train_params.device)

        if config.train_params.loss == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f"Loss {config.train_params.loss} not implemented!")
        
        # optimization parameters
        self.lr = float(config.train_params.lr)
        self.lr_sch_params = config.train_params.lr_scheduler
        self.automatic_optimization = False

        # Parameters
        self.albedo = nn.Parameter(torch.tensor(0.25, requires_grad=True))

        self.u0 = nn.Parameter(torch.tensor(25.0, requires_grad=True))
        self.u1 = nn.Parameter(torch.tensor(7.0, requires_grad=True))

        self.k_a = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.k_d = nn.Parameter(torch.tensor(-6.0, requires_grad=True))         
        self.tc_d = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.k_rs = nn.Parameter(torch.tensor(1e-3, requires_grad=True))
        self.k_rsh = nn.Parameter(torch.tensor(1e-3, requires_grad=True))

    def forward(self, x, metadata):
        dhi = x['dhi']
        ghi = x['ghi']
        dni = x['dni']
        wind_speed = x['wind_speed']
        temp_air = x['temp_air']
        unix_timestamps = x['unix_timestamps']

        batch_size = len(unix_timestamps)

        solar_zenith = torch.zeros_like(unix_timestamps)
        solar_azimuth = torch.zeros_like(unix_timestamps)
        tilt = torch.zeros(batch_size, 1).to(self.device)
        orient = torch.zeros(batch_size, 1).to(self.device)
        for i in range(batch_size):
            loc = location.Location(latitude=metadata["Latitude"][i].item(),
                                    longitude=metadata["Longitude"][i].item(),
                                    altitude=metadata["Elevation"][i].item(),
                                    tz="UTC")
            solpos = loc.get_solarposition(pd.to_datetime(unix_timestamps[i].cpu(), unit="s") - pd.Timedelta(minutes=7.5))
            solar_zenith[i] = torch.deg2rad(torch.tensor(solpos.apparent_zenith.values)).to(self.device)
            solar_azimuth[i] = torch.deg2rad(torch.tensor(solpos.azimuth.values)).to(self.device)

            tilt[i] = torch.deg2rad(torch.tensor(metadata['Array Tilt (degrees)'][i].item()))
            if metadata['Orientation'][i] == 'S':
                orient[i] = torch.deg2rad(torch.tensor(180))
            elif metadata['Orientation'][i] == 'SW':
                orient[i] = torch.deg2rad(torch.tensor(225))
            elif metadata['Orientation'][i] == 'SE':
                orient[i] = torch.deg2rad(torch.tensor(135))
            else:
                raise NotImplementedError(f"Orientation {metadata['Orientation'][i]} not implemented!")
        

        sky_diffuse = dhi * ((1 + torch.cos(tilt)) * 0.5)
        ground_diffuse = ghi * (self.albedo * (1 - torch.cos(tilt)) * 0.5)

        projection = (
            torch.cos(tilt) * torch.cos(solar_zenith) +
            torch.sin(tilt) * torch.sin(solar_zenith) *
            torch.cos(solar_azimuth - orient))

        projection = torch.clip(projection, -1, 1)

        aoi = torch.acos(projection)

        poa_direct = torch.maximum(dni * torch.cos(aoi), torch.tensor(0.0))
        poa_diffuse = sky_diffuse + ground_diffuse
        poa_global = poa_direct + poa_diffuse

        total_loss_factor = self.u0 + self.u1 * wind_speed
        heat_input = poa_global
        temp_difference = heat_input / total_loss_factor
        pv_temp = temp_air + temp_difference

        # normalize the irradiance
        s = poa_global / 1000.0

        # obtain the difference from reference temperature
        dt = pv_temp - 25.0

        s_o     = 10**(self.k_d + (dt * self.tc_d))
        s_o_ref = 10**(self.k_d)

        v  = torch.log(s / s_o     + 1)
        v /= torch.log(1 / s_o_ref + 1)

        eta = self.k_a * ((1 + self.k_rs + self.k_rsh) * v - self.k_rs * s - self.k_rsh * v**2)

        # Set the desired array size and the irradiance level needed to achieve this output:
        P_STC = metadata["System Size (watts)"].unsqueeze(1)
        G_STC = 1000.   # (W/m2)

        return P_STC * eta * (poa_global / G_STC)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        x, y, metadata = batch

        y_hat = self(x, metadata)
        loss = self.criterion(y_hat / 4000.0, y)
            
        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()
        self.log("train_loss", loss.detach())
        return {"prediction": y_hat, "loss": loss.detach()}

    def validation_step(self, batch, batch_idx):
        x, y, metadata = batch

        y_hat = self(x, metadata)
        loss = self.criterion(y_hat / 4000.0, y)

        self.log("val_loss", loss.detach())
        return {"prediction": y_hat, "loss": loss.detach()}

    def test_step(self, batch, batch_idx):
        x, y, metadata = batch

        y_hat = self(x, metadata)
        mse = nn.functional.mse_loss(y_hat / 4000.0, y)

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

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
        elif isinstance(sch, torch.optim.lr_scheduler.ExponentialLR):
            sch.step()
