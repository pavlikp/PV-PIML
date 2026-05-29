import torch
import torch.nn as nn


class ADRModule(nn.Module):
    
    def __init__(self, initialize_values={"albedo": 0.25, "u0": 25.0, "u1": 7.0, "k_a": 1.0, "k_d": -6.0, "tc_d": 0.0, "k_rs": 1e-3, "k_rsh": 1e-3}):
        
        super().__init__()

        self.albedo = nn.Parameter(torch.tensor(initialize_values["albedo"], requires_grad=True))

        self.u0 = nn.Parameter(torch.tensor(initialize_values["u0"], requires_grad=True))
        self.u1 = nn.Parameter(torch.tensor(initialize_values["u1"], requires_grad=True))

        self.k_a = nn.Parameter(torch.tensor(initialize_values["k_a"], requires_grad=True))
        self.k_d = nn.Parameter(torch.tensor(initialize_values["k_d"], requires_grad=True))
        self.tc_d = nn.Parameter(torch.tensor(initialize_values["tc_d"], requires_grad=True))
        self.k_rs = nn.Parameter(torch.tensor(initialize_values["k_rs"], requires_grad=True))
        self.k_rsh = nn.Parameter(torch.tensor(initialize_values["k_rsh"], requires_grad=True))
        
    def forward(self, x, metadata):
        dhi = x['dhi']
        ghi = x['ghi']
        dni = x['dni']
        wind_speed = x['wind_speed']
        temp_air = x['temp_air']

        tilt = metadata["tilt"].unsqueeze(1)
        orient = metadata["orientation"].unsqueeze(1)

        sky_diffuse = dhi * ((1 + torch.cos(tilt)) * 0.5)
        ground_diffuse = ghi * (self.albedo * (1 - torch.cos(tilt)) * 0.5)

        projection = (
            torch.cos(tilt) * torch.cos(x['solar_zenith']) +
            torch.sin(tilt) * torch.sin(x['solar_zenith']) *
            torch.cos(x['solar_azimuth'] - orient))

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

        return eta * s, {"pv_ref_temp_diff": dt, "s": s, "v": v, "projection": projection, "poa_direct": poa_direct / 1000.0, "poa_diffuse": poa_diffuse / 1000.0}