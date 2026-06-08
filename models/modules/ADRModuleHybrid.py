import torch
import torch.nn as nn


class ADRModuleHybrid(nn.Module):
    
    def __init__(self, num_sites=3713, initialize_values={"albedo": -1.0, "u0": 25.0, "u1": 7.0, "k_a": 0.55, "k_d": 0.0, "tc_d": 0.0, "k_rs": -1.0, "k_rsh": -1.0}):
        
        super().__init__()

        self.albedo_common = nn.Parameter(torch.tensor(initialize_values["albedo"], requires_grad=True))

        self.u0_common = nn.Parameter(torch.tensor(initialize_values["u0"], requires_grad=True))
        self.u1_common = nn.Parameter(torch.tensor(initialize_values["u1"], requires_grad=True))

        self.k_a_common = nn.Parameter(torch.tensor(initialize_values["k_a"], requires_grad=True))
        self.k_d_common = nn.Parameter(torch.tensor(initialize_values["k_d"], requires_grad=True))
        self.tc_d_common = nn.Parameter(torch.tensor(initialize_values["tc_d"], requires_grad=True))
        self.k_rs_common = nn.Parameter(torch.tensor(initialize_values["k_rs"], requires_grad=True))
        self.k_rsh_common = nn.Parameter(torch.tensor(initialize_values["k_rsh"], requires_grad=True))

        self.site_params = nn.Embedding(num_sites, initialize_values.keys().__len__())

        self.site_params.weight.data = torch.zeros_like(self.site_params.weight.data)

        self.param_names = list(initialize_values.keys())
        
    def forward(self, x, metadata):
        dhi = x['dhi']
        ghi = x['ghi']
        dni = x['dni']
        wind_speed = x['wind_speed']
        temp_air = x['temp_air']

        tilt = metadata["tilt"].unsqueeze(1)
        orient = metadata["orientation"].unsqueeze(1)

        subset_params = self.site_params(metadata["installation_embedding"])
        albedo = torch.nn.functional.sigmoid(subset_params[:, self.param_names.index("albedo")].unsqueeze(1) + self.albedo_common)
        u0 = subset_params[:, self.param_names.index("u0")].unsqueeze(1) + self.u0_common
        u1 = subset_params[:, self.param_names.index("u1")].unsqueeze(1) + self.u1_common
        k_a = torch.nn.functional.softplus(subset_params[:, self.param_names.index("k_a")].unsqueeze(1) + self.k_a_common)
        k_d = torch.nn.functional.tanh(subset_params[:, self.param_names.index("k_d")].unsqueeze(1) + self.k_d_common) * 6 - 6
        tc_d = torch.nn.functional.tanh(subset_params[:, self.param_names.index("tc_d")].unsqueeze(1) + self.tc_d_common) * 0.1
        k_rs = torch.nn.functional.softplus(subset_params[:, self.param_names.index("k_rs")].unsqueeze(1) + self.k_rs_common)
        k_rsh = torch.nn.functional.softplus(subset_params[:, self.param_names.index("k_rsh")].unsqueeze(1) + self.k_rsh_common)

        sky_diffuse = dhi * ((1 + torch.cos(tilt)) * 0.5)
        ground_diffuse = ghi * (albedo * (1 - torch.cos(tilt)) * 0.5)

        projection = (
            torch.cos(tilt) * torch.cos(x['solar_zenith']) +
            torch.sin(tilt) * torch.sin(x['solar_zenith']) *
            torch.cos(x['solar_azimuth'] - orient))

        projection = torch.clip(projection, -1, 1)

        aoi = torch.acos(projection)

        poa_direct = torch.maximum(dni * torch.cos(aoi), torch.tensor(0.0))
        poa_diffuse = sky_diffuse + ground_diffuse
        poa_global = poa_direct + poa_diffuse

        total_loss_factor = u0 + u1 * wind_speed
        heat_input = poa_global
        temp_difference = heat_input / total_loss_factor
        pv_temp = temp_air + temp_difference

        # normalize the irradiance
        s = poa_global / 1000.0

        # obtain the difference from reference temperature
        dt = pv_temp - 25.0

        s_o     = 10**(k_d + (dt * tc_d))
        s_o_ref = 10**(k_d)

        v  = torch.log(s / s_o     + 1)
        v /= torch.log(1 / s_o_ref + 1)

        eta = k_a * ((1 + k_rs + k_rsh) * v - k_rs * s - k_rsh * v**2)

        return eta * s, {"pv_ref_temp_diff": dt, "s": s, "v": v, "projection": projection, "poa_direct": poa_direct / 1000.0, "poa_diffuse": poa_diffuse / 1000.0}