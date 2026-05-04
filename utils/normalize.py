import torch

def normalize_inputs(x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for key in x.keys():
        if 'temp_air' in key:
            x[key] = (x[key] - 11) / 8
        elif 'ghi' in key or 'dni' in key or 'dhi' in key:
            x[key] = torch.log10(x[key] + 1) - 2
        elif 'wind_speed' in key:
            x[key] = (torch.log10(x[key] + 0.1) - 1/3) * 3
        elif 'solar_zenith' in key:
            x[key] = (x[key] - torch.pi / 2) * 2
        elif 'solar_azimuth' in key:
            x[key] = (x[key] - torch.pi) / 2
    return x