from models import TSMixerx
import torch

model = TSMixerx(h=96, input_size=480, n_series=1, futr_exog_size=7)

insample_y = torch.randn(128, 480, 1)
futr_exog = torch.randn(128, 7, 480 + 96, 1)
stat_exog = torch.randn(128, 2)

batch = {
    "insample_y": insample_y,
    "hist_exog": None,
    "futr_exog": futr_exog,
    "stat_exog": None
}

y = model(batch)

print(y.shape)