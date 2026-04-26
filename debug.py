from models.ADRInspired import ADRInspired as ADR
from PVDatamodule import PVDatamodule

from utils.config import load_config

config = load_config("./config/ADR.yaml")

model = ADR(config)

datamodule = PVDatamodule(config)

datamodule.setup("fit")

train_loader = datamodule.train_dataloader()

x, y, metadata = next(iter(train_loader))

y_hat = model(x, metadata)

print(y_hat)