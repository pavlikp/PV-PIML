from PVDatasetWithHistory import PVDatasetWithHistory

ds = PVDatasetWithHistory(r"C:\Users\peter\Repos\ENFIELD-data\processed_unpack\nwp_0h", previous_days=0, split="train")

x, y, meta = ds[0]

print(x.keys())