from importlib.metadata import metadata
from os import listdir

from torch.utils.data import Dataset
import os.path as osp
import pandas as pd
from datetime import datetime
import torch

import pvlib
from pvlib import location

class PVDataset(Dataset):

    def __init__(self,
                 path,
                 country,
                 installation="top",
                 split="train",):
        self.path = path
        self.country = country
        self.split = split
        self.installation = installation

        self.country_folder = osp.join(self.path, self.country)
        self.metadata = pd.read_csv(osp.join(self.country_folder, "metadata.csv"))

        self.metadata["Total Records"] = (self.metadata["Number of records 2020"] +
                                          self.metadata["Number of records 2021"] +
                                          self.metadata["Number of records 2022"] +
                                          self.metadata["Number of records 2023"])
        
        if self.installation == "top":
            self.metadata = self.metadata.sort_values("Total Records", ascending=False)
            self.installation_folder = osp.join(self.country_folder, str(self.metadata["System ID"].iloc[0]))

            train_samples, valid_samples, test_samples = self._split_samples_in_folder(self.installation_folder)
            self.installation_metadata = self.metadata.iloc[0]
        else:
            self.installation_folder = osp.join(self.country_folder, self.installation)
            train_samples, valid_samples, test_samples = self._split_samples_in_folder(self.installation_folder)
            self.installation_metadata = self.metadata[self.metadata["System ID"] == self.installation].iloc[0]
        
        if self.split == "train":
            self.datapaths = [osp.join(self.installation_folder, sample) for sample in train_samples]
        elif self.split == "valid":
            self.datapaths = [osp.join(self.installation_folder, sample) for sample in valid_samples]
        elif self.split == "test":
            self.datapaths = [osp.join(self.installation_folder, sample) for sample in test_samples]
        
        self.datapaths.sort()

    def __len__(self):
        return len(self.datapaths)

    def __getitem__(self, idx):
        date = self.datapaths[idx].split("\\")[-1].split(".")[0]
        df = pd.read_csv(self.datapaths[idx])
        df["date"] = datetime.strptime(date, "%Y%m%d")
        df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"])


        dhi = torch.tensor(df['aswdir_s_i'].values)
        ghi = torch.tensor(df['ghi'].values)
        dni = torch.tensor(df['aswdifd_s_i'].values)
        wind_speed = torch.tensor(df['wind_speed'].values)
        temp_air = torch.tensor(df['t_2m'].values) - 273.15 # convert from K to °C
        unix_timestamps = torch.tensor(df["datetime"].apply(lambda x: x.value // 10 ** 9).values)

        out = torch.tensor(df['production'].values)

        return {'dhi': dhi,
                'ghi': ghi,
                'dni': dni,
                'wind_speed': wind_speed,
                'temp_air': temp_air,
                'unix_timestamps': unix_timestamps}, out, self.installation_metadata.to_dict()

    def _split_samples_in_folder(self, folder):
        TEST_FRACTION = 5
        VALID_FRACTION = 8

        samples = listdir(folder)
        test_samples = samples[-len(samples)//TEST_FRACTION:]
        train_samples_full = samples[:-len(samples)//TEST_FRACTION]
        valid_samples = train_samples_full[::VALID_FRACTION]
        train_samples = list(set(train_samples_full) - set(valid_samples))
        return train_samples, valid_samples, test_samples