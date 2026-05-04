from importlib.metadata import metadata
from os import listdir

from torch.utils.data import Dataset
import os.path as osp
import pandas as pd
from datetime import date, datetime
import torch

import pvlib
from pvlib import location

class PVDatasetWithHistory(Dataset):

    def __init__(self,
                 path,
                 previous_days=5,
                 countries=None,
                 installations=None,
                 split="train",):
        self.path = path
        self.previous_days = previous_days
        self.countries = countries
        self.installations = installations
        self.split = split

        self.full_metadata = pd.read_csv(osp.join(self.path, "sample_selection.csv"))

        self.metadata = self.full_metadata[self.full_metadata["previous_days"] >= self.previous_days]

        if self.countries is not None:
            self.metadata = self.full_metadata[self.full_metadata["country"].isin(self.countries)]

        if self.installations is not None:
            self.metadata = self.full_metadata[self.full_metadata["installation"].isin(self.installations)]
        
        self.metadata = self.metadata.sort_values("date", ascending=False)

        TEST_FRACTION = 5
        VALID_FRACTION = 8

        test_samples = self.metadata.iloc[-len(self.metadata)//TEST_FRACTION:]
        train_samples_full = self.metadata.iloc[:-len(self.metadata)//TEST_FRACTION]
        valid_samples = train_samples_full.iloc[::VALID_FRACTION]
        train_samples = train_samples_full[~train_samples_full.index.isin(valid_samples.index)]

        if self.split == "train":
            self.samples = train_samples
        elif self.split == "valid":
            self.samples = valid_samples
        elif self.split == "test":
            self.samples = test_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        installation_metadata = pd.read_csv(osp.join(self.path, self.samples.iloc[idx].country, "metadata.csv"))
        installation_metadata = installation_metadata[installation_metadata["System ID"] == self.samples.iloc[idx].installation_ID].iloc[0]

        date = self.samples.iloc[idx].date
        df_present = pd.read_csv(osp.join(self.path, self.samples.iloc[idx].filepath))
        df_present["date"] = date
        df_present["datetime"] = pd.to_datetime(df_present["date"].astype(str) + " " + df_present["time"])

        inputs = {}

        inputs['dhi'] = torch.tensor(df_present['aswdir_s_i'].values)
        inputs['ghi'] = torch.tensor(df_present['ghi'].values)
        inputs['dni'] = torch.tensor(df_present['aswdifd_s_i'].values)
        inputs['wind_speed'] = torch.tensor(df_present['wind_speed'].values)
        inputs['temp_air'] = torch.tensor(df_present['t_2m'].values) - 273.15 # convert from K to °C
        inputs['unix_timestamps'] = torch.tensor(df_present["datetime"].apply(lambda x: x.value // 10 ** 9).values)
        
        inputs['solar_zenith'], inputs['solar_azimuth'] = self._solar_position(df_present["datetime"], installation_metadata)

        for i in range(1, self.previous_days + 1):
            meta_index = self.samples.iloc[idx:idx+1].index[0] - i
            previous_date = pd.to_datetime(date) - pd.Timedelta(days=i)
            df_previous = pd.read_csv(osp.join(self.path, self.full_metadata.loc[meta_index].filepath))
            df_previous["date"] = previous_date
            df_previous["datetime"] = pd.to_datetime(df_previous["date"].astype(str) + " " + df_previous["time"])

            inputs[f'dhi_t-{i}d'] = torch.tensor(df_previous['aswdir_s_i'].values)
            inputs[f'ghi_t-{i}d'] = torch.tensor(df_previous['ghi'].values)
            inputs[f'dni_t-{i}d'] = torch.tensor(df_previous['aswdifd_s_i'].values)
            inputs[f'wind_speed_t-{i}d'] = torch.tensor(df_previous['wind_speed'].values)
            inputs[f'temp_air_t-{i}d'] = torch.tensor(df_previous['t_2m'].values) - 273.15 # convert from K to °C
            inputs[f'unix_timestamps_t-{i}d'] = torch.tensor(df_previous["datetime"].apply(lambda x: x.value // 10 ** 9).values)
            inputs[f'production_t-{i}d'] = torch.tensor(df_previous['production'].values)

            inputs[f'solar_zenith_t-{i}d'], inputs[f'solar_azimuth_t-{i}d'] = self._solar_position(df_previous["datetime"], installation_metadata)


        out = torch.tensor(df_present['production'].values * 4000) # convert from kW to W
        out = out / installation_metadata["System Size (watts)"] # normalize by system size

        meta = {}
        meta["system_size"] = installation_metadata["System Size (watts)"]
        meta["latitude"] = installation_metadata["Latitude"]
        meta["longitude"] = installation_metadata["Longitude"]
        meta["elevation"] = installation_metadata["Elevation"]
        meta["country"] = self.samples.iloc[idx].country
        meta["installation"] = self.samples.iloc[idx].installation_ID
        meta["date"] = self.samples.iloc[idx].date

        return inputs, out, meta
    
    def _solar_position(self, datetimes, installation_metadata):
        solar_zenith = torch.zeros(len(datetimes))
        solar_azimuth = torch.zeros(len(datetimes))
        loc = location.Location(latitude=installation_metadata["Latitude"],
                                longitude=installation_metadata["Longitude"],
                                altitude=installation_metadata["Elevation"],
                                tz="UTC")
        solpos = loc.get_solarposition(datetimes - pd.Timedelta(minutes=7.5))
        solar_zenith = torch.deg2rad(torch.tensor(solpos.apparent_zenith.values))
        solar_azimuth = torch.deg2rad(torch.tensor(solpos.azimuth.values))

        return solar_zenith, solar_azimuth