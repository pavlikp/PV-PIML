from importlib.metadata import metadata
from os import listdir
import pathlib

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
                 target_variables=['dhi', 'ghi', 'dni', 'wind_speed', 'temp_air', 'unix_timestamps', 'solar_zenith', 'solar_azimuth'],
                 history_variables=['dhi', 'ghi', 'dni', 'wind_speed', 'temp_air', 'unix_timestamps', 'solar_zenith', 'solar_azimuth', 'production'],
                 countries=None,
                 installations=None,
                 split="train",
                 continuous=False,
                 test_fraction=5,
                 valid_fraction=100,
                 train_fraction=1):
        self.path = path
        self.previous_days = previous_days
        self.target_variables = target_variables
        self.history_variables = history_variables
        self.countries = countries
        self.installations = installations
        self.split = split
        self.continuous = continuous

        self.full_metadata = pd.read_csv(osp.join(self.path, "sample_selection.csv"))

        self.metadata = self.full_metadata[self.full_metadata["previous_days"] >= self.previous_days]

        installation_ids = self.full_metadata["installation_ID"].unique()
        installation_ids.sort()
        installation_idx = torch.arange(len(installation_ids))
        self.installation_id_to_idx = dict(zip(installation_ids, installation_idx))

        if self.countries is not None:
            self.metadata = self.metadata[self.metadata["country"].isin(self.countries)]

        if self.installations is not None:
            self.metadata = self.metadata[self.metadata["installation_ID"].isin(self.installations)]
        
        self.metadata = self.metadata.sort_values("date", ascending=True)

        self.train_fraction = train_fraction
        self.valid_fraction = valid_fraction
        self.test_fraction = test_fraction

        test_samples = self.metadata.iloc[-len(self.metadata)//self.test_fraction:]
        train_samples_full = self.metadata.iloc[:-len(self.metadata)//self.test_fraction]
        valid_samples = train_samples_full.iloc[::self.valid_fraction]
        train_samples = train_samples_full[~train_samples_full.index.isin(valid_samples.index)]
        train_samples = train_samples.iloc[::self.train_fraction]

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
        df_present = pd.read_csv(osp.join(self.path, pathlib.PureWindowsPath(self.samples.iloc[idx].filepath).as_posix()))
        df_present["date"] = date
        df_present["datetime"] = pd.to_datetime(df_present["date"].astype(str) + " " + df_present["time"])

        inputs = {}

        if 'dhi' in self.target_variables:
            inputs['dhi'] = torch.tensor(df_present['aswdir_s_i'].values).float()
        if 'ghi' in self.target_variables:
            inputs['ghi'] = torch.tensor(df_present['ghi'].values).float()
        if 'dni' in self.target_variables:
            inputs['dni'] = torch.tensor(df_present['aswdifd_s_i'].values).float()
        if 'wind_speed' in self.target_variables:
            inputs['wind_speed'] = torch.tensor(df_present['wind_speed'].values).float()
        if 'temp_air' in self.target_variables:
            inputs['temp_air'] = torch.tensor(df_present['t_2m'].values).float() - 273.15 # convert from K to °C
        if 'unix_timestamps' in self.target_variables:
            inputs['unix_timestamps'] = torch.tensor(df_present["datetime"].apply(lambda x: x.value // 10 ** 9).values).float()
        if 'solar_zenith' in self.target_variables or 'solar_azimuth' in self.target_variables:
            inputs['solar_zenith'], inputs['solar_azimuth'] = self._solar_position(df_present["datetime"], installation_metadata)

        if 'production' in self.history_variables and self.continuous and self.previous_days > 0:
            inputs['production'] = torch.tensor([]).float()
        for i in range(1, self.previous_days + 1):
            meta_index = self.samples.iloc[idx:idx+1].index[0] - i
            previous_date = pd.to_datetime(date) - pd.Timedelta(days=i)
            df_previous = pd.read_csv(osp.join(self.path, pathlib.PureWindowsPath(self.full_metadata.loc[meta_index].filepath).as_posix()))
            df_previous["date"] = previous_date
            df_previous["datetime"] = pd.to_datetime(df_previous["date"].astype(str) + " " + df_previous["time"])

            if self.continuous:
                if 'dhi' in self.history_variables:
                    inputs[f'dhi'] = torch.cat((torch.tensor(df_previous['aswdir_s_i'].values).float(), inputs[f'dhi']), dim=0)
                if 'ghi' in self.history_variables:
                    inputs[f'ghi'] = torch.cat((torch.tensor(df_previous['ghi'].values).float(), inputs[f'ghi']), dim=0)
                if 'dni' in self.history_variables:
                    inputs[f'dni'] = torch.cat((torch.tensor(df_previous['aswdifd_s_i'].values).float(), inputs[f'dni']), dim=0)
                if 'wind_speed' in self.history_variables:
                    inputs[f'wind_speed'] = torch.cat((torch.tensor(df_previous['wind_speed'].values).float(), inputs[f'wind_speed']), dim=0)
                if 'temp_air' in self.history_variables:
                    inputs[f'temp_air'] = torch.cat((torch.tensor(df_previous['t_2m'].values).float() - 273.15, inputs[f'temp_air']), dim=0) # convert from K to °C
                if 'unix_timestamps' in self.history_variables:
                    inputs[f'unix_timestamps'] = torch.cat((torch.tensor(df_previous["datetime"].apply(lambda x: x.value // 10 ** 9).values).float(), inputs[f'unix_timestamps']), dim=0)
                if 'production' in self.history_variables:
                    prod = torch.tensor(df_previous['production'].values).float()
                    prod = prod * 4000 / installation_metadata["System Size (watts)"] # convert from kW to W and normalize by system size
                    inputs[f'production'] = torch.cat((prod, inputs[f'production']), dim=0)

                if 'solar_zenith' in self.history_variables or 'solar_azimuth' in self.history_variables:
                    solar_zenith_t, solar_azimuth_t = self._solar_position(df_previous["datetime"], installation_metadata)
                    inputs[f'solar_zenith'] = torch.cat((solar_zenith_t, inputs[f'solar_zenith']), dim=0)
                    inputs[f'solar_azimuth'] = torch.cat((solar_azimuth_t, inputs[f'solar_azimuth']), dim=0)
            else:
                if 'dhi' in self.history_variables:
                    inputs[f'dhi_t-{i}d'] = torch.tensor(df_previous['aswdir_s_i'].values).float()
                if 'ghi' in self.history_variables:
                    inputs[f'ghi_t-{i}d'] = torch.tensor(df_previous['ghi'].values).float()
                if 'dni' in self.history_variables:
                    inputs[f'dni_t-{i}d'] = torch.tensor(df_previous['aswdifd_s_i'].values).float()
                if 'wind_speed' in self.history_variables:
                    inputs[f'wind_speed_t-{i}d'] = torch.tensor(df_previous['wind_speed'].values).float()
                if 'temp_air' in self.history_variables:
                    inputs[f'temp_air_t-{i}d'] = torch.tensor(df_previous['t_2m'].values).float() - 273.15 # convert from K to °C
                if 'unix_timestamps' in self.history_variables:
                    inputs[f'unix_timestamps_t-{i}d'] = torch.tensor(df_previous["datetime"].apply(lambda x: x.value // 10 ** 9).values).float()
                if 'production' in self.history_variables:
                    prod = torch.tensor(df_previous['production'].values).float()
                    prod = prod * 4000 / installation_metadata["System Size (watts)"] # convert from kW to W and normalize by system size
                    inputs[f'production_t-{i}d'] = prod

                if 'solar_zenith' in self.history_variables or 'solar_azimuth' in self.history_variables:
                    inputs[f'solar_zenith_t-{i}d'], inputs[f'solar_azimuth_t-{i}d'] = self._solar_position(df_previous["datetime"], installation_metadata)


        out = torch.tensor(df_present['production'].values * 4000).float() # convert from kW to W
        out = out / installation_metadata["System Size (watts)"] # normalize by system size

        meta = {}
        meta["system_size"] = installation_metadata["System Size (watts)"]
        meta["latitude"] = installation_metadata["Latitude"]
        meta["longitude"] = installation_metadata["Longitude"]
        meta["elevation"] = installation_metadata["Elevation"]
        meta["country"] = self.samples.iloc[idx].country
        meta["installation"] = self.samples.iloc[idx].installation_ID
        meta["installation_embedding"] = self.installation_id_to_idx[self.samples.iloc[idx].installation_ID]
        meta["date"] = self.samples.iloc[idx].date
        meta["tilt"] = torch.deg2rad(torch.tensor(installation_metadata['Array Tilt (degrees)'])).float()
        if meta["tilt"].isnan():
            meta["tilt"] = torch.deg2rad(torch.tensor(installation_metadata['Latitude'] * 0.85)).float() # rule of thumb for tilt if not provided
        if installation_metadata['Orientation'] == 'N':
            meta["orientation"] = torch.deg2rad(torch.tensor(0)).float()
        elif installation_metadata['Orientation'] == 'NE':
            meta["orientation"] = torch.deg2rad(torch.tensor(45)).float()
        elif installation_metadata['Orientation'] == 'E':
            meta["orientation"] = torch.deg2rad(torch.tensor(90)).float()
        elif installation_metadata['Orientation'] == 'SE':
            meta["orientation"] = torch.deg2rad(torch.tensor(135)).float()
        elif installation_metadata['Orientation'] == 'S':
            meta["orientation"] = torch.deg2rad(torch.tensor(180)).float()
        elif installation_metadata['Orientation'] == 'SW':
            meta["orientation"] = torch.deg2rad(torch.tensor(225)).float()
        elif installation_metadata['Orientation'] == 'W':
            meta["orientation"] = torch.deg2rad(torch.tensor(270)).float()
        elif installation_metadata['Orientation'] == 'NW':
            meta["orientation"] = torch.deg2rad(torch.tensor(315)).float()
        elif installation_metadata['Orientation'] == 'EW': # I guess?
            meta["orientation"] = torch.deg2rad(torch.tensor(180)).float()
            meta["tilt"] = torch.deg2rad(torch.tensor(0)).float()
        else:
            raise NotImplementedError(f"Orientation {installation_metadata['Orientation']} not implemented!")

        return inputs, out, meta
    
    def _solar_position(self, datetimes, installation_metadata):
        solar_zenith = torch.zeros(len(datetimes))
        solar_azimuth = torch.zeros(len(datetimes))
        loc = location.Location(latitude=installation_metadata["Latitude"],
                                longitude=installation_metadata["Longitude"],
                                altitude=installation_metadata["Elevation"],
                                tz="UTC")
        solpos = loc.get_solarposition(datetimes - pd.Timedelta(minutes=7.5))
        solar_zenith = torch.deg2rad(torch.tensor(solpos.apparent_zenith.values)).float()
        solar_azimuth = torch.deg2rad(torch.tensor(solpos.azimuth.values)).float()

        return solar_zenith, solar_azimuth