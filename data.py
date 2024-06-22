import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import datetime

CLOSING_PRICE = "Close"
DATES = "Date"

class DataProcessing:
    
    def __init__(self, file_name:str, asset_names:list, start_date:datetime, end_date:datetime) -> None :
        self.asset_names = asset_names
        self.data = self.__load_data(file_name, start_date, end_date)
        
    def __load_data(self, file_name:str, start_date:datetime, end_date:datetime) -> dict :
        try:
            raw_data = pd.read_excel(file_name + ".xlsx", sheet_name=self.asset_names)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load Excel file: {e}")
        
        results = {}
        for asset in self.asset_names:
            try:
                results[asset] = self.__process_data(raw_data[asset], asset, start_date, end_date)
            except KeyError:
                raise ValueError(f"Asset data for {asset} is not available in the loaded Excel sheets.")
        return results
    
    def __process_data(self, data:pd.DataFrame, asset_name:str, start_date:datetime, end_date:datetime) -> pd.DataFrame :
        if DATES not in data.columns:
            raise ValueError(f"The column '{DATES}' is missing in data for {asset_name}.")
        data[DATES] = pd.to_datetime(data[DATES])
        data = data[(data[DATES] >= start_date) & (data[DATES] <= end_date)]
        return data
    
    def get_data_loader(self, asset_name:str, sequence_lenght:int, split_date:str, batch_size:int=64) -> tuple :
        asset_data = self.data[asset_name]
        train_data = Asset(asset_data[asset_data[DATES] < split_date], sequence_lenght, split_date)
        test_data = Asset(asset_data[asset_data[DATES] >= split_date], sequence_lenght, split_date)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
        return test_data, train_loader, test_loader
            

class Asset(Dataset):
    
    def __init__(self, asset_data, sequence_lenght:int, split_date:str):
        self.scaler = MinMaxScaler()       
        test_scaled = self.scaler.fit_transform(asset_data[[CLOSING_PRICE]])
        
        self.data = torch.tensor(test_scaled, dtype=torch.float32)
        self.data = [self.data[i:i+sequence_lenght+1] for i in range(len(self.data)-sequence_lenght)]
        
        self.actual_prices = asset_data[asset_data[DATES] >= split_date][[CLOSING_PRICE]]
        self.dates = asset_data[asset_data[DATES] >= split_date][[DATES]]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index][:-1]  # Features: all entries except the last
        y = self.data[index][-1]   # Labels: last entry in the sequence
        return x, y

    def get_scaler(self):
        return self.scaler