import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import datetime

CLOSING_PRICE = "Close"
DATES = "Date"

class DataProcessing:
    """
    Class for loading and processing financial data from an Excel file.
    
    Attributes:
        asset_names (list): List of asset names to load.
        data (dict): Dictionary containing processed data for each asset.
    """
    
    def __init__(self, file_name:str, asset_names:list, start_date:datetime, end_date:datetime) -> None :
        """
        Initializes the class with the file name, list of assets, and start and end dates.

        Inputs:
            file_name (str): Name of the Excel file (without the extension).
            asset_names (list): List of asset names to load.
            start_date (datetime): Start date for filtering the data.
            end_date (datetime): End date for filtering the data.
        """
        
        self.asset_names = asset_names
        self.data = self.__load_data(file_name, start_date, end_date)
        
    def __load_data(self, file_name:str, start_date:datetime, end_date:datetime) -> dict :
        """
        Loads data from an Excel file and filters it between specified start and end dates.
        
        Inputs:
            file_name (str): Name of the Excel file including asset names as sheet names.
            start_date (datetime): Start date for data filtering.
            end_date (datetime): End date for data filtering.
        
        Outputs:
            dict: Dictionary with keys as asset names and values as processed dataframes.
        """
        
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
        """
        Processes individual asset data, filters by date, and ensures required columns are present.

        Inputs:
            data (pd.DataFrame): Raw data for an asset.
            asset_name (str): Name of the asset for which data is processed.
            start_date (datetime): Start date for data filtering.
            end_date (datetime): End date for data filtering.
        
        Outputs:
            pd.DataFrame: Processed data for the specified asset within the date range.
        """
        
        if DATES not in data.columns:
            raise ValueError(f"The column '{DATES}' is missing in data for {asset_name}.")
        data[DATES] = pd.to_datetime(data[DATES])
        data = data[(data[DATES] >= start_date) & (data[DATES] <= end_date)]
        return data
    
    def get_data_loader(self, asset_name:str, sequence_lenght:int, split_date:str, batch_size:int) -> tuple :
        """
        Creates data loaders for the training and testing datasets of a specific asset.

        Inputs:
            asset_name (str): Name of the asset to create data loaders for.
            sequence_length (int): Number of consecutive data points used as input.
            split_date (str): Date to split the data into training and testing datasets.
            batch_size (int): Number of samples in each batch of data. 

        Outputs:
            tuple: Contains the test dataset, training DataLoader, and testing DataLoader.
        """
        
        asset_data = self.data[asset_name]
        train_data = Asset(asset_data[asset_data[DATES] < split_date], sequence_lenght, split_date)
        test_data = Asset(asset_data[asset_data[DATES] >= split_date], sequence_lenght, split_date)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
        return test_data, train_loader, test_loader
            

class Asset(Dataset):
    """
    Dataset class for an individual asset, providing scaling and sequential data preparation for deep learning models.

    Attributes:
        data (list): List of tensors representing scaled and sequential price data.
        actual_prices (pd.DataFrame): Actual closing prices for comparison post-prediction.
        dates (pd.DataFrame): Dates corresponding to the actual prices.
    """
    
    def __init__(self, asset_data, sequence_lenght:int, split_date:str) -> None :
        """
        Initializes the dataset with scaled and sequential financial data for the model input.

        Inputs:
            asset_data (pd.DataFrame): DataFrame containing the price data of the asset.
            sequence_length (int): Number of consecutive data points to use for each sequence.
            split_date (str): Date to split the dataset into training and testing periods.
        """
        self.scaler = MinMaxScaler()       
        test_scaled = self.scaler.fit_transform(asset_data[[CLOSING_PRICE]])
        
        self.data = torch.tensor(test_scaled, dtype=torch.float32)
        self.data = [self.data[i:i+sequence_lenght+1] for i in range(len(self.data)-sequence_lenght)]
        
        self.actual_prices = asset_data[asset_data[DATES] >= split_date][[CLOSING_PRICE]]
        self.dates = asset_data[asset_data[DATES] >= split_date][[DATES]]
    
    def __len__(self) -> int :
        """
        Returns the number of sequences available in the dataset.

        Outputs:
            int: Total number of sequences.
        """
        return len(self.data)

    def __getitem__(self, index) -> tuple :
        """
        Retrieves an item at a specified index in the dataset.

        Inputs:
            index (int): Index of the data sequence to retrieve.

        Outputs:
            tuple: Contains input features 'x' (all but the last in sequence) and label 'y' (last in sequence).
        """
        x = self.data[index][:-1]  
        y = self.data[index][-1]  
        return x, y

    def get_scaler(self) :
        """
        Retrieves the MinMaxScaler instance used for scaling the dataset.

        Outputs:
            MinMaxScaler: Scaler used for transforming the data.
        """
        return self.scaler