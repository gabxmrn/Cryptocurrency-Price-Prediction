import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

DATES_COL = "Date"  
        
class DataProcessing:
    """
    Class for processing and analyzing financial asset data from an Excel file.
    The class handles data loading, processing, and visualization for multiple assets over a specified date range.
    """
    
    def __init__(self, file_name:str, asset_names:list, start_date:datetime, end_date:datetime, price_type:str) -> None:
        """
        Initializes the DataProcessing class with specified file, assets, date range, and price type.
        
        Parameters:
        - file_name: str - the name of the Excel file to load.
        - asset_names: list - a list of asset names whose data will be loaded.
        - start_date: datetime - the starting date for the data selection.
        - end_date: datetime - the ending date for the data selection.
        - price_type: str - the type of price data to extract.
        """
        self.asset_names = asset_names
        self.asset_data = self.__load_data(file_name, start_date, end_date, price_type)

    def __load_data(self, file_name:str, start_date:datetime, end_date:datetime, price_type:str) -> dict:
        """
        Loads data from an Excel file for the specified assets, date range, and price type.
        
        Parameters are the same as in __init__.

        Returns:
        - dict: A dictionary of DataFrames, with asset names as keys and processed data as values.
        
        Raises:
        - FileNotFoundError: If there is an error loading the Excel file, e.g., file not found or incorrect file name.
        - ValueError: If data for any of the specified assets is not found in the loaded Excel sheets.
        """
        
        try:
            raw_data = pd.read_excel(file_name + ".xlsx", sheet_name=self.asset_names)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load Excel file: {e}")
        
        results = {}
        for asset in self.asset_names:
            try:
                results[asset] = self.__process_data(raw_data[asset], asset, price_type, start_date, end_date)
            except KeyError:
                raise ValueError(f"Asset data for {asset} is not available in the loaded Excel sheets.")
        return results

    def __process_data(self, data:pd.DataFrame, asset_name:str, price_type:str, start_date:datetime, end_date:datetime) -> pd.DataFrame :
        """
        Processes data for a single asset, filtering and adjusting as specified.
        
        Parameters:
        - data: pd.DataFrame - the DataFrame containing data for the asset.
        - asset_name: str - the name of the asset.
        - price_type: str - the type of price data to extract.
        - start_date: datetime - the start date for filtering data.
        - end_date: datetime - the end date for filtering data.

        Returns:
        - pd.DataFrame: Processed data set with dates as index and specified price type as column.

        Raises:
        - ValueError: If the 'Date' column is missing from the data for the specified asset.
        - ValueError: If the specified price type is not available in the data for the asset.
        """
        
        if DATES_COL not in data.columns:
            raise ValueError(f"The column '{DATES_COL}' is missing in data for {asset_name}.")
        data[DATES_COL] = pd.to_datetime(data[DATES_COL])
        data = data[(data[DATES_COL] >= start_date) & (data[DATES_COL] <= end_date)]
        
        if price_type not in data.columns:
            raise ValueError(f"The price type '{price_type}' is not available in the data for {asset_name}.")
        result = data.loc[:, [DATES_COL, price_type]]
        result.columns = [DATES_COL, asset_name]
        
        return result.set_index(DATES_COL, drop=True)
    
    def get_splitted_data(self, asset_name:str, splitting_date: datetime) -> tuple :
        """
        Splits the data for a given asset into training and testing sets based on a splitting date.
        
        Parameters:
        - asset_name: str - the name of the asset to split data for.
        - splitting_date: datetime - the date to split the training and testing data.

        Returns:
        - tuple: (train, test) where both are DataFrames.
        """
        data = self.asset_data[asset_name]
        train = data[data.index < splitting_date]
        test = data[data.index >= splitting_date]
        return (train, test) 
    
    def get_prepared_data(self, asset_name:str, splitting_date:datetime, n_steps:int) -> tuple:
        """
        Prepare data for time series forecasting models.

        Parameters:
        - asset_name: str - the name of the asset to prepare data for.
        - splitting_date: datetime - the date to split the training and testing data.
        - n_steps: int - number of time steps to include in the input features.

        Returns:
        - tuple (np.array, np.array, np.array, np.array): Arrays of input features and output labels for train & test series.
        """
        train, test = self.get_splitted_data(asset_name, splitting_date)
        train, test = train.values.flatten(), test.values.flatten()
        
        X_train, X_test, y_train, y_test = [], [], [], []
        
        for i in range(len(train) - n_steps):
            X_train.append(train[i:i+n_steps])
            y_train.append(train[i+n_steps])
            
        for j in range(len(test) - n_steps):
            X_test.append(test[j:j+n_steps])
            y_test.append(test[j+n_steps])
        
        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    
    def plot_data(self, asset_name:str, splitting_date: datetime) -> None:
        """
        Plots the training and testing datasets for a given asset.
        
        Parameters:
        - asset_name: str - the name of the asset to plot.
        - splitting_date: datetime - the date to split the data for visualization.
        """
        train, test = self.get_splitted_data(asset_name, splitting_date)
        plt.figure(figsize=(10, 5))
        plt.plot(train.index, train, label='Training Set', color='green')
        plt.plot(test.index, test, label='Testing Set', color='red')
        plt.title(f'Training and Testing Dataset for {asset_name}')
        plt.xlabel('Dates')
        plt.ylabel('Price - USD')
        plt.legend()
        plt.grid(False)
        plt.show()
        
    def plot_all_data(self) -> None:
        """
        Plots the closing prices for all assets managed by this instance.
        """
        plt.figure(figsize=(10, 5))
        for asset in self.asset_names :
            data = self.asset_data[asset]
            plt.plot(data.index, data, label=asset)

        plt.title(f'Time series with closing prices for all assets')
        plt.xlabel('Price Date')
        plt.ylabel('Price US Dollar')
        plt.legend()
        plt.grid(False)
        plt.show()
        
    def plot_correlation(self, asset_names:list) -> None:
        """
        Plots a correlation matrix for the specified assets to visualize the relationships between them.

        Parameters:
        - asset_names: list - a list of asset names whose correlations are to be analyzed and visualized.
        
        Raises:
        - ValueError: If an asset name provided in the list does not exist in the asset_data dictionary.
        """
        try:
            full_data = pd.concat([self.asset_data[asset] for asset in asset_names], axis=1)
        except KeyError as e:
            raise ValueError(f"Asset data for {str(e)} is not available in the loaded data.")
        
        
        full_data = pd.concat([self.asset_data[asset] for asset in asset_names], axis=1)
        correlation_matrix = full_data.corr()
        color = sns.cubehelix_palette(as_cmap=True)
        sns.heatmap(correlation_matrix, annot=True, cmap=color, fmt=".2f")
        plt.title(f'Correlation between {asset_names}')
        plt.show()