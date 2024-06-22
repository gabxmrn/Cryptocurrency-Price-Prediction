import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns

from data import DATES

def plot_asset(asset_name:str, asset_data:pd.DataFrame, splitting_date: datetime,  price_type:str='Close') -> None :
    """
    Plots the price data of a single asset, dividing it into training and testing sets based on a splitting date.

    Inputs:
        asset_name (str): Name of the asset.
        asset_data (pd.DataFrame): DataFrame containing the price data of the asset.
        price_type (str): The column name of the price data to plot (e.g., 'Close').
        splitting_date (datetime): Date to split the data into training and testing sets.
    """
    train = asset_data[asset_data[DATES] < splitting_date]
    test = asset_data[asset_data[DATES] >= splitting_date]
    plt.figure(figsize=(10, 5))
    plt.plot(train[DATES], train[price_type], label='Training Set', color='green')
    plt.plot(test[DATES], test[price_type], label='Testing Set', color='red')
    plt.title(f'Training and Testing Dataset for {asset_name}')
    plt.xlabel('Dates')
    plt.ylabel('Price - USD')
    plt.legend()
    plt.show()
        
def plot_all_assets(asset_names:list, data:dict, price_type:str) -> None :
    """
    Plots time series data for all assets.

    Inputs:
        asset_names (list): List of asset names.
        data (dict): Dictionary where keys are asset names and values are their respective DataFrames.
        price_type (str): The column name of the price data to plot (e.g., 'Close').
    """
    plt.figure(figsize=(10, 5))
    for asset in asset_names :
        asset_data = data[asset]
        plt.plot(asset_data[DATES], asset_data[price_type], label=asset)
    plt.title(f'Time series with closing prices for all assets')
    plt.xlabel('Price Date')
    plt.ylabel('Price US Dollar')
    plt.legend()
    plt.show()
        
def plot_correlation(asset_names:list, data:dict, price_type:str) -> None :
    """
    Plots a heatmap of the correlation matrix between the prices of different assets.

    Inputs:
        asset_names (list): List of asset names.
        data (dict): Dictionary containing the data for each asset.
        price_type (str): The type of price data to correlate (e.g., 'Close').
    """
    data_price = {}
    for asset in asset_names :
        try:
            data_price[asset] = data[asset][price_type]
        except KeyError as e:
            raise ValueError(f"Asset data for {str(e)} is not available in the loaded data.")
        data_price[asset].name = asset
    full_data = pd.concat([data[asset][price_type] for asset in asset_names], axis=1)
    correlation_matrix = full_data.corr()
    color = sns.cubehelix_palette(as_cmap=True)
    sns.heatmap(correlation_matrix, annot=True, cmap=color, fmt=".2f")
    plt.title('Correlation between cryptocurrencies')
    plt.show()
    
    
def plot_results(asset, predictions, targets, dates) -> None :
    """
    Plots the predicted vs actual prices for an asset.

    Inputs:
        asset (str): Asset name.
        predictions (array): Array of predicted prices.
        targets (array): Array of actual prices.
        dates (array): Array of dates corresponding to the prices.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(dates[:-1], predictions[1:], label='Predicted Price', color='red')
    plt.plot(dates[:-1], targets[:-1], label='Actual Price', color='green')
    plt.title(f'{asset} Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.gcf().autofmt_xdate() 
    plt.show()
    
    
def get_rmse(predictions, targets) -> float :
    """
    Calculates the Root Mean Square Error between predictions and targets.

    Inputs:
        predictions (array): Predicted values.
        targets (array): Actual values.

    Outputs:
        float: Calculated RMSE value.
    """
    return np.sqrt(np.mean((targets - predictions)**2))
    
def get_mape(predictions, targets) -> float :
    """
    Calculates the Mean Absolute Percentage Error between predictions and targets.

    Inputs:
        predictions (array): Predicted values.
        targets (array): Actual values.

    Outputs:
        float: Calculated MAPE value in percentage.
    """
    return np.mean(np.abs((targets - predictions) / targets)) * 100


def run_process(asset_name:str, data_args:dict, models:dict, models_args:dict, charts:bool=False) -> pd.DataFrame :
    """
    Runs the model training and testing process for given asset models and data arguments, optionally plotting results.

    Inputs:
        asset_name (str): Name of the asset being analyzed.
        data_args (dict): Dictionary containing DataLoader instances and other relevant data like scaler.
        models (dict): Dictionary of model instances ('LSTM', 'GRU', 'BI_LSTM').
        models_args (dict): Dictionary containing model arguments like epochs and optimizers.
        charts (bool): Flag to determine whether to plot results or not.

    Outputs:
        pd.DataFrame: DataFrame containing the RMSE and MAPE values for each model type.
    """
            
    results = pd.DataFrame(index=["LSTM", "bi-LSTM", "GRU"], columns=["RMSE", "MAPE"])
    targets = data_args["asset"].actual_prices[models_args["sequence_lenght"]:]
    dates = data_args["asset"].dates[models_args["sequence_lenght"]:]
    
    ### LSTM Model : 
    models["LSTM"].train_model(models_args["epochs"], data_args["train_loader"], models_args["criterion"], models["LSTM_optimizer"])
    lstm_predictions = models["LSTM"].test_model(data_args["test_loader"], data_args["scaler"])
    results.at['LSTM', 'RMSE'] = round(get_rmse(lstm_predictions[1:], targets[:-1]), 3)
    results.at['LSTM', 'MAPE'] = round(get_mape(lstm_predictions[1:], targets[:-1]), 4)
    
    ### GRU Model : 
    models["GRU"].train_model(models_args["epochs"], data_args["train_loader"], models_args["criterion"], models["GRU_optimizer"])
    gru_predictions = models["GRU"].test_model(data_args["test_loader"], data_args["scaler"])
    results.at['GRU', 'RMSE'] = round(get_rmse(gru_predictions[1:], targets[:-1]), 3)
    results.at['GRU', 'MAPE'] = round(get_mape(gru_predictions[1:], targets[:-1]), 4)
    
    ### BI-LSTM Model : 
    models["BI_LSTM"].train_model(models_args["epochs"], data_args["train_loader"], models_args["criterion"], models["BI_LSTM_optimizer"])
    bi_lstm_predictions = models["BI_LSTM"].test_model(data_args["test_loader"], data_args["scaler"])
    results.at['bi-LSTM', 'RMSE'] = round(get_rmse(bi_lstm_predictions[1:], targets[:-1]), 3)
    results.at['bi-LSTM', 'MAPE'] = round(get_mape(bi_lstm_predictions[1:], targets[:-1]), 4)

    if charts:
        plot_results(asset_name, lstm_predictions, targets, dates)
        plot_results(asset_name, gru_predictions, targets, dates)
        plot_results(asset_name, bi_lstm_predictions, targets, dates)

    return results 