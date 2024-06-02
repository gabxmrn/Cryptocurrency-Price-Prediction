from typing import Tuple, List
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(start_date: datetime, end_date: datetime, price_type: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and filter cryptocurrency price data for a specified date range and a price type.

    Args:
        start_date (datetime): start date from which to filter the data.
        end_date (datetime): end date up to which to filter the data.
        price_type (str): The type of price data to retrieve.

    Raises:
        ValueError: _description_

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containinf three pandas DataFrame for BTC, LTC, and ETH respectively.
    """

    if price_type.lower() not in ['open', 'high', 'low', 'close']:
        raise ValueError('Price should be one of: open, high, low or close.')

    xls = pd.read_excel(r'data.xlsx', sheet_name=['BTC', 'LTC', 'ETH'])

    def process_data(df: pd.DataFrame, crypto_name: str):
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        df = df.loc[:, ['Date', price_type]]
        df.columns = ['Date', crypto_name]
        return df

    btc = process_data(xls['BTC'], 'BTC')
    ltc = process_data(xls['LTC'], 'LTC')
    eth = process_data(xls['ETH'], 'ETH')

    return btc, ltc, eth


def split_data(df: pd.DataFrame, split_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the data into a training and test subsets.

    Args:
        df (pd.DataFrame): dataframe containing a time series with dates and prices for one cryptocurrency.
        split_date (datetime): date on which we split the data in a training and a test subset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test sets.
    """
    train = df[df['Date'] < split_date]
    test = df[df['Date'] >= split_date]
    return (train, test)  


def plot_data(df: pd.DataFrame, split_date: datetime) -> None:
    """Plots the price data with different colors for the training and the test data.

    Args:
        df (pd.DataFrame): dataframe containing a time series with dates and prices for one cryptocurrency.
        split_date (datetime): date on which we split the data in a training and a test subset.
    """
    
    # Data processing
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    train, test = split_data(df, split_date)

    # Plot graph
    plt.figure(figsize=(10, 5))
    plt.plot(train['Date'], train.iloc[:,1], label='Training Set', color='green')
    plt.plot(test['Date'], test.iloc[:,1], label='Testing Set', color='red')
    plt.title(f'Training and Testing Dataset for {train.columns[1]}')
    plt.xlabel('Date')
    plt.ylabel('Price - USD')
    plt.legend()
    plt.grid(False)
    plt.show()


def plot_correlation(btc: pd.DataFrame, ltc: pd.DataFrame, eth: pd.DataFrame) -> None:
    
    # Correlation matrix
    df_combined = pd.concat([btc.iloc[:,1], ltc.iloc[:,1], eth.iloc[:,1]], axis=1)
    correlation_matrix = df_combined.corr()

    # Heatmap
    color = sns.cubehelix_palette(as_cmap=True)
    sns.heatmap(correlation_matrix, annot=True, cmap=color, fmt=".2f")
    plt.title('Correlation between cryptocurrencies')
    plt.show()