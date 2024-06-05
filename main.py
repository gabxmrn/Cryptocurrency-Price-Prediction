from data import DataProcessing


### 1) Collecting historical cryptocurrency data 

database = DataProcessing(file_name='Database', 
                          asset_names=['BTC', 'LTC', 'ETH'], 
                          start_date='2018-01-01', 
                          end_date='2021-06-30', 
                          price_type='Close')

btc = database.asset_data['BTC'] # Bitcoin
ltc = database.asset_data['LTC'] # Litecoin
eth = database.asset_data['ETH'] # Ethereum


### 2) Data exploration and visualization

database.plot_data(asset_name='BTC', splitting_date='2020-10-22') # Figure 2
database.plot_data(asset_name='LTC', splitting_date='2020-10-22') # Figure 3
database.plot_data(asset_name='ETH', splitting_date='2020-10-22') # Figure 4

database.plot_all_data() # Figure 8

database.plot_correlation(asset_names=['BTC', 'LTC', 'ETH']) # Figure 9


### 3) Training three types of models 
    # Long short-term memory (LSTM) 
    # Gated recurrent unit (GRU) 
    # Bidirectional LSTM (bi-LSTM)
    # Training -> from 22/01/2018 until 22/10/2020 (80%)
    
btc_train, btc_test = database.get_splitted_data(asset_name='BTC', splitting_date='2020-10-22')
ltc_train, ltc_test = database.get_splitted_data(asset_name='LTC', splitting_date='2020-10-22')
eth_train, eth_test = database.get_splitted_data(asset_name='ETH', splitting_date='2020-10-22')



### 4) Testing the models 
    # Testing -> from 22/10/2020 until 30/06/2021 (20%)

### 5) Extracting and comparing the results 