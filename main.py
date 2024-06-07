from data import DataProcessing
from models import LSTM

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


### 1) Collecting historical cryptocurrency data 

database = DataProcessing(file_name='Database', 
                          asset_names=['BTC', 'LTC', 'ETH'], 
                          start_date='2018-01-01', 
                          end_date='2021-06-30', 
                          price_type='Close')

btc = database.asset_data['BTC'] # Bitcoin
ltc = database.asset_data['LTC'] # Litecoin
eth = database.asset_data['ETH'] # Ethereum

n_steps = 10 # number of time steps to include in the input features
splitting_date = '2020-10-22' # the date to split the training and testing data

X_btc_train, y_btc_train, X_btc_test, y_btc_test, = database.get_prepared_data(asset_name='BTC', splitting_date=splitting_date, n_steps=n_steps)
X_ltc_train, y_ltc_train, X_ltc_test, y_ltc_test, = database.get_prepared_data(asset_name='LTC', splitting_date=splitting_date, n_steps=n_steps)
X_eth_train, y_eth_train, X_eth_test, y_eth_test, = database.get_prepared_data(asset_name='ETH', splitting_date=splitting_date, n_steps=n_steps)


### 2) Data exploration and visualization

# database.plot_data(asset_name='BTC', splitting_date='2020-10-22') # Figure 2
# database.plot_data(asset_name='LTC', splitting_date='2020-10-22') # Figure 3
# database.plot_data(asset_name='ETH', splitting_date='2020-10-22') # Figure 4

# database.plot_all_data() # Figure 8

# database.plot_correlation(asset_names=['BTC', 'LTC', 'ETH']) # Figure 9


### 3) Training three types of models 
    # Gated recurrent unit (GRU) 
    # Bidirectional LSTM (bi-LSTM)
    # Training -> from 22/01/2018 until 22/10/2020 (80%)

    # Long short-term memory (LSTM) :

LTSM_model = LSTM(input_dim=n_steps, 
                  hidden_dim=50, 
                  output_dim=1, 
                  nb_layers=2)

X_btc_train_tensor = torch.tensor(X_btc_train, dtype=torch.float32)
y_btc_train_tensor = torch.tensor(y_btc_train, dtype=torch.float32).unsqueeze(1)
X_btc_test_tensor = torch.tensor(X_btc_test, dtype=torch.float32)
y_btc_test_tensor = torch.tensor(y_btc_test, dtype=torch.float32).unsqueeze(1)

optimizer = torch.optim.Adam(LTSM_model.parameters(), lr=0.01) # lr = learning rate 
loss_fct = nn.MSELoss()
LTSM_model.train_model(optimizer, y_btc_train_tensor, X_btc_train_tensor, loss_fct)
btc_predictions = LTSM_model.test_model(y_btc_train_tensor, X_btc_train_tensor, loss_fct)
print(btc_predictions)

plt.figure(figsize=(10, 5))  # Définit la taille du graphique
plt.plot(btc_predictions, label='Data')  # Trace les données avec une légende
plt.title('Example Plot')  # Ajoute un titre au graphique
plt.xlabel('Time')  # Nomme l'axe des x
plt.ylabel('Value')  # Nomme l'axe des y
plt.show()  # Affiche le graphique

### 4) Testing the models 
    # Testing -> from 22/10/2020 until 30/06/2021 (20%)

### 5) Extracting and comparing the results 