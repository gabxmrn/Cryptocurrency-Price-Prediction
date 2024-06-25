from data import DataProcessing
from models import Model
import tools

import torch
import torch.nn as nn
from datetime import datetime


### STEP 1 - Collecting historical cryptocurrency data

dataset = DataProcessing(file_name='Database', 
                          asset_names=['BTC', 'LTC', 'ETH'], 
                          start_date='2018-01-01', 
                          end_date='2021-06-30')

split_date = datetime.strptime('2020-10-22', '%Y-%m-%d')


### STEP 2 - Data exploration and visualization 

tools.plot_asset("BTC", dataset.data["BTC"], split_date, "Close")
tools.plot_asset("ETH", dataset.data["ETH"], split_date, "Close")
tools.plot_asset("LTC", dataset.data["LTC"], split_date, "Close")

tools.plot_all_assets(['BTC', 'LTC', 'ETH'], dataset.data, "Close")

tools.plot_correlation(['BTC', 'LTC', 'ETH'], dataset.data, "Close")


### STEP 3 - Deep Learning Models

models_args = {"sequence_lenght":10,
                "batch_size":64, 
                "input_dim": 1,
                "hidden_dim":64,
                "layer_dim":1,
                "output_dim":1,
                "epochs":75, 
                "criterion":nn.MSELoss()
            }

LSTM = Model(models_args["input_dim"], models_args["hidden_dim"], models_args["layer_dim"], models_args["output_dim"], model_type="LSTM")
GRU = Model(models_args["input_dim"], models_args["hidden_dim"], models_args["layer_dim"], models_args["output_dim"], model_type="GRU")
BI_LSTM = Model(models_args["input_dim"], models_args["hidden_dim"], models_args["layer_dim"], models_args["output_dim"], model_type="bi-LSTM")

models = {"LSTM":LSTM,
            "LSTM_optimizer":torch.optim.Adam(LSTM.parameters(), lr=0.01),
            "GRU":GRU,
            "GRU_optimizer":torch.optim.Adam(GRU.parameters(), lr=0.01),
            "BI_LSTM":BI_LSTM,
            "BI_LSTM_optimizer":torch.optim.Adam(BI_LSTM.parameters(), lr=0.01) 
        }

    # BTC 

btc_asset, btc_train_loader, btc_test_loader = dataset.get_data_loader("BTC", models_args["sequence_lenght"], split_date, models_args["batch_size"]) 

btc_args = {"asset":btc_asset, 
                "train_loader":btc_train_loader, 
                "test_loader":btc_test_loader,
                "scaler":btc_asset.get_scaler()
            }

btc_results = tools.run_process("BTC", btc_args, models, models_args, True)
print(btc_results)

    # ETH 
    
eth_asset, eth_train_loader, eth_test_loader = dataset.get_data_loader("ETH", models_args["sequence_lenght"], split_date, models_args["batch_size"]) 

eth_args = {"asset":eth_asset, 
                "train_loader":eth_train_loader, 
                "test_loader":eth_test_loader,
                "scaler":eth_asset.get_scaler()
            }

eth_results = tools.run_process("ETH", eth_args, models, models_args, True)
print(eth_results)

    # LTC    

ltc_asset, ltc_train_loader, ltc_test_loader = dataset.get_data_loader("LTC", models_args["sequence_lenght"], split_date, models_args["batch_size"]) 

ltc_args = {"asset":ltc_asset, 
                "train_loader":ltc_train_loader, 
                "test_loader":ltc_test_loader,
                "scaler":ltc_asset.get_scaler()
            }

ltc_results = tools.run_process("LTC", ltc_args, models, models_args, True)
print(ltc_results)