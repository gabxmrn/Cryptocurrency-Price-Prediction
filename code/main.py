from data import load_data, split_data, plot_data, plot_correlation
from LSTM import LSTM


# Import data
btc, ltc, eth = load_data("2018-01-01", "2021-06-30", "Close")

# Visualise data
plot_data(btc, "2020-10-22")
plot_data(ltc, "2020-10-22")
plot_data(eth, "2020-10-22")

# Correlation heathmap
plot_correlation(btc, ltc, eth)

# Split Data
btc_train, btc_test = split_data(btc, "2020-10-22")
ltc_train, ltc_test = split_data(ltc, "2020-10-22")
eth_train, eth_test = split_data(eth, "2020-10-22")
