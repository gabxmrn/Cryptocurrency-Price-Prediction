import torch 
import torch.nn as nn
import pandas as pd
import numpy as np


class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1,:])
        return out
    

def train(model, optimizer, y_train, Xtrain, loss_fn):
    for epoch in range(100):
        model.train()

        optimizer.zero_grad()
        outputs = model(Xtrain)
        loss =  loss_fn(outputs, y_train)
        print(f'loss : {loss}')
        loss.backward()
        optimizer.step()


def load_data(dataset: pd.DataFrame, look_back: int):
    data_raw = dataset.values.astype('float32')
    data = []

    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data)
    X = data[:, :-1, :]
    y = data[:, -1, 0].reshape(-1, 1)

    return (X, y)


if __name__ == '__main':
    x_train = torch.randn((10000, 10, 3))
    y_train = torch.rand((10000,1))
    lstm = LSTM(2, 64, 1, 2)
    train(lstm, torch.optim.Adam(lstm.parameters(), lr=0.01), y_train, x_train, nn.MSELoss())