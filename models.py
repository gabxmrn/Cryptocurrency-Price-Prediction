import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Models(nn.Module):
    def __init__(self):
        super(Models, self).__init__()

    def get_rmse(self, predictions, targets) -> float :
        # return np.sqrt(mean_squared_error(targets, predictions))
        return np.sqrt(np.mean((targets - predictions)**2))
    
    def get_mape(self, predictions, targets) -> float :
        return np.mean(np.abs((targets - predictions) / targets)) * 100

    def train_model(self, optimizer, y, X, loss_fct):
        for t in range(10) :
            self.train()
            optimizer.zero_grad()
            outputs = self(X)
            loss = loss_fct(outputs, y)
            # print(f'loss : {loss}')
            loss.backward()
            optimizer.step()
    
    def test_model(self, y, X, loss_fct) :
        self.eval()  
        with torch.no_grad(): 
            predictions = self(X)  
            loss = loss_fct(predictions, y)  
            print(f'Loss: {loss.item()}')
        # return predictions 
        return predictions.cpu().detach().numpy()


class LSTM(Models):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, nb_layers: int):
        super(Models, self).__init__()
        self.hidden_dim = hidden_dim
        self.nb_layers = nb_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, nb_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a batch dimension if missing
        h0 = torch.zeros(self.nb_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.nb_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])
