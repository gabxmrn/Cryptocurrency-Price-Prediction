import torch
import torch.nn as nn
import numpy as np

class Models(nn.Module):
    def __init__(self):
        super(self, Models).__init__()

    def get_rmse(self, predictions, targets) -> float :
        # return np.sqrt(mean_squared_error(targets, predictions))
        return np.sqrt(np.mean((targets - predictions)**2))
    
    def get_mape(self, predictions, targets) -> float :
        return np.mean(np.abs((targets - predictions) / targets)) * 100

    def train_model(self, train_loader, criterion, optimizer):
        self.train()  # Set the model to training mode
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def test_model(self, test_loader):
        self.eval()  # Set the model to evaluation mode
        predictions, targets = [], []
        with torch.no_grad():
            for inputs, target in test_loader:
                outputs = self(inputs)
                predictions.extend(outputs.numpy())
                targets.extend(target.numpy())
        return predictions, targets


class LSTM(Models):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, nb_layers: int):
        super(self, Models).__init__()
        self.hidden_dim = hidden_dim
        self.nb_layers = nb_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, nb_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.nb_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.nb_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])
