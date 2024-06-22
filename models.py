import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, model_type):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        if model_type == "LSTM":
            self.model = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        elif model_type == "GRU":
            self.model = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        elif model_type == "bi-LSTM":
            self.model = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)
            hidden_dim *= 2 
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.model(x)
        out = self.fc(out[:, -1, :])
        return out

    def train_model(self, epochs, loader, criterion, optimizer):
        self.train()
        for epoch in range(epochs):
            for i, (features, labels) in enumerate(loader):
                optimizer.zero_grad()
                outputs = self(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            # if epoch%10 == 0 :
            #     print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
            
    def test_model(self, loader, scaler):
        self.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for features, labels in loader:
                outputs = self(features)
                predictions.extend(outputs.numpy())
                actuals.extend(labels.numpy())

        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))