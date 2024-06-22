import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    """
    General purpose RNN model class supporting LSTM, GRU, and bi-LSTM architectures.

    Attributes:
        hidden_dim (int): Dimensionality of the hidden state.
        layer_dim (int): Number of recurrent layers.
        model (nn.Module): The chosen RNN model from PyTorch's module library.
        fc (nn.Linear): Linear layer to map the hidden state output to the desired output dimension.
    """
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, model_type) -> None :
        """
        Initializes the RNN model with specified dimensions and type.

        Inputs:
            input_dim (int): Number of expected features in the input `x`.
            hidden_dim (int): Dimensionality of the hidden state.
            layer_dim (int): Number of recurrent layers.
            output_dim (int): Dimensionality of the output space.
            model_type (str): Type of RNN model ('LSTM', 'GRU', 'bi-LSTM').
        """
        
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
        
    def forward(self, x) :
        """
        Defines the forward pass of the model.

        Inputs:
            x (Tensor): Input data tensor containing features.

        Outputs:
            Tensor: Output data tensor from the model after applying the linear layer to the last hidden state.
        """
        out, _ = self.model(x)
        out = self.fc(out[:, -1, :])
        return out

    def train_model(self, epochs, loader, criterion, optimizer) -> None :
        """
        Trains the model using the provided parameters.

        Inputs:
            epochs (int): Number of training epochs.
            loader (DataLoader): DataLoader providing batches of input data.
            criterion (Loss): Loss function to use for training.
            optimizer (Optimizer): Optimizer to use for training.
        """
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
            
    def test_model(self, loader, scaler) -> np.array :
        """
        Tests the model and returns predictions.

        Inputs:
            loader (DataLoader): DataLoader providing batches of test data.
            scaler (Scaler): Scaler used to revert the data back to the original scale.

        Outputs:
            np.array: Inverse transformed array of model predictions.
        """
        
        self.eval()
        predictions = []
        with torch.no_grad():
            for features, labels in loader:
                outputs = self(features)
                predictions.extend(outputs.numpy())

        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))