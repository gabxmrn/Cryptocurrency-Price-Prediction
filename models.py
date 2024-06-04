import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, nb_layers: int):
        """ A simple LSTM (Long Short-Term Memory) network for sequence modeling.

        Args:
            input_dim (int): The size of input vocabulary.
            hidden_dim (int): The size of the hidden state, also the output dimension of the embedding layer.
            output_dim (int): The size of the output layer.
            n_layers (int): The number of stacked LSTM layers.
        """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = nb_layers
        
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, nb_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    

    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor containing sequences of indices, shaped (batch_size, sequence_length).

        Returns:
            out (torch.Vector): Output prediction of the mdoel, shaped (batch_size, output_dim).
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1,:])
        return out
