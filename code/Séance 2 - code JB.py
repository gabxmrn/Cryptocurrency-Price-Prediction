import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import pandas as pd

# Fetching data from yfinance
data = yf.download(
    tickers="^FCHI",
    start="2021-01-07",
    end=datetime.now().strftime("%Y-%m-%d"),
    interval="1d",
)

# Convert the data to a pandas DataFrame
data_df = pd.DataFrame(data.values, columns=data.columns)

# Feature Engineering
data_df['Open-Close'] = data_df['Open'] - data_df['Close']
data_df['High-Low'] = data_df['High'] - data_df['Low']

# Target variable (next day's Close price)
data_df['Target'] = data_df['Close'].shift(-1)

# Dropping rows with NaN values
data_df.dropna(inplace=True)

# Selecting features and target
features = ['Open-Close', 'High-Low', 'Open', 'High', 'Low', 'Close']
X = data_df[features]
y = data_df['Target']

# Splitting the data into training and testing sets
split_ratio = 0.8
split = int(split_ratio * len(data_df))
Xtrain = torch.tensor(X[:split].values, dtype=torch.float32)
ytrain = torch.tensor(y[:split].values, dtype=torch.float32).unsqueeze(1)
Xtest = torch.tensor(X[split:].values, dtype=torch.float32)
ytest = torch.tensor(y[split:].values, dtype=torch.float32).unsqueeze(1)


# Define the neural network model
class FinancialModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FinancialModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


# Training function
def fit(model, loss_fn, Xtrain, ytrain, Xtest, ytest, optimizer, n_epochs):
    batch_size = 100
    mean_train_losses = []
    train_losses, test_losses = [], []
    for epoch in range(n_epochs):
        model.train()
        permutation = torch.randperm(Xtrain.size()[0])
        for i in range(0, Xtrain.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = Xtrain[indices], ytrain[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            train_losses.append(loss.item()/batch_size)
            loss.backward()
            optimizer.step()
        mean_train_losses.append(np.mean(train_losses))
        model.eval()
        with torch.no_grad():

            test_loss = loss_fn(model(Xtest), ytest)

            test_losses.append(test_loss.item())
            print(f'Epoch {epoch + 1}: Train Loss = {mean_train_losses}, Test Loss = {test_loss.item()}')

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')

    plt.title('Training and Testing Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig("loss.png")

    return train_losses, test_losses


# Model initialization and training
input_size = 6  # Based on the number of features used
hidden_size = 128  # Example hidden layer size
output_size = 1  # Predicting a single value
model = FinancialModel(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Assuming n_epochs is defined
n_epochs = 10
train_losses, test_losses = fit(model, loss_fn, Xtrain, ytrain, Xtest, ytest, optimizer, n_epochs)

# si loi normale >= std scalaire sinon minmax et on peut mettre des clips si les min / max varient dans le futur


