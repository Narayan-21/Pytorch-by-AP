import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Using MNIST dataset and interpreting the data as having 28 sequences and each sequence having 28 features.
# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2
# input_size => The number of expected features in the input x.
# hidden_size => The number of features in the hidden state a.
# In LSTM => a<t> != c<t>


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward Propagation
        out, _ = self.lstm(x, (h0,c0)) # _ refers (hidden state, cell state)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


# Load data
train_dataset = datasets.MNIST(
    root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(
    root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)

# Initialize Network
model = LSTM(input_size, hidden_size, num_layers, num_classes)

# Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for (data, targets) in train_loader:
        data = data.squeeze(1)  # The data from which we need to predict thing
        targets = targets  # The target value

        # forward part
        scores = model(data)  # Prediction of the model
        # Loss, that is cross entropy loss which is calculated given two args: 'predicted value' &'target value'
        loss = criterion(scores, targets)

        # Backward part
        optimizer.zero_grad()  # Setting the optimized GD to zero
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()

# Checking the model accuracy:


def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x=x.squeeze(1)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(
            f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
