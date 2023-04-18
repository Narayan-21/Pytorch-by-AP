# Imports
import torch
import torch.nn as nn
import torch.optim as optim # Package used to implement various optimization algorithms.
import torch.nn.functional as F # Contains various Nonlinear activation, convolutional functions etc.
from torch.utils.data import DataLoader, Dataset # dataset management
import torchvision.datasets as datasets # Provides many built-in datasets as well as utility to create your own dataset.
import torchvision.transforms as transforms

# Create fully connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50) # applies linear transformation on incoming data: y=x.AT+b
        self.fc2 = nn.Linear(50, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

#Load data
train_dataset = datasets.MNIST(root='dataset/',train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset= train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(
    root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size, shuffle=True)

# Initialize Network
model = NN(input_size=input_size, num_classes=num_classes)

# Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for (data, targets) in train_loader:
        data = data # The data from which we need to predict thing
        targets = targets  # The target value

        # Getting to correct shape eg. flattening the layer.
        data = data.reshape(data.shape[0], -1)

        # forward part
        scores = model(data) # Prediction of the model
        loss = criterion(scores, targets) # Loss, that is cross entropy loss which is calculated given two args: 'predicted value' &'target value'

        # Backward part
        optimizer.zero_grad() # Setting the optimized GD to zero
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
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)