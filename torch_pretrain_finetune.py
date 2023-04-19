import torch
import torchvision
from torchvision.models import VGG16_Weights
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Hyperparameters
in_channel = 3
num_classes = 10
learning_rate = 0.001
batch_size = 1024
num_epochs = 5

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

#Load pre-trained model and modify it
model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity() # In-case we have multiple avgpool layers, we would use model.avgpool[i] instead
model.classifier = nn.Sequential(nn.Linear(512,100),
                                 nn.ReLU(),
                                 nn.Linear(100,10))
print(model)


# Load data
train_dataset = datasets.CIFAR10(
    root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)


# Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train Network
for epoch in range(num_epochs):
    losses = []

    for (data, targets) in train_loader:
        data = data  # The data from which we need to predict thing
        targets = targets  # The target value

        # forward part
        scores = model(data)  # Prediction of the model
        # Loss, that is cross entropy loss which is calculated given two args: 'predicted value' &'target value'
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # Backward part
        optimizer.zero_grad()  # Setting the optimized GD to zero
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()

    mean_loss = sum(losses)/len(losses)
    print(f'Loss at epoch {epoch} was {mean_loss:.5f}')

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
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(
            f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')


check_accuracy(train_loader, model)
