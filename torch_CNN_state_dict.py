import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10): # in_channel means the convolutional layers (for b&w images here 1 o.w. it is 3 for RGB)
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # Same convolution
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) # Pooling layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # Same Convolution
        self.fc1 = nn.Linear(16*7*7, num_classes) # 16*7*7 cause we will use the pooling layer twice in forward()
        # Here we defined various layers but have not implemented them yet.
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_dict'])

# Hyperparameters
in_channel = 1
num_classes = 10
learning_rate = 0.0001
batch_size = 1024
num_epochs = 10
load_model = True

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
model = CNN(in_channels=in_channel, num_classes=num_classes)

# Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'))

# Train Network
for epoch in range(num_epochs):
    losses=[]

    if epoch % 3 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer_dict': optimizer.state_dict()}
        save_checkpoint(checkpoint)

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
check_accuracy(test_loader, model)
