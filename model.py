from numpy import mod
import torch
from torch import nn
from torch import optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from statistics import mean
import os
from progress.bar import Bar

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = CIFAR10(root="data", download=True, train=True, transform=ToTensor())

test_dataset = CIFAR10(root="data", download=True, train=False, transform=ToTensor())

BATCH_SIZE = 1000

train_data, test_data = DataLoader(train_dataset, BATCH_SIZE), DataLoader(
    test_dataset, BATCH_SIZE
)


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.ZeroPad2d(1), nn.Conv2d(3, 32, 3), nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d(1), nn.Conv2d(32, 32, 3), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Sequential(nn.ZeroPad2d(1), nn.Conv2d(32, 64, 3), nn.ReLU())
        self.conv4 = nn.Sequential(nn.ZeroPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.conv5 = nn.Sequential(nn.ZeroPad2d(1), nn.Conv2d(64, 128, 3), nn.ReLU())
        self.conv6 = nn.Sequential(nn.ZeroPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU())
        self.pool3 = nn.MaxPool2d(2)
        # self.drop = nn.Dropout(0.1)
        self.fc1 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(128, 10))


    def forward(self, input):
        
        x = self.conv1(input)
        x = self.conv2(x)    
        x = self.pool1(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        
        x = torch.flatten(x, start_dim=1)
        # x = self.drop(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


LR = 0.005
N_EPOCHS = 20

model = NN().to(device)
optimizer = optim.Adam(model.parameters(), LR)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(N_EPOCHS):
    model.train()
    print(f"Epoch: {epoch + 1} / {N_EPOCHS}")
    bar = Bar("Training", max=len(train_data))
    for step, (input, target) in enumerate(train_data):
        bar.next()
        input, target = input.to(device), target.to(device)
        output = model(input)
        output = output.to(device)
        optimizer.zero_grad()
        loss = loss_fn(output, target)
        loss = loss.to(device)
        loss.backward()
        optimizer.step()
    bar.finish()
    print(f"Loss: {loss.item()}")

    model.eval()
    accuracies = []
    for step, (input, target) in enumerate(test_data):
        input, target = input.to(device), target.to(device)
        accuracies.append(0)
        output = model(input)
        for values, label in zip(output,target):
            if torch.argmax(values) == label:
                accuracies[-1] += 1
    avg_accuracy = mean(accuracies) / 10
    print(f"ACCURACY: {avg_accuracy}")
    print("-" * 20) 
