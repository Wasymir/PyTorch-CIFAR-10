import torch
from torch import nn
from torch import optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

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
        self.conv1 = nn.Sequential(nn.ZeroPad2d(2), nn.Conv2d(3, 32, 5), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(nn.ZeroPad2d(2), nn.Conv2d(32, 64, 5), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.output = nn.Linear(64 * 8 * 8, 10)

    def forward(self, input):
        x = self.conv1(input)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.output(x)
        return x


LR = 0.01
N_EPOCHS = 10

model = NN().to(device)
optimizer = optim.Adam(model.parameters(), LR)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(N_EPOCHS):
    print(f"Epoch: {epoch + 1} / {N_EPOCHS}")
    for step, (input, target) in enumerate(train_data):
        input, target = input.to(device), target.to(device)
        output = model(input)
        output = output.to(device)
        optimizer.zero_grad()
        loss = loss_fn(output, target)
        loss = loss.to(device)
        loss.backward()
        optimizer.step()
        print(f"Step: {step}; Loss: {loss.item()}")




