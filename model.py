import torch
from torch import nn
from torch import optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = CIFAR10(root="data", download=True, train=True, transform=ToTensor())

test_dataset = CIFAR10(root="data", download=True, train=False, transform=ToTensor())

BATCH_SIZE = 100

train_data, test_data = DataLoader(train_dataset, BATCH_SIZE), DataLoader(
    test_dataset, BATCH_SIZE
)


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.ZeroPad2d(2), nn.Conv2d(3, 16, 5), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(nn.ZeroPad2d(2), nn.Conv2d(16, 32, 5), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.output = nn.Linear(32 * 8 * 8, 10)

    def forward(self, input):
        x = self.conv1(input)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        return x


