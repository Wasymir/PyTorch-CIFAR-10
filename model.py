import torch
from torch import nn
from torch import optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

train_dataset = CIFAR10(
    root="data",
    download=True,
    train=True,
    transform=ToTensor()
)

test_dataset = CIFAR10(
    root="data",
    download=True,
    train=False,
    transform=ToTensor()
)

