import torch
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import PILToTensor
from torch.utils.data import Dataset

class PILDataSet(Dataset):
    def __init__(self, train=True, DS="MNIST"):
        self.train = train
        transform = PILToTensor()
        if DS.upper() == "CIFAR10":
            self.dataset = CIFAR10("datasets", train=train, download=True, transform=transform)
        elif DS.upper() == "FASHION":
            self.dataset = FashionMNIST("datasets", train=train, download=True, transform=transform)
        else:  # Default to MNIST
            self.dataset = MNIST("datasets", train=train, download=True, transform=transform)
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
