import torch
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import PILToTensor
from torch.utils.data import Dataset

class PILDataSet(Dataset):
    def __init__(self, train=True, DS="MNIST"):
        self.train = train
        transform = PILToTensor()
        # Choose the underlying dataset based on DS (case-insensitive)
        if DS.upper() == "CIFAR10":
            self.dataset = CIFAR10("datasets", train=train, download=True, transform=transform)
        elif DS.upper() == "FASHION":
            self.dataset = FashionMNIST("datasets", train=train, download=True, transform=transform)
        elif DS.upper() == "MNIST":
            self.dataset = MNIST("datasets", train=train, download=True, transform=transform)
        else:
            raise ValueError("Unknown dataset: " + DS)

    def __getitem__(self, index):
        # Delegate fetching to the underlying dataset.
        return self.dataset[index]

    def __len__(self):
        # Delegate length to the underlying dataset.
        return len(self.dataset)
