import torch
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import PILToTensor

from torch.utils.data import Dataset

class PILDataSet(Dataset):
    def __init__(self, train=True, DS="MNIST"):
        # Initialize and load your data here.
        self.train = train
        # For example, load images and labels into a list or tensor:
        self.data = ...  # Your data loading logic
        self.labels = ...  # Your labels

    def __getitem__(self, index):
        # Retrieve one sample and its corresponding label
        image = self.data[index]
        label = self.labels[index]
        # You might need to do additional processing here.
        return image, label

    def __len__(self):
        return len(self.data)

