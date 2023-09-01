import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import PILToTensor

class PILDataSet(Subset):
  def __init__(self, train, DS = None):
    assert isinstance(train, bool), type(train)

    if DS is None or "CIFAR10".startswith(DS): DS = CIFAR10("datasets", train=train, download=True)
    if DS == "FASHION": DS = FashionMNIST("datasets", train=train, download=True) 
    if DS == "MNIST": DS = MNIST("datasets", train=train, download=True)
    self.DS = DS
    self.p2t = PILToTensor()

  def __len__(self):
    return len(self.DS)

  def __getitem__(self, item):
    d, l = self.DS[item]
    d = self.p2t(d)
    return (d.to(torch.float32)/255, l)

