# import torch
# from torch.utils.data import Subset
# from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
# from torchvision.transforms import PILToTensor

# class PILDataSet(Subset):
#   def __init__(self, train, DS = None):
#     assert isinstance(train, bool), type(train)

#     if DS is None or "CIFAR10".startswith(DS): DS = CIFAR10("datasets", train=train, download=True)
#     if DS == "FASHION": DS = FashionMNIST("datasets", train=train, download=True) 
#     if DS == "MNIST": DS = MNIST("datasets", train=train, download=True)
#     self.DS = DS
#     self.p2t = PILToTensor()

#   def __len__(self):
#     return len(self.DS)

#   def __getitem__(self, item):
#     d, l = self.DS[item]
#     d = self.p2t(d)
#     return (d.to(torch.float32)/255, l)

#debugged and changed version 

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import PILToTensor

class PILDataSet(Dataset):
    def __init__(self, train, DS=None):
        # we wnat to choose dataset based on DS parameter (default to MNIST if DS=="MNIST")
        if DS is None or DS.upper() == "CIFAR10":
            ds_obj = CIFAR10("datasets", train=train, download=True)
        elif DS.upper() == "FASHION":
            ds_obj = FashionMNIST("datasets", train=train, download=True)
        elif DS.upper() == "MNIST":
            ds_obj = MNIST("datasets", train=train, download=True)
        else:
            raise ValueError("Unknown dataset type: " + str(DS))
        self.dataset = ds_obj
        self.p2t = PILToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        img, label = self.dataset[index]
        img = self.p2t(img)  #
        img = img.to(torch.float32) / 255.
        label = torch.tensor(label, dtype=torch.long)
        return img, label


