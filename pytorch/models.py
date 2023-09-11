import torch
import torch.nn.functional as F

class BaseModel(torch.nn.Module):
    def __init__(self, H, W, C, categories):
        super(BaseModel, self).__init__()
        self.H, self.W, self.C = H, W, C
        self.categories = categories

        if not hasattr(self, "op"): self.op = None

    def input_dims(self):
        return (self.C, self.H, self.W)

    def forward(self, x):
        # return F.softmax(self.op(x), dim=1)
        return self.op(x)
    
    def children(self):
        return [c for c in self.op.children() if isinstance(c, (torch.nn.Linear, torch.nn.Conv2d))]

    
class MLP3(BaseModel):
    # 2 hidden layers, not counting the output layer.
    def __init__(self, widths = (300, 100),
                 H=32, W=32, C=3,
                 categories=10):
        assert len(widths) == 2, widths
        super(MLP3, self).__init__(H, W, C, categories)

        self.w1, self.w2 = widths

        from torch.nn import Flatten, Linear, ReLU, Sequential
        C_in = self.H * self.W * self.C
        self.op = Sequential(
            Flatten(),
            Linear(C_in,    self.w1, bias=True), ReLU(),
            Linear(self.w1, self.w2, bias=True), ReLU(),
            Linear(self.w2, self.categories, bias=True)
        )
        
        # Pytorch has different initialization from keras.
        for c in self.children():
            with torch.no_grad():
                torch.nn.init.xavier_normal_(c.weight)
                c.bias[:] = 0

