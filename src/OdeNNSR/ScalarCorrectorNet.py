import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalarCorrector(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out

        self.linear1 = nn.Linear(dim, dim * 2, dtype=torch.float64)
        self.linear2 = nn.Linear(dim * 2, dim * 2, dtype=torch.float64)
        self.linear3 = nn.Linear(dim * 2, dim_out, dtype=torch.float64)
        self.linearR = nn.Linear(dim, dim_out, dtype=torch.float64)

        linearLayers = [
            self.linear1,
            self.linear2,
            self.linear3,
            self.linearR,
        ]
        # for layer in linearLayers:
        #     nn.init.zeros_(layer.weight)
        #     nn.init.zeros_(layer.bias)

    def forward(self, u):
        v = F.tanh(self.linear1(u))
        v = F.tanh(self.linear2(v))
        w = self.linear3(v) 
        # return w * u[..., 0 : self.dim_out]
        return w
