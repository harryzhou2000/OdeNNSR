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
        # nn.init.zeros_(self.linear1.weight)
        # nn.init.zeros_(self.linear2.weight)
        # nn.init.zeros_(self.linear3.weight)
        # nn.init.zeros_(self.linear1.bias)
        # nn.init.zeros_(self.linear2.bias)
        # nn.init.zeros_(self.linear3.bias)

    def forward(self, u):
        v = self.linear1(u)
        v = F.tanh(v)
        v = self.linear2(v)
        v = F.tanh(v)
        v = self.linear3(v)
        # return v * u[..., 0 : self.dim_out]
        return v
