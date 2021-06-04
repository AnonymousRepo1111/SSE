import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Linear', 'BMM']

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        
    def forward(self, x):
        return self.linear(x)

    def flops(self):
        #NOTE: We ignore activation funcitons.
        MAC = self.out_features * self.in_features
        ADD = 0
        if self.bias:
            ADD = self.out_features
        flops = 2 * MAC + ADD
        return flops

class BMM(nn.Module):
    def __init__(self):
        super(BMM, self).__init__()
        self.A_m = 0
        self.A_n = 0
        self.B_p = 0

    def forward(self, A, B):
        if not self.training:
            if self.A_m == 0:
                _, self.A_m, self.A_n = A.size()
                _, _, self.B_p = B.size()
        
        return torch.bmm(A, B)

    def flops(self):
        return 2 * self.A_m * self.A_n * self.B_p
