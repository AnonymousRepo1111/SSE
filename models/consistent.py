import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models import SlotSetEncoder

__all__ = ['ConsistentAggregator']

def list2string(l):
    delim = '_'
    return delim.join(map(str, l))

class ConsistentAggregator(nn.Module):
    def __init__(self, K, h, d, d_hat, g='sum', ln=True, _slots='Random'):
        super(ConsistentAggregator, self).__init__()
        self.K = K                  #Number of slots in each stage
        self.h = h                  #The dimension of each slot
        self.d = d                  #Input dimension to each stage
        self.g = g                  #Choice of aggregation function g: sum, mean, max, min
        self.d_hat = d_hat          #Projection dimension in each stage
        self.ln = ln                #Use LayerNorm or Not

        self.name = 'Consistent/{}/{}/{}/{}/{}'.format(_slots, list2string(K), list2string(h), list2string(d), list2string(d_hat))

        self.enc = []
        for i in range(len(K)):
            self.enc.append(
                    SlotSetEncoder(K=K[i], h=h[i], d=d[i], d_hat=d_hat[i]),
                    )
        self.enc = nn.Sequential(*self.enc)

        if self.ln:
            self.norm = nn.LayerNorm(normalized_shape=d_hat[-1])
            self.name = '{}/{}'.format(self.name, 'LN')
        else:
            self.name = '{}/{}'.format(self.name, 'NO_LN')

    def forward(self, x, split_size=None):
        if split_size is None:
            enc = self.enc(x)
            if enc.size(1) > 1:
                if self.g == 'mean':
                    enc = enc.mean(dim=1, keepdims=True)
                elif self.g == 'sum':
                    enc = enc.sum(dim=1, keepdims=True)
                elif self.g == 'max':
                    enc, _ = enc.max(dim=1, keepdims=True)
                elif self.g == 'min':
                    enc, _ = enc.min(dim=1, keepdims=True)
                else:
                    raise NotImplementedError
            return self.norm(enc.squeeze(1)) if self.ln else enc.squeeze(1)
        else:
            B, _, _, device = *x.size(), x.device
            x = torch.split(x, split_size_or_sections=split_size, dim=1)
            
            enc = []
            for split in x:
                enc.append(self.enc(split))
            enc = torch.cat(enc, dim=1)
            if self.g == 'mean':
                enc = enc.mean(dim=1, keepdims=True)
            elif self.g == 'sum':
                enc = enc.sum(dim=1, keepdims=True)
            elif self.g == 'max':
                enc, _ = enc.max(dim=1, keepdims=True)
            elif self.g == 'min':
                enc, _ = enc.min(dim=1, keepdims=True)
            else:
                raise NotImplementedError
            return self.norm(enc.squeeze(1)) if self.ln else enc.squeeze(1)
