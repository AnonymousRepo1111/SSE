import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Linear

__all__ = ['MeanAgg', 'SumAgg', 'CNPEncoder', 'CNPDecoder', 'CNP']

class MeanAgg(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.name = 'DeepSets/Mean'
        self.dim = dim

    def forward(self, x, split_size=None):
        return x.mean(dim=1)

class SumAgg(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.name = 'DeepSets/Sum'

    def forward(self, x, split_size=None):
        return x.sum(dim=1)

class CNPEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim, num_layers, aggregator):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.aggregator = aggregator

        self.encoder = []
        for i in range(num_layers - 1):
            if i == 0:
                in_dim = x_dim + y_dim
            else:
                in_dim = hidden_dim
            self.encoder.append(Linear(in_features=in_dim, out_features=hidden_dim))
            self.encoder.append(nn.ReLU(inplace=True))
        self.encoder.append(Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, context_x, context_y, split_size=None):
        x = torch.cat([context_x, context_y], dim=-1)
        h = self.encoder(x)
        representation = self.aggregator(h, split_size=split_size)
        return representation

class CNPDecoder(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim, num_layers):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.decoder = []
        for i in range(num_layers):
            if i == 0:
                in_dim = x_dim + hidden_dim
            else:
                in_dim = hidden_dim
            self.decoder.append(Linear(in_features=in_dim, out_features=hidden_dim))
            self.decoder.append(nn.ReLU(inplace=True))
        self.decoder.append(Linear(in_features=hidden_dim, out_features=2 * y_dim))
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, representation, target_x):
        representation = torch.unsqueeze(representation, dim=1).repeat(1, target_x.shape[1], 1)

        x = torch.cat([representation, target_x], dim=-1)
        h = self.decoder(x)
        mu, log_sigma = torch.split(h, self.y_dim, dim=-1)
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        dist = torch.distributions.normal.Normal(loc=mu, scale=sigma)
        return dist, mu, sigma

class CNP(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.name = 'CNP/' + encoder.aggregator.name
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_dim = self.encoder.hidden_dim

        self.num_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

    def forward(self, context_x, context_y, target_x, target_y=None, split_size=None):
        encoder = self.encoder(context_x, context_y, split_size=split_size)
        dist, mu, sigma = self.decoder(encoder, target_x)
       
        if target_y is not None:
            log_prob = dist.log_prob(target_y).sum(dim=1)
        else:
            log_prob = None
        output = {}
        
        output['loss'] = -log_prob
        output['mu'] = mu
        output['sigma'] = sigma
        return output
