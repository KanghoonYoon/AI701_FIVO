import numpy as np
import torch as th
from torch import nn as nn
from torch.nn import functional as F


class VRNN(nn.Module):

    def __init__(self):

        super(VRNN, self).__init__()

        self.prior_net = nn.Sequential()
        self.encoder = nn.Sequential()
        self.rnn = nn.GRU()
        self.decoder = nn.Sequential()

    def forward(self):

        h_prior = th.randn()

        z_prior = self.prior_net(h_prior)
        x_t = self.encoder(th.concat([h_prior, z_prior], dim=-1))
        h_next, _ = nn.rnn(th.concat([x_t, z_prior], dim=-1))

        z_posterior = self.decoder(th.concat([h_prior, x_t], dim=-1))





