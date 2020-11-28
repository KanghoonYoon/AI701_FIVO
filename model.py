import numpy as np
import torch as th
from torch import nn as nn
from torch.nn import functional as F


class VRNN(nn.Module):

    def __init__(self):

        super(VRNN, self).__init__()

        self.prior_mu = nn.Sequential(nn.Linear(), nn.ReLU(),
                                       nn.Linear(), nn.ReLU())
        self.prior_std = nn.Sequential(nn.Linear(), nn.ReLU(),
                                      nn.Linear(), nn.Softplus())

        self.encoder_mu = nn.Sequential(nn.Linear(), nn.ReLU(),
                                       nn.Linear(), nn.ReLU())
        self.encoder_std = nn.Sequential(nn.Linear(), nn.ReLU(),
                                       nn.Linear(), nn.Softplus())

        self.rnn = nn.GRU()

        self.decoder_mu = nn.Sequential(nn.Linear(), nn.ReLU(),
                                       nn.Linear(), nn.ReLU())
        self.decoder_std =  nn.Sequential(nn.Linear(), nn.ReLU(),
                                       nn.Linear(), nn.Softplus())


    def forward(self, x):

        h_prev = th.randn()

        for t in range(x.size(1)):

            mu_prior = self.prior_mu(h_prev[-1])
            std_prior = self.prior_std(h_prev[-1])

            z_prior = sample(mu_prior, mu_std)

            mu = self.encoder_mu(th.concat([h_prev[-1], z_prior], dim=-1))
            std = self.encoder_std(th.concat([h_prev[-1], z_prior], dim=-1))


            h_next, _ = nn.rnn(th.concat([x_t, z_prior], dim=-1))

            z_posterior = self.decoder(th.concat([h_prior, x_t], dim=-1))





