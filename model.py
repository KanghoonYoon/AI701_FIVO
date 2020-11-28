import numpy as np
import torch as th
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from loss import ELBO, IWAE, FIVO

class VRNN(nn.Module):

    def __init__(self, config):

        super(VRNN, self).__init__()

        self.nll_type = config.decoder_dist

        self.h_dim = config.h_dim

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

        if self.nll_type == "gauss":
            self.decoder_std =  nn.Sequential(nn.Linear(), nn.ReLU(),
                                           nn.Linear(), nn.Softplus())

        self.optimizer = th.optim.Adam(self.parameters(), lr=config.lr)


    def forward(self, x):

        Z_t = []
        PRIOR_mu = []
        PRIOR_std = []
        ENC_mu = []
        ENC_std = []
        DEC_mu = []
        DEC_std = []

        h = th.randn((x.size(0), 1, self.h_dim))

        for t in range(x.size(1)):

            prior_mu = self.prior_mu(h[:,-1,:])
            prior_std = self.prior_std(h[:,-1,:])

            z_prior = self.reparam_sample(prior_mu, prior_std)

            z_mu = self.encoder_mu(th.concat([h[-1], z_prior], dim=-1))
            z_std = self.encoder_std(th.concat([h[-1], z_prior], dim=-1))

            z_t = self.reparam_sample(z_mu, z_std)

            x_hat_mu = self.decoder_mu(th.concat([h[:, -1, :], z_t], dim=-1))

            if self.nll_type == "gauss":
                x_hat_std = self.decoder_std(th.concat([h[:, -1, :]], dim=-1))

            h, _ = self.rnn(th.concat([x[:, t], z_t], dim=-1))

            PRIOR_mu.append(prior_mu)
            PRIOR_std.append(prior_std)
            ENC_mu.append(z_mu)
            ENC_mu.append(z_std)
            DEC_mu.append(x_hat_mu)
            if self.nll_type == "gauss":
                DEC_std.append(x_hat_std)


        return Z_t, PRIOR_mu, PRIOR_std, \
               ENC_mu, ENC_std, DEC_mu, DEC_std


    def reparam_sample(self, mu, std):

        eps = th.FloatTensor(std.size(0)).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu)


    def fit(self, batch, train=True):

        Z, prior_mu, prior_std, enc_mu, enc_std, dec_mu, dec_std = self.forward(batch)

        loss, kld, nll = ELBO(x, prior_mu=prior_mu, prior_std=prior_std, enc_mu=enc_mu, enc_std=enc_std,
                              dec_mu=dec_mu, dec_std=dec_std, seq_len=self.seq_len, device='cuda', nll_type =self.nll_type)


        if train:
            self.optimizer.zero_grad()
            loss.bacwkard()
            self.optimzier.step()

        return loss, kld, nll