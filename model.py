import numpy as np
import torch as th
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from loss import ELBO, IWAE, FIVO

class VRNN(nn.Module):

    def __init__(self, config, device):

        super(VRNN, self).__init__()

        self.device = device

        self.loss_type = config.loss_type
        self.nll_type = config.nll_type

        self.input_dim = config.input_dim
        self.h_dim = config.h_dim
        self.z_dim = config.z_dim
        self.seq_len = config.seq_len

        self.prior_mu = nn.Sequential(nn.Linear(self.h_dim, self.h_dim), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.z_dim), nn.ReLU())
        self.prior_std = nn.Sequential(nn.Linear(self.h_dim, self.h_dim), nn.ReLU(),
                                      nn.Linear(self.h_dim, self.z_dim), nn.Softplus())

        self.encoder_mu = nn.Sequential(nn.Linear(self.h_dim + self.h_dim, self.h_dim), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.z_dim), nn.ReLU())
        self.encoder_std = nn.Sequential(nn.Linear( self.h_dim + self.h_dim, self.h_dim), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.z_dim), nn.Softplus())

        self.phi_x = nn.Sequential(nn.Linear(self.input_dim, self.h_dim), nn.ReLU())
        self.rnn = nn.GRU(input_size=self.h_dim+self.z_dim, hidden_size=self.h_dim, batch_first=True)


        self.decoder_mu = nn.Sequential(nn.Linear(self.h_dim + self.z_dim, self.h_dim), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.input_dim), nn.Sigmoid())

        if self.nll_type == "gauss":
            self.decoder_std = nn.Sequential(nn.Linear(), nn.ReLU(),
                                           nn.Linear(), nn.Softplus())

        self.optimizer = th.optim.Adam(self.parameters(), lr=config.lr)


    def forward(self, x):

        loss = 0

        Z_t = []
        PRIOR_mu = []
        PRIOR_std = []
        ENC_mu = []
        ENC_std = []
        DEC_mu = []
        DEC_std = []

        h = Variable(th.randn((x.size(0), 1, self.h_dim)).to(self.device))

        for t in range(x.size(1)):

            x_t = self.phi_x(x[:, t, :])

            prior_mu = self.prior_mu(h[:,-1,:])
            prior_std = self.prior_std(h[:,-1,:])

            z_prior = self.reparam_sample(prior_mu, prior_std)

            z_mu = self.encoder_mu(th.cat([h[:, -1, :], x_t], dim=-1))
            z_std = self.encoder_std(th.cat([h[:, -1, :], x_t], dim=-1))

            z_t = self.reparam_sample(z_mu, z_std)

            x_hat_mu = self.decoder_mu(th.cat([h[:, -1, :], z_t], dim=-1))

            if self.nll_type == "gaussian":
                x_hat_std = self.decoder_std(th.cat([h[:, -1, :]], dim=-1))


            _, h = self.rnn(th.cat([x_t, z_t], dim=-1).unsqueeze(dim=1))

            loss_t, kld, nll = ELBO(x[:, t, :], prior_mu, prior_std, z_mu, z_std, x_hat_mu, dec_std=None,
                                  device=self.device , nll_type=self.nll_type)

            loss += loss_t

            Z_t.append(z_t)
            PRIOR_mu.append(prior_mu)
            PRIOR_std.append(prior_std)
            ENC_mu.append(z_mu)
            ENC_std.append(z_std)
            DEC_mu.append(x_hat_mu)
            if self.nll_type == "gauss":
                DEC_std.append(x_hat_std)


        return loss, kld, nll, \
               Z_t, (ENC_mu, ENC_std), (DEC_mu, DEC_std)


    def reparam_sample(self, mu, std):

        eps = th.FloatTensor(std.size(0)).normal_().to(self.device)
        eps = Variable(eps)

        return eps.mul(std).add_(mu)


    def fit(self, batch, loss_type="ELBO", train=True, device='cuda'):

        loss, kld, nll, _, _, _  = self.forward(batch)

        # if loss_type == "ELBO":
        #     loss, kld, nll = ELBO(batch, prior_mu=prior_mu, prior_std=prior_std, enc_mu=enc_mu, enc_std=enc_std,
        #                           dec_mu=dec_mu, dec_std=dec_std, device=self.device, nll_type =self.nll_type)
        #
        # elif loss_type == "IWAE":
        #     print("NOT IMPLEMENTED ERROR")
        #
        # elif loss_type == "FIVO":
        #     print("NOT IMPLEMENTED ERROR")

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, kld, nll