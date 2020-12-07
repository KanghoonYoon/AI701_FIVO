import torch as th
from torch import nn as nn

def ELBO(x, prior_mu, prior_std, enc_mu, enc_std, dec_mu, dec_std, device='cuda', nll_type='bernoulli'):

    kld_loss = th.tensor([0.]).to(device)
    nll_loss = th.tensor([0.]).to(device)

    # for t in range(x.size(1)):
    #
    #     kld_loss += _kld_gauss(enc_mu[t], enc_std[t], prior_mu[t], prior_std[t])
    #
    #     if nll_type == 'bernoulli':
    #         nll_loss += _nll_loss(x=x[:, t, :], params=dec_mu[t], nll_type=nll_type)
    #
    #     elif nll_type == "gaussian":
    #         nll_loss += nll_loss(x=x[:, t, :], params=(dec_mu[t], dec_std[t]), nll_type=nll_type)

    kld_loss += _kld_gauss(enc_mu, enc_std, prior_mu, prior_std)

    if nll_type == 'bernoulli':
        nll_loss += _nll_loss(x=x, params=dec_mu, nll_type=nll_type)

    ELBO = kld_loss + nll_loss

    return ELBO, kld_loss, nll_loss

def IWAE():

    pass


def FIVO():

    pass


def _kld_gauss(mu1, std1, mu2, std2):

    KL_div = (2*th.log(std2) - 2* th.log(std1) + (std1.pow(2) + (mu1 - mu2).pow(2))/ std2.pow(2)-1)

    return 0.5* th.sum(KL_div)


def _nll_loss(x, params, nll_type):

    """
    :param x:
    :param params: if bernoulli, params represent "theta" which is list with length 1
                    if gauss, params represent "mu, sigma" which is list with length 2
    :param nll_type: Bernoulli or Gauss
    :return: nll_loss
    """

    assert len(params) < 3

    if nll_type == "bernoulli":

        return nn.BCELoss()(params, x)

    elif nll_type == "gauss":

        return NotImplementedError
