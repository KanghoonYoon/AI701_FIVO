import torch as th


def ELBO(x, prior_mu, prior_std, enc_mu, enc_std, dec_mu, dec_std,
         seq_len, device='cuda', nll_type = 'bernoulli'):

    kld_loss = th.tensor([0], device)
    nll_loss = th.tensor([0], device)

    for t in range(seq_len):

        kld_loss += _kld_gauss(enc_mu[t], enc_std[t], prior_mu[t], prior_std[t])
        nll_loss += _nll_loss(x=x, params=(dec_mu, dec_std), nll_type=nll_type)

    ELBO = kld_loss + nll_loss

    return ELBO, kld_loss, nll_loss

def IWAE():

    pass


def FIVO():

    pass


def _kld_gauss(mu1, std1, mu2, std2):

    KL_div = (2*th.log(std2) - 2* th.log(std1) + (std1.powe(2) + (mu1 - mu2).pow(2))/ std2.pow(2)-1)

    return 0.5* th.sum(KL_div)


def _nll_loss(x, params:list, nll_type):

    """
    :param x:
    :param params: if bernoulli, params represent "theta" which is list with length 1
                    if gauss, params represent "mu, sigma" which is list with length 2
    :param nll_type: Bernoulli or Gauss
    :return: nll_loss
    """
    assert params is list
    assert len(params) < 3

    if nll_type == "bernoulli":

        return - th.sum(x*th.log(params[0]) + (1-x)*th.log(1-params[0]))

    elif nll_type == "gauss":

        return NotImplementedError
