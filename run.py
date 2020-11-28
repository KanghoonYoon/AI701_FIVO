from src.utils import mk_dir
from model import VRNN

from argparse import ArgumentParser
import numpy as np
import torch as th
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--loss_type", type=str, default='ELBO')
    parser.add_argument("--save_name", type=str, default=None)

    parser.add_argument("--decoder_dist", type=str, default="bernoulli")
    parser.add_argument("--h_dim", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--prt_evry", type=int, default=1)
    parser.add_argument("--save_evry", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)

    config = parser.parse_args()
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    dir_name = mk_dir(str(config.data) + 'experiment')

    x = open('a', 'r')

    x = th.FloatTensor(x, device=device)

    if config.model == 'VRNN':
        model = VRNN(config)
    else:
        print("NotImplementedERROR")

    trainloader = DataLoader(x, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.epochs):

        RANGE_LOSS = 0

        for x in trainloader:
            loss, kld, nll = model(x)

            RANGE_LOSS += loss.item()

        if epoch%config.prt_evry:
            print("Training LOSS:{}, KLD:{}, NLL:{}".format(loss/len(traindata)*config.batch_size,
                                                            kld/len(traindata)*config.batch_size,
                                                            nll/len(traindata)*config.batch_size))
        if epoch%config.save_evry:
            th.save(model, dir_name + str(config.model) + config.save_name + '.pth')



