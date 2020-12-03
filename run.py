from data.data_utils import read_data, data2seq
from src.utils import mk_dir
from model import VRNN

from argparse import ArgumentParser
import numpy as np
import torch as th
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--model", type=str, default='VRNN')
    parser.add_argument("--loss_type", type=str, default='ELBO')
    parser.add_argument("--save_name", type=str, default=None)

    parser.add_argument("--decoder_dist", type=str, default="bernoulli")

    parser.add_argument("--input_dim", type=int, default=87) ## max_note - min_note - train_split(108 - 21 - 1 = 87)
    parser.add_argument("--h_dim", type=int, default=64)
    parser.add_argument("--z_dim", type=int, default=32)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--prt_evry", type=int, default=1)
    parser.add_argument("--save_evry", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)

    config = parser.parse_args()
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    dir_name = mk_dir(str(config.data) + 'experiment')

    data = read_data('data/pianorolls/piano-midi.de.pkl')
    train_data, test_data = data2seq(data=data, split='train')

    train_loader = iter(train_data)


    if config.model == "VRNN":
        model = VRNN(config)

    for idx, train_mat in train_loader:

        x = th.FloatTensor(train_mat, device=device)
        x = x.unsqueeze(dim=0) ## 1 batch tensor
        out = model(x)


    if config.model == 'VRNN':
        model = VRNN(config)
    else:
        print("NotImplementedERROR")

    trainloader = DataLoader(x, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.epochs):

        RANGE_LOSS = 0

        for x in trainloader:
            loss, kld, nll = model.fit(x, loss_type=config.loss_type, device=device)

            RANGE_LOSS += loss.item()

        if epoch%config.prt_evry:
            print("Training LOSS:{}, KLD:{}, NLL:{}".format(loss/len(train_data)*config.batch_size,
                                                            kld/len(train_data)*config.batch_size,
                                                            nll/len(train_data)*config.batch_size))
        if epoch%config.save_evry:
            th.save(model, dir_name + str(config.model) + config.save_name + '.pth')



