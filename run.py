from data.data_utils import read_data, data2seq
from src.utils import mk_dir
from model import VRNN

from argparse import ArgumentParser
import numpy as np
import torch as th
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data", type=str, default='piano')
    parser.add_argument("--model", type=str, default='VRNN')
    parser.add_argument("--loss_type", type=str, default='IWAE')
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--nll_type", type=str, default="bernoulli")
    parser.add_argument("--save_name", type=str, default='')

    parser.add_argument("--input_dim", type=int, default=88)  ## max_note - min_note - train_split(108 - 21 = 88)
    parser.add_argument("--h_dim", type=int, default=64)
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=10)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prt_evry", type=int, default=1)
    parser.add_argument("--save_evry", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    config = parser.parse_args()
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    dir_name = mk_dir(config.data + 'experiment')

    print(config, "DEVICE", device)

    if config.data == 'piano':
        data = read_data('data/pianorolls/piano-midi.de.pkl')

    train_data, test_data = data2seq(data=data, split='train', seq_len=config.seq_len)

    if config.model == "VRNN":
        model = VRNN(config, device)
    else:
        print("NotImplementedERROR")

    model.to(device)

    epoch = 0

    while (epoch < config.epochs):

        train_loader = iter(train_data)

        RANGE_LOSS1 = 0
        RANGE_LOSS2 = 0
        RANGE_LOSS3 = 0

        for idx, train_mat in train_loader:

            if idx % 20 == 0:
                print("{}/{} BATCH".format(idx + 1, len(train_data)))

            x = th.FloatTensor(train_mat).to(device)
            x = x.unsqueeze(dim=0)  ## 1 batch tensor
            loss, kld, nll = model.fit(x)

            RANGE_LOSS1 += loss.item() / x.size(1)
            RANGE_LOSS2 += kld.item() / x.size(1)
            RANGE_LOSS3 += nll.item() / x.size(1)

        if (epoch % config.prt_evry) == 0:
            print("-------------------------------------------------------------------------------------")
            print("{} EPOCH".format(epoch))
            print("Training LOSS:{}, KLD:{}, NLL:{}".format(-RANGE_LOSS1,
                                                            RANGE_LOSS2,
                                                            RANGE_LOSS3))
        if (epoch % config.save_evry) == 0:
            th.save(model, dir_name + '/' + str(config.model) + config.save_name +
                    '_' + str(epoch) + 'epoch'+ '.pth')

        epoch += 1








