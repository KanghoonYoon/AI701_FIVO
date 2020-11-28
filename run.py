from src.utils import mk_dir

from argparse import ArgumentParser
import numpy as np
import torch as th


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data", type=str, default=None)

    parser.add_argument("--h_dim", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--prt_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)

    config = parser.parse_args()

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    dir_name = mk_dir(str(config.data) + 'experiment')

