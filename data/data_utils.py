import numpy as np
import torch as th
import pickle as pickle
from scipy.sparse import coo_matrix

def read_data(path):

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def calculate_mean(data):

    MEAN = 0
    tot_timesteps = 0
    for i in range(len(data)):
        mat, num_timesteps = pianoroll2array(data[i])

        tot_timesteps += num_timesteps
        np.mean(mat)

    return MEAN

def data2seq(data, split='train', seq_len=10, min_note=21, max_note=108):

    """
    sparse pianorolls -> dense numpy array [num_timesteps, num_notes]
    where if note j is activate on timestep i, give 1. Otherwise, 0

    :param data:
    :param split:
    :return:
    """
    data = data[split]

    # mean = calculate_mean(data)

    train_seq = []
    test_seq = []

    idx = 0
    for i in range(len(data)):
        mat, num_timesteps = pianoroll2array(data[i])

        train_seq.append((i, mat[:-1, :]))
        test_seq.append((i, mat[1:, :]))


        # train_seq.append((idx, mat[:-1, :]))
        # test_seq.append((idx, mat[1:, :]))

        # for t in range(num_timesteps-seq_len):
        #     train_seq.append((idx, mat[t:t+seq_len, :]))
        #     test_seq.append((idx, mat[t+1:t+seq_len, :]))
        #     idx+=1

    return train_seq, test_seq


def pianoroll2array(pianorolls, min_note=21, max_note=108):

    num_timesteps = len(pianorolls)
    inds = []
    for time, chord in enumerate(pianorolls):
        inds.extend((time, note - min_note) for note in chord)
    shape = [num_timesteps, (max_note - min_note + 1)]
    values = [1.] * len(inds)
    sparse_pianoroll = coo_matrix((values, ([x[0] for x in inds], [x[1] for x in inds])), shape=shape)

    return sparse_pianoroll.toarray(), num_timesteps

