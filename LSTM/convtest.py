import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from dataset import HDF5Dataset
from torch.utils.data import DataLoader

# from model import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():

    batch_size = 32
    input_size = 1000
    hidden_size = 10
    num_layers = 1

    conv1 = nn.Conv1d(1, 10, 2, stride=2)
    conv2 = nn.Conv1d(10, 100, 2, stride=2)
    conv3 = nn.Conv1d(100, 500, 2, stride=2)
    conv4 = nn.Conv1d(500, 1000, 10)
    l1 = nn.Linear(1000, 100)
    l2 = nn.Linear(100, 10)
    l3 = nn.Linear(10, 1)
    p1 = nn.AvgPool1d(3)
    p2 = nn.AvgPool1d(5)
    lstm = nn.LSTM(input_size, hidden_size, num_layers)

    a = torch.zeros((batch_size, 1, 6000))
    a = a.view(-1, 1, 6000)

    h0 = torch.randn(num_layers, batch_size, hidden_size)
    c0 = torch.randn(num_layers, batch_size, hidden_size)

    out_c1 = conv1(a)
    out_p1 = p1(out_c1)
    out_c2 = conv2(out_p1)
    out_p2 = p2(out_c2)
    out_c3 = conv3(out_p2)
    out_p3 = p2(out_c3)
    out_c4 = conv4(out_p3)
    out_c4 = out_c4.view(-1, batch_size, input_size)
    out_lstm, _ = lstm(out_c4, (h0, c0))

    print(f'shape data: {a.shape}\n'
          f'out_c1: {out_c1.shape}\n'
          f'out_p1: {out_p1.shape}\n'
          f'out_c2: {out_c2.shape}\n'
          f'out_p2: {out_p2.shape}\n'
          f'out_c3: {out_c3.shape}\n'
          f'out_p3: {out_p3.shape}\n'
          f'out_c4: {out_c4.shape}\n'
          f'out_lstm: {out_lstm.shape}')

    # train_dataset = HDF5Dataset('/home/ph/PycharmProjects/STEAD_ANN/MiniTrain.hdf5')
    # train_dataset = HDF5Dataset('../../PycharmProjects/STEAD_ANN/Train_data.hdf5')
    # trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    #
    # trace, label = next(iter(trainloader))
    # trace, label = next(iter(trainloader))
    #
    # print(label)
    # print(trace)
    # print(trace.shape)
    # print(type(trace))
    # print(type(trace.numpy()))
    #
    # plt.figure()
    # plt.plot(trace.squeeze().numpy())
    # plt.show()
    #
    # # eq, labels = next(iter(trainloader))
    # #
    # # print(f'input size: {eq.shape}')
    # #
    # # eq = torch.unsqueeze(eq, 0)
    # #
    # # print(f'input size: {eq.shape}')
    # #
    # # convl1 = nn.Conv1d(1, , 2, stride=2)
    # #
    # # eq = convl1(eq)
    # #
    # # print(f'output size: {eq.shape}')
    #
    # # data = torch.ones((5, 1, 6000))
    #
    # # net = ClassConv()
    #
    # # out = net(data)
    #
    # # print(out.shape)


if __name__ == "__main__":
    main()