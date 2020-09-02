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

    a = torch.zeros((batch_size, 6000))
    a = a.view((batch_size, 1, 60, 100))

    conv1 = nn.Conv1d(1, 16, 3, padding=1, stride=1)
    conv2 = nn.Conv1d(16, 32, 3, padding=1, stride=2)
    conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
    conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

    p1 = nn.MaxPool1d(3)
    p2 = nn.MaxPool1d(5)

    bn1 = nn.BatchNorm1d(16)
    bn2 = nn.BatchNorm1d(32)
    bn3 = nn.BatchNorm1d(64)
    bn4 = nn.BatchNorm1d(128)

    lstm = nn.LSTM(128, 128, 10, batch_first=True)

    l1 = nn.Linear(128, 64)
    l2 = nn.Linear(64, 1)

    sigmoid = nn.Sigmoid()

    # Pasar cada una por una CNN pequeña
    out_convs = []

    for i in range(100):
        trozo = a[:, :, :, i]
        trozo = bn1(F.relu(conv1(trozo)))
        trozo = bn2(F.relu(conv2(trozo)))
        trozo = p1(trozo)
        trozo = bn3(F.relu(conv3(trozo)))
        trozo = bn4(F.relu(conv4(trozo)))
        trozo = p2(trozo)
        out_convs.append(trozo)

    # Concatenar las salidas
    out_convs = torch.cat(out_convs, dim=2)
    out_convs = out_convs.view(batch_size, 100, 128)

    out_lstm, _ = lstm(out_convs)

        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = l1(out_lstm)
        out_l2 = l2(out_l1)

    out = sigmoid(out_l2)

    print(f'out: {out.shape}')

    # conv1 = nn.Conv1d(1, 256, 3, padding=1, stride=1)
    # conv2 = nn.Conv1d(256, 256, 3, padding=1, stride=2)
    # conv3 = nn.Conv1d(256, 256, 3, padding=1, stride=2)
    # conv3 = nn.Conv1d(256, 256, 3, padding=1, stride=2)
    #
    # pool1 = nn.MaxPool1d(3)
    #
    # out_c1 = conv1(a)
    # out_c2 = conv2(out_c1)
    # out_p1 = pool1(out_c2)
    #
    # out_c3 = conv3(out_p1)

    # conv3 = nn.Conv1d(256, 256, 2, stride=2)

    # p1 = nn.AvgPool1d(3)
    # p2 = nn.AvgPool1d(5)

    # bn = nn.BatchNorm1d(256)

    # lstm = nn.LSTM(256, 256, 2, batch_first=True)

    # l1 = nn.Linear(256, 100)
    # l2 = nn.Linear(100, 1)

    # sigmoid = nn.Sigmoid()

    # out_c1 = conv1(a)
    # out_p1 = p1(out_c1)
    # out_c2 = conv2(out_p1)
    # out_p2 = p2(out_c2)
    # out_c3 = conv3(out_p2)
    # out_p3 = p2(out_c3)
    #
    # out_view = out_p3.view(batch_size, 10, 256)
    #
    # out_lstm1, (out_lstm2, out_lstm3) = lstm(out_view)
    #
    # wanted = out_lstm1[:, -1, :]

    # out_c4 = out_c4.view(1000, batch_size, 1)
    # out_lstm, _ = lstm(out_c4)

    # print(f'shape data: {a.shape}\n'
    #       f'out_c1: {out_c1.shape}\n'
    #       f'out_c2: {out_c2.shape}\n'
    #       f'out_p1: {out_p1.shape}\n'
    #       f'out_c3: {out_c3.shape}\n')
    # #       f'out_c2: {out_c2.shape}\n'
    # #       f'out_p2: {out_p2.shape}\n'
    # #       f'out_c3: {out_c3.shape}\n'
    # #       f'out_p3: {out_p3.shape}\n'
    # #       f'out_view: {out_view.shape}\n'
    # #       f'out_lstm1: {out_lstm1.shape}\n'
    # #       f'out_lstm2: {out_lstm2.shape}\n'
    # #       f'out_lstm3: {out_lstm3.shape}\n'
    # #       f'wanted: {wanted.shape}')
    #       # f'out_c4: {out_c4.shape}\n')
    #       # f'out_lstm: {out_lstm.shape}')

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
