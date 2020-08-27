import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()

        # LSTM PARAMETERS
        self.input_size = 1000
        self.hidden_size = 1000
        self.num_layers = 10

        self.conv1 = nn.Conv1d(1, 10, 2, stride=2)
        self.conv2 = nn.Conv1d(10, 100, 2, stride=2)
        self.conv3 = nn.Conv1d(100, 500, 2, stride=2)
        self.conv4 = nn.Conv1d(500, 1000, 10)
        self.l1 = nn.Linear(1000, 100)
        self.l2 = nn.Linear(100, 10)
        self.l3 = nn.Linear(10, 1)
        self.p1 = nn.AvgPool1d(3)
        self.p2 = nn.AvgPool1d(5)
        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d(500)
        self.bn4 = nn.BatchNorm1d(1000)

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to('cuda:0')
        c0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to('cuda:0')

        wave = wave.view(-1, 1, 6000)
        wave = self.bn1(F.relu(self.conv1(wave)))
        wave = self.p1(wave)
        wave = self.bn2(F.relu(self.conv2(wave)))
        wave = self.p2(wave)
        wave = self.bn3(F.relu(self.conv3(wave)))
        wave = self.p2(wave)
        wave = self.bn4(F.relu(self.conv4(wave)))

        wave = wave.view(-1, batch_size, self.input_size)
        wave, _ = self.lstm(wave, (h0, c0))
        wave = wave.view(batch_size, self.hidden_size, 1)
        wave = wave.squeeze()

        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


# class CNNLSTM_v2(nn.Module):
#     def __init__(self):
#         super(CNNLSTM_v2, self).__init__()
#
#         self.conv1 = nn.Conv1d(1, 10, 2, stride=2)
#         self.conv2 = nn.Conv1d(10, 100, 2, stride=2)
#         self.conv3 = nn.Conv1d(100, 500, 2, stride=2)
#         self.conv4 = nn.Conv1d(500, 1000, 10)
#         self.l1 = nn.Linear(1000, 100)
#         self.l2 = nn.Linear(100, 10)
#         self.l3 = nn.Linear(10, 1)
#         self.p1 = nn.AvgPool1d(3)
#         self.p2 = nn.AvgPool1d(5)
#         self.bn1 = nn.BatchNorm1d(10)
#         self.bn2 = nn.BatchNorm1d(100)
#         self.bn3 = nn.BatchNorm1d(500)
#         self.bn4 = nn.BatchNorm1d(1000)
#         self.lstm = nn.LSTM(input_size=100, hidden_size=100, num_layers=1, batch_first=True)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, wave):
#         a = wave + self.hidden_size
#         return a
#
#
# class ClassConvBN(nn.Module):
#     def __init__(self):
#         super(ClassConvBN, self).__init__()
#
#         self.conv1 = nn.Conv1d(1, 10, 2, stride=2)
#         self.conv2 = nn.Conv1d(10, 100, 2, stride=2)
#         self.conv3 = nn.Conv1d(100, 500, 2, stride=2)
#         self.conv4 = nn.Conv1d(500, 1000, 10)
#         self.l1 = nn.Linear(1000, 100)
#         self.l2 = nn.Linear(100, 10)
#         self.l3 = nn.Linear(10, 1)
#         self.p1 = nn.AvgPool1d(3)
#         self.p2 = nn.AvgPool1d(5)
#         self.bn1 = nn.BatchNorm1d(10)
#         self.bn2 = nn.BatchNorm1d(100)
#         self.bn3 = nn.BatchNorm1d(500)
#         self.bn4 = nn.BatchNorm1d(1000)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, wave):
#         wave = wave.view(-1, 1, 6000)
#         wave = self.bn1(F.relu(self.conv1(wave)))
#         wave = self.p1(wave)
#         wave = self.bn2(F.relu(self.conv2(wave)))
#         wave = self.p2(wave)
#         wave = self.bn3(F.relu(self.conv3(wave)))
#         wave = self.p2(wave)
#         wave = self.bn4(F.relu(self.conv4(wave)))
#         wave = wave.squeeze()
#         wave = F.relu(self.l1(wave))
#         wave = F.relu(self.l2(wave))
#         wave = self.l3(wave)
#         return self.sigmoid(wave)
#
#
# class CBN_v2(nn.Module):
#     def __init__(self):
#         super(CBN_v2, self).__init__()
#
#         self.conv1 = nn.Conv1d(1, 10, 3, padding=1, stride=1)
#         self.conv2 = nn.Conv1d(10, 50, 3, padding=1, stride=2)
#         self.conv3 = nn.Conv1d(50, 100, 3, padding=1, stride=1)
#         self.conv4 = nn.Conv1d(100, 300, 3, padding=1, stride=2)
#         self.conv5 = nn.Conv1d(300, 500, 3, padding=1, stride=1)
#         self.conv6 = nn.Conv1d(500, 1000, 3, padding=1, stride=2)
#         self.conv7 = nn.Conv1d(1000, 1500, 3, padding=1, stride=1)
#         self.conv8 = nn.Conv1d(1500, 2000, 3, padding=1, stride=2)
#         self.l1 = nn.Linear(2000, 2000)
#         self.l2 = nn.Linear(2000, 1000)
#         self.l3 = nn.Linear(1000, 100)
#         self.l4 = nn.Linear(100, 10)
#         self.l5 = nn.Linear(10, 1)
#         self.p1 = nn.MaxPool1d(3)
#         self.p2 = nn.MaxPool1d(5)
#         self.p3 = nn.MaxPool1d(5)
#         self.p4 = nn.MaxPool1d(5)
#         self.bn1 = nn.BatchNorm1d(10)
#         self.bn2 = nn.BatchNorm1d(50)
#         self.bn3 = nn.BatchNorm1d(100)
#         self.bn4 = nn.BatchNorm1d(300)
#         self.bn5 = nn.BatchNorm1d(500)
#         self.bn6 = nn.BatchNorm1d(1000)
#         self.bn7 = nn.BatchNorm1d(1500)
#         self.bn8 = nn.BatchNorm1d(2000)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, wave):
#         wave = wave.view(-1, 1, 6000)
#         wave = self.bn1(F.relu(self.conv1(wave)))
#         wave = self.bn2(F.relu(self.conv2(wave)))
#         wave = self.p1(wave)
#         wave = self.bn3(F.relu(self.conv3(wave)))
#         wave = self.bn4(F.relu(self.conv4(wave)))
#         wave = self.p2(wave)
#         wave = self.bn5(F.relu(self.conv5(wave)))
#         wave = self.bn6(F.relu(self.conv6(wave)))
#         wave = self.p3(wave)
#         wave = self.bn7(F.relu(self.conv7(wave)))
#         wave = self.bn8(F.relu(self.conv8(wave)))
#         wave = self.p4(wave)
#         wave = wave.squeeze()
#         wave = F.relu(self.l1(wave))
#         wave = F.relu(self.l2(wave))
#         wave = F.relu(self.l3(wave))
#         wave = F.relu(self.l4(wave))
#         wave = self.l5(wave)
#         return self.sigmoid(wave)
