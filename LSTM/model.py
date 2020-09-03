import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 10, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(10, 50, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(50, 100, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(100, 200, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(50)
        self.bn3 = nn.BatchNorm1d(100)
        self.bn4 = nn.BatchNorm1d(200)

        self.l1 = nn.Linear(200, 100)
        self.l2 = nn.Linear(100, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = wave.view(-1, 1, 6000)
        wave = self.bn1(F.relu(self.conv1(wave)))
        wave = self.bn2(F.relu(self.conv2(wave)))
        wave = self.p1(wave)
        wave = self.bn3(F.relu(self.conv3(wave)))
        wave = self.bn4(F.relu(self.conv4(wave)))
        wave = self.p2(wave)
        wave = wave.squeeze()
        wave = F.relu(self.l1(wave))
        wave = self.l2(wave)
        return self.sigmoid(wave)


class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()

        self.cnn = CNN()
        self.lstm = nn.LSTM(320, 64, 1, batch_first=True)
        self.l1 = nn.Linear(64, 1)

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 100, 1, 60)
        batch_size, timesteps, C, L = wave.size()
        c_in = wave.view(batch_size * timesteps, C, L)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.lstm(r_in)
        r_out2 = self.l1(r_out[:, -1, :])

        return F.sigmoid(r_out2)


class CNNLSTMANN(nn.Module):
    def __init__(self):
        super(CNNLSTMANN, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 128, 10, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.view(batch_size, 100, 128)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = self.l2(out_l1)

        return self.sigmoid(out_l2)


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

        self.lstm = nn.LSTM(1000, 1000, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]

        wave = wave.view(-1, 1, 6000)
        wave = self.bn1(F.relu(self.conv1(wave)))
        wave = self.p1(wave)
        wave = self.bn2(F.relu(self.conv2(wave)))
        wave = self.p2(wave)
        wave = self.bn3(F.relu(self.conv3(wave)))
        wave = self.p2(wave)
        wave = self.bn4(F.relu(self.conv4(wave)))

        wave, _ = self.lstm1(wave)
        # wave = wave.view(batch_size, -1, 1)
        # wave = wave.squeeze()

        wave = self.l1(wave[:, -1, :, :])
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
